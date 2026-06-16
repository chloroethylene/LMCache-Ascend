# SPDX-License-Identifier: Apache-2.0
"""Tests for the fused NPU MP gather/scatter override.

Pure-Python tests (slot-mapping construction, override installation and the
CPU fallback path) run on any host.  Correctness tests against the fused
``fused_multi_layer_kv_transfer`` kernel require Ascend NPU and are skipped
otherwise.
"""

# Standard
from typing import Optional

# Third Party
import pytest
import torch

# First Party
from lmcache_ascend.v1.multiprocess import npu_gather

_NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
requires_npu = pytest.mark.skipif(
    not _NPU_AVAILABLE, reason="requires Ascend NPU"
)


# ---------------------------------------------------------------------------
# Pure-Python tests (no NPU required)
# ---------------------------------------------------------------------------


def test_build_slot_mapping_dense_layout():
    """slot[t] = block_id * block_size + offset, in block-then-offset order."""
    sm = npu_gather._build_slot_mapping([3, 7], block_size=2, device="cpu")
    # block 3 -> 6,7 ; block 7 -> 14,15
    assert sm.tolist() == [6, 7, 14, 15]


def test_build_slot_mapping_single_block():
    sm = npu_gather._build_slot_mapping([11], block_size=4, device="cpu")
    assert sm.tolist() == [44, 45, 46, 47]


def test_build_slot_mapping_rejects_empty():
    with pytest.raises(ValueError):
        npu_gather._build_slot_mapping([], block_size=2, device="cpu")


def test_get_descriptor_is_none_for_cpu():
    """CPU kv_caches must not build an NPU descriptor (fallback to upstream)."""
    kv = {
        f"layer_{i}": (
            torch.empty(4, 2, 2, 8),
            torch.empty(4, 2, 2, 8),
        )
        for i in range(2)
    }
    assert npu_gather._get_descriptor(kv) is None


def test_install_overrides_patches_both_namespaces_and_falls_back():
    """install_overrides patches base+worker_transfer; CPU input hits upstream."""
    # Third Party
    import lmcache.v1.multiprocess.transfer_context.base as base
    import lmcache.v1.multiprocess.transfer_context.worker_transfer as wt

    npu_gather.install_overrides()
    assert base.gather_paged_kv_to_cpu is npu_gather._gather_wrapper
    assert wt.gather_paged_kv_to_cpu is npu_gather._gather_wrapper
    assert base.scatter_cpu_to_paged_kv is npu_gather._scatter_wrapper

    # CPU kv_caches -> descriptor None -> wrapper must delegate to _orig_gather.
    called: dict[str, bool] = {}
    real = npu_gather._orig_gather

    def spy(*_args: object, **_kwargs: object) -> list[torch.Tensor]:
        called["yes"] = True
        return []

    npu_gather._orig_gather = spy  # type: ignore[assignment]
    try:
        kv = {"layer_0": (torch.empty(2, 2, 2, 2), torch.empty(2, 2, 2, 2))}
        base.gather_paged_kv_to_cpu(kv, [0], 1)
        assert called.get("yes") is True
    finally:
        npu_gather._orig_gather = real  # type: ignore[assignment]


def test_install_overrides_is_idempotent():
    """Calling install_overrides twice must not nest wrappers over the original."""
    npu_gather.install_overrides()
    first_orig = npu_gather._orig_gather
    npu_gather.install_overrides()
    assert npu_gather._orig_gather is first_orig


# ---------------------------------------------------------------------------
# Reference gather (mirrors the upstream TWO_X layout, on CPU)
# ---------------------------------------------------------------------------


def _ref_gather(
    kv_caches: dict[str, tuple[torch.Tensor, torch.Tensor]],
    block_ids: list[int],
    blocks_per_chunk: int,
    num_layers: int,
    hidden: int,
) -> list[torch.Tensor]:
    num_chunks = len(block_ids) // blocks_per_chunk
    out: list[torch.Tensor] = []
    for c in range(num_chunks):
        cbi = block_ids[c * blocks_per_chunk : (c + 1) * blocks_per_chunk]
        idx = torch.tensor(cbi, dtype=torch.long)
        k_layers, v_layers = [], []
        for layer in range(num_layers):
            k_t, v_t = kv_caches[f"layer_{layer}"]
            tokens = len(cbi) * k_t.shape[1]
            k_layers.append(k_t[idx].reshape(tokens, hidden))
            v_layers.append(v_t[idx].reshape(tokens, hidden))
        k_stacked = torch.stack(k_layers, dim=0)
        v_stacked = torch.stack(v_layers, dim=0)
        out.append(torch.stack([k_stacked, v_stacked], dim=0))
    return out


def _make_separate_kv(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_heads: int,
    head_size: int,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    kv: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in range(num_layers):
        shape = (num_blocks, block_size, num_heads, head_size)
        k = torch.randn(*shape, device=device, dtype=dtype)
        v = torch.randn(*shape, device=device, dtype=dtype)
        kv[f"layer_{layer}"] = (k, v)
    return kv


# ---------------------------------------------------------------------------
# NPU correctness tests (skipped without Ascend NPU)
# ---------------------------------------------------------------------------


_NUM_LAYERS = 3
_NUM_BLOCKS = 8
_BLOCK_SIZE = 2
_NUM_HEADS = 4
_HEAD_SIZE = 8
_HIDDEN = _NUM_HEADS * _HEAD_SIZE


@requires_npu
def test_descriptor_separate_kv_fields():
    kv = _make_separate_kv(
        _NUM_LAYERS, _NUM_BLOCKS, _BLOCK_SIZE, _NUM_HEADS, _HEAD_SIZE, "npu"
    )
    desc = npu_gather._NPUTransferDescriptor(list(kv.values()))
    assert desc.kv_format.value == 2  # SEPARATE_KV
    assert desc.block_size == _BLOCK_SIZE
    assert desc.page_buffer_size == _NUM_BLOCKS * _BLOCK_SIZE
    assert desc.num_layers == _NUM_LAYERS
    assert desc.hidden == _HIDDEN
    assert desc.use_mla is False
    # Interleaved [k0, v0, ...] -> 2 pointers per layer.
    assert desc.ptr_table.shape[0] == 2 * _NUM_LAYERS
    assert desc.ptr_table.device.type == "npu"


@requires_npu
def test_gather_matches_reference():
    blocks_per_chunk = 2
    block_ids = [0, 1, 4, 5]  # two chunks
    kv = _make_separate_kv(
        _NUM_LAYERS, _NUM_BLOCKS, _BLOCK_SIZE, _NUM_HEADS, _HEAD_SIZE, "npu"
    )
    desc = npu_gather._get_descriptor(kv)
    assert desc is not None

    chunks = npu_gather._npu_gather_paged_kv_to_cpu(
        desc, block_ids, blocks_per_chunk, out=None, chunk_indices=None
    )

    kv_cpu = {k: (k_t.cpu(), v_t.cpu()) for k, (k_t, v_t) in kv.items()}
    ref = _ref_gather(kv_cpu, block_ids, blocks_per_chunk, _NUM_LAYERS, _HIDDEN)

    assert len(chunks) == len(ref)
    for got, want in zip(chunks, ref):
        assert got.shape == want.shape
        torch.testing.assert_close(got, want)


@requires_npu
def test_gather_into_out_with_chunk_indices():
    """SHM-style path: preallocated `out` filled only for `chunk_indices`."""
    blocks_per_chunk = 2
    block_ids = [0, 1, 4, 5]
    kv = _make_separate_kv(
        _NUM_LAYERS, _NUM_BLOCKS, _BLOCK_SIZE, _NUM_HEADS, _HEAD_SIZE, "npu"
    )
    desc = npu_gather._get_descriptor(kv)
    assert desc is not None

    tokens = blocks_per_chunk * _BLOCK_SIZE
    out = [
        torch.empty(2, _NUM_LAYERS, tokens, _HIDDEN, dtype=torch.bfloat16)
    ]
    returned = npu_gather._npu_gather_paged_kv_to_cpu(
        desc, block_ids, blocks_per_chunk, out=out, chunk_indices=[1]
    )
    assert returned is out

    kv_cpu = {k: (k_t.cpu(), v_t.cpu()) for k, (k_t, v_t) in kv.items()}
    ref = _ref_gather(kv_cpu, block_ids, blocks_per_chunk, _NUM_LAYERS, _HIDDEN)
    # chunk_indices=[1] -> out[0] holds chunk index 1 (the reference's [1]).
    torch.testing.assert_close(out[0], ref[1])


@requires_npu
def test_scatter_gather_roundtrip():
    """gather then scatter restores the original paged KV for stored blocks."""
    blocks_per_chunk = 2
    block_ids = [0, 3, 4, 7]
    kv = _make_separate_kv(
        _NUM_LAYERS, _NUM_BLOCKS, _BLOCK_SIZE, _NUM_HEADS, _HEAD_SIZE, "npu"
    )
    kv_before = {
        k: (k_t.clone(), v_t.clone()) for k, (k_t, v_t) in kv.items()
    }
    desc = npu_gather._get_descriptor(kv)
    assert desc is not None

    chunks = npu_gather._npu_gather_paged_kv_to_cpu(
        desc, block_ids, blocks_per_chunk, out=None, chunk_indices=None
    )

    # Zero the stored blocks, then scatter the gathered chunks back.
    for k_t, v_t in kv.values():
        k_t.zero_()
        v_t.zero_()
    npu_gather._npu_scatter_cpu_to_paged_kv(
        desc, block_ids, chunks, blocks_per_chunk, skip_first_n_tokens=0
    )

    # Only the blocks in ``block_ids`` are written back by scatter; the rest
    # stay zeroed, so compare just the stored blocks against the original.
    for k, (k_t, v_t) in kv.items():
        kb, vb = kv_before[k]
        for block in block_ids:
            torch.testing.assert_close(k_t[block].cpu(), kb[block].cpu())
            torch.testing.assert_close(v_t[block].cpu(), vb[block].cpu())


@requires_npu
def test_scatter_skip_first_n_tokens():
    """skip_first_n_tokens (block-aligned) leaves the leading blocks untouched."""
    blocks_per_chunk = 2
    block_ids = [0, 1]
    kv = _make_separate_kv(
        _NUM_LAYERS, _NUM_BLOCKS, _BLOCK_SIZE, _NUM_HEADS, _HEAD_SIZE, "npu"
    )
    desc = npu_gather._get_descriptor(kv)
    assert desc is not None

    chunks = npu_gather._npu_gather_paged_kv_to_cpu(
        desc, block_ids, blocks_per_chunk, out=None, chunk_indices=None
    )

    # Snapshot, zero, then scatter skipping the first block (_BLOCK_SIZE tokens).
    kv_before = {k: (k_t.clone(), v_t.clone()) for k, (k_t, v_t) in kv.items()}
    for k_t, v_t in kv.values():
        k_t.zero_()
        v_t.zero_()
    npu_gather._npu_scatter_cpu_to_paged_kv(
        desc,
        block_ids,
        chunks,
        blocks_per_chunk,
        skip_first_n_tokens=_BLOCK_SIZE,
    )

    first_block = block_ids[0]
    for layer in range(_NUM_LAYERS):
        k_t, _ = kv[f"layer_{layer}"]
        kb, _ = kv_before[f"layer_{layer}"]
        # Skipped block must remain zeroed (untouched by scatter).
        assert torch.equal(
            k_t[first_block].cpu(), torch.zeros_like(k_t[first_block].cpu())
        )
        # The non-skipped block must match the original.
        torch.testing.assert_close(
            k_t[block_ids[1]].cpu(), kb[block_ids[1]].cpu()
        )
