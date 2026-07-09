# SPDX-License-Identifier: Apache-2.0
"""Tests for the fused NPU MP gather/scatter override.

Pure-Python tests (slot-mapping construction, override installation, MLA/DSA
plane geometry and the layout-negotiation patch) run on any host.  Correctness
tests against the fused ``multi_layer_kv_transfer`` kernel require Ascend NPU
and are skipped otherwise.
"""

# Standard

# Third Party
import pytest
import torch

# First Party
from lmcache_ascend.v1.kv_format import KVCacheFormat
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
# MLA / DSA plane-geometry derivation (no NPU required)
# ---------------------------------------------------------------------------


def test_derive_plane_geometry_separate_kv():
    """SEPARATE_KV: hidden = num_heads * head_size, two equal planes."""
    k = torch.empty(4, 2, 8, 16)  # [NB, BS, NH, HS]
    v = torch.empty(4, 2, 8, 16)
    geo = npu_gather._derive_plane_geometry(KVCacheFormat.SEPARATE_KV, (k, v))
    assert geo.kv_lora_rank == 0
    assert geo.qk_rope_head_dim == 0
    assert geo.dsa_head_dim == 0
    assert geo.use_mla is False
    assert geo.staging_kv_lead == 2
    assert geo.hidden == 8 * 16


def test_derive_plane_geometry_mla():
    """MLA: hidden = kv_lora_rank + qk_rope_head_dim, one flat plane."""
    k = torch.empty(4, 2, 1, 512)  # [NB, BS, num_kv_heads, kv_lora_rank]
    v = torch.empty(4, 2, 1, 64)  # [NB, BS, num_kv_heads, qk_rope_head_dim]
    geo = npu_gather._derive_plane_geometry(KVCacheFormat.MLA_KV, (k, v))
    assert geo.kv_lora_rank == 512
    assert geo.qk_rope_head_dim == 64
    assert geo.dsa_head_dim == 0
    assert geo.use_mla is True
    assert geo.staging_kv_lead == 1
    assert geo.hidden == 576


def test_derive_plane_geometry_dsa():
    """DSA: hidden = kv_lora_rank + qk_rope_head_dim + dsa_head_dim."""
    k = torch.empty(4, 2, 1, 512)
    v = torch.empty(4, 2, 1, 64)
    dsa = torch.empty(4, 2, 1, 128)
    geo = npu_gather._derive_plane_geometry(KVCacheFormat.DSA_KV, (k, v, dsa))
    assert geo.kv_lora_rank == 512
    assert geo.qk_rope_head_dim == 64
    assert geo.dsa_head_dim == 128
    assert geo.use_mla is True
    assert geo.staging_kv_lead == 1
    assert geo.hidden == 704


def test_derive_plane_geometry_rejects_unsupported():
    with pytest.raises(ValueError):
        npu_gather._derive_plane_geometry(
            KVCacheFormat.MERGED_KV, (torch.empty(1),)
        )


# ---------------------------------------------------------------------------
# MLA / DSA SHM shape reconciliation (no NPU required)
# ---------------------------------------------------------------------------


def test_view_as_unifies_rank3_shm_slot_to_rank4_staging():
    """out[out_idx].view_as(staging): rank-3 server slot -> rank-4 kernel
    staging (MLA/DSA) is a zero-copy view; rank-4 (SEPARATE_KV) is identity."""
    num_layers, tokens, hidden = 3, 4, 576
    # MLA/DSA: server allocates rank-3 [L, tokens, hidden].
    shm_slot = torch.arange(
        num_layers * tokens * hidden, dtype=torch.float32
    ).reshape(num_layers, tokens, hidden)
    staging = torch.empty(1, num_layers, tokens, hidden, dtype=torch.float32)
    viewed = shm_slot.view_as(staging)
    assert tuple(viewed.shape) == (1, num_layers, tokens, hidden)
    # Zero-copy: writing through the view mutates the SHM slot storage.
    assert viewed.data_ptr() == shm_slot.data_ptr()

    # SEPARATE_KV: both rank-4 [2, L, tokens, H] -> view_as is identity.
    shm_sep = torch.empty(2, num_layers, tokens, 8)
    staging_sep = torch.empty(2, num_layers, tokens, 8)
    assert tuple(shm_sep.view_as(staging_sep).shape) == (2, num_layers, tokens, 8)


def test_scatter_canonicalises_rank3_src_for_mla():
    """scatter reshapes a rank-3 retrieved MLA chunk to [1, L, t, hidden]
    so the existing token-slice logic applies unchanged."""
    num_layers, tokens, hidden = 3, 4, 576
    src = torch.arange(
        num_layers * tokens * hidden, dtype=torch.float32
    ).reshape(num_layers, tokens, hidden)
    canonical = src.reshape(1, num_layers, -1, hidden)
    assert tuple(canonical.shape) == (1, num_layers, tokens, hidden)
    # Token slicing (the next scatter step) then works on the rank-4 view.
    sliced = canonical[:, :, 2:]
    assert tuple(sliced.shape) == (1, num_layers, tokens - 2, hidden)


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


def _make_mla_kv(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """DeepSeek-V2/V3 MLA: per-layer (k_cache, v_cache) with differing widths.

    k_cache[...,kv_lora_rank], v_cache[...,qk_rope_head_dim]; num_kv_heads is 1.
    """
    kv: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in range(num_layers):
        k = torch.randn(
            num_blocks, block_size, 1, kv_lora_rank, device=device, dtype=dtype
        )
        v = torch.randn(
            num_blocks, block_size, 1, qk_rope_head_dim, device=device, dtype=dtype
        )
        kv[f"layer_{layer}"] = (k, v)
    return kv


def _make_dsa_kv(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dsa_head_dim: int,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """DeepSeek-V3.2 DSA: MLA plus a third sparse-attention key plane."""
    kv: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for layer in range(num_layers):
        k = torch.randn(
            num_blocks, block_size, 1, kv_lora_rank, device=device, dtype=dtype
        )
        v = torch.randn(
            num_blocks, block_size, 1, qk_rope_head_dim, device=device, dtype=dtype
        )
        dsa = torch.randn(
            num_blocks, block_size, 1, dsa_head_dim, device=device, dtype=dtype
        )
        kv[f"layer_{layer}"] = (k, v, dsa)
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


@requires_npu
def test_gather_scatter_multichunk_roundtrip():
    """Batched gather+scatter across >=3 chunks restores the paged KV.

    Exercises the P1 batched path: all chunks share one paged->staging kernel
    launch (one concatenated slot-mapping), then per-chunk D2H; scatter reverses
    it. Verifies both the gathered CPU chunks (vs the reference) and that
    scattering them back restores every stored block.
    """
    blocks_per_chunk = 2
    block_ids = [0, 1, 3, 4, 6, 7]  # 3 chunks, non-contiguous blocks
    kv = _make_separate_kv(
        _NUM_LAYERS, _NUM_BLOCKS, _BLOCK_SIZE, _NUM_HEADS, _HEAD_SIZE, "npu"
    )
    kv_before = {k: (k_t.clone(), v_t.clone()) for k, (k_t, v_t) in kv.items()}
    desc = npu_gather._get_descriptor(kv)
    assert desc is not None

    chunks = npu_gather._npu_gather_paged_kv_to_cpu(
        desc, block_ids, blocks_per_chunk, out=None, chunk_indices=None
    )
    assert len(chunks) == 3

    # Gathered chunks must match the reference layout exactly.
    kv_cpu = {k: (k_t.cpu(), v_t.cpu()) for k, (k_t, v_t) in kv.items()}
    ref = _ref_gather(kv_cpu, block_ids, blocks_per_chunk, _NUM_LAYERS, _HIDDEN)
    for got, exp in zip(chunks, ref, strict=True):
        torch.testing.assert_close(got, exp)

    for k_t, v_t in kv.values():
        k_t.zero_()
        v_t.zero_()
    npu_gather._npu_scatter_cpu_to_paged_kv(
        desc, block_ids, chunks, blocks_per_chunk, skip_first_n_tokens=0
    )
    for k, (k_t, v_t) in kv.items():
        kb, vb = kv_before[k]
        for block in block_ids:
            torch.testing.assert_close(k_t[block].cpu(), kb[block].cpu())
            torch.testing.assert_close(v_t[block].cpu(), vb[block].cpu())


@requires_npu
def test_gather_subbatching_matches_single_batch(monkeypatch):
    """Forcing many sub-batches yields the same gathered chunks as one batch.

    Lowers the staging cap so each chunk becomes its own sub-batch (exercising
    the per-sub-batch slicing and per-chunk D2H offset math), then checks the
    result is identical to the default single-sub-batch gather.
    """
    blocks_per_chunk = 2
    block_ids = [0, 2, 3, 5, 6, 7]  # 3 chunks
    kv = _make_separate_kv(
        _NUM_LAYERS, _NUM_BLOCKS, _BLOCK_SIZE, _NUM_HEADS, _HEAD_SIZE, "npu"
    )
    desc = npu_gather._get_descriptor(kv)
    assert desc is not None

    expected = npu_gather._npu_gather_paged_kv_to_cpu(
        desc, block_ids, blocks_per_chunk, out=None, chunk_indices=None
    )

    # Force a fresh descriptor (the cache keys on data pointers) and a tiny cap
    # so each chunk is its own sub-batch.
    npu_gather._descriptor_cache.clear()
    desc2 = npu_gather._get_descriptor(kv)
    assert desc2 is not None and desc2 is not desc
    monkeypatch.setattr(npu_gather, "_STAGING_CAP_BYTES", 1)  # 1 byte -> 1 chunk/sub
    chunked = npu_gather._npu_gather_paged_kv_to_cpu(
        desc2, block_ids, blocks_per_chunk, out=None, chunk_indices=None
    )

    assert len(chunked) == len(expected)
    for got, exp in zip(chunked, expected, strict=True):
        torch.testing.assert_close(got, exp)


@requires_npu
def test_scatter_subbatching_restores_paged(monkeypatch):
    """Forcing many scatter sub-batches still restores the paged KV correctly."""
    blocks_per_chunk = 2
    block_ids = [1, 2, 4, 5, 6, 7]  # 3 chunks
    kv = _make_separate_kv(
        _NUM_LAYERS, _NUM_BLOCKS, _BLOCK_SIZE, _NUM_HEADS, _HEAD_SIZE, "npu"
    )
    kv_before = {k: (k_t.clone(), v_t.clone()) for k, (k_t, v_t) in kv.items()}
    desc = npu_gather._get_descriptor(kv)
    assert desc is not None

    chunks = npu_gather._npu_gather_paged_kv_to_cpu(
        desc, block_ids, blocks_per_chunk, out=None, chunk_indices=None
    )
    for k_t, v_t in kv.values():
        k_t.zero_()
        v_t.zero_()

    npu_gather._descriptor_cache.clear()
    desc2 = npu_gather._get_descriptor(kv)
    monkeypatch.setattr(npu_gather, "_STAGING_CAP_BYTES", 1)
    npu_gather._npu_scatter_cpu_to_paged_kv(
        desc2, block_ids, chunks, blocks_per_chunk, skip_first_n_tokens=0
    )

    for k, (k_t, v_t) in kv.items():
        kb, vb = kv_before[k]
        for block in block_ids:
            torch.testing.assert_close(k_t[block].cpu(), kb[block].cpu())
            torch.testing.assert_close(v_t[block].cpu(), vb[block].cpu())


# ---------------------------------------------------------------------------
# MLA / DSA correctness tests (skipped without Ascend NPU)
# ---------------------------------------------------------------------------

_MLA_KV_LORA_RANK = 512
_MLA_QK_ROPE_HEAD_DIM = 64
_DSA_HEAD_DIM = 128


@requires_npu
def test_descriptor_mla_fields():
    kv = _make_mla_kv(
        _NUM_LAYERS,
        _NUM_BLOCKS,
        _BLOCK_SIZE,
        _MLA_KV_LORA_RANK,
        _MLA_QK_ROPE_HEAD_DIM,
        "npu",
    )
    desc = npu_gather._NPUTransferDescriptor(list(kv.values()))
    assert desc.kv_format == KVCacheFormat.MLA_KV
    assert desc.kv_lora_rank == _MLA_KV_LORA_RANK
    assert desc.qk_rope_head_dim == _MLA_QK_ROPE_HEAD_DIM
    assert desc.dsa_head_dim == 0
    assert desc.hidden == _MLA_KV_LORA_RANK + _MLA_QK_ROPE_HEAD_DIM
    assert desc.staging_kv_lead == 1
    assert desc.use_mla is True
    assert desc.plane_extras == (_MLA_KV_LORA_RANK, _MLA_QK_ROPE_HEAD_DIM, 0, 0)
    # MLA is a 2-tuple -> 2 pointers per layer.
    assert desc.ptr_table.shape[0] == 2 * _NUM_LAYERS


@requires_npu
def test_descriptor_dsa_fields():
    kv = _make_dsa_kv(
        _NUM_LAYERS,
        _NUM_BLOCKS,
        _BLOCK_SIZE,
        _MLA_KV_LORA_RANK,
        _MLA_QK_ROPE_HEAD_DIM,
        _DSA_HEAD_DIM,
        "npu",
    )
    desc = npu_gather._NPUTransferDescriptor(list(kv.values()))
    assert desc.kv_format == KVCacheFormat.DSA_KV
    assert desc.hidden == _MLA_KV_LORA_RANK + _MLA_QK_ROPE_HEAD_DIM + _DSA_HEAD_DIM
    assert desc.staging_kv_lead == 1
    assert desc.plane_extras == (
        _MLA_KV_LORA_RANK,
        _MLA_QK_ROPE_HEAD_DIM,
        _DSA_HEAD_DIM,
        0,
    )
    # DSA is a 3-tuple -> 3 pointers per layer.
    assert desc.ptr_table.shape[0] == 3 * _NUM_LAYERS


@requires_npu
def test_scatter_gather_roundtrip_mla():
    """MLA gather -> scatter restores the original paged KV for stored blocks."""
    blocks_per_chunk = 2
    block_ids = [0, 3, 4, 7]
    kv = _make_mla_kv(
        _NUM_LAYERS,
        _NUM_BLOCKS,
        _BLOCK_SIZE,
        _MLA_KV_LORA_RANK,
        _MLA_QK_ROPE_HEAD_DIM,
        "npu",
    )
    kv_before = {k: (kt.clone(), vt.clone()) for k, (kt, vt) in kv.items()}
    desc = npu_gather._get_descriptor(kv)
    assert desc is not None

    chunks = npu_gather._npu_gather_paged_kv_to_cpu(
        desc, block_ids, blocks_per_chunk, out=None, chunk_indices=None
    )
    # MLA staging is [1, L, tokens, kv_lora_rank + qk_rope_head_dim].
    tokens = blocks_per_chunk * _BLOCK_SIZE
    assert chunks[0].shape == (
        1,
        _NUM_LAYERS,
        tokens,
        _MLA_KV_LORA_RANK + _MLA_QK_ROPE_HEAD_DIM,
    )

    for kt, vt in kv.values():
        kt.zero_()
        vt.zero_()
    npu_gather._npu_scatter_cpu_to_paged_kv(
        desc, block_ids, chunks, blocks_per_chunk, skip_first_n_tokens=0
    )

    for k, (kt, vt) in kv.items():
        kb, vb = kv_before[k]
        for block in block_ids:
            torch.testing.assert_close(kt[block].cpu(), kb[block].cpu())
            torch.testing.assert_close(vt[block].cpu(), vb[block].cpu())


@requires_npu
def test_scatter_gather_roundtrip_dsa():
    """DSA gather -> scatter restores all three planes for stored blocks."""
    blocks_per_chunk = 2
    block_ids = [0, 3, 4, 7]
    kv = _make_dsa_kv(
        _NUM_LAYERS,
        _NUM_BLOCKS,
        _BLOCK_SIZE,
        _MLA_KV_LORA_RANK,
        _MLA_QK_ROPE_HEAD_DIM,
        _DSA_HEAD_DIM,
        "npu",
    )
    kv_before = {
        k: (kt.clone(), vt.clone(), dt.clone())
        for k, (kt, vt, dt) in kv.items()
    }
    desc = npu_gather._get_descriptor(kv)
    assert desc is not None

    chunks = npu_gather._npu_gather_paged_kv_to_cpu(
        desc, block_ids, blocks_per_chunk, out=None, chunk_indices=None
    )
    tokens = blocks_per_chunk * _BLOCK_SIZE
    assert chunks[0].shape == (
        1,
        _NUM_LAYERS,
        tokens,
        _MLA_KV_LORA_RANK + _MLA_QK_ROPE_HEAD_DIM + _DSA_HEAD_DIM,
    )

    for kt, vt, dt in kv.values():
        kt.zero_()
        vt.zero_()
        dt.zero_()
    npu_gather._npu_scatter_cpu_to_paged_kv(
        desc, block_ids, chunks, blocks_per_chunk, skip_first_n_tokens=0
    )

    for k, (kt, vt, dt) in kv.items():
        kb, vb, db = kv_before[k]
        for block in block_ids:
            torch.testing.assert_close(kt[block].cpu(), kb[block].cpu())
            torch.testing.assert_close(vt[block].cpu(), vb[block].cpu())
            torch.testing.assert_close(dt[block].cpu(), db[block].cpu())


@requires_npu
def test_gather_into_rank3_out_then_scatter_roundtrip_mla():
    """SHM path end-to-end: gather MLA into a rank-3 [L, tokens, k+v] ``out``
    slot (the shape the server's MLA branch allocates), then scatter it back.

    Exercises the gather's ``out[out_idx].view_as(staging)`` rank-3->rank-4
    reconciliation and the scatter's ``reshape`` back, which the non-SHM
    pickle round-trip (rank-4 throughout) does not cover.
    """
    blocks_per_chunk = 2
    block_ids = [0, 3]
    kv = _make_mla_kv(
        _NUM_LAYERS,
        _NUM_BLOCKS,
        _BLOCK_SIZE,
        _MLA_KV_LORA_RANK,
        _MLA_QK_ROPE_HEAD_DIM,
        "npu",
    )
    kv_before = {k: (kt.clone(), vt.clone()) for k, (kt, vt) in kv.items()}
    desc = npu_gather._get_descriptor(kv)
    assert desc is not None

    # The server's MLA branch allocates a rank-3 [L, tokens, k+v] slot.
    tokens = blocks_per_chunk * _BLOCK_SIZE
    out = [
        torch.empty(
            _NUM_LAYERS,
            tokens,
            _MLA_KV_LORA_RANK + _MLA_QK_ROPE_HEAD_DIM,
            dtype=torch.bfloat16,
            device="cpu",
        )
    ]
    returned = npu_gather._npu_gather_paged_kv_to_cpu(
        desc, block_ids, blocks_per_chunk, out=out, chunk_indices=None
    )
    assert returned is out

    for kt, vt in kv.values():
        kt.zero_()
        vt.zero_()
    npu_gather._npu_scatter_cpu_to_paged_kv(
        desc, block_ids, out, blocks_per_chunk, skip_first_n_tokens=0
    )

    for k, (kt, vt) in kv.items():
        kb, vb = kv_before[k]
        for block in block_ids:
            torch.testing.assert_close(kt[block].cpu(), kb[block].cpu())
            torch.testing.assert_close(vt[block].cpu(), vb[block].cpu())
