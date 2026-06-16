# SPDX-License-Identifier: Apache-2.0
"""Fused NPU gather/scatter for the multiprocess non-GPU transfer path.

The upstream multiprocess (MP) ``DataTransferContext`` moves KV cache between
a worker's paged NPU memory and CPU chunks via the generic, per-layer PyTorch
helpers :func:`gather_paged_kv_to_cpu` / :func:`scatter_cpu_to_paged_kv`
(``lmcache/v1/multiprocess/transfer_context/base.py``).  Those helpers are
O(num_chunks * num_layers) kernel launches with several intermediate
allocations each — they never reach the fused Ascend transfer kernel that the
in-process connector already uses.

This module monkey-patches those two callables (on both the ``base`` and the
``worker_transfer`` namespaces) so that, **for SEPARATE_KV caches on 910B/C
NPU**, each chunk's paged transfer is a single
:func:`lmcache_ascend.c_ops.multi_layer_kv_transfer` call between the paged KV
and an NPU staging buffer; the host leg (D2H/H2D) is done with ordinary torch
``copy_``/``cpu`` so it works with any host buffer (SHM views or freshly
allocated CPU tensors).  The fused variant was avoided because Ascend
``aclrtMemcpyAsync`` requires registered/pinned host memory, which the MP SHM
and pickle buffers are not.

Unsupported cases (CPU/other devices, 310P, MLA, DSA family, SGLang) fall
through to the original upstream implementation unchanged.

.. note::
   MLA_KV is intentionally routed to the original path for now: the upstream
   ``compute_kv_layout`` reports only the K-plane hidden dim for Ascend
   ``(K, V)`` tuples (both map to ``TWO_X_NL_X_NB_BS_NH_HS``), so the
   chunk-shape metadata the SHM server allocates does not match the kernel's
   ``kv_lora_rank + qk_rope_head_dim`` layout.  Fixing MLA additionally
   requires correcting that shape contract; tracked as a follow-up.

Heavy dependencies (``c_ops``, the NPU connector helpers) are imported lazily
so the module and its pure-Python helpers stay importable on hosts without a
built extension — this keeps the slot-mapping and fallback logic unit-testable.
"""

# Standard
from typing import Optional

# Third Party
from lmcache.logging import init_logger
import torch

# First Party
from lmcache_ascend.v1.kv_format import KVCacheFormat

logger = init_logger(__name__)

# Formats whose chunk-shape contract is compatible with the fused kernel.
# MLA_KV is excluded for now — see the module docstring.
_SUPPORTED_FORMATS: tuple[KVCacheFormat, ...] = (KVCacheFormat.SEPARATE_KV,)

# Reusable per-worker transfer descriptors, keyed by the full data-pointer
# signature of the paged KV tensors. In production ``kv_caches`` is registered
# once per worker and lives for the engine's lifetime, so the signature is
# stable and the descriptor is reused across store/retrieve calls. Keying on
# the full signature (rather than ``id(kv_caches)``) avoids stale hits when a
# Python object id or a device address is reused — e.g. tests that build and
# destroy KV caches back-to-back in one process.
_descriptor_cache: dict[tuple[int, ...], "_NPUTransferDescriptor"] = {}


def _descriptor_signature(kv_caches: dict[str, object]) -> tuple[int, ...]:
    """Tuple of every paged-tensor data pointer, in layer-then-plane order."""
    sig: list[int] = []
    for value in kv_caches.values():
        if isinstance(value, (tuple, list)):
            for tensor in value:
                sig.append(tensor.data_ptr())  # type: ignore[union-attr]
        else:
            sig.append(value.data_ptr())  # type: ignore[union-attr]
    return tuple(sig)


def _first_layer_tensor(layers: list[object]) -> torch.Tensor:
    """Return the representative tensor of the first per-layer entry."""
    first = layers[0]
    if isinstance(first, (tuple, list)):
        return first[0]  # type: ignore[no-any-return]
    return first  # type: ignore[no-any-return]


class _NPUTransferDescriptor:
    """Precomputed kernel inputs for one worker's paged KV cache.

    Built once per ``kv_caches`` registration and reused across store/retrieve
    calls so the device-resident pointer table and the staging buffer are not
    rebuilt per chunk.

    Attributes:
        device: NPU device of the paged KV tensors.
        kv_format: Detected Ascend :class:`KVCacheFormat`.
        ptr_table: Flat int64 NPU tensor of interleaved per-layer (K, V) data
            pointers in the order ``_pointers_for_entry`` produces
            (``[k0, v0, k1, v1, ...]``).
        block_size: vLLM block size (tokens per block).
        page_buffer_size: ``num_blocks * block_size`` (slots per layer).
        num_layers: Number of KV layers.
        hidden: Contiguous-buffer hidden dim (``num_heads * head_size`` for
            SEPARATE_KV).
        use_mla: Always ``False`` for the currently supported formats.
        kv_lora_rank / qk_rope_head_dim: MLA plane widths (0 for SEPARATE_KV).
        dtype: Element dtype of the paged KV / contiguous buffer.
    """

    def __init__(self, layers: list[object]) -> None:
        # First Party — lazy: avoids importing the connector / c_ops at module
        # import time so this module stays unit-testable on non-NPU hosts.
        from lmcache_ascend.v1.npu_connector.npu_connectors import _pointers_for_entry

        first = layers[0]
        if not isinstance(first, (tuple, list)):
            # Bare-tensor entries (SGLang NPU 5-D, MERGED, ...) are not handled
            # here; let the caller fall back to the upstream path.
            raise ValueError("NPU fused gather/scatter requires per-layer tuples")

        ref = _first_layer_tensor(layers)
        self.device: torch.device = ref.device
        self.dtype: torch.dtype = ref.dtype
        self.kv_format: KVCacheFormat = KVCacheFormat.detect(layers)

        if self.kv_format not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"NPU fused gather/scatter does not support {self.kv_format.name}"
            )

        k0 = first[0]
        self.block_size: int = int(k0.shape[1])
        num_blocks = int(k0.shape[0])
        self.page_buffer_size: int = num_blocks * self.block_size
        self.num_layers: int = len(layers)

        # SEPARATE_KV: K and V share shape [num_blocks, block_size, nh, hs].
        self.use_mla: bool = False
        self.kv_lora_rank: int = 0
        self.qk_rope_head_dim: int = 0
        self.hidden: int = int(k0.shape[2]) * int(k0.shape[3])

        # Interleaved [k0, v0, k1, v1, ...] device-resident pointer table.
        ptrs: list[int] = []
        for entry in layers:
            ptrs.extend(_pointers_for_entry(entry, self.kv_format))
        cpu_ptrs = torch.tensor(ptrs, dtype=torch.int64)
        self.ptr_table: torch.Tensor = torch.empty(
            cpu_ptrs.shape, dtype=torch.int64, device=self.device
        )
        self.ptr_table.copy_(cpu_ptrs)

        self._staging: Optional[torch.Tensor] = None

    @property
    def plane_extras(self) -> tuple[int, int, int, int]:
        """The (k, v, dsa, scale) hidden-dim extras passed to the kernel."""
        return (self.kv_lora_rank, self.qk_rope_head_dim, 0, 0)

    def staging_for(self, kv_lead: int, tokens: int) -> torch.Tensor:
        """Return a (re)allocated NPU staging buffer ``[kv_lead, L, tokens, H]``."""
        shape = torch.Size([kv_lead, self.num_layers, tokens, self.hidden])
        if self._staging is None or self._staging.shape != shape:
            self._staging = torch.empty(shape, dtype=self.dtype, device=self.device)
        return self._staging


def _build_descriptor(kv_caches: dict[str, object]) -> Optional[_NPUTransferDescriptor]:
    """Build a descriptor for ``kv_caches`` if it is a supported NPU layout.

    Returns ``None`` for non-NPU devices, 310P, or unsupported formats so the
    caller can route to the upstream implementation.
    """
    # Third Party / First Party — lazy (see _NPUTransferDescriptor.__init__).
    from lmcache.v1.gpu_connector.utils import get_device
    from lmcache_ascend.v1.npu_connector.npu_connectors import is_310p

    values = list(kv_caches.values())
    if not values:
        return None
    try:
        device = get_device(values)  # type: ignore[arg-type]
    except (AttributeError, IndexError):
        return None
    if device.type != "npu" or is_310p():
        return None
    try:
        return _NPUTransferDescriptor(values)
    except ValueError:
        return None


def _get_descriptor(
    kv_caches: dict[str, object],
) -> Optional[_NPUTransferDescriptor]:
    """Return the cached descriptor for ``kv_caches``, building it if needed."""
    sig = _descriptor_signature(kv_caches)
    cached = _descriptor_cache.get(sig)
    if cached is not None:
        return cached
    desc = _build_descriptor(kv_caches)
    if desc is not None:
        _descriptor_cache[sig] = desc
    return desc


def _build_slot_mapping(
    chunk_block_ids: list[int], block_size: int, device: torch.device
) -> torch.Tensor:
    """Build a dense ``[num_tokens]`` slot-mapping ``block_id * block_size + j``.

    The kernel does not handle ``-1`` sentinels, so the mapping must be dense
    over the active chunk — which it is here since ``chunk_block_ids`` holds
    only the vLLM block ids that back the chunk's tokens.
    """
    if not chunk_block_ids:
        raise ValueError("chunk_block_ids must be non-empty")
    bids = torch.tensor(chunk_block_ids, dtype=torch.int64)
    offsets = torch.arange(block_size, dtype=torch.int64)
    slot_cpu = (bids[:, None] * block_size + offsets[None, :]).reshape(-1)
    slot = torch.empty(slot_cpu.shape, dtype=torch.int64, device=device)
    slot.copy_(slot_cpu, non_blocking=True)
    return slot


def _npu_gather_paged_kv_to_cpu(
    desc: _NPUTransferDescriptor,
    block_ids: list[int],
    blocks_per_chunk: int,
    out: Optional[list[torch.Tensor]],
    chunk_indices: Optional[list[int]],
) -> list[torch.Tensor]:
    """Gather paged NPU KV into CPU chunks via the fused kernel (one call/chunk).

    Honours the upstream ``out`` / ``chunk_indices`` contract: when ``out`` is
    provided (SHM path) each gathered chunk is written in place into
    ``out[out_idx]``; otherwise freshly allocated CPU tensors are returned.
    """
    # First Party — lazy.
    import lmcache_ascend.c_ops as lmc_ops

    num_chunks = len(block_ids) // blocks_per_chunk
    iter_indices = chunk_indices if chunk_indices is not None else range(num_chunks)
    chunks: list[torch.Tensor] = [] if out is None else out

    k1, k2, k3, k4 = desc.plane_extras
    for out_idx, chunk_idx in enumerate(iter_indices):
        chunk_block_ids = block_ids[
            chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
        ]
        tokens = len(chunk_block_ids) * desc.block_size
        slot_mapping = _build_slot_mapping(chunk_block_ids, desc.block_size, desc.device)
        staging = desc.staging_for(kv_lead=2, tokens=tokens)

        # Paged KV -> NPU staging (device-to-device; no host memory involved).
        lmc_ops.multi_layer_kv_transfer(
            key_value=staging,
            key_value_ptrs=desc.ptr_table,
            slot_mapping=slot_mapping,
            paged_memory_device=desc.device,
            page_buffer_size=desc.page_buffer_size,
            direction=True,  # from_gpu: paged -> staging
            use_mla=desc.use_mla,
            kvcache_format_raw=desc.kv_format.value,
            k_hidden_dims=k1,
            v_hidden_dims=k2,
            dsa_hidden_dims=k3,
            dsa_c8_scale_plane_bytes=k4,
            paged_kv_block_size=desc.block_size,
        )

        # D2H via torch (handles host allocation/registration internally), so
        # any host buffer — SHM views or freshly allocated CPU tensors — works.
        if out is not None:
            out[out_idx].copy_(staging)
        else:
            chunks.append(staging.cpu())

    return chunks


def _npu_scatter_cpu_to_paged_kv(
    desc: _NPUTransferDescriptor,
    block_ids: list[int],
    chunks: list[torch.Tensor],
    blocks_per_chunk: int,
    skip_first_n_tokens: int,
) -> None:
    """Scatter CPU chunks into paged NPU KV via the fused kernel (one call/chunk).

    Block-aligned ``skip_first_n_tokens`` handling mirrors the upstream helper.
    """
    # First Party — lazy.
    import lmcache_ascend.c_ops as lmc_ops

    num_chunks = len(block_ids) // blocks_per_chunk
    k1, k2, k3, k4 = desc.plane_extras

    for chunk_idx in range(min(num_chunks, len(chunks))):
        chunk_block_ids = list(
            block_ids[chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk]
        )
        chunk_start = chunk_idx * blocks_per_chunk * desc.block_size
        chunk_end = chunk_start + len(chunk_block_ids) * desc.block_size
        effective_start = max(chunk_start, skip_first_n_tokens)
        if effective_start >= chunk_end:
            continue

        skip_blocks = (effective_start - chunk_start) // desc.block_size
        skip_tokens = skip_blocks * desc.block_size
        eff_block_ids = chunk_block_ids[skip_blocks:]
        eff_tokens = len(eff_block_ids) * desc.block_size

        # H2D the effective (post-skip) tokens into a *packed* contiguous
        # staging buffer. A sliced view ``src[:, :, skip_tokens:]`` of a
        # ``[kv, layers, tokens, hidden]`` chunk is NOT contiguous across
        # layers, but the kernel assumes layer ``L`` lives at
        # ``L * eff_tokens * hidden`` from the base — so we copy the slice
        # into a freshly sized contiguous buffer instead of passing the view.
        src = chunks[chunk_idx]
        src_slice = src[:, :, skip_tokens:] if skip_tokens else src
        staging = desc.staging_for(kv_lead=2, tokens=eff_tokens)
        staging.copy_(src_slice)

        slot_mapping = _build_slot_mapping(eff_block_ids, desc.block_size, desc.device)
        lmc_ops.multi_layer_kv_transfer(
            key_value=staging,
            key_value_ptrs=desc.ptr_table,
            slot_mapping=slot_mapping,
            paged_memory_device=desc.device,
            page_buffer_size=desc.page_buffer_size,
            direction=False,  # to_gpu: staging -> paged
            use_mla=desc.use_mla,
            kvcache_format_raw=desc.kv_format.value,
            k_hidden_dims=k1,
            v_hidden_dims=k2,
            dsa_hidden_dims=k3,
            dsa_c8_scale_plane_bytes=k4,
            paged_kv_block_size=desc.block_size,
        )


# --- Override installation -------------------------------------------------

_orig_gather: Optional[object] = None
_orig_scatter: Optional[object] = None


def _gather_wrapper(
    kv_caches: dict[str, object],
    block_ids: list[int],
    blocks_per_chunk: int,
    layout_hints: Optional[object] = None,
    engine_kv_format: Optional[object] = None,
    out: Optional[list[torch.Tensor]] = None,
    chunk_indices: Optional[list[int]] = None,
) -> list[torch.Tensor]:
    """Dispatch gather to the fused NPU path when applicable, else upstream."""
    desc = _get_descriptor(kv_caches)
    if desc is not None:
        return _npu_gather_paged_kv_to_cpu(
            desc, block_ids, blocks_per_chunk, out, chunk_indices
        )
    assert _orig_gather is not None
    return _orig_gather(  # type: ignore[misc]
        kv_caches,
        block_ids,
        blocks_per_chunk,
        layout_hints=layout_hints,
        engine_kv_format=engine_kv_format,
        out=out,
        chunk_indices=chunk_indices,
    )


def _scatter_wrapper(
    kv_caches: dict[str, object],
    block_ids: list[int],
    chunks: list[torch.Tensor],
    blocks_per_chunk: int,
    skip_first_n_tokens: int = 0,
    layout_hints: Optional[object] = None,
    engine_kv_format: Optional[object] = None,
) -> None:
    """Dispatch scatter to the fused NPU path when applicable, else upstream."""
    desc = _get_descriptor(kv_caches)
    if desc is not None:
        _npu_scatter_cpu_to_paged_kv(
            desc, block_ids, chunks, blocks_per_chunk, skip_first_n_tokens
        )
        return
    assert _orig_scatter is not None
    _orig_scatter(  # type: ignore[misc]
        kv_caches,
        block_ids,
        chunks,
        blocks_per_chunk,
        skip_first_n_tokens=skip_first_n_tokens,
        layout_hints=layout_hints,
        engine_kv_format=engine_kv_format,
    )


def install_overrides() -> None:
    """Replace the upstream gather/scatter callables with the NPU dispatcher.

    Idempotent.  Patches both ``base`` (definition site) and ``worker_transfer``
    (import binding used by ``DataTransferContext`` at call time) because the
    upstream adapter imports the names by value.
    """
    global _orig_gather, _orig_scatter

    # Third Party
    import lmcache.v1.multiprocess.transfer_context.base as base
    import lmcache.v1.multiprocess.transfer_context.worker_transfer as wt

    if _orig_gather is None:
        _orig_gather = base.gather_paged_kv_to_cpu
        _orig_scatter = base.scatter_cpu_to_paged_kv

    base.gather_paged_kv_to_cpu = _gather_wrapper  # type: ignore[assignment]
    base.scatter_cpu_to_paged_kv = _scatter_wrapper  # type: ignore[assignment]
    wt.gather_paged_kv_to_cpu = _gather_wrapper  # type: ignore[assignment]
    wt.scatter_cpu_to_paged_kv = _scatter_wrapper  # type: ignore[assignment]
    logger.info(
        "Installed NPU fused gather/scatter override for MP non-GPU transfer "
        "(supported formats: %s)",
        ", ".join(f.name for f in _SUPPORTED_FORMATS),
    )
