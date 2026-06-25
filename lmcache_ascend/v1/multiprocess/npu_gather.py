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

In addition, ``EngineDrivenTransferContext.submit_store`` / ``submit_retrieve``
are patched so the fused path runs on a dedicated NPU stream
(``_NPUTransferDescriptor.transfer_stream``): the whole-device
``torch_dev.synchronize()`` that orders the gather against the model forward is
replaced with ``transfer_stream.wait_stream(torch.npu.current_stream())``
(mirrors the in-process connector at npu_connectors.py:1126). The "forward-
completion event" the engine passes in is not used: the deployed vLLM connector
resolves it to a CPU-runner ``_EventPlaceholder`` with no ``.wait`` method, so
ordering via the current (forward) stream is the robust choice; the two
pre-commit syncs become stream-scoped ``transfer_stream.synchronize()``. The
per-chunk D2H/H2D copies are issued ``non_blocking`` on that stream so N host
syncs collapse to one. Unsupported layouts (and CPU/310P workers) fall back to the
original upstream methods unchanged.

Heavy dependencies (``c_ops``, the NPU connector helpers) are imported lazily
so the module and its pure-Python helpers stay importable on hosts without a
built extension — this keeps the slot-mapping and fallback logic unit-testable.
"""

# Standard
from typing import Optional

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.futures import MessagingFuture
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
        transfer_stream: Dedicated NPU stream for the paged<->staging DMA and
            the D2H/H2D leg, so MP transfer does not contend with model
            kernels on the default stream.
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

        # LMC-A: dedicated stream for the paged<->staging DMA + the D2H/H2D
        # leg so KV transfer no longer contends with model kernels on the
        # default NPU stream. Created eagerly (mirrors the channel
        # ``transport_stream`` at hccl_channel.py:104); the descriptor is only
        # built lazily once a supported NPU layout is registered, so the device
        # is live by this point.
        self.transfer_stream: torch.npu.Stream = torch.npu.Stream(
            device=self.device
        )

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
    # LMC-A: run the whole per-chunk loop on the dedicated transfer stream so
    # the paged->staging DMA and the D2H leg do not contend with model kernels
    # on the default stream. No per-chunk host sync: completion is awaited once
    # by the caller (EngineDrivenTransferContext.submit_store, via
    # ``desc.transfer_stream.synchronize()``) before commit. A
    # ``wait_stream(torch.npu.current_stream())`` is also issued there to order
    # this stream against the model forward that wrote the paged KV.
    with torch.npu.stream(desc.transfer_stream):
        for out_idx, chunk_idx in enumerate(iter_indices):
            chunk_block_ids = block_ids[
                chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
            ]
            tokens = len(chunk_block_ids) * desc.block_size
            slot_mapping = _build_slot_mapping(
                chunk_block_ids, desc.block_size, desc.device
            )
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

            # D2H on the transfer stream. ``non_blocking=True`` needs a pinned
            # host dst for a truly async copy; SHM mmap views are not pinned so
            # torch falls back to a synchronous copy there, but the call still
            # lands on the transfer stream and the single post-loop stream sync
            # (caller) guarantees completion before commit. The non-SHM path
            # uses a pinned CPU buffer so it is genuinely async.
            if out is not None:
                out[out_idx].copy_(staging, non_blocking=True)
            else:
                dst = torch.empty(
                    staging.shape,
                    dtype=staging.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                dst.copy_(staging, non_blocking=True)
                chunks.append(dst)

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

    # LMC-A: run the scatter on the dedicated transfer stream (mirrors the
    # gather path); completion is awaited once by the caller
    # (EngineDrivenTransferContext.submit_retrieve, via
    # ``desc.transfer_stream.synchronize()``) before the SHM slot is released.
    with torch.npu.stream(desc.transfer_stream):
        for chunk_idx in range(min(num_chunks, len(chunks))):
            chunk_block_ids = list(
                block_ids[
                    chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
                ]
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
            staging.copy_(src_slice, non_blocking=True)

            slot_mapping = _build_slot_mapping(
                eff_block_ids, desc.block_size, desc.device
            )
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
# LMC-A: originals of EngineDrivenTransferContext.submit_store / submit_retrieve,
# saved so the NPU-aware wrappers below can fall back to them for non-NPU /
# unsupported-layout workers.
_orig_submit_store: Optional[object] = None
_orig_submit_retrieve: Optional[object] = None


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


def _flatten_single_group_block_ids(block_ids: list[list[int]]) -> list[int]:
    """Flatten the single per-group block-id list for engine-driven transfer.

    Mirrors upstream ``_single_group_block_ids`` (worker_transfer.py) without
    importing that private helper. The NPU fused path, like upstream's
    engine-driven path, handles one KV cache group only; multi-group transfers
    are rejected here.
    """
    if len(block_ids) != 1:
        raise RuntimeError(
            "engine-driven transfer does not support hybrid KV cache groups"
        )
    return block_ids[0]


def _submit_store_wrapper(
    self,
    _request_id: object,
    key: object,
    instance_id: int,
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[list[int]],
    _event: object,
    blocks_in_chunk: int,
) -> MessagingFuture:
    """NPU-aware ``EngineDrivenTransferContext.submit_store``.

    Falls back to the upstream method when ``kv_caches`` is not a supported NPU
    layout (``_get_descriptor`` returns ``None``). Otherwise it replaces
    upstream's whole-device pre-gather sync (worker_transfer.py:429) with a
    non-blocking ``transfer_stream.wait_stream(torch.npu.current_stream())``
    (mirrors the in-process connector at npu_connectors.py:1126). The passed
    ``_event`` is intentionally unused: the deployed vLLM connector resolves the
    "forward-completion event" to a CPU-runner ``_EventPlaceholder`` (which has
    no ``.wait``), so ordering via the current (forward) stream is both robust
    and exactly what the in-process path already does. The fused gather then
    runs on the transfer stream, and the post-gather whole-device sync (:448)
    becomes a single ``transfer_stream.synchronize()`` before commit.
    """
    if self._engine_driven_context is None:
        raise RuntimeError(
            "Engine-driven transfer context is not registered. "
            "Call register() before submit_store()."
        )
    desc = _get_descriptor(kv_caches)
    if desc is None:
        assert _orig_submit_store is not None
        return _orig_submit_store(  # type: ignore[misc]
            self,
            _request_id,
            key,
            instance_id,
            kv_caches,
            block_ids,
            _event,
            blocks_in_chunk,
        )

    # Order the transfer stream against the forward that wrote the paged KV
    # (which ran on the current/compute stream), replacing the whole-device
    # pre-gather sync (worker_transfer.py:429). The passed _event is a vLLM
    # _EventPlaceholder and is intentionally unused here.
    desc.transfer_stream.wait_stream(torch.npu.current_stream())
    result = self._engine_driven_context.prepare_store(key, instance_id)
    out_buffers, chunk_indices = result if result is not None else (None, None)
    # All chunks already in cache — nothing to gather or commit.
    if chunk_indices is not None and len(chunk_indices) == 0:
        future: MessagingFuture[bool] = MessagingFuture()
        future.set_result(True)
        return future
    cpu_chunks = _gather_wrapper(
        kv_caches,
        _flatten_single_group_block_ids(block_ids),
        blocks_in_chunk,
        layout_hints=self._layout_hints,
        engine_kv_format=self._engine_kv_format,
        out=out_buffers,
        chunk_indices=chunk_indices,
    )
    # Complete the async D2H on the transfer stream before commit, replacing
    # the whole-device sync at worker_transfer.py:448.
    desc.transfer_stream.synchronize()
    ok = self._engine_driven_context.commit_store(key, instance_id, cpu_chunks)

    future = MessagingFuture()
    future.set_result(ok)
    return future


def _submit_retrieve_wrapper(
    self,
    _request_id: object,
    key: object,
    instance_id: int,
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[list[int]],
    _event: object,
    blocks_in_chunk: int,
    skip_first_n_tokens: int = 0,
) -> MessagingFuture:
    """NPU-aware ``EngineDrivenTransferContext.submit_retrieve``.

    Falls back to the upstream method for unsupported layouts. Otherwise it runs
    the fused scatter on the transfer stream and replaces the whole-device
    post-scatter sync (worker_transfer.py:490) with a single
    ``transfer_stream.synchronize()`` before the SHM slot is released
    (``commit_retrieve``). ``event`` is unused here, matching upstream.
    """
    if self._engine_driven_context is None:
        raise RuntimeError(
            "Engine-driven transfer context is not registered. "
            "Call register() before submit_retrieve()."
        )
    desc = _get_descriptor(kv_caches)
    if desc is None:
        assert _orig_submit_retrieve is not None
        return _orig_submit_retrieve(  # type: ignore[misc]
            self,
            _request_id,
            key,
            instance_id,
            kv_caches,
            block_ids,
            _event,
            blocks_in_chunk,
            skip_first_n_tokens=skip_first_n_tokens,
        )

    src_buffers = self._engine_driven_context.prepare_retrieve(key, instance_id)
    ok = src_buffers is not None
    if src_buffers is not None:
        try:
            _scatter_wrapper(
                kv_caches,
                _flatten_single_group_block_ids(block_ids),
                src_buffers,
                blocks_in_chunk,
                skip_first_n_tokens=skip_first_n_tokens,
                layout_hints=self._layout_hints,
                engine_kv_format=self._engine_kv_format,
            )
        except (RuntimeError, ValueError, TypeError, IndexError):
            logger.exception("Failed to scatter retrieved CPU context chunks")
            ok = False
        # Ensure the scatter's device writes are complete before releasing the
        # SHM slot, replacing the whole-device sync at worker_transfer.py:490.
        desc.transfer_stream.synchronize()
    self._engine_driven_context.commit_retrieve(key, instance_id)

    future: MessagingFuture[bool] = MessagingFuture()
    future.set_result(ok)
    return future


def install_overrides() -> None:
    """Replace the upstream gather/scatter + submit callables with NPU dispatchers.

    Idempotent.  Patches both ``base`` (definition site) and ``worker_transfer``
    (import binding used by ``DataTransferContext`` at call time) for the
    gather/scatter names, because the upstream adapter imports them by value.

    Additionally patches ``EngineDrivenTransferContext.submit_store`` /
    ``submit_retrieve`` so the NPU path orders the fused transfer on a dedicated
    stream via the forward-completion event instead of upstream's whole-device
    ``torch_dev.synchronize()`` calls. Non-NPU / unsupported-layout workers fall
    back to the saved originals unchanged.
    """
    global _orig_gather, _orig_scatter, _orig_submit_store, _orig_submit_retrieve

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

    if _orig_submit_store is None:
        _orig_submit_store = wt.EngineDrivenTransferContext.submit_store
        _orig_submit_retrieve = wt.EngineDrivenTransferContext.submit_retrieve
    wt.EngineDrivenTransferContext.submit_store = (  # type: ignore[assignment]
        _submit_store_wrapper
    )
    wt.EngineDrivenTransferContext.submit_retrieve = (  # type: ignore[assignment]
        _submit_retrieve_wrapper
    )

    logger.info(
        "Installed NPU fused gather/scatter + transfer-stream submit override "
        "for MP non-GPU transfer (supported formats: %s)",
        ", ".join(f.name for f in _SUPPORTED_FORMATS),
    )
