# SPDX-License-Identifier: Apache-2.0
"""Tests for the NPU host-memory pin backend (``NpuPinMemoryBackend``).

The backend page-locks host memory via AscendCL ``aclrtHostRegister`` so the MP
SHM pool supports async D2H/H2D. Library/construction tests run wherever
``libascendcl`` is loadable; registration + async-copy tests require a live NPU
device context and are skipped otherwise.
"""

# Standard
import ctypes
import ctypes.util
import time

# Third Party
import pytest
import torch

# First Party
from lmcache_ascend.v1.platform.npu_pin_memory import NpuPinMemoryBackend

_NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
requires_npu = pytest.mark.skipif(
    not _NPU_AVAILABLE, reason="requires Ascend NPU device context"
)


def _page_aligned_buffer(num_bytes: int) -> tuple[int, int]:
    """Allocate a page-aligned, page-multiple host region via anonymous mmap.

    Mirrors an SHM pool base (page-aligned), which is what upstream's
    ``_pin_shm_buffer`` registers. Returns ``(ptr, size)``.
    """
    page = 4096
    npg = (num_bytes + page - 1) // page
    size = npg * page
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
    mmap = libc.mmap
    mmap.restype = ctypes.c_void_p
    mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    PROT_READ, PROT_WRITE = 1, 2
    MAP_PRIVATE, MAP_ANONYMOUS = 2, 0x20
    base = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
    assert base % page == 0, "mmap must return a page-aligned base"
    return base, size


def test_is_pin_supported_when_libascendcl_present():
    """The backend reports support when libascendcl + symbols load."""
    backend = NpuPinMemoryBackend()
    assert backend.is_pin_supported() is True


@requires_npu
def test_pin_unpin_roundtrip_page_aligned():
    """Pinning a page-aligned region succeeds and unpins cleanly."""
    _ = torch.randn(8, device="npu")  # initialize the NPU device context
    backend = NpuPinMemoryBackend()
    ptr, size = _page_aligned_buffer(2 * 1024 * 1024)  # 2 MiB
    try:
        assert backend.pin_memory(ptr, size) is True
        assert ptr in backend._registered_bases
        assert backend.unpin_memory(ptr) is True
        assert ptr not in backend._registered_bases
    finally:
        backend.unpin_memory(ptr)


@requires_npu
def test_pin_auto_aligns_non_page_aligned_pointer():
    """A non-page-aligned pointer is registered via its enclosing page range."""
    _ = torch.randn(8, device="npu")
    backend = NpuPinMemoryBackend()
    page = 4096
    ptr, size = _page_aligned_buffer(4 * 1024 * 1024)
    # Offset into the region so the base pointer is not page-aligned.
    unaligned_ptr = ptr + (page // 2)
    try:
        assert backend.pin_memory(unaligned_ptr, size - page) is True
        # The registered base is the page-aligned enclosing base.
        assert backend._registered_bases[unaligned_ptr] == ptr
    finally:
        backend.unpin_memory(unaligned_ptr)


@requires_npu
def test_pin_ensures_context_without_prior_device_op():
    """A fresh thread with no live context still pins (the backend ensures it).

    Replicates the cache-server thread state: ``aclrtHostRegister`` would fail
    with ``107002`` (no current context) absent a prior device op, so the
    backend must establish one itself. Pinning must succeed, not degrade.
    """
    backend = NpuPinMemoryBackend()
    ptr, size = _page_aligned_buffer(1024 * 1024)
    try:
        # No torch.npu op has run on this test's main thread yet at module
        # import; the backend ensures the context on first pin.
        assert backend.pin_memory(ptr, size) is True
    finally:
        backend.unpin_memory(ptr)


@requires_npu
def test_pin_ensures_per_thread_context():
    """Pinning from a thread with no prior device op also succeeds.

    The ACL context is thread-local; the allocator's expand worker thread has
    none until the backend establishes it. Verifies the per-thread ensure.
    """
    # Standard
    import threading

    backend = NpuPinMemoryBackend()
    ptr, size = _page_aligned_buffer(1024 * 1024)
    result = {}

    def _pin_from_thread():
        result["ok"] = backend.pin_memory(ptr, size)

    t = threading.Thread(target=_pin_from_thread)
    t.start()
    t.join()
    try:
        assert result.get("ok") is True
    finally:
        if result.get("ok"):
            backend.unpin_memory(ptr)


@requires_npu
def test_pin_short_circuits_when_npu_unavailable(monkeypatch):
    """When NPU is unavailable, pin returns False without raising."""
    # Standard
    import types

    backend = NpuPinMemoryBackend()
    backend._no_npu = False  # reset any prior probe state
    # Force is_available() False on the torch.npu module the backend consults.
    monkeypatch.setattr(torch.npu, "is_available", lambda: False)
    ptr, size = _page_aligned_buffer(1024 * 1024)
    assert backend.pin_memory(ptr, size) is False
    assert backend._no_npu is True


@requires_npu
def test_registered_copy_is_async():
    """After pinning, non_blocking D2H returns immediately (issue << total)."""
    dev = torch.randn(1 << 25, device="npu", dtype=torch.float32)  # 128 MiB; inits ctx
    backend = NpuPinMemoryBackend()
    ptr, size = _page_aligned_buffer((1 << 25) * 4)
    host = torch.frombuffer(
        (ctypes.c_char * size).from_address(ptr), dtype=torch.float32
    ).view(1 << 25)
    try:
        assert backend.pin_memory(ptr, size) is True
        for _ in range(3):  # warm up
            host.copy_(dev)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        host.copy_(dev, non_blocking=True)
        issue = time.perf_counter() - t0
        torch.npu.synchronize()
        total = time.perf_counter() - t0
        # Async copy returns near-instantly; the bulk of the time is the sync.
        assert issue < total / 4, f"copy not async: issue={issue*1e3}ms total={total*1e3}ms"
    finally:
        backend.unpin_memory(ptr)


@requires_npu
def test_plugin_wires_backend_onto_torch_dev_ext():
    """Importing the plugin installs the NPU backend on ``torch_dev.ext._pin``."""
    # Third Party
    import lmcache_ascend  # noqa: F401  -- triggers plugin activation
    from lmcache import torch_dev

    assert isinstance(torch_dev.ext._pin, NpuPinMemoryBackend)
    assert torch_dev.ext.is_pin_supported is True


@requires_npu
def test_lazy_memory_allocator_pool_uses_aclrtmallochost(caplog):
    """The server's LazyMemoryAllocator pool is aclrtMallocHost-pinned (no 507899).

    Regression for the ``torch.empty`` pool whose malloc'd memory
    ``aclrtHostRegister`` rejects intermittently (``507899``). The plugin swaps
    ``__init__``/``_pin_memory_chunk``/``close`` so the whole pool is allocated
    via ``alloc_pinned_ptr`` (aclrtMallocHost + internal register_ptr) and needs
    no per-chunk pin -- so expansion produces zero pin warnings.
    """
    # Third Party
    import lmcache_ascend  # noqa: F401  -- applies _patch_lazy_memory_allocator
    from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator

    chunk = LazyMemoryAllocator.PIN_CHUNK_SIZE
    with caplog.at_level("WARNING", logger="lmcache"):
        alloc = LazyMemoryAllocator(
            init_size=chunk, final_size=chunk * 2, numa_mapping=None
        )
        try:
            # The pool is aclrtMallocHost-backed, not the torch.empty/NUMA path.
            assert alloc._use_numa is False
            assert hasattr(alloc, "_ascend_pool_ptr")
            assert alloc._buffer.data_ptr() % 4096 == 0  # page-aligned
            assert alloc._pin_record == []  # no per-chunk pin: already pinned
        finally:
            alloc.close()

    pin_warnings = [
        r.message
        for r in caplog.records
        if "aclrtHostRegister failed" in r.message or "pin_memory failed" in r.message
    ]
    assert pin_warnings == [], f"unexpected pin warnings: {pin_warnings[:2]}"
