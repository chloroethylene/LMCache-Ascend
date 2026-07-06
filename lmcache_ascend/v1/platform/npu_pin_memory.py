# SPDX-License-Identifier: Apache-2.0
"""NPU host-memory pinning via AscendCL ``aclrtHostRegister``.

Mirrors :class:`lmcache.v1.platform.cuda.pin_memory.CudaPinMemoryBackend` so the
upstream SHM engine-driven transfer path
(:meth:`EngineDrivenContextShm._pin_shm_buffer`) can page-lock the worker's SHM
pool on NPU. Once the pool is registered, ``copy_(..., non_blocking=True)``
issues a genuinely async D2H/H2D (verified on 910: ``issue`` drops to ~0 while
``synchronize()`` still waits for completion), removing the per-chunk host stall
in the MP gather/scatter path.

The AscendCL ``aclrtHostRegister`` signature is::

    aclError aclrtHostRegister(void *ptr, uint64_t size,
                               aclrtHostRegisterType type, void **devPtr);

with ``ACL_HOST_REGISTER_MAPPED = 0`` the only defined type. ``devPtr`` (a
device-mapped alias of the host region) is unused here -- torch's ``copy_`` keeps
addressing the original host pointer, which the registration page-locks for DMA.

Page alignment is mandatory: registering a non-page-aligned pointer fails
(observed AscendCL error ``507899``). SHM mmap pools are page-aligned at their
base, but this backend page-aligns defensively for any caller.
"""

# Standard
import ctypes
import ctypes.util
import glob
import os
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base_pin_memory import PinMemoryBackend

logger = init_logger(__name__)

#: ``aclrtHostRegisterType``: ``ACL_HOST_REGISTER_MAPPED`` is the sole defined value.
_ACL_HOST_REGISTER_MAPPED = 0

#: Host page size used for alignment. AscendCL requires page-aligned ``ptr`` and
#: a page-multiple ``size``; SHM mmap pools satisfy both at their base.
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

_ACL_SUCCESS = 0


def _candidate_lib_paths() -> list[str]:
    """Return AscendCL library search candidates, most-specific first.

    Honors ``$ASCEND_HOME_PATH`` and the standard CANN install layout, then falls
    back to :func:`ctypes.util.find_library`. ``dlopen`` returns the instance
    already loaded by ``torch_npu`` when one exists, so the loaded handle shares
    ``torch_npu``'s device context -- no separate ``aclrtSetDevice`` is needed.
    """
    paths: list[str] = []
    home = os.environ.get("ASCEND_HOME_PATH")
    if home:
        paths.append(os.path.join(home, "lib64", "libascendcl.so"))
    # Standard CANN installs: /usr/local/Ascend/cann-<ver>/<arch>-linux/lib64/...
    paths.extend(
        sorted(
            glob.glob("/usr/local/Ascend/cann*/aarch64-linux/lib64/libascendcl.so"),
            reverse=True,
        )
    )
    found = ctypes.util.find_library("ascendcl")
    if found:
        paths.append(found)
    return paths


def _load_libascendcl() -> ctypes.CDLL | None:
    """Load ``libascendcl`` and bind the host-(un)register symbols.

    Returns:
        The loaded library with bound symbols on success, or ``None`` if the
        library or symbols are unavailable (in which case pinning is unsupported
        and callers gracefully degrade to synchronous copies).
    """
    last_exc: Exception | None = None
    for path in _candidate_lib_paths():
        try:
            lib = ctypes.CDLL(path)
        except OSError as exc:
            last_exc = exc
            continue
        try:
            lib.aclrtHostRegister.restype = ctypes.c_int32
            lib.aclrtHostRegister.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            lib.aclrtHostUnregister.restype = ctypes.c_int32
            lib.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
            logger.info("NpuPinMemoryBackend: loaded libascendcl from %s", path)
            return lib
        except AttributeError as exc:
            last_exc = exc
            continue
    logger.warning(
        "NpuPinMemoryBackend: libascendcl not available (last error: %r); "
        "NPU host pinning disabled, D2H/H2D will be synchronous",
        last_exc,
    )
    return None


class NpuPinMemoryBackend(PinMemoryBackend):
    """Pin host memory for NPU DMA via AscendCL ``aclrtHostRegister``.

    Page-aligns the region defensively (AscendCL requires a page-aligned
    ``ptr`` and a page-multiple ``size``). The caller's original pointer is
    mapped to the registered (aligned) base so :meth:`unpin_memory` can reverse
    the registration when handed the original pointer (matching the upstream
    SHM lifecycle, which stores and later unpins the pointer it passed in).
    """

    def __init__(self) -> None:
        self._lib: ctypes.CDLL | None = _load_libascendcl()
        # Maps the caller's original pointer -> the registered (page-aligned)
        # base, so unpin_memory(original_ptr) resolves to the right base.
        self._registered_bases: dict[int, int] = {}
        # Process-level flag set when NPU is genuinely unavailable, so repeated
        # pin attempts short-circuit without re-probing.
        self._no_npu: bool = False
        # Per-thread "context ensured" flag. The ACL context is thread-local;
        # a worker thread has one (torch_npu creates it on first device op) but a
        # cache-server thread may not, so each pinning thread must ensure its own.
        self._tls = threading.local()

    def is_pin_supported(self) -> bool:
        """Whether AscendCL host registration is available.

        Returns:
            True if ``libascendcl`` and its register/unregister symbols loaded.
        """
        return self._lib is not None

    def _ensure_context(self) -> bool:
        """Ensure an ACL device context is current on the calling thread.

        ``aclrtHostRegister`` fails with AscendCL ``107002`` (no current
        context) when the thread has never touched the device. Worker threads
        are fine (torch_npu creates the context on first device op), but a
        cache-server thread that only manages host memory may have no context,
        so the first pin attempt on each thread must establish one.

        Uses ``torch.npu.set_device(current_device())`` rather than a hardcoded
        device so worker ranks already bound to device N are not shifted. The
        check is per-thread (the context is thread-local) and runs at most once
        per thread; on the worker it is a no-op idempotent call.

        Returns:
            True if a context is current (or was just established), False if the
            NPU is unavailable (in which case :meth:`pin_memory` short-circuits).
        """
        if self._lib is None or self._no_npu:
            return False
        if getattr(self._tls, "ensured", False):
            return True
        # Third Party
        import torch

        if not torch.npu.is_available():
            self._no_npu = True
            logger.info(
                "NpuPinMemoryBackend: NPU unavailable; host pinning disabled "
                "(copies will be synchronous)"
            )
            return False
        try:
            torch.npu.set_device(torch.npu.current_device())
        except (RuntimeError, AssertionError) as exc:
            # current_device() can be 0 pre-init; fall back to device 0.
            try:
                torch.npu.set_device(0)
            except RuntimeError as exc2:
                self._no_npu = True
                logger.warning(
                    "NpuPinMemoryBackend: cannot establish NPU context for "
                    "pinning: %r / %r; copies will be synchronous",
                    exc,
                    exc2,
                )
                return False
        self._tls.ensured = True
        return True

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        """Page-lock ``[ptr, ptr + size)`` for async NPU DMA.

        Args:
            ptr: Raw host pointer to the memory region.
            size: Region size in bytes.
            flags: Unused (AscendCL's only register type is ``MAPPED``).

        Returns:
            True if registration succeeded, False otherwise (callers degrade to
            synchronous copies).
        """
        if self._lib is None or size <= 0 or ptr == 0:
            return False
        if not self._ensure_context():
            return False

        base = _align_down(ptr, _PAGE_SIZE)
        end = _align_up(ptr + size, _PAGE_SIZE)
        reg_size = end - base
        dev_ptr = ctypes.c_void_p(0)
        ret = self._lib.aclrtHostRegister(
            ctypes.c_void_p(base),
            ctypes.c_uint64(reg_size),
            ctypes.c_int32(_ACL_HOST_REGISTER_MAPPED),
            ctypes.byref(dev_ptr),
        )
        if ret != _ACL_SUCCESS:
            logger.warning(
                "aclrtHostRegister failed: ptr=%#x base=%#x size=%d ret=%d; "
                "D2H/H2D will be synchronous",
                ptr,
                base,
                reg_size,
                ret,
            )
            return False
        self._registered_bases[ptr] = base
        logger.debug(
            "aclrtHostRegister ok: ptr=%#x base=%#x size=%d devptr=%#x",
            ptr,
            base,
            reg_size,
            dev_ptr.value or 0,
        )
        return True

    def unpin_memory(self, ptr: int) -> bool:
        """Unregister a previously pinned region.

        Args:
            ptr: The original pointer passed to :meth:`pin_memory`.

        Returns:
            True if unregistration succeeded (or the region was never pinned),
            False on AscendCL error.
        """
        if self._lib is None:
            return False
        base = self._registered_bases.pop(ptr, ptr)
        ret = self._lib.aclrtHostUnregister(ctypes.c_void_p(base))
        if ret != _ACL_SUCCESS:
            logger.warning("aclrtHostUnregister failed: ptr=%#x base=%#x ret=%d", ptr, base, ret)
            return False
        return True


def _align_down(value: int, alignment: int) -> int:
    return value - (value % alignment)


def _align_up(value: int, alignment: int) -> int:
    remainder = value % alignment
    return value if remainder == 0 else value + (alignment - remainder)
