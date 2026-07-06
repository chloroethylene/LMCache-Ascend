# SPDX-License-Identifier: Apache-2.0
"""Ascend platform primitives.

Wires the NPU host-memory pin backend (:class:`NpuPinMemoryBackend`) into the
upstream platform so the SHM engine-driven transfer path can page-lock worker
SHM pools for async D2H/H2D. Mirrors the CUDA platform package's pattern at
``lmcache/v1/platform/cuda/__init__.py``.

Upstream selects a device's pin backend via ``DeviceInfo.pin_memory_backend``,
read when ``DeviceExt`` is constructed. ``DeviceExt`` is built at
``import lmcache`` (before this plugin loads) with the no-op base backend, so
the plugin additionally overwrites the already-built instance's ``_pin`` at
activation time (see :func:`lmcache_ascend._patch_pin_memory`); the
class-level patch below covers any ``DeviceExt`` built from the registry
afterwards.
"""

# First Party
from lmcache_ascend.v1.platform.npu_pin_memory import NpuPinMemoryBackend

# Expose the NPU pin backend on ``NpuDeviceInfo`` via upstream's
# ``DeviceInfo.pin_memory_backend`` mechanism. This replaces the
# ``register_pin_memory_backend`` registration API removed by the upstream
# platform refactor. Best-effort: if ``NpuDeviceInfo`` is absent (older
# upstream without NPU platform support), the live-instance patch in
# ``_patch_pin_memory`` still installs the backend.
try:
    # First Party
    from lmcache.v1.platform.npu import NpuDeviceInfo

    NpuDeviceInfo.pin_memory_backend = property(  # type: ignore[method-assign,assignment]
        lambda self: NpuPinMemoryBackend
    )
except ImportError:
    pass

__all__ = ["NpuPinMemoryBackend"]
