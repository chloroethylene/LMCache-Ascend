# SPDX-License-Identifier: Apache-2.0
"""Device-plugin entry point for early, deterministic plugin activation.

LMCache discovers this module through the ``lmcache.device_plugins``
entry-point group while ``lmcache.v1.platform`` is initializing, which
happens before any downstream LMCache module (connectors, storage,
cache engine) is imported. Importing this module first imports the
``lmcache_ascend`` package parent, so every monkey patch lands at that
same deterministic point instead of wherever an accidental import chain
(first ``lmcache.v1.offload_server.zmq_server``) happens to reach us.

The spec itself stands down: ``is_available()`` returns False so the
in-tree ``npu`` backend (``lmcache.v1.platform.devices.npu``) keeps the
default selection and provides the same factories. This entry point only
changes when the plugin activates, not which backend is dispatched to.
"""

# Future
from __future__ import annotations

# First Party
from lmcache.v1.platform.devices.npu import NpuDeviceSpec


class AscendDeviceSpec(NpuDeviceSpec):
    """Entry-point carrier delegating every capability to the in-tree spec."""

    @property
    def backend_name(self) -> str:
        return "ascend"

    def is_available(self) -> bool:
        return False
