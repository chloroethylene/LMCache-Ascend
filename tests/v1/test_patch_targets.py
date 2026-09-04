# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
"""Regression tests for monkey-patch targets after the 0.5.x upstream sync.

Each test pins one patch to the module/symbol where current upstream defines
it, so an upstream move turns into a clear failure here instead of a silent
no-op (or an AttributeError at serving time).
"""

# Standard
import subprocess
import sys
import textwrap

# Third Party
import pytest

# First Party
from lmcache_ascend.v1.memory_management import GPUMemoryAllocator__init__
from lmcache_ascend.v1.multiprocess.custom_types import AscendIPCWrapper
from lmcache_ascend.v1.npu_connector.utils import permute_kv_caches_to_contiguous
import lmcache_ascend


def test_npu_ipc_wrapper_registered_on_device_spec():
    """The KV-cache IPC wrapper must resolve through DeviceSpec.ipc_wrapper_cls.

    Upstream dispatches wrapper creation via
    ``resolve_kv_wrapper_factory(device_type)``; before the 0.5.x sync the
    plugin rebound the removed ``custom_types.CudaIPCWrapper`` instead, so
    ``resolve_kv_wrapper_factory("npu")`` raised ``ValueError``.
    """
    # Third Party
    from lmcache.v1.platform import resolve_kv_wrapper_factory
    from lmcache.v1.platform.npu import NpuDeviceSpec

    assert NpuDeviceSpec().ipc_wrapper_cls is AscendIPCWrapper
    # The wrapper must report the true device type instead of the inherited
    # CUDA one, so server-side device detection resolves the NPU spec.
    assert AscendIPCWrapper.device_type == "npu"
    # Binding the classmethod gives a bound method whose __self__ is the class
    factory = resolve_kv_wrapper_factory("npu")
    assert getattr(factory, "__self__", None) is AscendIPCWrapper


def test_permute_rebound_only_at_list_call_site():
    """Only gpu_connectors' kvcaches-list call site takes the NPU permute.

    The upstream rename to ``attempt_permute_to_contiguous_view`` also gave
    the contiguity module single-tensor callers (cuda/cpu IPC wrappers, shm);
    rebinding the module-level name would hand those a list-only function.
    """
    # Third Party
    import lmcache.v1.gpu_connector.gpu_connectors as gpu_connectors_mod
    import lmcache.v1.gpu_connector.kv_format.contiguity as contiguity_mod

    # _patch_gpu_connector is vllm-runtime gated; it is pure rebinding, so
    # invoke it directly to make the assertion deterministic.
    lmcache_ascend._patch_gpu_connector()

    assert (
        gpu_connectors_mod.attempt_permute_to_contiguous_view
        is permute_kv_caches_to_contiguous
    )
    # The defining module keeps upstream single-tensor semantics.
    assert contiguity_mod.attempt_permute_to_contiguous_view is not (
        permute_kv_caches_to_contiguous
    )


def test_sgl_gpuram_allocator_patch_target_exists():
    """_patch_sgl must find GPUMemoryAllocator where upstream now defines it.

    The class moved to ``lmcache.v1.memory_allocators.gpu_memory_allocator``
    in the #4077 refactor; patching the old module raised ``AttributeError``
    under SGLang runtimes (actual application is covered by the subprocess
    activation test below).
    """
    # Third Party
    from lmcache.v1.memory_allocators import gpu_memory_allocator
    from lmcache.v1.memory_allocators.gpu_memory_allocator import (
        GPUMemoryAllocator,
    )
    import lmcache.v1.memory_management as lmc_memory_management

    # The patch target exists at the new location and the old one is gone.
    assert gpu_memory_allocator.GPUMemoryAllocator is GPUMemoryAllocator
    assert not hasattr(lmc_memory_management, "GPUMemoryAllocator")

    # The replacement init is signature-compatible with the class it patches.
    # Standard
    import inspect

    replacement = inspect.signature(GPUMemoryAllocator__init__)
    original = inspect.signature(GPUMemoryAllocator.__init__)
    assert list(replacement.parameters) == list(original.parameters)


def test_sgl_runtime_activation_does_not_raise():
    """Full plugin activation on an SGLang-detected runtime stays clean.

    Runs ``import lmcache_ascend`` in a subprocess after importing the real
    ``sglang`` (the plugin's own sglang adapter imports ``sglang.srt`` at
    module level, so a stub cannot stand in) so the SGLang-only patch branch
    executes; the pre-sync bug (AttributeError from _patch_sgl) surfaces as
    a non-zero exit code. Skipped where sglang is not installed.
    """
    # Standard
    import importlib.util

    if importlib.util.find_spec("sglang") is None:
        pytest.skip("sglang is not installed; _patch_sgl needs its adapter")

    code = textwrap.dedent(
        """
        import sglang.srt.configs.model_config  # noqa: F401  # runtime marker
        import lmcache_ascend  # noqa: F401
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"plugin activation failed under sglang runtime:\n{result.stderr[-2000:]}"
    )


def test_upstream_tag_is_a_ref_not_stale_release():
    """LMCACHE_UPSTREAM_TAG must reflect the actually required upstream ref.

    After the 0.5.x sync the plugin imports post-#4077 module paths, so a
    stale v0.4.x tag here would mislead CI cloning and the bootstrap sync.
    """
    tag = lmcache_ascend.LMCACHE_UPSTREAM_TAG
    assert tag != "v0.4.5"
    assert tag.startswith(("v0.", "dev", "feature/", "release/"))
