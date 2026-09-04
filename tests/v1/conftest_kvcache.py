# SPDX-License-Identifier: Apache-2.0
"""Generic KV cache / NPU kernel test helpers (format-agnostic)."""

# Future
from __future__ import annotations

# Third Party
import torch


def npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def device() -> torch.device:
    if npu_available():
        return torch.device("npu:0")
    return torch.device("cpu")
