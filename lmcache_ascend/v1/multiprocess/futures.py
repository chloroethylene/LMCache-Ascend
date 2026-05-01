# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Generic, Optional, TypeVar

# Third Party
import torch

# First Party
# Reuse MessagingFuture from LMCache
from lmcache.v1.multiprocess.futures import MessagingFuture

T = TypeVar("T")


class NPUMessagingFuture(MessagingFuture[T]):
    """
    The future class that wraps both result and an NPU IPC event.
    The `query`, `wait`, and `result` methods will pend on both the
    original future and the NPU event.
    The original future should return tuple[bytes, T], where the first
    element is the serialized NPU event.
    """

    def __init__(
        self,
        raw_future: MessagingFuture[tuple[bytes, T]],
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.raw_future_ = raw_future
        self.event_: torch.npu.Event | None = None
        self.result_: T | None = None
        self.device_ = device if device is not None else torch.npu.current_device()

    def _on_raw_future_complete(self):
        """
        Update the NPU event and result when the raw future is complete.
        """
        event_bytes, result = self.raw_future_.result()
        self.result_ = result

        # Deserialize the NPU event
        self.event_ = torch.npu.Event.from_ipc_handle(self.device_, event_bytes)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the future to be done, with the NPU stream.

        Args:
            timeout (Optional[float]): Maximum time to wait for the UNDERLYING
                RAW FUTURE in seconds. The exact timeout is not guaranteed
                when waiting on the NPU event. (NOTE: this could be improved
                with careful threading management)

        Returns:
            bool: True if the future is done, False if the timeout was reached.

        Raises:
            ValueError: if the timeout is not None.

        Notes:
            This function does not support waiting for a specific time.
        """
        if self.event_:
            self.event_.synchronize()
            return True

        flag = self.raw_future_.wait(timeout)
        if not flag:
            return False

        self._on_raw_future_complete()

        assert self.event_ is not None
        self.event_.synchronize()

        return True

    def result(self, timeout: Optional[float] = None) -> T:
        """
        Get the result of the future.

        Args:
            timeout (Optional[float]): Maximum time to wait for the UNDERLYING
                RAW FUTURE in seconds. The exact timeout is not guaranteed
                when waiting on the NPU event. (NOTE: this could be improved
                with careful threading management)

        Returns:
            T: The result of the future.

        Raises:
            TimeoutError: If the future is not done within the timeout.
        """
        flag = self.wait(timeout)
        if not flag:
            raise TimeoutError(
                "NPUMessagingFuture result not available within timeout"
            )

        assert self.result_ is not None
        return self.result_

    def query(self) -> bool:
        """
        Check if the future is done.

        Returns:
            bool: True if the future is done, False otherwise.
        """
        if self.event_:
            return self.event_.query()

        if self.raw_future_.query():
            self._on_raw_future_complete()
            assert self.event_ is not None
            return self.event_.query()

        return False

    def set_result(self, result: T) -> None:
        raise NotImplementedError(
            "NPUMessagingFuture does not support set_result directly"
        )

    @staticmethod
    def FromMessagingFuture(
        raw_future: MessagingFuture[tuple[bytes, T]],
        device: torch.device | None = None,
    ) -> "NPUMessagingFuture[T]":
        return NPUMessagingFuture(raw_future, device)