# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Generic, Optional, TypeVar
import threading

# Third Party
import torch

T = TypeVar("T")


class MessagingFuture(Generic[T]):
    def __init__(self):
        self.is_done_ = threading.Event()
        self.result_ = None

    def query(self) -> bool:
        """
        Check if the future is done.

        Returns:
            bool: True if the future is done, False otherwise.
        """
        return self.is_done_.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the future to be done.

        Args:
            timeout (Optional[float]): Maximum time to wait in seconds.
                If None, wait indefinitely.

        Returns:
            bool: True if the future is done, False if the timeout was reached.
        """
        return self.is_done_.wait(timeout)

    def result(self, timeout: Optional[float] = None) -> T:
        """
        Get the result of the future.

        Args:
            timeout (Optional[float]): Maximum time to wait in seconds.
                If None, wait indefinitely.

        Returns:
            T: The result of the future.

        Raises:
            TimeoutError: If the future is not done within the timeout.
        """
        flag = self.wait(timeout)
        if not flag:
            raise TimeoutError("Future result not available within timeout")
        return self.result_

    def set_result(self, result: T) -> None:
        """
        Set the result of the future and mark it as done. This function is NOT
        SUPPOSED TO BE CALLED by users directly. It should be only called by
        the messaging system when the result is available.

        Args:
            result (T): The result to set.
        """
        self.result_ = result
        self.is_done_.set()

    def to_npu_future(
        self,
        device: torch.device | None = None,
    ) -> "NPUMessagingFuture":
        # TODO: need extra type checking for the future type
        return NPUMessagingFuture.FromMessagingFuture(self, device)  # type: ignore


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