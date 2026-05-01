# SPDX-License-Identifier: Apache-2.0
"""
Stub for native_storage_ops C++ extension on Ascend NPU.

This module provides lightweight Python fallbacks for the native_storage_ops
C++ extension when it is not available (e.g., when NO_CUDA_EXT=1 is used).
"""

# Standard
from typing import Any, Set, overload
import threading
import time


class TTLLock:
    """
    A thread-safe lock with TTL (Time-To-Live) support.

    The lock maintains a counter that can be incremented (lock) and decremented
    (unlock). If the TTL expires, the lock is considered unlocked regardless
    of the counter value.
    """

    def __init__(self, ttl_second: int = 300) -> None:
        """
        Construct a TTLLock with the specified TTL duration in seconds.

        Args:
            ttl_second: TTL duration in seconds. Default is 300.
        """
        self._ttl = ttl_second
        self._counter = 0
        self._lock = threading.Lock()
        self._expiry = time.time() + ttl_second

    def lock(self) -> None:
        """
        Increment the lock counter by 1 and update the TTL.
        If the previous TTL has expired, reset counter to 1.
        """
        with self._lock:
            now = time.time()
            if now > self._expiry:
                self._counter = 1
            else:
                self._counter += 1
            self._expiry = now + self._ttl

    def unlock(self) -> None:
        """Decrement the lock counter by 1 (minimum 0)."""
        with self._lock:
            self._counter = max(0, self._counter - 1)

    def is_locked(self) -> bool:
        """
        Check if the lock is held (counter > 0 and TTL not expired).

        Returns:
            True if the lock is held, False otherwise.
        """
        with self._lock:
            return self._counter > 0 and time.time() <= self._expiry

    def reset(self) -> None:
        """Reset the lock to initial state (counter = 0, TTL expired)."""
        with self._lock:
            self._counter = 0
            self._expiry = 0


class Bitmap:
    """
    A bitmap for tracking the state of L2 storage operation results.

    Each bit represents the success or failure of a key.
    """

    @overload
    def __init__(self, size: int) -> None:
        """
        Construct a Bitmap with the specified number of bits.

        Args:
            size: The number of bits in the bitmap.
        """
        ...

    @overload
    def __init__(self, size: int, prefix_bits: int) -> None:
        """
        Construct a Bitmap with the specified number of bits and prefix.

        Args:
            size: The number of bits in the bitmap.
            prefix_bits: The first N bits are set to 1.
        """
        ...

    def __init__(self, size: int, prefix_bits: int = 0) -> None:
        self._size = size
        self._bits = [False] * size
        for i in range(min(prefix_bits, size)):
            self._bits[i] = True

    def set(self, index: int) -> None:
        """Set the bit at the specified index to 1."""
        if 0 <= index < self._size:
            self._bits[index] = True

    def clear(self, index: int) -> None:
        """Clear the bit at the specified index to 0."""
        if 0 <= index < self._size:
            self._bits[index] = False

    def test(self, index: int) -> bool:
        """
        Test the bit at the specified index.

        Returns:
            True if the bit is set to 1, False otherwise.
        """
        if 0 <= index < self._size:
            return self._bits[index]
        return False

    def popcount(self) -> int:
        """Return the number of bits set to 1."""
        return sum(self._bits)

    def count_leading_zeros(self) -> int:
        """Return the number of leading zeros."""
        for i, bit in enumerate(self._bits):
            if bit:
                return i
        return self._size

    def count_leading_ones(self) -> int:
        """Return the number of leading ones."""
        for i, bit in enumerate(self._bits):
            if not bit:
                return i
        return self._size

    def __and__(self, other: "Bitmap") -> "Bitmap":
        """
        Bitwise AND with another bitmap.
        If sizes differ, the result is truncated to the smaller size.
        """
        size = min(self._size, other._size)
        result = Bitmap(size)
        for i in range(size):
            result._bits[i] = self._bits[i] and other._bits[i]
        return result

    def __invert__(self) -> "Bitmap":
        """Bitwise NOT (flip all bits)."""
        result = Bitmap(self._size)
        for i in range(self._size):
            result._bits[i] = not self._bits[i]
        return result

    def __or__(self, other: "Bitmap") -> "Bitmap":
        """
        Bitwise OR with another bitmap.
        If sizes differ, the result is truncated to the smaller size.
        """
        size = min(self._size, other._size)
        result = Bitmap(size)
        for i in range(size):
            result._bits[i] = self._bits[i] or other._bits[i]
        return result

    def get_indices_list(self) -> list[int]:
        """Return a list of indices where the bit is set to 1, in ascending order."""
        return [i for i, bit in enumerate(self._bits) if bit]

    def get_indices_set(self) -> Set[int]:
        """Return a set of indices where the bit is set to 1."""
        return {i for i, bit in enumerate(self._bits) if bit}

    def gather(self, items: list[Any]) -> list[Any]:
        """
        Return elements from items at indices where the bit is set to 1.

        Args:
            items: A list of objects. Length should match the bitmap size.

        Returns:
            A list of objects from items at positions where the bitmap bit is 1.
        """
        return [items[i] for i in self.get_indices_list()]

    def __repr__(self) -> str:
        """String representation: '1' for set bits, '0' for clear bits."""
        return "".join("1" if bit else "0" for bit in self._bits)


class ParallelPatternMatcher:
    """
    Pattern matcher for integer vectors.

    This class performs pattern matching on a vector of integers.
    It finds all positions where a given pattern occurs in the input data.
    """

    def __init__(self, pattern: list[int]) -> None:
        """
        Construct a ParallelPatternMatcher with the specified pattern.

        Args:
            pattern: The pattern to search for. Must not be empty.

        Raises:
            ValueError: If pattern is empty.
        """
        if not pattern:
            raise ValueError("Pattern must not be empty")
        self._pattern = pattern

    def match(self, data: list[int]) -> list[int]:
        """
        Match the pattern in the given data.

        Args:
            data: The data to search in.

        Returns:
            A sorted list of positions where the pattern starts.
            Returns an empty list if no matches are found.
        """
        pattern_len = len(self._pattern)
        result = []
        for i in range(len(data) - pattern_len + 1):
            if data[i:i + pattern_len] == self._pattern:
                result.append(i)
        return result


class RangePatternMatcher:
    """
    Range pattern matcher for integer vectors.

    This class performs range pattern matching on a vector of integers.
    It finds ranges that start with a start pattern and end with an end pattern.
    When multiple end patterns exist after a start pattern, it matches the first
    one (minimal range).
    """

    def __init__(self, start_pattern: list[int], end_pattern: list[int]) -> None:
        """
        Construct a RangePatternMatcher with start and end patterns.

        Args:
            start_pattern: The pattern marking the start of a range.
            end_pattern: The pattern marking the end of a range.

        Raises:
            ValueError: If either pattern is empty or has more than 5 elements.
        """
        if not start_pattern or not end_pattern:
            raise ValueError("Patterns must not be empty")
        if len(start_pattern) > 5 or len(end_pattern) > 5:
            raise ValueError("Patterns must have at most 5 elements")
        self._start = start_pattern
        self._end = end_pattern

    def match(self, data: list[int]) -> list[tuple[int, int]]:
        """
        Match ranges in the given data.

        Finds all ranges that start with the start pattern and end with the end
        pattern. When multiple end patterns exist after a start pattern, matches
        the first one (minimal range).

        Args:
            data: The data to search in.

        Returns:
            A list of (start_pos, end_pos) tuples where:
            - start_pos is the beginning index of the start pattern
            - end_pos is the exclusive index after the end pattern
            Returns an empty list if no ranges are found.
        """
        start_len = len(self._start)
        end_len = len(self._end)
        result = []

        i = 0
        while i <= len(data) - start_len:
            if data[i:i + start_len] == self._start:
                start_pos = i
                j = i + start_len
                while j <= len(data) - end_len:
                    if data[j:j + end_len] == self._end:
                        result.append((start_pos, j + end_len))
                        i = j + end_len
                        break
                    j += 1
                else:
                    i += 1
            else:
                i += 1

        return result