"""Progress reporting and cancellation hook passed through the pipeline."""

from __future__ import annotations

from typing import Callable


class Cancelled(Exception):
    """Raised inside the pipeline when the job was cancelled."""


class Job:
    """No-op by default. `report` raises Cancelled once cancellation is requested."""

    def __init__(self, on_progress: Callable[[float, str], None] | None = None,
                 is_cancelled: Callable[[], bool] | None = None,
                 start: float = 0.0, end: float = 1.0):
        self._on_progress = on_progress
        self._is_cancelled = is_cancelled
        self._start, self._end = start, end

    def report(self, fraction: float, message: str = "") -> None:
        if self._is_cancelled and self._is_cancelled():
            raise Cancelled()
        if self._on_progress:
            self._on_progress(self._start + (self._end - self._start) * fraction, message)

    def sub(self, start: float, end: float) -> "Job":
        """A job that maps its 0..1 onto start..end of this one."""
        span = self._end - self._start
        return Job(self._on_progress, self._is_cancelled,
                   self._start + span * start, self._start + span * end)
