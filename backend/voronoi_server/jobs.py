"""Background generation jobs with progress and cancel.

One worker thread runs jobs in order. Starting a job cancels every older job that
has not finished: only the newest request matters to the UI.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from voronoi_engine.generate import GenerationError, GenerationResult
from voronoi_engine.progress import Cancelled, Job

log = logging.getLogger("voronoi.jobs")

MAX_RECORDS = 30
MAX_KEPT_RESULTS = 2
FINISHED = ("done", "failed", "cancelled")


@dataclass
class JobRecord:
    id: str
    model_id: str
    mode: str
    state: str = "queued"  # queued | running | done | failed | cancelled
    progress: float = 0.0
    message: str = "Waiting"
    error: str | None = None
    result: bytes | None = None
    meta: dict = field(default_factory=dict)  # what the export endpoints need
    cancel: threading.Event = field(default_factory=threading.Event)

    def status(self) -> dict:
        return {"id": self.id, "model_id": self.model_id, "mode": self.mode, "state": self.state,
                "progress": round(self.progress, 3), "message": self.message, "error": self.error}


class JobManager:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="generate")
        self._lock = threading.Lock()
        self._jobs: "OrderedDict[str, JobRecord]" = OrderedDict()

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(self, model_id: str, mode: str,
               work: Callable[[Job], tuple[GenerationResult, bytes]],
               meta: dict | None = None) -> JobRecord:
        """`work(job)` runs the pipeline and returns (result, packed bytes)."""
        record = JobRecord(id=uuid.uuid4().hex[:12], model_id=model_id, mode=mode, meta=meta or {})
        with self._lock:
            for other in self._jobs.values():
                if other.state not in FINISHED:
                    other.cancel.set()
            self._jobs[record.id] = record
            while len(self._jobs) > MAX_RECORDS:
                self._jobs.popitem(last=False)
        self._executor.submit(self._run, record, work)
        return record

    def cancel(self, job_id: str) -> JobRecord | None:
        record = self.get(job_id)
        if record and record.state not in FINISHED:
            record.cancel.set()
            if record.state == "queued":
                record.state, record.message = "cancelled", "Cancelled"
        return record

    def _run(self, record: JobRecord, work) -> None:
        if record.cancel.is_set():
            record.state, record.message = "cancelled", "Cancelled"
            return
        record.state, record.message = "running", "Starting"

        def on_progress(fraction: float, message: str) -> None:
            record.progress = max(record.progress, min(1.0, fraction))
            if message:
                record.message = message

        try:
            result, packed = work(Job(on_progress, record.cancel.is_set))
            record.result = packed
            record.meta["result_info"] = result.info()
            record.progress, record.state, record.message = 1.0, "done", "Done"
            self._drop_old_results(keep=record)
        except Cancelled:
            record.state, record.message = "cancelled", "Cancelled"
        except GenerationError as error:
            record.state, record.error, record.message = "failed", str(error), "Failed"
        except MemoryError:
            record.state, record.message = "failed", "Failed"
            record.error = ("Not enough memory for this part at this detail level. "
                            "Try a lower quality, or thicker features.")
        except Exception as error:  # noqa: BLE001 - reported to the user, logged with traceback
            log.exception("generation job %s failed", record.id)
            record.state, record.message = "failed", "Failed"
            record.error = f"Generation failed unexpectedly ({type(error).__name__}: {error})."

    def _drop_old_results(self, keep: JobRecord) -> None:
        with self._lock:
            with_result = [j for j in self._jobs.values() if j.result is not None and j is not keep]
            for old in with_result[:max(0, len(with_result) - (MAX_KEPT_RESULTS - 1))]:
                old.result = None
