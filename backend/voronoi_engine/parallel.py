"""Run chunked work on all CPU cores.

The heavy steps evaluate a function at millions of independent points. They are cut
into chunks that run in threads: numpy and scipy release Python's lock during the
numerical work, and threads (unlike processes) share the large arrays without
copying. Every chunk writes only its own slice of the result, so the output is the
same, byte for byte, whatever the number of cores.

Work started from inside a chunk runs serially (no nested pools), and KD-tree
queries inside a chunk use one core, so the cores are not oversubscribed.

Set the environment variable VORONOI_THREADS to limit the number of threads.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

_local = threading.local()
_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()
MIN_CHUNK = 20_000


def thread_count() -> int:
    try:
        wanted = int(os.environ.get("VORONOI_THREADS", "0"))
    except ValueError:
        wanted = 0
    return max(1, wanted or os.cpu_count() or 1)


def in_worker() -> bool:
    return getattr(_local, "active", False)


def kd_workers() -> int:
    """`workers` for cKDTree queries: all cores, or one inside a parallel chunk."""
    return 1 if in_worker() or thread_count() == 1 else -1


def _get_pool() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=thread_count(), thread_name_prefix="voronoi")
        return _pool


def run_chunks(work: Callable[[int, int], None], total: int, max_chunk: int = 200_000,
               done: Callable[[int], None] | None = None) -> None:
    """Call work(start, stop) for consecutive chunks covering range(total).

    `done(stop)` is called on the calling thread after each chunk, in order (for
    progress reports and cancellation). Exceptions from `work` or `done` stop the run.
    """
    if total <= 0:
        return
    threads = thread_count()
    size = max(MIN_CHUNK, min(max_chunk, -(-total // (threads * 4))))
    parts = [(start, min(total, start + size)) for start in range(0, total, size)]
    if in_worker() or threads == 1 or len(parts) == 1:
        for start, stop in parts:
            work(start, stop)
            if done:
                done(stop)
        return

    def task(start: int, stop: int) -> None:
        _local.active = True
        try:
            work(start, stop)
        finally:
            _local.active = False

    futures = [_get_pool().submit(task, start, stop) for start, stop in parts]
    try:
        for (_, stop), future in zip(parts, futures):
            future.result()
            if done:
                done(stop)
    finally:
        for future in futures:
            future.cancel()  # anything not started yet, e.g. after a cancellation
        for future in futures:
            if not future.cancelled():
                try:
                    future.result()  # let running chunks finish before returning
                except Exception:  # noqa: BLE001 - the first error is already raised
                    pass


def map_in_parallel(work: Callable, items: list) -> list:
    """[work(item) for item in items], on all cores, results in order. For work that
    releases Python's lock (compression, numpy)."""
    if in_worker() or thread_count() == 1 or len(items) <= 1:
        return [work(item) for item in items]
    return list(_get_pool().map(work, items))
