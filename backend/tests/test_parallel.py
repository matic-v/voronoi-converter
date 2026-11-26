"""Multi-core execution: same results whatever the number of threads."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from voronoi_engine import parallel, test_models
from voronoi_engine.export import to_binary_stl
from voronoi_engine.generate import generate
from voronoi_engine.mesh_io import load_stl
from voronoi_engine.progress import Cancelled, Job
from voronoi_engine.settings import Settings


def test_run_chunks_covers_everything_once_and_reports_in_order(monkeypatch) -> None:
    monkeypatch.setenv("VORONOI_THREADS", "4")
    seen = np.zeros(1_000_003, dtype=np.int64)
    threads = set()

    def work(start: int, stop: int) -> None:
        seen[start:stop] += 1
        threads.add(threading.get_ident())

    done = []
    parallel.run_chunks(work, len(seen), max_chunk=50_000, done=done.append)
    assert (seen == 1).all()
    assert done == sorted(done) and done[-1] == len(seen)
    assert len(threads) > 1  # really ran in parallel


def test_nested_work_runs_serially(monkeypatch) -> None:
    monkeypatch.setenv("VORONOI_THREADS", "4")
    inner_ok = []

    def outer(start: int, stop: int) -> None:
        assert parallel.in_worker() and parallel.kd_workers() == 1
        hits = np.zeros(100_000)
        parallel.run_chunks(lambda a, b: hits.__setitem__(slice(a, b), 1), len(hits))
        inner_ok.append(hits.all())

    parallel.run_chunks(outer, 200_000, max_chunk=50_000)
    assert inner_ok and all(inner_ok)


def test_an_error_or_cancellation_stops_the_run(monkeypatch) -> None:
    monkeypatch.setenv("VORONOI_THREADS", "4")

    def work(start: int, stop: int) -> None:
        if start >= 100_000:
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        parallel.run_chunks(work, 400_000, max_chunk=50_000)

    def cancel_after_first(stop: int) -> None:
        raise Cancelled()

    with pytest.raises(Cancelled):
        parallel.run_chunks(lambda a, b: None, 400_000, max_chunk=50_000, done=cancel_after_first)


@pytest.mark.parametrize("settings", [
    Settings(),
    Settings(surface_style="rib_lattice", smoothness="smooth", keep_edges=[0, 4],
             grading_mode="from_skin", grading_target="both"),
])
def test_results_do_not_depend_on_the_number_of_threads(monkeypatch, settings: Settings) -> None:
    mesh = load_stl(test_models.to_binary_stl(test_models.cube()), "cube.stl")
    stl = {}
    for threads in ("1", "4"):
        monkeypatch.setenv("VORONOI_THREADS", threads)
        result = generate(mesh, settings, "final", job=Job())
        stl[threads] = to_binary_stl(result.vertices, result.faces)
    assert stl["1"] == stl["4"]
