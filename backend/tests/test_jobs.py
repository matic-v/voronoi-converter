"""Step 7 tests: background jobs, progress, cancel, automatic quality and memory cap."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voronoi_engine import memory, test_models
from voronoi_engine.generate import generate
from voronoi_engine.grid import choose_resolution
from voronoi_engine.mesh_io import load_stl
from voronoi_engine.progress import Cancelled, Job
from voronoi_engine.settings import Settings
from voronoi_server.app import app

client = TestClient(app)
SHELL = Settings(surface_style="solid", interior_style="hollow").model_dump(mode="json")


def upload(mesh) -> str:
    response = client.post("/api/models", files={
        "file": ("part.stl", test_models.to_binary_stl(mesh), "model/stl")})
    assert response.status_code == 200
    return response.json()["id"]


def start(model_id: str, settings: dict, mode: str = "preview") -> str:
    response = client.post(f"/api/models/{model_id}/jobs", json={"settings": settings, "mode": mode})
    assert response.status_code == 202, response.text
    return response.json()["id"]


def wait(job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["state"] in ("done", "failed", "cancelled"):
            return status
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {status}")


def unpack(data: bytes):
    length = int(np.frombuffer(data, dtype="<u4", count=1)[0])
    info = json.loads(data[4:4 + length])
    offset = 4 + length
    vertices = np.frombuffer(data, dtype="<f4", count=info["vertices"] * 3, offset=offset)
    faces = np.frombuffer(data, dtype="<u4", count=info["triangles"] * 3,
                          offset=offset + info["vertices"] * 12)
    assert offset + info["vertices"] * 12 + info["triangles"] * 12 == len(data)
    return info, vertices.reshape(-1, 3), faces.reshape(-1, 3)


# --- engine: progress and cancel ----------------------------------------------------

@pytest.fixture(scope="module")
def cube():
    return load_stl(test_models.to_binary_stl(test_models.cube()), "cube.stl")


def test_progress_runs_from_start_to_one_without_going_back(cube) -> None:
    seen: list[tuple[float, str]] = []
    generate(cube, Settings(), "preview", job=Job(lambda f, m: seen.append((f, m))))
    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions)
    assert fractions[0] < 0.1 and fractions[-1] == 1.0 and len(fractions) >= 8
    assert {"Measuring the body", "Building struts", "Meshing"} <= {m for _, m in seen}


def test_cancel_stops_the_pipeline_at_the_next_checkpoint(cube) -> None:
    calls = {"n": 0}

    def cancelled() -> bool:
        calls["n"] += 1
        return calls["n"] > 4

    started = time.perf_counter()
    with pytest.raises(Cancelled):
        generate(cube, Settings(), "final", job=Job(is_cancelled=cancelled))
    assert time.perf_counter() - started < 5.0
    assert generate(cube, Settings(), "preview").valid  # nothing left in a broken state


# --- API: jobs ----------------------------------------------------------------------

def test_job_runs_and_returns_the_mesh() -> None:
    model = upload(test_models.cube())
    job = start(model, SHELL)
    status = wait(job)
    assert status["state"] == "done" and status["progress"] == 1.0 and status["error"] is None
    info, vertices, faces = unpack(client.get(f"/api/jobs/{job}/result").content)
    assert info["valid"] and info["mode"] == "preview" and faces.max() < len(vertices)
    assert info["size_mm"] == pytest.approx([50, 50, 50], abs=info["voxel_mm"])


def test_job_failure_is_reported_in_plain_language() -> None:
    model = upload(test_models.cube(6.0))
    job = start(model, Settings(surface_cell_size=50.0, interior_style="hollow").model_dump(mode="json"))
    status = wait(job)
    assert status["state"] == "failed" and "surface cell size is too large" in status["error"]
    assert client.get(f"/api/jobs/{job}/result").status_code == 409


def test_invalid_requests() -> None:
    model = upload(test_models.cube())
    bad = dict(SHELL, shell_thickness=99)
    assert client.post(f"/api/models/{model}/jobs", json={"settings": bad}).status_code == 422
    empty = dict(SHELL, surface_style="none")
    refused = client.post(f"/api/models/{model}/jobs", json={"settings": empty})
    assert refused.status_code == 422 and "nothing to generate" in refused.text
    assert client.post("/api/models/missing/jobs", json={"settings": SHELL}).status_code == 404
    assert client.get("/api/jobs/missing").status_code == 404


def test_cancel_a_running_job() -> None:
    model = upload(test_models.cube(100.0))
    job = start(model, Settings().model_dump(mode="json"), "final")
    deadline = time.monotonic() + 30
    while client.get(f"/api/jobs/{job}").json()["state"] == "queued" and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.5)
    assert client.post(f"/api/jobs/{job}/cancel").status_code == 200
    started = time.monotonic()
    status = wait(job, timeout=30)
    assert status["state"] == "cancelled"
    assert time.monotonic() - started < 10.0  # stops at the next checkpoint
    assert client.get(f"/api/jobs/{job}/result").status_code == 409
    # The engine is usable straight after.
    assert wait(start(model, SHELL))["state"] == "done"


def test_a_new_job_cancels_the_older_one() -> None:
    model = upload(test_models.cube(100.0))
    first = start(model, Settings().model_dump(mode="json"), "final")
    time.sleep(0.3)
    second = start(model, SHELL, "preview")
    assert wait(second)["state"] == "done"
    assert client.get(f"/api/jobs/{first}").json()["state"] == "cancelled"


def test_progress_is_visible_while_running() -> None:
    model = upload(test_models.cube(100.0))
    job = start(model, Settings().model_dump(mode="json"), "final")
    seen = set()
    while True:
        status = client.get(f"/api/jobs/{job}").json()
        if status["state"] in ("done", "failed", "cancelled"):
            break
        seen.add((status["progress"], status["message"]))
        time.sleep(0.1)
    assert status["state"] == "done"
    assert len({p for p, _ in seen}) >= 4 and len({m for _, m in seen}) >= 3


# --- automatic quality and memory cap -------------------------------------------------

def test_final_resolution_follows_quality_and_thinnest_feature() -> None:
    size = (100, 40, 60)
    normal = choose_resolution(size, Settings(), "final")
    assert normal.voxel_mm == pytest.approx(0.5) and normal.voxels_across_feature == pytest.approx(4.0)
    assert not normal.capped
    assert choose_resolution(size, Settings(quality="draft"), "final").voxels_across_feature == pytest.approx(3.0, abs=0.01)
    assert choose_resolution(size, Settings(quality="fine"), "final").voxels_across_feature == pytest.approx(6.0, abs=0.01)
    thin = choose_resolution(size, Settings(strut_diameter=1.0), "final")
    assert thin.voxel_mm == pytest.approx(0.25)


def test_quality_limit_caps_large_parts_and_says_so() -> None:
    big = choose_resolution((250, 250, 250), Settings(strut_diameter=0.8), "final")
    assert big.limited_by == "quality" and big.voxel_mm > big.wanted_mm
    mesh = load_stl(test_models.to_binary_stl(test_models.cube(120.0)), "big.stl")
    result = generate(mesh, Settings(surface_style="solid", interior_style="hollow",
                                     shell_thickness=0.8, quality="draft"), "final")
    assert result.quality_capped
    assert any("limit of this quality level" in w and "A higher quality" in w for w in result.warnings)


def test_memory_budget_caps_the_grid(monkeypatch) -> None:
    monkeypatch.setenv("VORONOI_MEMORY_MB", "200")
    assert memory.memory_budget_mb() == 200
    capped = choose_resolution((100, 100, 100), Settings(), "final")
    assert capped.limited_by == "memory"
    nodes = np.prod(np.array([100, 100, 100]) / capped.voxel_mm + 10)
    assert nodes * memory.BYTES_PER_NODE <= 200 * 2**20 * 1.02
    mesh = load_stl(test_models.to_binary_stl(test_models.cube()), "cube.stl")
    monkeypatch.setenv("VORONOI_MEMORY_MB", "64")
    result = generate(mesh, Settings(surface_style="solid", interior_style="hollow"), "final")
    assert result.valid and any("memory budget" in w for w in result.warnings)


def test_memory_budget_default_is_a_share_of_system_memory(monkeypatch) -> None:
    monkeypatch.delenv("VORONOI_MEMORY_MB", raising=False)
    assert memory.system_memory_mb() > 256
    assert memory.memory_budget_mb() == pytest.approx(
        min(0.4 * memory.system_memory_mb(), 8192))


def test_preview_spends_more_to_keep_thin_features_visible() -> None:
    size = (100, 100, 100)
    default = choose_resolution(size, Settings(), "preview")
    assert default.voxel_mm == pytest.approx(1.04, abs=0.02)  # normal preview budget
    thin = choose_resolution(size, Settings(strut_diameter=1.5), "preview")
    assert thin.voxels_across_feature >= 1.49  # stretched budget keeps 1.5 voxels across
    too_thin = choose_resolution(size, Settings(strut_diameter=0.8), "preview")
    assert too_thin.voxels_across_feature < 1.5  # beyond the stretch: stays coarse and warns


def test_preview_warns_when_it_cannot_show_the_thinnest_feature() -> None:
    mesh = load_stl(test_models.to_binary_stl(test_models.cube(100.0)), "cube.stl")
    result = generate(mesh, Settings(surface_style="solid", interior_style="hollow",
                                     shell_thickness=0.8), "preview")
    assert any("too coarse to show the thinnest feature" in w for w in result.warnings)
    fine = generate(mesh, Settings(surface_style="solid", interior_style="hollow"), "preview")
    assert not any("too coarse" in w for w in fine.warnings)
