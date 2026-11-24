"""Step 2 tests: STL loading and validation."""

from __future__ import annotations

import time

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from voronoi_engine import test_models
from voronoi_engine.mesh_io import MeshValidationError, load_stl
from voronoi_server.app import app

client = TestClient(app)


def stl(mesh: trimesh.Trimesh, ascii_format: bool = False) -> bytes:
    data = mesh.export(file_type="stl_ascii" if ascii_format else "stl")
    return data.encode() if isinstance(data, str) else data


def problems_of(data: bytes) -> str:
    with pytest.raises(MeshValidationError) as caught:
        load_stl(data, "bad.stl")
    return "\n".join(caught.value.problems)


@pytest.mark.parametrize("name", list(test_models.TEST_MODELS))
@pytest.mark.parametrize("ascii_format", [False, True])
def test_valid_models_load(name: str, ascii_format: bool) -> None:
    source = test_models.build(name)
    loaded = load_stl(stl(source, ascii_format), f"{name}.stl")
    assert loaded.info.triangles == len(source.faces)
    np.testing.assert_allclose(loaded.info.size_mm, source.extents, atol=1e-4)
    assert loaded.info.volume_mm3 == pytest.approx(source.volume, rel=1e-5)
    assert loaded.info.bodies == 1
    assert loaded.warnings == []
    assert loaded.to_trimesh().is_watertight


def test_open_edges_are_named_and_counted() -> None:
    text = problems_of(stl(test_models.broken_open_cube()))
    assert "mesh has 4 open edges" in text


def test_non_manifold_edge_is_named() -> None:
    text = problems_of(stl(test_models.broken_non_manifold()))
    assert "1 non-manifold edge " in text


def test_flipped_face_is_named() -> None:
    text = problems_of(stl(test_models.broken_flipped_face()))
    assert "normals are inconsistent" in text and "3 edges" in text


def test_inside_out_mesh_is_refused() -> None:
    mesh = test_models.cube()
    mesh.invert()
    assert "inside-out" in problems_of(stl(mesh))


def test_flat_mesh_is_refused() -> None:
    mesh = test_models.cube()
    mesh.vertices[:, 2] = 0.0
    assert "flat" in problems_of(stl(mesh))


def test_several_problems_are_all_listed() -> None:
    mesh = trimesh.util.concatenate(
        [test_models.broken_open_cube(), test_models.broken_non_manifold().apply_translation((200, 0, 0))])
    text = problems_of(stl(mesh))
    assert "open edges" in text and "non-manifold" in text


@pytest.mark.parametrize("data", [b"", b"this is not an stl file at all", b"\x00" * 50])
def test_garbage_is_refused(data: bytes) -> None:
    with pytest.raises(MeshValidationError):
        load_stl(data, "garbage.stl")


def test_error_message_lists_problems_and_suggests_repair() -> None:
    with pytest.raises(MeshValidationError) as caught:
        load_stl(stl(test_models.broken_open_cube()), "lid.stl")
    message = caught.value.message
    assert "lid.stl" in message and "open edges" in message and "does not repair" in message


def test_two_bodies_are_refused() -> None:
    two = trimesh.util.concatenate(
        [test_models.cube(), test_models.cube().apply_translation((100, 0, 0))])
    assert "2 separate bodies" in problems_of(stl(two))


def test_cavity_is_one_body() -> None:
    cavity = trimesh.creation.box(extents=(20, 20, 20)).apply_translation((0, 0, 25))
    cavity.invert()
    loaded = load_stl(stl(trimesh.util.concatenate([test_models.cube(), cavity])))
    assert loaded.info.bodies == 1
    assert loaded.info.volume_mm3 == pytest.approx(50**3 - 20**3)


def test_warning_for_large_part() -> None:
    loaded = load_stl(stl(test_models.cube(300.0)))
    assert any("larger than" in w for w in loaded.warnings)


def test_500k_triangles_load_in_under_5_seconds() -> None:
    data = stl(test_models.sphere(100.0, segments=500, rings=501))
    started = time.perf_counter()
    loaded = load_stl(data)
    elapsed = time.perf_counter() - started
    assert loaded.info.triangles == 500_000
    assert elapsed < 5.0, f"took {elapsed:.2f} s"


def test_api_upload_valid() -> None:
    response = client.post(
        "/api/models", files={"file": ("cube.stl", stl(test_models.cube()), "model/stl")})
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "cube.stl" and body["id"]
    assert body["info"]["size_mm"] == [50.0, 50.0, 50.0]
    assert body["info"]["triangles"] == 12
    assert body["info"]["volume_mm3"] == pytest.approx(125000.0)


def test_api_upload_broken_is_refused() -> None:
    response = client.post(
        "/api/models",
        files={"file": ("open.stl", stl(test_models.broken_open_cube()), "model/stl")})
    assert response.status_code == 422
    body = response.json()
    assert any("4 open edges" in p for p in body["problems"])
    assert "open.stl" in body["message"]
