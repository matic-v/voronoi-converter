"""Step 8 tests: validation, automatic repair, STL and settings export."""

from __future__ import annotations

import io
import json
import time

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from voronoi_engine import test_models
from voronoi_engine.export import (export_stem, settings_document, settings_from_document,
                                   to_binary_stl)
from voronoi_engine.generate import generate
from voronoi_engine.mesh_io import load_stl
from voronoi_engine.repair import repair, validate
from voronoi_engine.settings import Settings
from voronoi_server.app import app

client = TestClient(app)


def cube_mesh():
    mesh = test_models.cube()
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


# --- validation and repair ----------------------------------------------------------

def test_validate_accepts_a_good_mesh_and_names_defects() -> None:
    vertices, faces = cube_mesh()
    assert validate(vertices, faces) == []
    assert any("open edge" in p for p in validate(vertices, faces[1:]))
    flipped = faces.copy()
    flipped[0] = flipped[0][::-1]
    assert any("inconsistent" in p for p in validate(vertices, flipped))
    assert any("inside-out" in p for p in validate(vertices, faces[:, ::-1]))
    assert validate(vertices, faces[:0]) == ["the result has no triangles"]


def test_repair_removes_stray_pieces_but_keeps_cavities() -> None:
    vertices, faces = cube_mesh()
    cavity = trimesh.creation.box(extents=(10, 10, 10)).apply_translation((0, 0, 25))
    cavity.invert()
    stray = trimesh.creation.box(extents=(2, 2, 2)).apply_translation((60, 0, 0))
    both = trimesh.util.concatenate([trimesh.Trimesh(vertices, faces, process=False), cavity, stray])
    v, f = np.asarray(both.vertices), np.asarray(both.faces)
    assert any("2 separate bodies" in p for p in validate(v, f))
    v2, f2, done = repair(v, f)
    assert validate(v2, f2) == [] and done == ["removed 1 stray piece(s)"]
    assert trimesh.Trimesh(v2, f2, process=False).volume == pytest.approx(50**3 - 10**3)


def test_repair_closes_a_small_hole_and_fixes_winding() -> None:
    sphere = test_models.sphere()
    vertices, faces = np.asarray(sphere.vertices), np.asarray(sphere.faces)
    holed = np.delete(faces, 100, axis=0)
    v, f, done = repair(vertices, holed)
    assert validate(v, f) == [] and any("closed hole" in d for d in done)

    flipped = faces.copy()
    flipped[[5, 50, 500]] = flipped[[5, 50, 500]][:, ::-1]
    v, f, done = repair(vertices, flipped)
    assert validate(v, f) == [] and "made the winding consistent (3 triangle(s) turned)" in done
    assert trimesh.Trimesh(v, f, process=False).volume == pytest.approx(sphere.volume)


def test_repair_removes_collapsed_and_duplicate_triangles() -> None:
    vertices, faces = cube_mesh()
    extra = np.vstack([faces, faces[:1], [[0, 0, 1]]])
    v, f, done = repair(vertices, extra)
    assert validate(v, f) == [] and len(f) == len(faces) and len(done) == 2


def test_unrepairable_mesh_stays_invalid() -> None:
    sphere = test_models.sphere()
    big_hole = np.asarray(sphere.faces)[200:]  # a large ragged opening: not closed blindly
    v, f, _ = repair(np.asarray(sphere.vertices), big_hole)
    assert validate(v, f) != []


def test_generate_repairs_its_own_defects_and_says_so(monkeypatch) -> None:
    import voronoi_engine.generate as pipeline
    mesh = load_stl(test_models.to_binary_stl(test_models.cube()), "cube.stl")
    real_fit = pipeline.fit_to_envelope

    def damaged(vertices, faces, body, passes=2):
        vertices, faces = real_fit(vertices, faces, body, passes)
        faces = faces.copy()
        faces[10] = faces[10][::-1]  # one flipped triangle
        return vertices, np.delete(faces, 20, axis=0)  # and one missing

    monkeypatch.setattr(pipeline, "fit_to_envelope", damaged)
    result = generate(mesh, Settings(surface_style="solid", interior_style="hollow", cleanup="off"),
                      "final")
    assert result.valid and result.problems == []
    assert any("repaired automatically" in w for w in result.warnings)

    def ruined(vertices, faces, body, passes=2):
        vertices, faces = real_fit(vertices, faces, body, passes)
        return vertices, faces[: len(faces) // 2]

    monkeypatch.setattr(pipeline, "fit_to_envelope", ruined)
    result = generate(mesh, Settings(surface_style="solid", interior_style="hollow"), "final")
    assert not result.valid and any("open edge" in p for p in result.problems)


# --- STL and settings documents -----------------------------------------------------

def test_binary_stl_is_well_formed_and_round_trips() -> None:
    vertices, faces = cube_mesh()
    data = to_binary_stl(vertices, faces)
    assert len(data) == 84 + 50 * len(faces) and not data.startswith(b"solid")
    assert int(np.frombuffer(data, dtype="<u4", count=1, offset=80)[0]) == len(faces)
    loaded = trimesh.load(io.BytesIO(data), file_type="stl")
    assert loaded.is_watertight and loaded.volume == pytest.approx(125000.0)
    np.testing.assert_allclose(loaded.face_normals, trimesh.Trimesh(vertices, faces).face_normals,
                               atol=1e-6)
    assert data == to_binary_stl(vertices, faces)
    assert load_stl(data, "again.stl").info.triangles == len(faces)  # our own loader accepts it


def test_export_file_names() -> None:
    assert export_stem("bracket.stl") == "bracket_voronoi"
    assert export_stem("My Part v2.STL") == "My Part v2_voronoi"
    assert export_stem("C:\\\\parts\\\\a:b?.stl") .endswith("_voronoi")
    assert "/" not in export_stem("../../etc/passwd.stl") and export_stem("") == "model_voronoi"


def test_settings_document_reproduces_the_settings() -> None:
    settings = Settings(surface_style="rib_lattice", random_seed=77, grading_mode="z")
    document = settings_document(settings, "part.stl",
                                 {"size_mm": [1, 2, 3], "triangles": 12, "volume_mm3": 6.0},
                                 {"voxel_mm": 0.5, "triangles": 100, "seconds": 3.2})
    assert json.loads(json.dumps(document)) == document
    assert document["input"]["file"] == "part.stl" and document["settings"]["random_seed"] == 77
    assert "seconds" not in document["result"]  # nothing that changes from run to run
    assert settings_from_document(document) == settings
    assert settings_from_document(document["settings"]) == settings


# --- API ----------------------------------------------------------------------------

def run_job(model_id: str, settings: dict, mode: str) -> str:
    job = client.post(f"/api/models/{model_id}/jobs", json={"settings": settings, "mode": mode}).json()["id"]
    deadline = time.monotonic() + 120
    while client.get(f"/api/jobs/{job}").json()["state"] not in ("done", "failed", "cancelled"):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert client.get(f"/api/jobs/{job}").json()["state"] == "done"
    return job


def test_export_through_the_api() -> None:
    upload = client.post("/api/models", files={
        "file": ("my bracket.stl", test_models.to_binary_stl(test_models.bracket()), "model/stl")})
    model = upload.json()["id"]
    settings = Settings(random_seed=5).model_dump(mode="json")

    preview = run_job(model, settings, "preview")
    refused = client.get(f"/api/jobs/{preview}/export.stl")
    assert refused.status_code == 409 and "Generate final" in refused.json()["message"]
    assert client.get(f"/api/jobs/{preview}/export.json").status_code == 409

    final = run_job(model, settings, "final")
    stl = client.get(f"/api/jobs/{final}/export.stl")
    assert stl.status_code == 200
    assert stl.headers["content-disposition"] == 'attachment; filename="my bracket_voronoi.stl"'
    exported = trimesh.load(io.BytesIO(stl.content), file_type="stl")
    assert exported.is_watertight and exported.is_winding_consistent and exported.volume > 0
    np.testing.assert_allclose(exported.extents, [100, 40, 60], atol=0.3)

    saved = client.get(f"/api/jobs/{final}/export.json")
    assert saved.headers["content-disposition"] == 'attachment; filename="my bracket_voronoi.json"'
    document = saved.json()
    assert document["settings"] == settings and document["input"]["file"] == "my bracket.stl"
    assert document["result"]["triangles"] == len(exported.faces)

    # Reproduce from the saved document: byte-identical STL.
    again = run_job(model, settings_from_document(document).model_dump(mode="json"), "final")
    assert client.get(f"/api/jobs/{again}/export.stl").content == stl.content
    assert client.get("/api/jobs/missing/export.stl").status_code == 404


# --- fast winding fix (the old library routine took minutes on large meshes) -------------

def test_fix_winding_turns_exactly_the_flipped_triangles() -> None:
    from voronoi_engine.repair import fix_winding
    sphere = test_models.sphere()
    faces = np.asarray(sphere.faces).copy()
    rng = np.random.default_rng(1)
    flipped = rng.choice(len(faces), 300, replace=False)
    faces[flipped] = faces[flipped][:, ::-1]
    fixed, turned = fix_winding(faces, len(sphere.vertices))
    assert turned == 300
    assert np.array_equal(fixed, np.asarray(sphere.faces))
    assert fix_winding(np.asarray(sphere.faces), len(sphere.vertices))[1] == 0


def test_fix_winding_keeps_cavities_wound_inwards() -> None:
    from voronoi_engine.repair import fix_winding
    outer = trimesh.creation.box(extents=(50, 50, 50))
    cavity = trimesh.creation.box(extents=(10, 10, 10))
    cavity.invert()
    both = trimesh.util.concatenate([outer, cavity])
    faces = np.asarray(both.faces).copy()
    faces[3] = faces[3][::-1]  # one outer triangle
    faces[len(outer.faces) + 2] = faces[len(outer.faces) + 2][::-1]  # one cavity triangle
    fixed, turned = fix_winding(faces, len(both.vertices))
    assert turned == 2
    assert validate(np.asarray(both.vertices), fixed) == []
    assert trimesh.Trimesh(both.vertices, fixed, process=False).volume == pytest.approx(50**3 - 10**3)


def test_repair_is_fast_on_millions_of_triangles() -> None:
    sphere = test_models.sphere(100.0, segments=950, rings=950)  # about 1.8 M triangles
    vertices, faces = np.asarray(sphere.vertices), np.asarray(sphere.faces).copy()
    rng = np.random.default_rng(2)
    flipped = rng.choice(len(faces), 1000, replace=False)
    faces[flipped] = faces[flipped][:, ::-1]
    started = time.perf_counter()
    v, f, done = repair(vertices, faces)
    assert time.perf_counter() - started < 20.0
    assert validate(v, f) == [] and "1000 triangle(s) turned" in done[-1]


# --- 3MF ------------------------------------------------------------------------------

def read_3mf(data: bytes):
    """Independent reader using only the standard library."""
    import xml.etree.ElementTree as ET
    import zipfile
    archive = zipfile.ZipFile(io.BytesIO(data))
    assert archive.testzip() is None
    assert set(archive.namelist()) == {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"}
    assert "3dmodel.model" in archive.read("_rels/.rels").decode()
    root = ET.fromstring(archive.read("3D/3dmodel.model"))
    ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    assert root.get("unit") == "millimeter"
    vertices = np.array([[float(v.get(k)) for k in "xyz"] for v in root.iterfind(".//m:vertex", ns)])
    faces = np.array([[int(t.get(k)) for k in ("v1", "v2", "v3")]
                      for t in root.iterfind(".//m:triangle", ns)])
    assert root.find(".//m:build/m:item", ns).get("objectid") == "1"
    return vertices, faces


def test_3mf_holds_the_mesh_exactly() -> None:
    from voronoi_engine.export import to_3mf
    mesh = load_stl(test_models.to_binary_stl(test_models.bracket()), "bracket.stl")
    result = generate(mesh, Settings(surface_style="solid", interior_style="hollow"), "final")
    data = to_3mf(result.vertices, result.faces, "bracket_voronoi")
    vertices, faces = read_3mf(data)
    assert np.array_equal(faces, result.faces)
    assert np.abs(vertices - result.vertices).max() <= 0.00005 + 1e-9  # written to 0.0001 mm
    assert validate(vertices, faces) == []
    assert data == to_3mf(result.vertices, result.faces, "bracket_voronoi")  # same bytes
    assert len(data) < 0.35 * len(to_binary_stl(result.vertices, result.faces))


def test_3mf_does_not_depend_on_the_number_of_threads(monkeypatch) -> None:
    from voronoi_engine import export
    sphere = test_models.sphere(60.0, segments=400, rings=400)  # big enough for several blocks
    monkeypatch.setattr(export, "BLOCK_VERTICES", 20_000)
    monkeypatch.setattr(export, "BLOCK_TRIANGLES", 30_000)
    files = {}
    for threads in ("1", "4"):
        monkeypatch.setenv("VORONOI_THREADS", threads)
        files[threads] = export.to_3mf(np.asarray(sphere.vertices), np.asarray(sphere.faces))
    assert files["1"] == files["4"]
    vertices, faces = read_3mf(files["4"])
    assert len(faces) == len(sphere.faces) and validate(vertices, faces) == []


def test_negative_and_small_coordinates_are_written_correctly() -> None:
    from voronoi_engine.export import to_3mf
    vertices = np.array([[-0.00004, -1.23456, 12.0], [0.5, -0.04999, -100.00005], [0, 0, 0], [1, 1, 1]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    back, _ = read_3mf(to_3mf(vertices, faces))
    np.testing.assert_allclose(back, np.round(vertices, 4), atol=1e-9)


def test_3mf_export_through_the_api() -> None:
    upload = client.post("/api/models", files={
        "file": ("part.stl", test_models.to_binary_stl(test_models.cube()), "model/stl")})
    model = upload.json()["id"]
    settings = Settings(surface_style="solid", interior_style="hollow").model_dump(mode="json")
    assert settings["export_format"] == "3mf"
    preview = run_job(model, settings, "preview")
    assert client.get(f"/api/jobs/{preview}/export.3mf").status_code == 409
    final = run_job(model, settings, "final")
    response = client.get(f"/api/jobs/{final}/export.3mf")
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="part_voronoi.3mf"'
    vertices, faces = read_3mf(response.content)
    assert validate(vertices, faces) == []
    for chosen in ("3mf", "stl"):
        document = client.get(f"/api/jobs/{final}/export.json?format={chosen}").json()
        assert document["settings"]["export_format"] == chosen
    assert client.get(f"/api/jobs/{final}/export.json?format=obj").status_code == 422


def test_fit_splits_a_corner_only_once() -> None:
    from voronoi_engine.meshing import _new_and_apart

    corner = np.array([[1.0, 2.0, 3.0]])
    found = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0 + 1e-6], [7.0, 0.0, 0.0]])
    assert _new_and_apart(found, corner, 1e-4).tolist() == [False, True, False, True]
