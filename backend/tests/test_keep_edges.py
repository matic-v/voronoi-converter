"""Kept edges: sharp-edge detection, numbering, and solid zones round the kept edges."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from pydantic import ValidationError

import voronoi_engine.generate as pipeline
from voronoi_engine import test_models
from voronoi_engine.edges import find_edges
from voronoi_engine.generate import GenerationError, generate
from voronoi_engine.mesh_io import LoadedMesh, load_stl
from voronoi_engine.settings import Settings
from voronoi_server.app import app


def load(mesh: trimesh.Trimesh, name: str = "part.stl") -> LoadedMesh:
    return load_stl(test_models.to_binary_stl(mesh), name)


@pytest.fixture(scope="module")
def cube() -> LoadedMesh:
    return load(test_models.cube())


# --- finding and numbering edges ----------------------------------------------------

def test_cube_has_twelve_edges_of_50mm(cube) -> None:
    edges = cube.edges
    assert edges.count == 12
    np.testing.assert_allclose(edges.lengths(), 50.0)


def test_smooth_sphere_has_no_sharp_edges() -> None:
    assert load(test_models.sphere()).edges.count == 0


def test_bracket_has_24_edges() -> None:
    edges = load(test_models.bracket()).edges
    assert edges.count == 24  # 8 along the extrusion, 8 round each end profile
    assert sorted(np.round(edges.lengths(), 3))[:4] == [2.0, 2.0, 6.0, 6.0]  # flange end, step


def test_a_round_rim_is_one_edge() -> None:
    cylinder = trimesh.creation.cylinder(radius=10.0, height=20.0, sections=64)
    edges = load(cylinder).edges
    # Top and bottom rim; the side (radius 10 mm) is too gently curved to count as an edge.
    assert (edges.sharp_count, edges.curved_count) == (2, 0)
    np.testing.assert_allclose(edges.lengths(), 2 * 64 * 10 * np.sin(np.pi / 64), rtol=1e-6)


# --- curved (filleted) edges ----------------------------------------------------------

def test_fillet_and_round_are_curved_edges() -> None:
    edges = load(test_models.filleted_bracket()).edges
    assert (edges.sharp_count, edges.curved_count) == (12, 2)
    assert edges.curved_from == 12
    # Each is drawn as the 11 lines between its 12 facets, along the 40 mm extrusion.
    for curved in range(12, 14):
        assert (edges.edge_of_segment == curved).sum() == 11
        np.testing.assert_allclose(edges.lengths()[curved], 11 * 40.0)
    # The lines lie on the arcs: the 5 mm fillet centred at (13, 13), the 8 mm round at (8, 8)
    # (x, z of the profile; the part is centred in x).
    radii = set()
    for curved in range(12, 14):
        points = edges.segments[edges.edge_of_segment == curved].reshape(-1, 3)
        for cx, cz, r in ((13.0, 13.0, 5.0), (8.0, 8.0, 8.0)):
            centre_x = cx - 30.0  # the part is placed with x from -30 to 30
            if np.allclose(np.hypot(points[:, 0] - centre_x, points[:, 2] - cz), r, atol=1e-4):
                radii.add(r)
    assert radii == {5.0, 8.0}


def test_small_holes_and_rounds_count_large_curves_do_not() -> None:
    for radius, curved in ((3.0, 1), (5.0, 1), (10.0, 0)):
        cylinder = trimesh.creation.cylinder(radius=radius, height=20.0, sections=64)
        assert load(cylinder).edges.curved_count == curved
    assert load(test_models.sphere()).edges.count == 0
    assert load(test_models.sphere(100.0, 500, 501)).edges.count == 0


def test_curved_edges_do_not_renumber_sharp_edges() -> None:
    from voronoi_engine.edges import _MeshEdges, _sharp_edges
    mesh = load(test_models.filleted_bracket())
    _, sharp_ids, sharp_count = _sharp_edges(_MeshEdges(mesh.vertices, mesh.faces), 30.0)
    assert np.array_equal(mesh.edges.edge_of_segment[: len(sharp_ids)], sharp_ids)
    assert sharp_count == mesh.edges.sharp_count


def test_kept_fillet_is_solid() -> None:
    mesh = load(test_models.filleted_bracket())
    settings = Settings(surface_style="rib_lattice", interior_style="hollow", keep_edges=[12, 13])
    result, field = exact_field_of(mesh, settings)
    assert result.valid
    # Points 1 mm inside both arcs, all along the extrusion, are solid.
    t = np.radians(np.linspace(185.0, 265.0, 9))
    y = np.linspace(-18.0, 18.0, 7)
    probes = []
    for cx, cz, r in ((13.0, 13.0, 5.0), (8.0, 8.0, 8.0)):
        rr = r + 1.0 if cx == 13.0 else r - 1.0  # fillet: material outside the arc; round: inside
        for angle in t:
            for yy in y:
                probes.append((cx - 30.0 + rr * np.cos(angle), yy, cz + rr * np.sin(angle)))
    assert (field(np.array(probes)) < 0).all()
    plain = generate(mesh, settings.model_copy(update={"keep_edges": []}), "final")
    assert result.info()["volume_mm3"] > plain.info()["volume_mm3"]


def test_edge_numbers_are_stable_for_the_same_file() -> None:
    first = load(test_models.bracket()).edges
    again = load(test_models.bracket()).edges
    assert np.array_equal(first.edge_of_segment, again.edge_of_segment)
    assert np.array_equal(first.segments, again.segments)


def test_find_edges_angle_threshold() -> None:
    mesh = test_models.cube()
    assert find_edges(np.asarray(mesh.vertices), np.asarray(mesh.faces), angle_deg=95.0).count == 0


# --- settings -----------------------------------------------------------------------

def test_keep_settings() -> None:
    s = Settings(keep_edges=[5, 1, 5, 3])
    assert s.keep_edges == [1, 3, 5] and s.keep_width == 2.0
    assert Settings(**s.to_saved_json()["settings"]) == s
    with pytest.raises(ValidationError):
        Settings(keep_edges=[-1])
    with pytest.raises(ValidationError):
        Settings(keep_width=0.1)
    assert Settings().keep_edges == []


# --- geometry -----------------------------------------------------------------------

def exact_field_of(mesh: LoadedMesh, settings: Settings):
    captured = {}
    original = pipeline.ExactField

    def capture(*args, **kwargs):
        captured["field"] = original(*args, **kwargs)
        return captured["field"]

    pipeline.ExactField = capture
    try:
        result = generate(mesh, settings.model_copy(update={"smoothness": "smooth"}), "final")
    finally:
        pipeline.ExactField = original
    return result, captured["field"]


def edge_probe_points(cube: LoadedMesh, edge: int, inset_a: float, inset_b: float) -> np.ndarray:
    """Points along a cube edge, `inset_a` in from one face and `inset_b` from the other."""
    (a, b), = cube.edges.segments_of([edge])
    t = np.linspace(0.1, 0.9, 25)[:, None]
    along = a + (b - a) * t
    axis = int(np.argmax(np.abs(b - a)))
    others = [k for k in range(3) if k != axis]
    centre = np.array([0.0, 0.0, 25.0])
    points = along.copy()
    for k, inset in zip(others, (inset_a, inset_b)):
        points[:, k] -= np.sign(along[:, k] - centre[k]) * inset
    return points


def test_kept_edge_is_drawn_by_a_strut_not_filled_solid(cube) -> None:
    settings = Settings(surface_style="none", interior_style="struts", keep_edges=[0], keep_width=2.0)
    result, field = exact_field_of(cube, settings)
    _, plain = exact_field_of(cube, settings.model_copy(update={"keep_edges": []}))
    assert result.valid
    # The edge itself, and 0.8 mm either side of it on both faces, near the surface: solid.
    for inset_a, inset_b in ((0.3, 0.3), (0.5, 0.8), (0.8, 0.5)):
        assert (field(edge_probe_points(cube, 0, inset_a, inset_b)) < 0).all()
    # Deeper than the strut (2 mm deep) plus the blend radius: nothing was added.
    deep = edge_probe_points(cube, 0, 3.5, 3.5)
    assert np.array_equal(field(deep) < 0, plain(deep) < 0)
    # Further along the face than half the width plus the blend radius: nothing was added.
    beside = edge_probe_points(cube, 0, 0.5, 3.0)
    assert np.array_equal(field(beside) < 0, plain(beside) < 0)


def test_kept_edge_on_a_rib_lattice_is_a_rib(cube) -> None:
    """With ribs, the edge member is a rib like the others: as wide and as deep as the rib width."""
    settings = Settings(surface_style="rib_lattice", interior_style="hollow", keep_edges=[0],
                        keep_width=2.0)
    result, field = exact_field_of(cube, settings)
    assert result.valid
    assert (field(edge_probe_points(cube, 0, 0.5, 0.5)) < 0).all()
    assert (field(edge_probe_points(cube, 0, 1.5, 0.6)) < 0).all()  # 1.5 deep under one face
    assert (field(edge_probe_points(cube, 0, 0.6, 1.5)) < 0).all()  # ... and under the other
    assert (field(edge_probe_points(cube, 0, 3.0, 3.0)) > 0).all()  # hollow inside, not a block


def test_keeping_all_edges_restores_the_outer_dimensions(cube) -> None:
    """Surface None + Struts alone does not reach every extreme; a kept frame does."""
    every = list(range(cube.edges.count))
    result = generate(cube, Settings(surface_style="none", interior_style="struts",
                                     keep_edges=every), "final")
    out = trimesh.Trimesh(result.vertices, result.faces, process=False)
    assert result.valid and sum(1 for part in out.split(only_watertight=False) if part.volume > 0) == 1
    # (Thin edge struts do not catch every stray strut piece near a face; those are removed.)
    np.testing.assert_allclose(out.bounds[0], cube.info.bounds_min, atol=0.3)
    np.testing.assert_allclose(out.bounds[1], cube.info.bounds_max, atol=0.3)


@pytest.mark.parametrize("surface", ["perforated", "rib_lattice", "embossed", "none"])
def test_kept_edges_work_with_every_skin(cube, surface: str) -> None:
    plain = generate(cube, Settings(surface_style=surface), "preview")
    kept = generate(cube, Settings(surface_style=surface, keep_edges=[0, 5, 11]), "preview")
    assert kept.valid
    assert kept.info()["volume_mm3"] > plain.info()["volume_mm3"]


def test_kept_edges_that_do_not_belong_to_the_model_are_refused(cube) -> None:
    with pytest.raises(GenerationError, match="do not match this model"):
        generate(cube, Settings(keep_edges=[12]), "preview")


def test_wider_keep_zone_keeps_more(cube) -> None:
    volume = [generate(cube, Settings(surface_style="rib_lattice", keep_edges=[0], keep_width=w),
                       "preview").info()["volume_mm3"] for w in (2.0, 6.0)]
    assert volume[1] > volume[0]


# --- API ----------------------------------------------------------------------------

def test_edges_endpoint() -> None:
    client = TestClient(app)
    upload = client.post("/api/models", files={
        "file": ("bracket.stl", test_models.to_binary_stl(test_models.bracket()), "model/stl")})
    body = upload.json()
    assert body["edges"] == {"count": 24, "sharp": 24, "curved": 0, "too_many": False}
    data = client.get(f"/api/models/{body['id']}/edges").content
    segments, count = np.frombuffer(data, dtype="<u4", count=2)
    assert count == 24 and len(data) == 8 + segments * 24 + segments * 4
    ids = np.frombuffer(data, dtype="<u4", offset=8 + segments * 24)
    assert set(ids.tolist()) == set(range(24))
    assert client.get("/api/models/missing/edges").status_code == 404
