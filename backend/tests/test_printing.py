"""Printing without supports: bed face, self-supporting struts, pointed hole tops, suggestion."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scipy import ndimage

import voronoi_engine.generate as pipeline
import voronoi_engine.supports as supports
from voronoi_engine import test_models
from voronoi_engine.body_field import BodyField
from voronoi_engine.generate import generate
from voronoi_engine.grid import Grid
from voronoi_engine.lattice import poisson_seeds, stretched_voronoi_edges
from voronoi_engine.mesh_io import LoadedMesh, load_stl
from voronoi_engine.orientation import bed_rotation, oriented, suggest_bed_normal
from voronoi_engine.settings import Settings, field_metadata
from voronoi_server.app import app


def load(mesh: trimesh.Trimesh, name: str = "part.stl") -> LoadedMesh:
    return load_stl(test_models.to_binary_stl(mesh), name)


@pytest.fixture(scope="module")
def bracket() -> LoadedMesh:
    return load(test_models.bracket())


@pytest.fixture(scope="module")
def cube() -> LoadedMesh:
    return load(test_models.cube())


def solid_grid_of(mesh: LoadedMesh, settings: Settings, mode: str = "final"):
    """Generate, and return (result, the voxel solid before meshing, voxel size)."""
    captured = {}
    original = pipeline.keep_largest_body

    def capture(field, h):
        removed = original(field, h)
        captured["solid"], captured["h"] = field.copy(), h
        return removed

    pipeline.keep_largest_body = capture
    try:
        result = generate(mesh, settings, mode)
    finally:
        pipeline.keep_largest_body = original
    return result, captured["solid"], captured["h"]


def islands(solid: np.ndarray) -> int:
    """Slice into layers like a slicer: count pieces of a layer with nothing under them in
    the layer below (one voxel sideways per voxel up allowed, i.e. 45 degrees)."""
    occupied = solid < 0
    layers = np.flatnonzero(occupied.any(axis=(0, 1)))
    grow = ndimage.generate_binary_structure(2, 2)
    count = 0
    for k in layers[1:]:
        labels, n = ndimage.label(occupied[:, :, k])
        below = ndimage.binary_dilation(occupied[:, :, k - 1], structure=grow)
        held = np.zeros(n + 1, dtype=bool)
        held[np.unique(labels[below & (labels > 0)])] = True
        count += int((~held[1:]).sum())
    return count


# --- settings -------------------------------------------------------------------------

def test_printing_settings() -> None:
    s = Settings()
    assert s.bed_normal == [0.0, 0.0, -1.0] and s.self_supporting and s.pointed_holes
    assert (s.cell_stretch, s.overhang_angle, s.max_bridge) == (2.0, 45.0, 8.0)
    assert Settings(bed_normal=[0, 3, 0]).bed_normal == [0.0, 1.0, 0.0]
    for bad in ([0, 0, 0], [1, 2], [float("nan"), 0, 1]):
        with pytest.raises(ValidationError):
            Settings(bed_normal=bad)
    kinds = {m["key"]: m["kind"] for m in field_metadata()}
    assert kinds["self_supporting"] == kinds["pointed_holes"] == "bool"
    assert kinds["bed_normal"] == "list"


# --- bed face / orientation -----------------------------------------------------------

def test_bed_rotation_turns_the_face_down() -> None:
    rng = np.random.default_rng(0)
    normals = np.vstack([rng.normal(size=(50, 3)), [[0, 0, 1], [0, 0, -1], [1, 0, 0]]])
    for n in normals:
        r = bed_rotation(n)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0)
        np.testing.assert_allclose(r @ (n / np.linalg.norm(n)), [0, 0, -1], atol=1e-12)


def test_as_loaded_leaves_the_part_untouched(bracket) -> None:
    assert oriented(bracket, [0, 0, -1]) is bracket


def test_part_on_another_face(bracket) -> None:
    on_end = oriented(bracket, [1, 0, 0])
    np.testing.assert_allclose(on_end.info.size_mm, [60, 40, 100], atol=1e-9)
    assert on_end.info.bounds_min[2] == pytest.approx(bracket.info.bounds_min[2])
    assert on_end.edges.count == bracket.edges.count  # kept-edge numbers stay valid
    assert np.array_equal(on_end.faces, bracket.faces)
    result = generate(bracket, Settings(surface_style="rib_lattice", bed_normal=[1, 0, 0],
                                        keep_edges=[0, 1, 2]), "final")
    assert result.valid
    np.testing.assert_allclose(result.info()["size_mm"], [60, 40, 100], atol=0.3)


# --- self-supporting struts -----------------------------------------------------------

def test_stretched_cells_have_steeper_struts(cube) -> None:
    lo, hi = np.asarray(cube.info.bounds_min), np.asarray(cube.info.bounds_max)
    body = BodyField(cube.vertices, cube.faces, Grid.for_bounds(lo, hi, 1.0))
    seeds = poisson_seeds(body, lo, hi, 10.0, np.random.default_rng(1))

    def flat_share(stretch: float) -> float:
        e = stretched_voronoi_edges(seeds, stretch)
        e = e[(np.abs(e.mean(axis=1) - [0, 0, 25]) < 20).all(axis=1)]
        d = e[:, 1] - e[:, 0]
        return float((np.abs(d[:, 2]) / np.linalg.norm(d, axis=1) < np.sin(np.radians(45))).mean())

    assert flat_share(1.0) > 0.6 and flat_share(2.5) < 0.6 * flat_share(1.0)


@pytest.mark.parametrize("surface", ["perforated", "rib_lattice", "embossed", "solid", "none"])
def test_nothing_left_to_fix_after_the_fix(bracket, surface: str) -> None:
    """A second pass over the result finds no long bridge and no node to support."""
    captured = {}
    original = supports.make_self_supporting

    def spy(edges, body, **kwargs):
        out, report = original(edges, body, **kwargs)
        captured["second"] = original(out, body, **kwargs)[1]
        captured["first"] = report
        return out, report

    pipeline.make_self_supporting = spy
    try:
        result = generate(bracket, Settings(surface_style=surface), "preview")
    finally:
        pipeline.make_self_supporting = original
    second = captured["second"]
    assert result.valid
    assert second.removed_bridges == 0 and second.added_struts == 0 and second.added_drops == 0
    assert second.removed_nodes == 0
    if surface != "none":
        assert captured["first"].unsupported == 0 and second.unsupported == 0


def test_no_islands_with_a_skin(bracket) -> None:
    for surface in ("rib_lattice", "perforated"):
        result, solid, _ = solid_grid_of(bracket, Settings(surface_style=surface, pointed_holes=False))
        assert result.valid and islands(solid) <= 1


def test_bare_lattice_has_far_fewer_islands() -> None:
    cube60 = load(test_models.cube(60.0))
    _, off, _ = solid_grid_of(cube60, Settings(surface_style="none", self_supporting=False))
    result, on, _ = solid_grid_of(cube60, Settings(surface_style="none"))
    assert result.valid
    assert islands(off) > 30 and islands(on) <= 4


def test_unsupportable_nodes_are_reported() -> None:
    sphere = load(test_models.sphere())  # a bare lattice ball has nothing under its lower half
    result = generate(sphere, Settings(surface_style="none"), "preview")
    assert result.valid and any("could not be held from below" in w for w in result.warnings)
    with_skin = generate(sphere, Settings(surface_style="rib_lattice"), "preview")
    assert not any("could not be held" in w for w in with_skin.warnings)


def test_self_supporting_off_keeps_the_old_lattice(cube) -> None:
    off = generate(cube, Settings(self_supporting=False, pointed_holes=False), "preview")
    on = generate(cube, Settings(pointed_holes=False), "preview")
    assert off.valid and on.valid and not np.array_equal(off.vertices, on.vertices)


# --- pointed hole tops ------------------------------------------------------------------

def wall_overhang_area(result) -> float:
    tri = result.vertices[result.faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    nz = cross[:, 2] / np.maximum(2 * area, 1e-12)
    c = tri.mean(axis=1)
    on_wall = ((np.abs(c[:, 0]) > 22.5) | (np.abs(c[:, 1]) > 22.5)) & (c[:, 2] > 3) & (c[:, 2] < 47)
    return float(area[on_wall & (nz < -np.cos(np.radians(45)) - 0.02)].sum())


@pytest.mark.parametrize("surface", ["perforated"])
def test_pointed_tops_remove_most_hole_ceilings(cube, surface: str) -> None:
    flat = generate(cube, Settings(surface_style=surface, interior_style="hollow", pointed_holes=False), "final")
    pointed = generate(cube, Settings(surface_style=surface, interior_style="hollow"), "final")
    assert pointed.valid
    assert wall_overhang_area(pointed) < 0.5 * wall_overhang_area(flat)


def test_pointed_tops_keep_the_smoothing_consistent(cube) -> None:
    """The exact field (used for smoothing) builds the same pointed tops as the grid."""
    captured = {}
    original_field, original_keep = pipeline.ExactField, pipeline.keep_largest_body

    def field(*args, **kwargs):
        captured["field"] = original_field(*args, **kwargs)
        return captured["field"]

    def keep(grid_field, h):
        captured["grid"] = grid_field.copy()
        return original_keep(grid_field, h)

    pipeline.ExactField, pipeline.keep_largest_body = field, keep
    try:
        generate(cube, Settings(surface_style="perforated", smoothness="smooth"), "final")
    finally:
        pipeline.ExactField, pipeline.keep_largest_body = original_field, original_keep
    exact, grid = captured["field"], captured["grid"]
    h = exact.body.grid.h
    nodes = np.nonzero(np.abs(grid) < 1.5 * h)
    assert np.abs(exact(exact.body.grid.points(nodes)) - grid[nodes]).max() < 0.15 * h


# --- bed face suggestion --------------------------------------------------------------------

def turned(mesh: trimesh.Trimesh, degrees: float, axis) -> trimesh.Trimesh:
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(degrees), axis))
    return mesh


@pytest.mark.parametrize("mesh,expected", [
    (test_models.bracket(), [0, 0, -1]),
    (turned(test_models.bracket(), 180, [1, 0, 0]), [0, 0, 1]),
    (turned(test_models.bracket(), 90, [1, 0, 0]), [0, 1, 0]),
    (test_models.sphere(), [0, 0, -1]),
])
def test_suggested_bed_face(mesh, expected) -> None:
    suggestion = suggest_bed_normal(load(mesh))
    np.testing.assert_allclose(suggestion["normal"], expected, atol=1e-6)
    assert suggestion["overhang_mm2"] <= suggestion["as_loaded_overhang_mm2"]


def test_suggestion_avoids_the_overhanging_round() -> None:
    suggestion = suggest_bed_normal(load(test_models.filleted_bracket()))
    assert suggestion["as_loaded_overhang_mm2"] > 100 and suggestion["overhang_mm2"] == 0


def test_suggest_bed_api() -> None:
    client = TestClient(app)
    upside_down = turned(test_models.bracket(), 180, [1, 0, 0])
    model = client.post("/api/models", files={
        "file": ("b.stl", test_models.to_binary_stl(upside_down), "model/stl")}).json()["id"]
    body = client.get(f"/api/models/{model}/suggest-bed").json()
    np.testing.assert_allclose(body["normal"], [0, 0, 1], atol=1e-6)
    assert body["as_loaded_overhang_mm2"] > 1000 and body["overhang_mm2"] == 0
    assert client.get("/api/models/missing/suggest-bed").status_code == 404


def test_rib_lattice_ignores_pointed_tops(cube) -> None:
    """Style B: the openings fill nearly the whole cell; pointed tops must not fill them."""
    on = generate(cube, Settings(surface_style="rib_lattice", interior_style="hollow"), "preview")
    off = generate(cube, Settings(surface_style="rib_lattice", interior_style="hollow",
                                  pointed_holes=False), "preview")
    assert np.array_equal(on.vertices, off.vertices) and np.array_equal(on.faces, off.faces)
