"""Step 5 tests: surface seeds, Worley pattern, styles A, B, C in every combination."""

from __future__ import annotations

import io

import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree

from voronoi_engine import test_models
from voronoi_engine.body_field import BodyField
from voronoi_engine.generate import generate
from voronoi_engine.grid import Grid
from voronoi_engine.lattice import poisson_seeds
from voronoi_engine.mesh_io import LoadedMesh, load_stl
from voronoi_engine.settings import INVALID_COMBINATIONS, Settings
from voronoi_engine.surface import SurfaceIndex
from voronoi_engine.surface_pattern import SurfacePattern, surface_seeds

MODELS = list(test_models.TEST_MODELS)
SURFACES = ["perforated", "rib_lattice", "embossed", "none", "solid"]
INTERIORS = ["struts", "solid", "hollow"]
COMBINATIONS = [(s, i) for s in SURFACES for i in INTERIORS
                if {"surface_style": s, "interior_style": i} not in INVALID_COMBINATIONS]


def load(name: str) -> LoadedMesh:
    return load_stl(test_models.to_binary_stl(test_models.build(name)), f"{name}.stl")


@pytest.fixture(scope="module")
def meshes() -> dict[str, LoadedMesh]:
    return {name: load(name) for name in MODELS}


def as_trimesh(result) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=result.vertices, faces=result.faces, process=False)


def solid_bodies(mesh: trimesh.Trimesh) -> int:
    return sum(1 for part in mesh.split(only_watertight=False) if part.volume > 0)


def volume(mesh: LoadedMesh, mode: str = "preview", **settings) -> float:
    return as_trimesh(generate(mesh, Settings(**settings), mode)).volume


# --- seeds ----------------------------------------------------------------------

def test_surface_seeds_lie_on_the_surface_at_the_right_spacing(meshes) -> None:
    mesh = meshes["sphere_60"]
    tri = mesh.vertices[mesh.faces]
    seeds = surface_seeds(tri, 10.0, np.random.default_rng(1))
    surface = SurfaceIndex(mesh.vertices, mesh.faces, 1.0)
    assert np.abs(surface.signed_distance(seeds)).max() < 1e-9
    nearest, _ = cKDTree(seeds).query(seeds, k=2)
    assert nearest[:, 1].min() >= 10.0
    assert nearest[:, 1].max() < 20.0  # no bald patches
    # Random dart throwing saturates at about 1.43 x spacing^2 of area per seed.
    saturated = 4 * np.pi * 30**2 / (1.43 * 10.0**2)
    assert 0.8 * saturated < len(seeds) <= 1.05 * saturated
    assert np.array_equal(seeds, surface_seeds(tri, 10.0, np.random.default_rng(1)))


def test_seeds_do_not_depend_on_the_voxel_size(meshes) -> None:
    """Preview and final must show the same pattern."""
    mesh = meshes["bracket_100"]
    lo, hi = np.asarray(mesh.info.bounds_min), np.asarray(mesh.info.bounds_max)
    sets = []
    for h in (1.0, 0.5):
        body = BodyField(mesh.vertices, mesh.faces, Grid.for_bounds(lo, hi, h))
        sets.append(poisson_seeds(body, lo, hi, 12.0, np.random.default_rng(3)))
    assert np.array_equal(sets[0], sets[1])


# --- pattern --------------------------------------------------------------------

def test_wall_distance_is_the_distance_to_the_cell_wall() -> None:
    seeds = np.array([[0.0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0], [5, 30, 0], [30, 5, 0]])
    pattern = SurfacePattern(seeds)
    points = np.array([[1.0, 2, 0], [4.5, 1, 0], [5.0, 2, 0], [8.0, 1, 3]])
    # Cell of seed 0 is x < 5, y < 5: distance to the nearer of those two walls.
    np.testing.assert_allclose(pattern.wall_distance(points, 0.0), [3.0, 0.5, 0.0, 3.0], atol=1e-12)
    # Inset 0.4 moves each wall 40 % of the way to the seed (2 mm here).
    np.testing.assert_allclose(pattern.wall_distance(points, 0.0, inset=0.4), [1.0, -1.5, -2.0, 1.0],
                               atol=1e-12)
    # Rounding only acts near corners, and only reduces the value.
    corner = np.array([[4.0, 4.0, 0], [1.0, 4.0, 0]])
    # Two walls at 1.0 give 0.5; the diagonal cell (wall at 1.414) rounds a little more.
    assert pattern.wall_distance(corner, 2.0)[0] == pytest.approx(0.3526, abs=1e-4)
    assert pattern.wall_distance(corner, 2.0)[1] == pytest.approx(1.0)


# --- every combination ----------------------------------------------------------

@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("surface,interior", COMBINATIONS)
def test_every_combination_generates(meshes, name: str, surface: str, interior: str) -> None:
    mesh = meshes[name]
    result = generate(mesh, Settings(surface_style=surface, interior_style=interior), "preview")
    out = as_trimesh(result)
    assert result.valid, result.problems
    assert out.is_watertight and out.is_winding_consistent and out.volume > 0
    assert solid_bodies(out) == 1
    assert out.volume <= mesh.info.volume_mm3 * 1.001
    h = result.voxel_mm
    lo, hi = result.vertices.min(axis=0), result.vertices.max(axis=0)
    if surface != "none":  # a skin keeps the outer dimensions
        assert np.abs(lo - mesh.info.bounds_min).max() <= h
        assert np.abs(hi - mesh.info.bounds_max).max() <= h
    index = SurfaceIndex(mesh.vertices, mesh.faces, spacing=0.5)
    assert index.signed_distance(result.vertices, k=12).max() <= 1e-3


@pytest.mark.parametrize("surface", ["perforated", "rib_lattice", "embossed"])
def test_final_quality_of_patterned_skins(meshes, surface: str) -> None:
    mesh = meshes["cube_50"]
    result = generate(mesh, Settings(surface_style=surface, interior_style="struts"), "final")
    out = as_trimesh(result)
    assert result.valid and out.is_watertight and solid_bodies(out) == 1
    assert result.fragments_removed == 0
    np.testing.assert_allclose(result.vertices.min(axis=0), mesh.info.bounds_min, atol=0.3)
    np.testing.assert_allclose(result.vertices.max(axis=0), mesh.info.bounds_max, atol=0.3)


# --- what each style does --------------------------------------------------------

def test_perforated_shell_has_holes_through_it(meshes) -> None:
    mesh = meshes["cube_50"]
    shell = volume(mesh, "final", surface_style="solid", interior_style="hollow")
    holes60 = volume(mesh, "final", surface_style="perforated", interior_style="hollow")
    holes30 = volume(mesh, "final", surface_style="perforated", interior_style="hollow", hole_size=30)
    # Cells scaled to 60 % remove about 0.36 of the shell, 30 % about 0.09
    # (less a little for rounded corners).
    assert holes60 / shell == pytest.approx(1 - 0.36, abs=0.08)
    assert holes30 / shell == pytest.approx(1 - 0.09, abs=0.05)
    # Through holes: the mesh is one closed surface, not an outer skin plus a cavity.
    result = generate(mesh, Settings(surface_style="perforated", interior_style="hollow"), "final")
    assert len(as_trimesh(result).split(only_watertight=False)) == 1


def test_rib_lattice_is_only_ribs_at_the_surface(meshes) -> None:
    mesh = meshes["cube_50"]
    result = generate(mesh, Settings(surface_style="rib_lattice", interior_style="hollow"), "final")
    index = SurfaceIndex(mesh.vertices, mesh.faces, spacing=0.5)
    depth = -index.signed_distance(result.vertices, k=8)
    # Rib depth = rib width; the inner corner along the cube's edges is rounded (blurred
    # inner face), which reaches about a voxel deeper there.
    assert depth.max() <= 2.0 + 1.5 * result.voxel_mm
    ribs2 = as_trimesh(result).volume
    ribs4 = volume(mesh, "final", surface_style="rib_lattice", interior_style="hollow", rib_width=4.0)
    bigger_cells = volume(mesh, "final", surface_style="rib_lattice", interior_style="hollow",
                          surface_cell_size=20.0)
    assert ribs4 > 2.5 * ribs2  # wider and deeper
    assert bigger_cells < 0.75 * ribs2  # fewer ribs


def test_embossed_keeps_a_closed_shell_and_the_envelope(meshes) -> None:
    mesh = meshes["cube_50"]
    solid = mesh.info.volume_mm3
    embossed = generate(mesh, Settings(surface_style="embossed", interior_style="solid"), "final")
    out = as_trimesh(embossed)
    # One closed surface, slightly lighter than the solid: only shallow recesses removed.
    assert len(out.split(only_watertight=False)) == 1
    removed = solid - out.volume
    max_removed = 6 * 50**2 * 0.8  # every face recessed by the full depth
    assert 0.3 * max_removed < removed < 0.9 * max_removed
    deeper = volume(mesh, "final", surface_style="embossed", interior_style="solid", emboss_depth=1.6)
    assert solid - deeper == pytest.approx(2 * removed, rel=0.2)
    # Hollow version stays closed: recesses do not break through a 2 mm shell.
    hollow = generate(mesh, Settings(surface_style="embossed", interior_style="hollow"), "final")
    assert len(as_trimesh(hollow).split(only_watertight=False)) == 2  # outside + cavity


def test_struts_do_not_poke_out_of_an_open_skin(meshes) -> None:
    mesh = meshes["cube_50"]
    with_struts = generate(mesh, Settings(surface_style="rib_lattice", interior_style="struts"), "final")
    ribs_only = generate(mesh, Settings(surface_style="rib_lattice", interior_style="hollow"), "final")
    assert as_trimesh(with_struts).volume > 1.3 * as_trimesh(ribs_only).volume
    # Material at the surface is the same with and without struts.
    index = SurfaceIndex(mesh.vertices, mesh.faces, spacing=0.5)
    def at_surface(result) -> float:
        tri = as_trimesh(result)
        on_skin = -index.signed_distance(tri.triangles_center, k=8) < 0.05
        return float(tri.area_faces[on_skin].sum())
    assert at_surface(with_struts) == pytest.approx(at_surface(ribs_only), rel=0.02)


def test_surface_pattern_is_independent_of_the_interior_style(meshes) -> None:
    mesh = meshes["cube_50"]
    index = SurfaceIndex(mesh.vertices, mesh.faces, spacing=0.5)
    def skin_area(interior: str) -> float:
        tri = as_trimesh(generate(mesh, Settings(surface_style="perforated", interior_style=interior),
                                  "final"))
        on_skin = -index.signed_distance(tri.triangles_center, k=8) < 0.05
        return float(tri.area_faces[on_skin].sum())
    assert skin_area("hollow") == pytest.approx(skin_area("struts"), rel=0.02)


def test_same_seed_gives_identical_stl_for_patterned_skin(meshes) -> None:
    def stl(seed: int) -> bytes:
        buffer = io.BytesIO()
        as_trimesh(generate(meshes["sphere_60"], Settings(random_seed=seed), "preview")).export(
            buffer, file_type="stl")
        return buffer.getvalue()
    assert stl(7) == stl(7)
    assert stl(7) != stl(8)


def test_surface_cell_size_too_large_is_reported(meshes) -> None:
    from voronoi_engine.generate import GenerationError
    small = load_stl(test_models.to_binary_stl(test_models.cube(6.0)), "tiny.stl")
    with pytest.raises(GenerationError, match="surface cell size is too large"):
        generate(small, Settings(surface_cell_size=50.0, interior_style="hollow"), "preview")


# --- struts are attached to an open skin -------------------------------------------

@pytest.mark.parametrize("surface", ["perforated", "rib_lattice"])
def test_struts_end_on_skin_material(meshes, surface: str) -> None:
    """Every strut that reaches an open skin is re-aimed at a wall or rib."""
    from voronoi_engine.lattice import interior_edges
    from voronoi_engine.skin_attach import attach_edges_to_skin

    mesh = meshes["cube_50"]
    lo, hi = np.asarray(mesh.info.bounds_min), np.asarray(mesh.info.bounds_max)
    body = BodyField(mesh.vertices, mesh.faces, Grid.for_bounds(lo, hi, 0.5))
    body.ensure_depth(4.0)
    pattern = SurfacePattern(surface_seeds(mesh.vertices[mesh.faces], 10.0, np.random.default_rng(1)))
    edges = interior_edges(body, lo, hi, 12.0, reach=1.5, rng=np.random.default_rng(2))
    depth = 2.0
    attached_edges, count = attach_edges_to_skin(edges, body, pattern, lambda p: np.full(len(p), depth))
    assert count > 40
    anchors = attached_edges[-count:, 1]
    inner = attached_edges[-count:, 0]
    # Anchors sit on a cell wall of the surface pattern...
    assert np.abs(pattern.wall_distance(anchors, 0.0)).max() < 0.25
    # ...inside the skin, 60 % of its depth below the surface...
    index = SurfaceIndex(mesh.vertices, mesh.faces, spacing=0.5)
    anchor_depth = -index.signed_distance(anchors)
    assert (anchor_depth > 0.3 * depth).all() and (anchor_depth < 0.9 * depth).all()
    assert np.median(anchor_depth) == pytest.approx(0.6 * depth, abs=0.02)
    # ...and the other end is below the skin. Nothing else comes near the skin.
    assert (-index.signed_distance(inner) > depth - 0.6).all()
    others = attached_edges[:-count].reshape(-1, 3)
    assert (-index.signed_distance(others) > depth - 0.6).all()


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("surface", ["perforated", "rib_lattice"])
def test_open_skin_with_struts_leaves_no_loose_stubs(meshes, name: str, surface: str) -> None:
    result = generate(meshes[name], Settings(surface_style=surface, interior_style="struts"), "final")
    assert result.valid and solid_bodies(as_trimesh(result)) == 1
    assert result.fragments_removed == 0


def test_attachment_holds_with_large_holes_and_thin_ribs(meshes) -> None:
    mesh = meshes["cube_50"]
    for settings in (Settings(hole_size=85), Settings(surface_style="rib_lattice", rib_width=1.2),
                     Settings(grading_mode="z", grading_strength=100, grading_target="both")):
        result = generate(mesh, settings, "final")
        assert result.valid and result.fragments_removed <= 1


def test_struts_do_not_show_on_the_recess_floors(meshes) -> None:
    """Regression: struts ended exactly at the recess floor and their blend pushed blobs
    up through it. Nothing may stand above the floor inside a recessed cell."""
    import voronoi_engine.generate as pipeline
    mesh = meshes["cube_50"]
    captured = {}
    original = pipeline.ExactField

    def capture(*args, **kwargs):
        captured["field"] = original(*args, **kwargs)
        return captured["field"]

    pipeline.ExactField = capture
    try:
        result = generate(mesh, Settings(surface_style="embossed", smoothness="smooth"), "final")
    finally:
        pipeline.ExactField = original
    index = SurfaceIndex(mesh.vertices, mesh.faces, spacing=0.3)
    depth = -index.signed_distance(result.vertices, k=6)
    wall = captured["field"].pattern.wall_distance(result.vertices, 0.0)
    inside_cell = wall > 0.5 * 2.0 + 0.6 + 0.5  # half rib + edge rounding margin + blend
    assert not (inside_cell & (depth > 0.05) & (depth < 0.8 - 0.1)).any()
