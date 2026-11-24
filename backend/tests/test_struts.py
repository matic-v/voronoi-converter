"""Step 4 tests: seeds, Voronoi edges, strut field, strut styles."""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree

from voronoi_engine import test_models
from voronoi_engine.body_field import BodyField
from voronoi_engine.generate import generate
from voronoi_engine.grid import Grid
from voronoi_engine.lattice import interior_edges, poisson_seeds, voronoi_edges
from voronoi_engine.mesh_io import LoadedMesh, load_stl
from voronoi_engine.settings import INVALID_COMBINATIONS, Settings
from voronoi_engine.strut_field import (StrutEvaluator, segment_distance, smooth_min,
                                        strut_field)
from voronoi_engine.surface import SurfaceIndex

MODELS = list(test_models.TEST_MODELS)


def load(name: str) -> LoadedMesh:
    return load_stl(test_models.to_binary_stl(test_models.build(name)), f"{name}.stl")


@pytest.fixture(scope="module")
def meshes() -> dict[str, LoadedMesh]:
    return {name: load(name) for name in MODELS}


@pytest.fixture(scope="module")
def cube_body(meshes) -> BodyField:
    mesh = meshes["cube_50"]
    grid = Grid.for_bounds(np.asarray(mesh.info.bounds_min), np.asarray(mesh.info.bounds_max), 1.0)
    return BodyField(mesh.vertices, mesh.faces, grid)


def as_trimesh(result) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=result.vertices, faces=result.faces, process=False)


def solid_bodies(mesh: trimesh.Trimesh) -> int:
    return sum(1 for part in mesh.split(only_watertight=False) if part.volume > 0)


def seeds_for(body: BodyField, mesh: LoadedMesh, spacing: float, seed: int) -> np.ndarray:
    return poisson_seeds(body, np.asarray(mesh.info.bounds_min), np.asarray(mesh.info.bounds_max),
                         spacing, np.random.default_rng(seed))


# --- seeds ----------------------------------------------------------------------

def test_poisson_seeds_keep_their_distance_and_fill_the_region(meshes, cube_body) -> None:
    spacing = 8.0
    seeds = seeds_for(cube_body, meshes["cube_50"], spacing, 1)
    nearest, _ = cKDTree(seeds).query(seeds, k=2)
    assert nearest[:, 1].min() >= spacing
    # Dense: no room left for another seed far from all existing ones, inside the body.
    rng = np.random.default_rng(9)
    probes = rng.uniform((-25, -25, 0), (25, 25, 50), size=(20000, 3))
    gap, _ = cKDTree(seeds).query(probes)
    assert gap.max() < 1.25 * spacing
    # Seeds also surround the body (ghost layer), and none are far away from it.
    outside = (np.abs(seeds[:, :2]) > 25).any(axis=1) | (seeds[:, 2] < 0) | (seeds[:, 2] > 50)
    assert outside.sum() > 50
    assert cube_body.surface.unsigned_distance_to_samples(seeds[outside]).max() <= 2.2 * spacing


def test_poisson_seeds_depend_only_on_the_random_seed(meshes, cube_body) -> None:
    mesh = meshes["cube_50"]
    assert np.array_equal(seeds_for(cube_body, mesh, 10.0, 5), seeds_for(cube_body, mesh, 10.0, 5))
    assert not np.array_equal(seeds_for(cube_body, mesh, 10.0, 5)[:20],
                              seeds_for(cube_body, mesh, 10.0, 6)[:20])


# --- Voronoi edges --------------------------------------------------------------

def test_voronoi_edges_are_equidistant_from_three_seeds() -> None:
    rng = np.random.default_rng(4)
    seeds = rng.uniform(0, 100, size=(400, 3))
    edges = voronoi_edges(seeds)
    centre_ok = (np.abs(edges - 50) < 30).all(axis=(1, 2))  # away from the unbounded rim
    middle = edges[centre_ok].mean(axis=1)
    assert len(middle) > 200
    distance, _ = cKDTree(seeds).query(middle, k=4)
    np.testing.assert_allclose(distance[:, 0], distance[:, 2], rtol=1e-6)
    assert (distance[:, 3] >= distance[:, 2] - 1e-9).all()


def test_voronoi_edge_count_matches_poisson_voronoi_statistics() -> None:
    rng = np.random.default_rng(5)
    seeds = rng.uniform(0, 100, size=(3000, 3))
    edges = voronoi_edges(seeds)
    inner = (np.abs(edges - 50) < 35).all(axis=(1, 2))
    seeds_inner = (np.abs(seeds - 50) < 35).all(axis=1).sum()
    # Theory for random seeds: 13.5 edges per cell (40.6 per cell, each shared by 3).
    assert inner.sum() / seeds_inner == pytest.approx(13.5, rel=0.12)


def test_interior_edges_stay_near_the_body(meshes, cube_body) -> None:
    mesh = meshes["cube_50"]
    edges = interior_edges(cube_body, np.asarray(mesh.info.bounds_min),
                           np.asarray(mesh.info.bounds_max), 10.0, reach=2.0,
                           rng=np.random.default_rng(1))
    assert len(edges) > 300
    surface = SurfaceIndex(mesh.vertices, mesh.faces, 1.0)
    assert surface.signed_distance(edges.reshape(-1, 3)).max() < 2.0 + 2 * cube_body.grid.h
    assert (surface.signed_distance(edges.mean(axis=1)) < 2.0).mean() > 0.98


# --- strut field ----------------------------------------------------------------

def random_segments(count: int = 60):
    rng = np.random.default_rng(7)
    start = rng.uniform(5, 35, size=(count, 3))
    return np.stack([start, start + rng.normal(0, 4, size=(count, 3))], axis=1)


def test_strut_evaluator_matches_brute_force_when_sharp() -> None:
    segments = random_segments()
    radius = np.full(len(segments), 1.0)
    evaluate = StrutEvaluator(segments, radius, blend=0.0, reach=2.5, min_spacing=0.5)
    points = np.random.default_rng(8).uniform(0, 40, size=(5000, 3))
    brute = np.min([segment_distance(points, np.broadcast_to(a, points.shape),
                                     np.broadcast_to(b, points.shape))[0]
                    for a, b in segments], axis=0) - 1.0
    got = evaluate(points)
    relevant = brute < 1.5  # exact wherever a strut is within reach
    np.testing.assert_allclose(got[relevant], brute[relevant], atol=1e-9)
    assert (got[~relevant] >= 1.5 - 1e-9).all()


def test_blend_only_adds_material_and_is_bounded() -> None:
    segments = random_segments()
    radius = np.full(len(segments), 1.0)
    points = np.random.default_rng(8).uniform(0, 40, size=(5000, 3))
    sharp = StrutEvaluator(segments, radius, 0.0, 3.0, 0.5)(points)
    blended = StrutEvaluator(segments, radius, 1.0, 3.0, 0.5)(points)
    near = sharp < 1.0
    assert (blended[near] <= sharp[near] + 1e-12).all()
    assert (sharp[near] - blended[near]).max() > 0.2  # joints did get thicker
    assert (sharp[near] - blended[near]).max() < 1.0  # ...by less than the blend radius


def test_smooth_min() -> None:
    a, b = np.array([0.0, 1.0, 5.0]), np.array([0.0, 1.5, 0.0])
    assert np.array_equal(smooth_min(a, b, 0.0), [0.0, 1.0, 0.0])
    np.testing.assert_allclose(smooth_min(a, b, 1.0), [-0.25, 1.0 - 0.0625, 0.0])


def test_strut_field_single_beam_has_the_right_diameter() -> None:
    grid = Grid.for_bounds(np.zeros(3), np.full(3, 20.0), 0.25)
    segment = np.array([[[10.0, 10.0, -5.0], [10.0, 10.0, 25.0]]])
    field = strut_field(grid, np.full(grid.shape, -1.0, dtype=np.float32), segment,
                        np.array([1.5]), 0.0)
    layer = field[:, :, grid.shape[2] // 2]
    area = (layer < 0).sum() * 0.25**2
    assert area == pytest.approx(np.pi * 1.5**2, rel=0.05)


# --- styles ---------------------------------------------------------------------

COMBINATIONS = [(s, i) for s in ("solid", "none") for i in ("struts", "solid", "hollow")
                if {"surface_style": s, "interior_style": i} not in INVALID_COMBINATIONS]


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("surface,interior", COMBINATIONS)
def test_every_built_combination_generates(meshes, name: str, surface: str, interior: str) -> None:
    mesh = meshes[name]
    result = generate(mesh, Settings(surface_style=surface, interior_style=interior), "preview")
    out = as_trimesh(result)
    assert result.valid, result.problems
    assert out.is_watertight and out.is_winding_consistent and out.volume > 0
    assert solid_bodies(out) == 1
    h = result.voxel_mm
    lo, hi = result.vertices.min(axis=0), result.vertices.max(axis=0)
    assert (lo >= np.asarray(mesh.info.bounds_min) - 1e-6).all()
    assert (hi <= np.asarray(mesh.info.bounds_max) + 1e-6).all()
    if surface == "solid":  # with a skin the outer dimensions are kept
        assert np.abs(lo - mesh.info.bounds_min).max() <= h
        assert np.abs(hi - mesh.info.bounds_max).max() <= h
    surface_index = SurfaceIndex(mesh.vertices, mesh.faces, spacing=0.5)
    assert surface_index.signed_distance(result.vertices, k=12).max() <= 1e-3


def test_struts_connect_to_the_skin_and_fill_the_interior(meshes) -> None:
    mesh = meshes["cube_50"]
    shell = generate(mesh, Settings(surface_style="solid", interior_style="hollow"), "final")
    lattice = generate(mesh, Settings(surface_style="solid", interior_style="struts"), "final")
    assert lattice.fragments_removed == 0
    shell_volume, lattice_volume = as_trimesh(shell).volume, as_trimesh(lattice).volume
    assert shell_volume * 1.1 < lattice_volume < mesh.info.volume_mm3 * 0.6
    # There is strut material right in the middle of the part.
    centre_distance = np.linalg.norm(lattice.vertices - (0, 0, 25), axis=1)
    assert centre_distance.min() < 8.0


def test_thicker_struts_and_smaller_cells_add_material(meshes) -> None:
    mesh = meshes["cube_50"]
    base = Settings(surface_style="none", interior_style="struts")
    volume = lambda s: as_trimesh(generate(mesh, s, "preview")).volume  # noqa: E731
    v0 = volume(base)
    assert volume(base.model_copy(update={"strut_diameter": 3.0})) > 1.6 * v0
    assert volume(base.model_copy(update={"interior_cell_size": 8.0})) > 1.5 * v0
    assert volume(base.model_copy(update={"blend_radius": 2.0})) > 1.05 * v0


def test_strut_diameter_is_respected(meshes) -> None:
    """Material fraction of a sharp lattice matches beams of the requested diameter."""
    mesh = meshes["cube_50"]
    s = Settings(surface_style="none", interior_style="struts", blend_radius=0.0,
                 strut_diameter=2.0, interior_cell_size=12.0)
    v2 = as_trimesh(generate(mesh, s, "final")).volume
    v3 = as_trimesh(generate(mesh, s.model_copy(update={"strut_diameter": 3.0}), "final")).volume
    # Beam volume grows with diameter squared, less the overlap at joints.
    assert 1.8 < v3 / v2 < 2.25


def test_same_seed_gives_identical_stl_and_other_seed_differs(meshes) -> None:
    def stl(seed: int) -> bytes:
        s = Settings(surface_style="solid", interior_style="struts", random_seed=seed)
        buffer = io.BytesIO()
        as_trimesh(generate(meshes["sphere_60"], s, "preview")).export(buffer, file_type="stl")
        return buffer.getvalue()
    assert stl(1) == stl(1)
    assert stl(1) != stl(2)


def test_100mm_part_meets_the_time_targets(meshes) -> None:
    mesh = meshes["bracket_100"]
    settings = Settings(surface_style="solid", interior_style="struts")
    started = time.perf_counter()
    generate(mesh, settings, "preview")
    assert time.perf_counter() - started < 5.0
    started = time.perf_counter()
    result = generate(mesh, settings, "final")
    assert time.perf_counter() - started < 60.0
    assert result.valid and result.voxel_mm == pytest.approx(0.5)
