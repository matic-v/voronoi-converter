"""Step 3 tests: distance field, marching cubes round trip, Solid/None/Hollow styles."""

from __future__ import annotations

import io

import numpy as np
import pytest
import trimesh

from voronoi_engine import test_models
from voronoi_engine.generate import GenerationError, generate
from voronoi_engine.grid import choose_voxel_size
from voronoi_engine.mesh_io import LoadedMesh, load_stl
from voronoi_engine.settings import Settings
from voronoi_engine.surface import SurfaceIndex

MODELS = list(test_models.TEST_MODELS)


def load(name: str) -> LoadedMesh:
    return load_stl(test_models.to_binary_stl(test_models.build(name)), f"{name}.stl")


@pytest.fixture(scope="module")
def meshes() -> dict[str, LoadedMesh]:
    return {name: load(name) for name in MODELS}


def as_trimesh(result) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=result.vertices, faces=result.faces, process=False)


def solid_bodies(mesh: trimesh.Trimesh) -> int:
    """Closed shells with positive volume (cavities are wound inwards)."""
    return sum(1 for part in mesh.split(only_watertight=False) if part.volume > 0)


# --- distance to the surface ------------------------------------------------

def test_signed_distance_matches_analytic_box() -> None:
    mesh = load("cube_50")
    surface = SurfaceIndex(mesh.vertices, mesh.faces, spacing=0.5)
    rng = np.random.default_rng(0)
    points = rng.uniform((-35, -35, -10), (35, 35, 60), size=(20000, 3))
    q = np.abs(points - (0, 0, 25)) - 25
    expected = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
    np.testing.assert_allclose(surface.signed_distance(points), expected, atol=1e-9)


def test_signed_distance_matches_sphere() -> None:
    mesh = load("sphere_60")
    surface = SurfaceIndex(mesh.vertices, mesh.faces, spacing=0.5)
    rng = np.random.default_rng(1)
    points = rng.uniform(-35, 35, size=(20000, 3)) + (0, 0, 30)
    expected = np.linalg.norm(points - (0, 0, 30), axis=1) - 30
    # The mesh is a faceted sphere: up to ~0.04 mm inside the true one.
    assert np.abs(surface.signed_distance(points) - expected).max() < 0.08


def test_surface_samples_cover_long_thin_triangles() -> None:
    from voronoi_engine.surface import sample_surface
    rng = np.random.default_rng(3)
    tri = np.array([[[0, 0, 0], [100, 0, 0], [50, 0.2, 0]],      # sliver
                    [[0, 0, 0], [30, 0, 0], [0, 20, 0]],         # right angle at A
                    [[0, 0, 0], [30, 0, 0], [29, 25, 0]]], float)
    points, face = sample_surface(tri, 0.5)
    assert len(points) < 6000
    for i, (a, b, c) in enumerate(tri):
        w = rng.dirichlet((1, 1, 1), size=4000)
        inside = w[:, :1] * a + w[:, 1:2] * b + w[:, 2:] * c
        corners = np.vstack([inside, a, b, c])
        gap = np.linalg.norm(corners[:, None] - points[face == i][None], axis=2).min(axis=1)
        assert gap.max() <= 0.5


def test_signed_distance_sign_on_bracket() -> None:
    mesh = load("bracket_100")
    surface = SurfaceIndex(mesh.vertices, mesh.faces, spacing=0.5)
    rng = np.random.default_rng(2)
    points = rng.uniform((-55, -25, -5), (55, 25, 65), size=(20000, 3))
    x, y, z = points[:, 0] + 50, points[:, 1], points[:, 2]
    inside_profile = ((x > 0) & (x < 8) & (z > 0) & (z < 60)) | \
                     ((x > 0) & (x < 60) & (z > 0) & (z < 8)) | \
                     ((x > 0) & (x < 100) & (z > 0) & (z < 2))
    inside = inside_profile & (np.abs(y) < 20)
    assert np.array_equal(surface.signed_distance(points) < 0, inside)


# --- round trip ---------------------------------------------------------------

@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("mode", ["preview", "final"])
def test_round_trip_matches_input(meshes, name: str, mode: str) -> None:
    mesh = meshes[name]
    settings = Settings(surface_style="solid", interior_style="solid")
    result = generate(mesh, settings, mode)
    h = result.voxel_mm
    out = as_trimesh(result)

    assert result.valid, result.problems
    assert out.is_watertight and out.is_winding_consistent and out.volume > 0
    assert solid_bodies(out) == 1 and result.fragments_removed == 0

    lo, hi = result.vertices.min(axis=0), result.vertices.max(axis=0)
    assert np.abs(lo - mesh.info.bounds_min).max() <= h
    assert np.abs(hi - mesh.info.bounds_max).max() <= h
    if mode == "final":
        assert h <= 0.5
        assert np.abs(lo - mesh.info.bounds_min).max() <= 0.3
        assert np.abs(hi - mesh.info.bounds_max).max() <= 0.3
    assert out.volume == pytest.approx(mesh.info.volume_mm3, rel=0.03 if mode == "final" else 0.08)

    # Nothing outside the envelope: vertices and face centres.
    surface = SurfaceIndex(mesh.vertices, mesh.faces, spacing=0.3)
    assert surface.signed_distance(result.vertices, k=12).max() <= 1e-3  # 1 micron
    # The final also refines concave edges; the preview may bridge them by a fraction of a voxel.
    limit = 0.1 * h if mode == "final" else 0.35 * h
    assert surface.signed_distance(out.triangles_center, k=12).max() <= limit
    edge_middles = result.vertices[out.edges_unique].mean(axis=1)
    assert surface.signed_distance(edge_middles, k=12).max() <= limit
    # ...and the surface stays within a voxel of the input everywhere.
    assert np.abs(surface.signed_distance(result.vertices, k=12)).max() <= h


# --- styles -------------------------------------------------------------------

@pytest.mark.parametrize("name", MODELS)
def test_hollow_shell(meshes, name: str) -> None:
    mesh = meshes[name]
    result = generate(mesh, Settings(surface_style="solid", interior_style="hollow"), "final")
    out = as_trimesh(result)
    assert result.valid and out.is_watertight and solid_bodies(out) == 1
    assert out.volume < mesh.info.volume_mm3
    # Wall thickness: the cavity surface is 2 mm below the input surface.
    surface = SurfaceIndex(mesh.vertices, mesh.faces, spacing=0.3)
    depth = -surface.signed_distance(result.vertices)
    inner = depth > 1.0
    if name != "bracket_100":  # the bracket's 2 mm flange has no room for a cavity
        assert inner.any()
    if inner.any():
        assert np.abs(depth[inner] - 2.0).max() < 0.5 * result.voxel_mm


def test_shell_volume_of_cube(meshes) -> None:
    result = generate(meshes["cube_50"], Settings(surface_style="solid", interior_style="hollow",
                                                  shell_thickness=3.0), "final")
    # Offset surface of a cube has rounded corners/edges; the box formula is a lower bound.
    assert as_trimesh(result).volume == pytest.approx(50**3 - 44**3, rel=0.03)


@pytest.mark.parametrize("surface,interior", [("solid", "solid"), ("none", "solid")])
def test_solid_interior_is_the_full_body(meshes, surface: str, interior: str) -> None:
    mesh = meshes["bracket_100"]
    result = generate(mesh, Settings(surface_style=surface, interior_style=interior), "preview")
    assert as_trimesh(result).volume == pytest.approx(mesh.info.volume_mm3, rel=0.08)


def test_none_with_hollow_is_rejected_by_the_schema() -> None:
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="nothing to generate"):
        Settings(surface_style="none", interior_style="hollow")


# --- determinism, cache, voxel size -------------------------------------------

def test_same_settings_give_identical_stl(meshes) -> None:
    settings = Settings(surface_style="solid", interior_style="hollow")
    stls = []
    for _ in range(2):
        buffer = io.BytesIO()
        as_trimesh(generate(meshes["sphere_60"], settings, "preview")).export(buffer, file_type="stl")
        stls.append(buffer.getvalue())
    assert stls[0] == stls[1]


def test_cache_reuses_body_field_and_gives_same_result(meshes) -> None:
    cache: dict = {}
    hollow = Settings(surface_style="solid", interior_style="hollow")
    first = generate(meshes["cube_50"], hollow, "preview", cache)
    assert len(cache) == 1
    generate(meshes["cube_50"], Settings(surface_style="solid", interior_style="solid"), "preview", cache)
    again = generate(meshes["cube_50"], hollow, "preview", cache)
    assert len(cache) == 1
    assert np.array_equal(first.vertices, again.vertices) and np.array_equal(first.faces, again.faces)


def test_voxel_size_choice() -> None:
    settings = Settings(surface_style="solid", interior_style="hollow")
    h, capped = choose_voxel_size((100, 40, 60), settings, "final")
    assert h == pytest.approx(0.5) and not capped  # 2 mm shell / 4 voxels
    h_fine, _ = choose_voxel_size((100, 40, 60), settings.model_copy(update={"quality": "fine"}), "final")
    assert h_fine < h
    h_big, capped = choose_voxel_size((250, 250, 250), settings.model_copy(update={"shell_thickness": 0.8}), "final")
    assert capped and h_big > 0.2
    h_preview, _ = choose_voxel_size((100, 40, 60), settings, "preview")
    assert h_preview >= 1.0
