"""Step 6 tests: grading, solid base, printability warnings."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree

from voronoi_engine import test_models
from voronoi_engine.body_field import BodyField
from voronoi_engine.generate import generate
from voronoi_engine.grading import Grading, end_ratio, scale_range
from voronoi_engine.grid import Grid, choose_voxel_size
from voronoi_engine.lattice import poisson_seeds
from voronoi_engine.mesh_io import LoadedMesh, load_stl
from voronoi_engine.printability import printability_warnings
from voronoi_engine.settings import Settings
from voronoi_engine.strut_field import StrutEvaluator
from voronoi_engine.surface import SurfaceIndex
from voronoi_engine.surface_pattern import surface_seeds


def load(name: str) -> LoadedMesh:
    return load_stl(test_models.to_binary_stl(test_models.build(name)), f"{name}.stl")


@pytest.fixture(scope="module")
def cube() -> LoadedMesh:
    return load("cube_50")


@pytest.fixture(scope="module")
def cube_body(cube) -> BodyField:
    grid = Grid.for_bounds(np.asarray(cube.info.bounds_min), np.asarray(cube.info.bounds_max), 1.0)
    return BodyField(cube.vertices, cube.faces, grid)


def bounds(mesh: LoadedMesh):
    return np.asarray(mesh.info.bounds_min), np.asarray(mesh.info.bounds_max)


def as_trimesh(result) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=result.vertices, faces=result.faces, process=False)


def solid_bodies(mesh: trimesh.Trimesh) -> int:
    return sum(1 for part in mesh.split(only_watertight=False) if part.volume > 0)


def material_in_slab(result, axis: int, low: float, high: float) -> float:
    """Surface area of the result between two planes: a cheap measure of how much is there."""
    tri = as_trimesh(result)
    c = tri.triangles_center[:, axis]
    return float(tri.area_faces[(c >= low) & (c < high)].sum())


# --- grading maths ----------------------------------------------------------------

def test_end_ratio_and_scale_range() -> None:
    assert end_ratio(Settings()) == 1.0
    graded = Settings(grading_mode="z", grading_strength=100, grading_target="both")
    assert end_ratio(graded) == 4.0
    assert scale_range(graded, "cell_size") == pytest.approx((0.5, 2.0))
    assert scale_range(graded, "thickness") == pytest.approx((0.5, 2.0))
    cells_only = Settings(grading_mode="z", grading_strength=50)
    assert scale_range(cells_only, "cell_size") == pytest.approx((2.5**-0.5, 2.5**0.5))
    assert scale_range(cells_only, "thickness") == (1.0, 1.0)
    assert end_ratio(Settings(grading_mode="z", grading_strength=0)) == 1.0


def test_axis_grading_scales(cube, cube_body) -> None:
    lo, hi = bounds(cube)
    points = np.array([[0.0, 0, 0], [0, 0, 25], [0, 0, 50]])
    start = Grading(Settings(grading_mode="z", grading_strength=100, grading_target="both"),
                    lo, hi, cube_body)
    np.testing.assert_allclose(start.cell_scale(points), [0.5, 1.0, 2.0])  # dense at Z min
    np.testing.assert_allclose(start.thickness_scale(points), [2.0, 1.0, 0.5])  # dense = thicker
    end = Grading(Settings(grading_mode="z", grading_strength=100, grading_target="both",
                           grading_direction="denser_at_end"), lo, hi, cube_body)
    np.testing.assert_allclose(end.cell_scale(points), [2.0, 1.0, 0.5])
    off = Grading(Settings(), lo, hi, cube_body)
    assert not off.active and np.all(off.cell_scale(points) == 1.0)


def test_from_skin_grading_uses_depth(cube, cube_body) -> None:
    lo, hi = bounds(cube)
    g = Grading(Settings(grading_mode="from_skin", grading_strength=100, grading_target="both"),
                lo, hi, cube_body)
    assert g.max_depth == pytest.approx(25.0, abs=1.5)  # centre of a 50 mm cube
    points = np.array([[0.0, 0, 0.0], [0, 0, 25.0]])  # on the skin, at the centre
    np.testing.assert_allclose(g.cell_scale(points), [0.5, 2.0], rtol=0.06)
    # On the grid the same numbers come from the distance field.
    scale = g.thickness_scale_on_grid(cube_body.grid, cube_body.sdf)
    assert scale.max() == pytest.approx(2.0, rel=0.05) and scale.min() == pytest.approx(0.5, rel=0.06)


def test_thickness_scale_on_grid_along_an_axis(cube, cube_body) -> None:
    lo, hi = bounds(cube)
    g = Grading(Settings(grading_mode="x", grading_strength=100, grading_target="thickness"),
                lo, hi, cube_body)
    scale = g.thickness_scale_on_grid(cube_body.grid, cube_body.sdf)
    assert scale.shape == (cube_body.grid.shape[0], 1, 1)
    assert scale[0, 0, 0] == pytest.approx(2.0) and scale[-1, 0, 0] == pytest.approx(0.5)


# --- graded seeds -----------------------------------------------------------------

def test_graded_interior_seeds_follow_the_local_spacing(cube, cube_body) -> None:
    lo, hi = bounds(cube)
    settings = Settings(grading_mode="z", grading_strength=100)
    g = Grading(settings, lo, hi, cube_body)
    seeds = poisson_seeds(cube_body, lo, hi, 8.0, np.random.default_rng(1),
                          scale=g.cell_scale, scale_range=scale_range(settings, "cell_size"))
    inside = (np.abs(seeds[:, :2]) < 25).all(axis=1) & (seeds[:, 2] > 0) & (seeds[:, 2] < 50)
    bottom = inside & (seeds[:, 2] < 12)
    top = inside & (seeds[:, 2] > 38)
    assert bottom.sum() > 8 * top.sum()  # spacing ratio ~3 between the two slabs -> ~27x density
    nearest, _ = cKDTree(seeds).query(seeds, k=2)
    # No seed is closer to another than (almost) its own local spacing.
    assert (nearest[:, 1] >= 8.0 * g.cell_scale(seeds) * 0.5 - 1e-9).all()
    assert nearest[bottom, 1].mean() < 0.5 * nearest[top, 1].mean()
    again = poisson_seeds(cube_body, lo, hi, 8.0, np.random.default_rng(1),
                          scale=g.cell_scale, scale_range=scale_range(settings, "cell_size"))
    assert np.array_equal(seeds, again)


def test_graded_seeds_do_not_depend_on_the_voxel_size() -> None:
    mesh = load("sphere_60")
    lo, hi = bounds(mesh)
    settings = Settings(grading_mode="from_skin", grading_strength=80)
    sets = []
    for h in (1.2, 0.6):
        body = BodyField(mesh.vertices, mesh.faces, Grid.for_bounds(lo, hi, h))
        g = Grading(settings, lo, hi, body)
        sets.append(poisson_seeds(body, lo, hi, 10.0, np.random.default_rng(2),
                                  scale=g.cell_scale, scale_range=scale_range(settings, "cell_size")))
    assert np.array_equal(sets[0], sets[1])


def test_graded_surface_seeds(cube, cube_body) -> None:
    lo, hi = bounds(cube)
    settings = Settings(grading_mode="x", grading_strength=100)
    g = Grading(settings, lo, hi, cube_body)
    seeds = surface_seeds(cube.vertices[cube.faces], 6.0, np.random.default_rng(1),
                          scale=g.cell_scale, scale_range=scale_range(settings, "cell_size"))
    assert (seeds[:, 0] < -10).sum() > 3 * (seeds[:, 0] > 10).sum()


def test_tapered_strut() -> None:
    segment = np.array([[[0.0, 0, 0], [20.0, 0, 0]]])
    evaluate = StrutEvaluator(segment, np.array([[2.0, 1.0]]), blend=0.0, reach=4.0, min_spacing=0.5)
    points = np.array([[0.0, 3, 0], [10.0, 3, 0], [20.0, 3, 0]])
    np.testing.assert_allclose(evaluate(points), [1.0, 1.5, 2.0])


# --- graded results ---------------------------------------------------------------

def test_cell_grading_along_z_moves_material(cube) -> None:
    base = dict(surface_style="none", interior_style="struts", interior_cell_size=10.0)
    dense_bottom = generate(cube, Settings(**base, grading_mode="z", grading_strength=100), "preview")
    dense_top = generate(cube, Settings(**base, grading_mode="z", grading_strength=100,
                                        grading_direction="denser_at_end"), "preview")
    assert dense_bottom.valid and dense_top.valid
    assert material_in_slab(dense_bottom, 2, 0, 15) > 2.5 * material_in_slab(dense_bottom, 2, 35, 50)
    assert material_in_slab(dense_top, 2, 35, 50) > 2.5 * material_in_slab(dense_top, 2, 0, 15)


def test_thickness_grading_of_struts_and_shell(cube) -> None:
    index = SurfaceIndex(cube.vertices, cube.faces, spacing=0.5)
    shell = generate(cube, Settings(surface_style="solid", interior_style="hollow",
                                    grading_mode="x", grading_strength=100,
                                    grading_target="thickness"), "final")
    assert shell.valid
    # 2 mm x 2 at X min falling to 2 mm x 0.5 at X max; the multiplier applies where the
    # inner wall sits, so the -X wall is 3.6 mm (t = 2 * 4^(0.5 - t/50)) and the +X wall 1.03 mm.
    depth = -index.signed_distance(shell.vertices, k=8)
    middle = (np.abs(shell.vertices[:, 1]) < 10) & (np.abs(shell.vertices[:, 2] - 25) < 10) & (depth > 0.3)
    left = middle & (shell.vertices[:, 0] < 0)
    right = middle & (shell.vertices[:, 0] > 0)
    assert depth[left].mean() == pytest.approx(3.6, abs=0.2)
    assert depth[right].mean() == pytest.approx(1.03, abs=0.2)

    struts = generate(cube, Settings(surface_style="none", interior_style="struts",
                                     grading_mode="x", grading_strength=100,
                                     grading_target="thickness"), "preview")
    assert material_in_slab(struts, 0, -25, -10) > 1.8 * material_in_slab(struts, 0, 10, 25)


def test_from_skin_grading_with_patterned_skin_is_valid(cube) -> None:
    for target in ("cell_size", "thickness", "both"):
        for direction in ("denser_at_start", "denser_at_end"):
            result = generate(cube, Settings(grading_mode="from_skin", grading_strength=60,
                                             grading_target=target, grading_direction=direction),
                              "preview")
            out = as_trimesh(result)
            assert result.valid and out.is_watertight and solid_bodies(out) == 1


@pytest.mark.parametrize("surface", ["perforated", "rib_lattice", "embossed"])
def test_axis_grading_of_every_patterned_skin(cube, surface: str) -> None:
    result = generate(cube, Settings(surface_style=surface, grading_mode="y", grading_strength=100,
                                     grading_target="both"), "preview")
    out = as_trimesh(result)
    assert result.valid and out.is_watertight and solid_bodies(out) == 1
    lo, hi = bounds(cube)
    assert np.abs(result.vertices.min(axis=0) - lo).max() <= result.voxel_mm
    assert np.abs(result.vertices.max(axis=0) - hi).max() <= result.voxel_mm


def test_grading_makes_voxels_fine_enough_for_the_thin_end() -> None:
    plain = Settings(surface_style="none")
    graded = plain.model_copy(update={"grading_mode": "z", "grading_strength": 100,
                                      "grading_target": "thickness"})
    assert choose_voxel_size((50, 50, 50), graded, "final")[0] == pytest.approx(
        0.5 * choose_voxel_size((50, 50, 50), plain, "final")[0])


# --- solid base -------------------------------------------------------------------

def test_solid_base_fills_the_bottom_layer(cube) -> None:
    open_bottom = generate(cube, Settings(surface_style="none", interior_style="struts",
                                          cleanup="off"), "final")
    with_base = generate(cube, Settings(surface_style="none", interior_style="struts",
                                        solid_base_thickness=3.0, cleanup="off"), "final")
    out = as_trimesh(with_base)
    assert with_base.valid and out.is_watertight and solid_bodies(out) == 1
    assert out.volume - as_trimesh(open_bottom).volume == pytest.approx(50 * 50 * 3.0, rel=0.12)
    # The base covers the whole footprint and is flat on the bed.
    bottom = with_base.vertices[with_base.vertices[:, 2] < 1e-6]
    # (sharp outer edges are rounded by up to a voxel, so the flat area stops just short)
    assert np.ptp(bottom[:, 0]) >= 50.0 - 2.2 * with_base.voxel_mm
    assert np.ptp(bottom[:, 1]) >= 50.0 - 2.2 * with_base.voxel_mm
    assert with_base.vertices[:, 2].min() == pytest.approx(0.0, abs=1e-6)
    # ...and is a full slab: inside it there is no surface except the four side walls.
    centre = out.triangles_center
    in_slab = (centre[:, 2] > 0.6) & (centre[:, 2] < 2.4)
    on_side_wall = np.abs(centre[:, :2]).max(axis=1) > 25.0 - 0.6
    assert in_slab.sum() > 100 and on_side_wall[in_slab].all()


def test_solid_base_follows_the_envelope(cube) -> None:
    sphere = load("sphere_60")
    result = generate(sphere, Settings(solid_base_thickness=5.0), "preview")
    index = SurfaceIndex(sphere.vertices, sphere.faces, spacing=0.5)
    assert result.valid and index.signed_distance(result.vertices, k=12).max() <= 1e-3


def test_solid_base_changes_nothing_on_a_solid_part(cube) -> None:
    plain = generate(cube, Settings(surface_style="solid", interior_style="solid"), "preview")
    based = generate(cube, Settings(surface_style="solid", interior_style="solid",
                                    solid_base_thickness=5.0), "preview")
    assert np.array_equal(plain.vertices, based.vertices)


# --- printability -----------------------------------------------------------------

def test_defaults_give_no_printability_warnings() -> None:
    assert printability_warnings(Settings()) == []
    for surface in ("rib_lattice", "embossed", "solid", "none"):
        assert printability_warnings(Settings(surface_style=surface)) == []


def test_thin_features_are_reported() -> None:
    text = "\n".join(printability_warnings(Settings(strut_diameter=1.0, shell_thickness=1.0)))
    assert "Strut diameter is 1.00 mm" in text and "Shell wall is 1.00 mm" in text
    assert "3 × nozzle (1.20 mm)" in text
    assert printability_warnings(Settings(strut_diameter=1.0, shell_thickness=1.0,
                                          nozzle_diameter=0.3)) == []
    assert any("Rib width" in w for w in printability_warnings(
        Settings(surface_style="rib_lattice", rib_width=1.0)))
    # Only settings of the selected styles are checked.
    assert printability_warnings(Settings(interior_style="hollow", strut_diameter=0.8)) == []


def test_small_holes_and_thin_walls_between_holes_are_reported() -> None:
    small = printability_warnings(Settings(surface_cell_size=3.0, hole_size=50))
    assert any("Holes are about 1.50 mm" in w and "5 × nozzle (2.00 mm)" in w for w in small)
    thin = printability_warnings(Settings(surface_cell_size=10.0, hole_size=90))
    assert any("Wall between holes is 1.00 mm" in w for w in thin)
    ribs = printability_warnings(Settings(surface_style="rib_lattice", surface_cell_size=3.0))
    assert any("Openings between ribs" in w for w in ribs)


def test_grading_is_taken_into_account() -> None:
    graded = Settings(grading_mode="z", grading_strength=100, grading_target="thickness")
    text = "\n".join(printability_warnings(graded))
    assert "Strut diameter is 1.00 mm at the thin end of the grading" in text


def test_warnings_reach_the_result(cube) -> None:
    result = generate(cube, Settings(strut_diameter=1.0), "preview")
    assert any("Strut diameter" in w for w in result.warnings)
    assert result.valid  # warnings never block generation
