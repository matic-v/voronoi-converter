"""Smoothness: subdivision and projection onto the exact shape."""

from __future__ import annotations

import io

import numpy as np
import pytest
import trimesh

import voronoi_engine.generate as pipeline
from voronoi_engine import test_models
from voronoi_engine.export import to_binary_stl
from voronoi_engine.generate import generate
from voronoi_engine.mesh_io import LoadedMesh, load_stl
from voronoi_engine.settings import Settings
from voronoi_engine.smoothing import allowed_levels, project, subdivide, vertex_normals
from voronoi_engine.surface import SurfaceIndex


def load(name: str) -> LoadedMesh:
    return load_stl(test_models.to_binary_stl(test_models.build(name)), f"{name}.stl")


@pytest.fixture(scope="module")
def cube() -> LoadedMesh:
    return load("cube_50")


def as_trimesh(result) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=result.vertices, faces=result.faces, process=False)


def generate_with_field(mesh: LoadedMesh, settings: Settings):
    """Run a final generation and also hand back the exact field it was smoothed against."""
    captured = {}
    original = pipeline.ExactField

    def Capturing(*args, **kwargs):  # noqa: N802 - stands in for the class
        captured["field"] = original(*args, **kwargs)
        return captured["field"]

    pipeline.ExactField = Capturing
    try:
        result = generate(mesh, settings, "final")
    finally:
        pipeline.ExactField = original
    return result, captured.get("field")


def facet_error(result, field) -> float:
    """Mean distance of the triangle centres from the exact surface: the size of the facets."""
    tri = result.vertices[result.faces]
    centres = tri.mean(axis=1)
    step = max(1, len(centres) // 60000)
    return float(np.abs(field(centres[::step])).mean())


# --- building blocks ------------------------------------------------------------

def test_subdivide_splits_every_triangle_in_four_and_stays_closed() -> None:
    mesh = test_models.sphere()
    v, f = subdivide(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    out = trimesh.Trimesh(v, f, process=False)
    assert len(f) == 4 * len(mesh.faces) and len(v) == len(mesh.vertices) + len(mesh.edges_unique)
    assert out.is_watertight and out.is_winding_consistent
    assert out.volume == pytest.approx(mesh.volume)  # flat split: same shape


def test_project_puts_vertices_on_the_exact_surface() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=10.0)
    v, f = subdivide(np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces))
    sphere = lambda p: np.linalg.norm(p, axis=1) - 10.0  # noqa: E731
    before = np.abs(sphere(v)).max()
    moved = project(v, vertex_normals(v, f), sphere, limit=2.0)
    assert before > 0.4 and np.abs(sphere(moved)).max() < 1e-3
    # No vertex ever moves further than the limit, and none moves away from the surface.
    held = project(v, vertex_normals(v, f), sphere, limit=0.1)
    assert np.linalg.norm(held - v, axis=1).max() <= 0.1 + 1e-12
    assert (np.abs(sphere(held)) <= np.abs(sphere(v)) + 1e-12).all()


def test_levels_are_capped_by_the_triangle_limit() -> None:
    assert allowed_levels(100_000, 2) == 2
    assert allowed_levels(2_000_000, 2) == 1  # 32 M would be too many, 8 M is fine
    assert allowed_levels(5_000_000, 2) == 0
    assert allowed_levels(100_000, 0) == 0


# --- the exact field agrees with the grid pipeline ---------------------------------

@pytest.mark.parametrize("settings", [
    Settings(smoothness="smooth"),
    Settings(smoothness="smooth", surface_style="rib_lattice", blend_radius=1.5),
    Settings(smoothness="smooth", surface_style="embossed", interior_style="solid"),
    Settings(smoothness="smooth", surface_style="embossed", interior_style="hollow"),
    Settings(smoothness="smooth", surface_style="perforated", interior_style="solid"),
    Settings(smoothness="smooth", surface_style="solid", interior_style="hollow"),
    Settings(smoothness="smooth", surface_style="none", solid_base_thickness=3.0),
    Settings(smoothness="smooth", grading_mode="z", grading_strength=80, grading_target="both"),
    Settings(smoothness="smooth", grading_mode="from_skin", grading_target="thickness"),
    Settings(smoothness="smooth", surface_style="rib_lattice", keep_edges=[0, 3, 7]),
])
def test_exact_field_matches_the_unsmoothed_mesh(cube, settings: Settings) -> None:
    """The marching-cubes surface must lie on the zero level of the exact field
    (to within the interpolation error of the grid), for every construction."""
    result, field = generate_with_field(cube, settings)
    plain = generate(cube, settings.model_copy(update={"smoothness": "off"}), "final")
    residual = np.abs(field(plain.vertices))
    h = plain.voxel_mm
    assert np.median(residual) < 0.02 * h
    assert np.quantile(residual, 0.99) < 0.25 * h


# --- results ------------------------------------------------------------------------

@pytest.mark.parametrize("surface,interior", [("perforated", "struts"), ("rib_lattice", "struts"),
                                              ("embossed", "hollow"), ("none", "struts")])
def test_smoothing_multiplies_triangles_and_reduces_the_facets(cube, surface: str, interior: str) -> None:
    base = Settings(surface_style=surface, interior_style=interior)
    plain = generate(cube, base, "final")
    smooth1, field = generate_with_field(cube, base.model_copy(update={"smoothness": "smooth"}))
    smooth2 = generate(cube, base.model_copy(update={"smoothness": "extra"}), "final")
    assert len(smooth1.faces) == pytest.approx(4 * len(plain.faces), rel=0.01)
    assert len(smooth2.faces) == pytest.approx(16 * len(plain.faces), rel=0.01)
    e0, e1, e2 = (facet_error(r, field) for r in (plain, smooth1, smooth2))
    assert e1 < 0.45 * e0 and e2 < 0.6 * e1
    from voronoi_engine.meshing import signed_volume
    from voronoi_engine.repair import validate
    plain_volume = signed_volume(plain.vertices, plain.faces)
    for result in (smooth1, smooth2):
        # validate() = closed, manifold, consistently wound, positive volume, one body
        assert result.valid and validate(result.vertices, result.faces) == []
        # round struts are slightly fatter than their faceted version
        assert signed_volume(result.vertices, result.faces) == pytest.approx(plain_volume, rel=0.05)


def test_smoothed_struts_are_round(cube) -> None:
    """Without a skin and without blending, every surface point away from the joints
    and the cut ends is exactly one strut radius from a strut axis."""
    settings = Settings(surface_style="none", interior_style="struts", blend_radius=0.0)
    plain, field = generate_with_field(cube, settings.model_copy(update={"smoothness": "smooth"}))
    rough = generate(cube, settings, "final")
    index = SurfaceIndex(cube.vertices, cube.faces, spacing=0.5)

    def radius_error(result) -> float:
        inside = -index.signed_distance(result.vertices) > 1.5  # away from the cut ends
        points = result.vertices[inside][::7]
        return float(np.quantile(np.abs(field.struts(points)), 0.9))

    assert radius_error(plain) < 0.25 * radius_error(rough)
    assert radius_error(plain) < 0.01


@pytest.mark.parametrize("name", ["cube_50", "sphere_60", "bracket_100"])
def test_smoothed_result_keeps_the_envelope_and_exports_cleanly(name: str) -> None:
    mesh = load(name)
    result = generate(mesh, Settings(smoothness="smooth"), "final")
    out = trimesh.load(io.BytesIO(to_binary_stl(result.vertices, result.faces)), file_type="stl")
    assert out.is_watertight and out.is_winding_consistent and out.volume > 0
    envelope = SurfaceIndex(mesh.vertices, mesh.faces, spacing=0.3)
    assert envelope.signed_distance(out.vertices, k=12).max() <= 1e-3
    lo, hi = np.asarray(mesh.info.bounds_min), np.asarray(mesh.info.bounds_max)
    tolerance = 0.6 if name == "sphere_60" else 0.3
    assert np.abs(out.bounds[0] - lo).max() <= tolerance and np.abs(out.bounds[1] - hi).max() <= tolerance


def test_smoothing_is_deterministic_and_final_only(cube) -> None:
    settings = Settings(smoothness="smooth")
    first = generate(cube, settings, "final")
    second = generate(cube, settings, "final")
    assert to_binary_stl(first.vertices, first.faces) == to_binary_stl(second.vertices, second.faces)
    preview = generate(cube, settings, "preview")
    plain_preview = generate(cube, Settings(), "preview")
    assert np.array_equal(preview.vertices, plain_preview.vertices)


def test_smoothness_is_reduced_with_a_warning_when_the_result_would_be_too_large(cube, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "allowed_levels", lambda triangles, wanted: min(wanted, 1))
    result = generate(cube, Settings(smoothness="extra", interior_style="hollow"), "final")
    assert result.valid
    warning = next(w for w in result.warnings if "Smoothness was reduced" in w)
    assert "from Extra smooth to Smooth" in warning and "With Draft quality" in warning


def test_smoothness_warning_wording() -> None:
    from voronoi_engine.generate import _smoothness_reduced_warning
    text = _smoothness_reduced_warning(Settings(smoothness="extra"), 775_000, 2, 1)
    assert "Extra smooth would give 12.4 million triangles, more than 12 million" in text
    assert "The result has 3.1 million triangles." in text and "With Draft quality" in text
    draft = _smoothness_reduced_warning(Settings(smoothness="extra", quality="draft"), 775_000, 2, 1)
    assert "Draft" not in draft.split("triangles.")[-1]


def test_smoothing_does_not_fold_triangles_over(cube) -> None:
    """Every triangle of the smoothed mesh still faces the way its parent triangle did."""
    from voronoi_engine.smoothing import face_normals
    settings = Settings(surface_style="rib_lattice")  # sharp-edged ribs: the hard case
    plain = generate(cube, settings, "final")
    smooth1 = generate(cube, settings.model_copy(update={"smoothness": "smooth"}), "final")
    assert len(smooth1.faces) == 4 * len(plain.faces)  # nothing merged: order is parent-wise
    parent = np.tile(face_normals(plain.vertices, plain.faces), (4, 1))
    child = face_normals(smooth1.vertices, smooth1.faces)
    assert (np.einsum("ij,ij->i", parent, child) > 0.0).all()


def test_undo_folds_restores_a_folded_triangle() -> None:
    from voronoi_engine.smoothing import undo_folds
    before = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    after = before.copy()
    after[3] = [-1.0, -1.0, 0.0]  # pulled through the opposite edge: second triangle flips
    fixed = undo_folds(before, after, faces)
    assert np.array_equal(fixed[3], before[3])
    gentle = before + [[0, 0, 0.05], [0, 0, 0], [0, 0, 0], [0, 0, -0.05]]
    assert np.array_equal(undo_folds(before, gentle, faces), gentle)  # small moves are kept


# --- edge rounding of the skin -----------------------------------------------------------

def sharp_share(result, angle: float = 60.0) -> float:
    """Share of mesh edges where the two triangles meet at more than `angle` degrees."""
    mesh = as_trimesh(result)
    return float((np.degrees(mesh.face_adjacency_angles) > angle).mean())


def spike_count(result) -> int:
    """Vertices further from the average of their neighbours than the mean edge length."""
    mesh = as_trimesh(result)
    offset = np.array([np.linalg.norm(mesh.vertices[i] - mesh.vertices[n].mean(axis=0)) if n else 0.0
                       for i, n in enumerate(mesh.vertex_neighbors)])
    return int((offset > mesh.edges_unique_length.mean()).sum())


@pytest.fixture(scope="module")
def sphere() -> LoadedMesh:
    return load("sphere_60")


@pytest.mark.parametrize("surface", ["rib_lattice", "perforated", "embossed"])
def test_edge_rounding_removes_the_saw_teeth_on_the_skin(sphere, surface: str) -> None:
    base = Settings(surface_style=surface, interior_style="hollow", smoothness="smooth")
    sharp = generate(sphere, base.model_copy(update={"edge_rounding": 0.0}), "final")
    rounded = generate(sphere, base, "final")  # default edge rounding 0.5 mm
    assert base.edge_rounding == 0.5
    assert rounded.valid and sharp.valid
    assert sharp_share(rounded) < 0.25 * sharp_share(sharp)
    assert spike_count(rounded) == 0


def test_smoothing_without_edge_rounding_makes_no_spikes(sphere) -> None:
    settings = Settings(surface_style="rib_lattice", interior_style="hollow", edge_rounding=0.0,
                        smoothness="smooth")
    result = generate(sphere, settings, "final")
    assert result.valid and spike_count(result) <= 5  # was ~200 before the spike guard


def test_edge_rounding_only_removes_material_from_an_open_skin(cube) -> None:
    from voronoi_engine.meshing import signed_volume
    base = Settings(surface_style="rib_lattice", interior_style="hollow")
    volumes = [signed_volume(r.vertices, r.faces) for r in
               (generate(cube, base.model_copy(update={"edge_rounding": v}), "final") for v in (0.0, 0.5, 1.0))]
    assert volumes[0] > volumes[1] > volumes[2] > 0.8 * volumes[0]


def test_edge_rounding_keeps_the_outer_dimensions(cube) -> None:
    for surface in ("perforated", "rib_lattice", "embossed"):
        result = generate(cube, Settings(surface_style=surface, edge_rounding=1.0, smoothness="smooth"),
                          "final")
        assert result.valid
        np.testing.assert_allclose(result.vertices.min(axis=0), cube.info.bounds_min, atol=0.3)
        np.testing.assert_allclose(result.vertices.max(axis=0), cube.info.bounds_max, atol=0.3)


def test_relax_spikes_moves_only_the_spike() -> None:
    from voronoi_engine.smoothing import relax_spikes
    grid = np.stack(np.meshgrid(np.arange(5.0), np.arange(5.0), indexing="ij"), axis=-1).reshape(-1, 2)
    vertices = np.column_stack([grid, np.zeros(len(grid))])
    faces = []
    for x in range(4):
        for y in range(4):
            v = x * 5 + y
            faces += [(v, v + 5, v + 6), (v, v + 6, v + 1)]
    faces = np.array(faces)
    spiky = vertices.copy()
    spiky[12, 2] = 3.0  # the middle vertex sticks up
    relaxed = relax_spikes(spiky, faces)
    assert relaxed[12, 2] < 0.5
    np.testing.assert_array_equal(np.delete(relaxed, 12, axis=0), np.delete(spiky, 12, axis=0))
    assert np.array_equal(relax_spikes(vertices, faces), vertices)



def test_grid_and_exact_field_agree_at_strut_joints(sphere) -> None:
    """Regression: blending several struts at a joint reaches beyond a single strut's
    reach; the exact field missed that, and smoothing then left flaps on the surface."""
    captured = {}
    original_field, original_keep = pipeline.ExactField, pipeline.keep_largest_body

    def capture_field(*args, **kwargs):
        captured["field"] = original_field(*args, **kwargs)
        return captured["field"]

    def capture_grid(field, h):
        captured["grid"] = field.copy()
        return original_keep(field, h)

    pipeline.ExactField, pipeline.keep_largest_body = capture_field, capture_grid
    try:
        generate(sphere, Settings(surface_style="rib_lattice", smoothness="smooth"), "final")
    finally:
        pipeline.ExactField, pipeline.keep_largest_body = original_field, original_keep
    field, grid = captured["field"], captured["grid"]
    h = field.body.grid.h
    nodes = np.nonzero(np.abs(grid) < 1.5 * h)
    difference = np.abs(field(field.body.grid.points(nodes)) - grid[nodes])
    assert difference.max() < 0.15 * h
