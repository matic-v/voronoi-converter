"""The generation pipeline: input mesh + settings -> output mesh.

Preview and final run the same code at different voxel sizes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, MutableMapping

import numpy as np
from scipy import ndimage

from .body_field import BodyField
from .grading import Grading, scale_range
from .grid import PREVIEW_MIN_VOXELS_ACROSS, Grid, Mode, Resolution, choose_resolution
from .lattice import interior_edges
from .mesh_io import LoadedMesh
from .orientation import oriented
from .meshing import extract_mesh, fit_to_envelope, keep_largest_body, signed_volume
from .printability import printability_warnings
from .progress import Job
from .repair import repair, validate, weld_as_exported
from .edge_members import member_depth, member_field, member_reach
from .exact_field import ExactField
from .settings import SMOOTHING_LEVELS, Settings
from .smoothing import MAX_TRIANGLES, allowed_levels, max_triangles, smooth
from .skin_attach import attach_edges_to_skin
from .supports import make_self_supporting
from .strut_field import make_evaluator, smooth_max, smooth_min, strut_field, strut_margin
from .surface_pattern import (SurfacePattern, pointed_tops, prepare_pointed_tops,
                              surface_seeds)

# Independent random streams, so the surface pattern does not change when the
# interior style does (and the other way round).
INTERIOR_STREAM, SURFACE_STREAM = 1, 2
# Corner rounding of the Voronoi cells, as a fraction of the surface cell size.
HOLE_ROUNDING = 0.3
RECESS_ROUNDING = 0.15


class GenerationError(Exception):
    """Generation cannot produce a result; the message is written for the user."""


@dataclass
class GenerationResult:
    vertices: np.ndarray  # (V, 3) float64
    faces: np.ndarray  # (F, 3) int64
    mode: Mode
    voxel_mm: float
    grid_shape: tuple[int, int, int]
    seconds: float
    fragments_removed: int
    quality_capped: bool
    valid: bool
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def info(self) -> dict[str, Any]:
        lo, hi = self.vertices.min(axis=0), self.vertices.max(axis=0)
        return {
            "mode": self.mode,
            "voxel_mm": self.voxel_mm,
            "grid_shape": list(self.grid_shape),
            "size_mm": [float(v) for v in hi - lo],
            "bounds_min": [float(v) for v in lo],
            "bounds_max": [float(v) for v in hi],
            "triangles": int(len(self.faces)),
            "vertices": int(len(self.vertices)),
            "volume_mm3": signed_volume(self.vertices, self.faces),
            "seconds": round(self.seconds, 3),
            "fragments_removed": self.fragments_removed,
            "quality_capped": self.quality_capped,
            "valid": self.valid,
            "problems": self.problems,
            "warnings": self.warnings,
        }


def generate(mesh: LoadedMesh, settings: Settings, mode: Mode = "preview",
             cache: MutableMapping[float, BodyField] | None = None,
             job: Job | None = None) -> GenerationResult:
    """Build the output mesh. `cache` (field_cache_key -> BodyField) lets repeated calls
    for the same input mesh skip the body distance field."""
    job = job or Job()
    started = time.perf_counter()
    warnings: list[str] = []
    mesh = oriented(mesh, settings.bed_normal)  # print orientation: +Z is up from here on

    surface, interior = settings.surface_style, settings.interior_style
    warnings += printability_warnings(settings)

    resolution = choose_resolution(mesh.info.size_mm, settings, mode)
    h, capped = resolution.voxel_mm, resolution.capped
    warnings += _resolution_warnings(resolution, mode)
    lo, hi = np.asarray(mesh.info.bounds_min), np.asarray(mesh.info.bounds_max)
    grid = Grid.for_bounds(lo, hi, h)
    job.report(0.02, "Measuring the body")
    cache_key = field_cache_key(h, settings)
    body = cache.get(cache_key) if cache is not None else None
    if body is None:
        body = BodyField(mesh.vertices, mesh.faces, grid, job.sub(0.02, 0.35))
        if cache is not None:
            cache[cache_key] = body
    d = body.sdf

    job.report(0.35, "Building the skin")
    blend = settings.blend_radius
    grading = Grading(settings, lo, hi, body)
    _, thickest = scale_range(settings, "thickness")
    cell_range = scale_range(settings, "cell_size")
    cell_scale = grading.cell_scale if grading.cells else None

    # Skin. Its depth is the shell thickness (the rib width for B), times the grading.
    skin, pattern, evaluator, skin_depth, band = None, None, None, np.float32(0.0), None
    # Edges of a patterned skin (rib edges, hole rims, recess edges) are rounded.
    rounding = settings.edge_rounding if surface in ("perforated", "rib_lattice", "embossed") else 0.0
    if surface != "none":
        nominal = settings.rib_width if surface == "rib_lattice" else settings.shell_thickness
        thickness_scale = grading.thickness_scale_on_grid(grid, d)
        skin_depth = np.float32(nominal) * thickness_scale
        if not (surface == "solid" and interior == "solid"):
            body.ensure_depth(nominal * thickest + blend + 2 * h + rounding)
            skin, pattern, band = _skin_field(surface, settings, body, skin_depth, thickness_scale, grading,
                               cell_range, job.sub(0.35, 0.42))

    if interior == "solid":
        if skin is None or surface == "solid":
            solid = d.copy()
        else:  # patterned skin on a solid core that starts below the skin
            solid = np.minimum(skin, d + skin_depth)
            if rounding > 0 and band is not None:  # round the bottoms of the pockets
                core = d[band] + (skin_depth[band] if np.ndim(skin_depth) == 3 else skin_depth)
                solid[band] = smooth_min(skin[band], core, rounding)
    elif interior == "struts":
        job.report(0.42, "Placing struts")
        radius = settings.strut_diameter / 2.0
        rng = np.random.default_rng([settings.random_seed, INTERIOR_STREAM])
        edges = interior_edges(body, lo, hi, settings.interior_cell_size,
                               reach=radius * thickest + h, rng=rng,
                               scale=cell_scale, scale_range=cell_range,
                               stretch=settings.cell_stretch if settings.self_supporting else 1.0)
        if len(edges) == 0:
            if skin is None:
                raise GenerationError(
                    "No struts fall inside the part. Use a smaller interior cell size or add a skin.")
            warnings.append("No struts fall inside the part: the interior cell size is too large.")
        if surface in ("perforated", "rib_lattice") and pattern is not None and len(edges):
            # An open skin has holes: aim every strut that reaches the skin at the
            # nearest wall or rib, so that it ends on material instead of in a hole.
            def depth_at(points: np.ndarray) -> np.ndarray:
                return nominal * grading.thickness_scale(points)

            edges, attached = attach_edges_to_skin(edges, body, pattern, depth_at)
            if attached == 0:
                warnings.append("No strut reaches the skin: the interior lattice is not "
                                "connected to it. Use a smaller interior cell size.")
        if settings.self_supporting and len(edges):
            # Every node held from below, no long flat bridges (print without supports).
            job.report(0.44, "Making the struts self-supporting")
            edges, support = make_self_supporting(
                edges, body, overhang_deg=settings.overhang_angle, max_bridge=settings.max_bridge,
                reach=settings.interior_cell_size * cell_range[1] * settings.cell_stretch,
                strut_radius=radius * thickest,
                skin_depth=None if surface == "none" else float(nominal * thickest), bed_z=lo[2],
                skin_value=None if skin is None else (
                    lambda points, skin=skin: ndimage.map_coordinates(
                        skin, grid.to_index(points).T, order=1, mode="nearest")))
            if support.unsupported:
                warnings.append(
                    f"{support.unsupported} strut node(s) could not be held from below (nothing "
                    "under them inside the part) and will need support when printing. A skin, a "
                    "solid base or another bed face avoids this.")
        # Radius at both ends of every strut (it tapers linearly when thickness is graded).
        end_radius = radius * grading.thickness_scale(edges.reshape(-1, 3)).reshape(-1, 2)
        if len(edges):
            evaluator = make_evaluator(edges, end_radius, blend, h)
        solid = strut_field(grid, d, edges, end_radius, blend, job.sub(0.45, 0.72),
                            evaluator=evaluator)
        if surface in ("perforated", "rib_lattice"):
            # An open skin would let strut ends poke out through its holes: stop the
            # struts half-way into the skin, where they still merge with its material.
            solid = np.maximum(solid, d + np.float32(0.5) * skin_depth)
        elif surface == "embossed":
            # End the struts half-way between the recess floor and the shell's inner face:
            # inside the floor material, so their blend never shows on the floor.
            emboss = np.float32(settings.emboss_depth)
            solid = np.maximum(solid, d + emboss + np.float32(0.5) * (skin_depth - emboss))
        if skin is not None:  # struts merge into the skin with the same fillet as strut joints
            solid = smooth_min(skin, solid, blend)
    elif skin is not None:
        solid = skin
    else:
        raise GenerationError(
            "Surface style None with interior style Hollow leaves nothing to generate. "
            "Choose a skin or an interior."
        )
    del skin

    keep_edges = None
    if settings.keep_edges:
        # Kept edges become members of the lattice (a rib or strut along the edge).
        model_edges = mesh.edges
        if max(settings.keep_edges) >= model_edges.count:
            raise GenerationError(
                f"The kept edges in these settings do not match this model (it has "
                f"{model_edges.count} edges to pick). Clear the kept edges, or load the model "
                "the settings were saved for.")
        job.report(0.72, "Adding struts along the kept edges")
        segments = model_edges.segments_of(settings.keep_edges)
        reach = member_reach(settings, h, rounding)
        depth_e = member_depth(settings)
        zero = np.zeros(len(segments))
        keep_edges = make_evaluator(segments, zero, 0.0, h, margin=reach)
        to_edges = strut_field(grid, d, segments, zero, 0.0, evaluator=keep_edges, margin=reach)
        nodes = np.nonzero((to_edges < reach) & (d > -(depth_e + 2 * h + rounding + blend)) & (d < 2 * h))
        members = member_field(d[nodes].astype(np.float64), to_edges[nodes].astype(np.float64),
                               settings.keep_width, depth_e, rounding)
        del to_edges
        # Joined to skin and struts like a strut joint.
        solid[nodes] = smooth_min(solid[nodes], members, blend)

    if settings.solid_base_thickness > 0:
        # Flat solid layer at the bottom of the part, for bed adhesion.
        z = grid.origin[2] + np.arange(grid.shape[2]) * h
        base = (z - (lo[2] + settings.solid_base_thickness)).astype(np.float32)
        solid = np.minimum(solid, base.reshape(1, 1, -1))
    solid = np.maximum(solid, d)  # never outside the original envelope

    job.report(0.72, "Removing loose fragments")
    fragments = keep_largest_body(solid, h)
    if not (solid < 0).any():
        raise GenerationError(
            "The result is empty: the features are too thin for the part at this detail level."
        )

    job.report(0.75, "Meshing")
    vertices, faces = extract_mesh(solid, grid)
    del solid
    job.report(0.82, "Fitting to the original surface")
    # The preview only moves stray vertices back; the final also refines concave edges.
    vertices, faces = fit_to_envelope(vertices, faces, body, passes=2 if mode == "final" else 0)
    wanted_levels = SMOOTHING_LEVELS[settings.smoothness] if mode == "final" else 0
    levels = allowed_levels(len(faces), wanted_levels)
    if levels < wanted_levels:
        warnings.append(_smoothness_reduced_warning(settings, len(faces), wanted_levels, levels))
    if levels:
        exact = ExactField(
            settings, body, grading, pattern, evaluator, strut_margin(blend, h),
            lo[2] + settings.solid_base_thickness if settings.solid_base_thickness > 0 else None,
            exact_depth=(nominal * thickest + 2 * h + rounding) if surface != "none" else 0.0,
            keep=keep_edges, keep_rounding=rounding)
        vertices, faces = smooth(vertices, faces, exact, levels, h, job.sub(0.85, 0.95))
        # Whatever the projection did, nothing may end outside the original envelope.
        vertices, faces = fit_to_envelope(vertices, faces, body, passes=0)
    job.report(0.95, "Checking the result")
    vertices, faces, _ = weld_as_exported(vertices, faces)
    problems = validate(vertices, faces)
    if problems:  # repair our own defects; if that fails the result must not be exported
        job.report(0.97, "Repairing the mesh")
        vertices, faces, repairs = repair(vertices, faces)
        remaining = validate(vertices, faces)
        if repairs and not remaining:
            warnings.append("Mesh defects were repaired automatically: " + "; ".join(repairs) + ".")
        problems = remaining
    if fragments:
        warnings.append(f"{fragments} floating fragment{'s' if fragments != 1 else ''} removed.")
    job.report(1.0, "Done")
    return GenerationResult(
        vertices=vertices, faces=faces, mode=mode, voxel_mm=h, grid_shape=grid.shape,
        seconds=time.perf_counter() - started, fragments_removed=fragments,
        quality_capped=capped, valid=not problems, problems=problems, warnings=warnings,
    )


def _smoothness_reduced_warning(settings: Settings, triangles: int, wanted: int, used: int) -> str:
    names = {0: "Off", 1: "Smooth", 2: "Extra smooth"}
    limit = max_triangles()
    reason = ("the limit for this computer's memory" if limit < MAX_TRIANGLES
              else "the limit (larger files are hard for slicers)")
    text = (f"Smoothness was reduced from {names[wanted]} to {names[used]}: {names[wanted]} would "
            f"give {triangles * 4 ** wanted / 1e6:.1f} million triangles, more than {limit / 1e6:.0f} "
            f"million, {reason}. The result has {triangles * 4 ** used / 1e6:.1f} million triangles.")
    if settings.quality != "draft":
        text += (f" With Draft quality there are fewer triangles before smoothing, so "
                 f"{names[wanted]} may fit.")
    return text


def opening_inradius(pattern: SurfacePattern, settings: Settings) -> np.ndarray:
    """Radius of the circle that fits in each seed's opening (hole or gap between ribs)."""
    half_gap = 0.5 * pattern.neighbour
    if settings.surface_style == "perforated":
        return half_gap * settings.hole_size / 100.0
    return np.clip(half_gap - 0.5 * settings.rib_width, 0.0, None)


def field_cache_key(h: float, settings: Settings) -> tuple:
    """Key of the body distance field: it depends on the voxel size and the orientation."""
    return (h, tuple(settings.bed_normal))


def _resolution_warnings(resolution: Resolution, mode: Mode) -> list[str]:
    feature, h = resolution.feature_mm, resolution.voxel_mm
    if mode == "preview":
        if resolution.voxels_across_feature < PREVIEW_MIN_VOXELS_ACROSS - 1e-6:
            return [f"The preview ({h:g} mm voxels) is too coarse to show the thinnest feature "
                    f"({feature:.2f} mm) properly; it may look broken. Generate final to see it."]
        return []
    if resolution.limited_by is None:
        return []
    reason = ("the memory budget" if resolution.limited_by == "memory"
              else "the limit of this quality level")
    advice = ("" if resolution.limited_by == "memory"
              else " A higher quality allows more detail.")
    return [f"Detail was limited by {reason}: {h:g} mm voxels instead of {resolution.wanted_mm:.2g} mm, "
            f"about {resolution.voxels_across_feature:.1f} voxels across the thinnest feature "
            f"({feature:.2f} mm). Thin features may be rough or break.{advice}"]


def _skin_field(surface: str, settings: Settings, body: BodyField, depth, thickness_scale,
                grading: Grading, cell_range: tuple[float, float], job: Job):
    """Field of the skin (negative in material) for every surface style except None.

    Returns (skin field, surface pattern or None, band): the band holds the grid nodes
    where the pattern was evaluated. `depth` and `thickness_scale` are numbers, or
    arrays over the grid when the thickness is graded.
    """
    d, grid, h = body.sdf, body.grid, body.grid.h
    shell = np.maximum(d, -(d + depth))  # between the surface and its inward offset
    if surface == "solid":
        return shell, None, None

    cell = settings.surface_cell_size
    rounding = settings.edge_rounding
    rng = np.random.default_rng([settings.random_seed, SURFACE_STREAM])
    seeds = surface_seeds(body.surface.tri, cell, rng,
                          scale=grading.cell_scale if grading.cells else None,
                          scale_range=cell_range)
    if len(seeds) < 4:
        raise GenerationError(
            "The surface cell size is too large for this part: fewer than 4 cells fit on it. "
            "Use a smaller surface cell size.")
    pattern = SurfacePattern(seeds)
    job.report(0.3, "Building the skin")

    # The pattern is only needed in the skin band (plus a margin for interpolation and
    # for the edge rounding). Outside it the skin is just the shell.
    band = np.nonzero((d > -(depth + 2 * h + rounding)) & (d < 2 * h))
    points = grid.points(band)
    # Cell corners are rounded in proportion to the local cell size.
    local_cell = cell * grading.cell_scale(points, depth=np.clip(-d[band], 0.0, None))
    at = lambda value: value[band] if np.ndim(value) == 3 else value  # noqa: E731
    shell_band = shell[band]
    # Holes only (A): the openings of a rib lattice fill nearly the whole cell, so a pointed
    # top there would fill each cell with a solid gable and hide the ribs.
    pointed = settings.pointed_holes and surface == "perforated"
    if pointed:
        prepare_pointed_tops(pattern, body.surface.closest(pattern.seeds, k=6)[2])
    if surface == "perforated":
        # Holes: each cell scaled about its seed to the hole size, corners rounded.
        fraction = settings.hole_size / 100.0
        wall = pattern.wall_distance(points, HOLE_ROUNDING * fraction * local_cell,
                                     inset=1.0 - fraction)
        if pointed:  # no flat ceilings on steep walls: print without support
            wall = pointed_tops(pattern, points, wall, opening_inradius(pattern, settings),
                                settings.overhang_angle)
    else:
        half_width = 0.5 * settings.rib_width * at(np.broadcast_to(thickness_scale, grid.shape)
                                                   if np.ndim(thickness_scale) else thickness_scale)
        if surface == "rib_lattice":
            # Ribs: within half a rib width of a cell wall; the blend radius fillets rib joints.
            wall = pattern.wall_distance(points, settings.blend_radius) - half_width
        else:
            # Embossed: closed shell; cell interiors are recessed from the outside, and
            # the corner between recess floor and rib side is filleted.
            wall = pattern.wall_distance(points, RECESS_ROUNDING * local_cell) - half_width
            wall = smooth_min(wall, d[band] + np.float32(settings.emboss_depth), rounding)
    # Shell minus holes / cells: the edges where they meet are rounded.
    shell[band] = smooth_max(shell_band, wall, rounding)
    return shell, pattern, band
