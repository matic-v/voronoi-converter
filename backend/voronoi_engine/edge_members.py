"""Kept edges become members of the lattice: a rib or strut that runs along the edge.

A kept edge is drawn by a member as wide as `width` and as deep as `depth`, lying
on the surface on both sides of the edge, like a cell wall of the surface pattern
that runs exactly along the edge: on each face a band half the width wide and
`depth` deep. The cells of the pattern end on it, struts join it, and nothing deeper
inside the part is filled.

For a curved (filleted) edge the "edge" is the band of lines along the fillet, so
the whole fillet plus width / 2 on each side becomes the member.

The same function is used on the voxel grid and by the exact field for smoothing,
so the two always describe the same shape.
"""

from __future__ import annotations

import numpy as np

from .strut_field import smooth_max, smooth_min


def member_field(d: np.ndarray, to_edge: np.ndarray, width: float, depth: float,
                 rounding: float) -> np.ndarray:
    """Field of the edge members (negative inside).

    `d` is the signed distance to the input surface (negative inside) and `to_edge` the
    distance to the kept edge. Next to a straight edge, a point `below` the nearest face
    lies `along` = sqrt(to_edge^2 - below^2) from the edge, measured in that face. The
    member is a band on each face of the edge, half the width wide and `depth` deep;
    seen from the nearest face, the band on the other face is the part at most half the
    width deep and at most `depth` along. Rib edges are rounded with `rounding`.
    """
    below = np.clip(-d, 0.0, None)
    along = np.sqrt(np.clip(to_edge**2 - below**2, 0.0, None))
    half = 0.5 * width
    on_this_face = smooth_max(below - depth, along - half, rounding)
    on_other_face = smooth_max(below - half, along - depth, rounding)
    return np.minimum(on_this_face, on_other_face)


def member_lateral(d: np.ndarray, to_edge: np.ndarray, width: float, depth,
                   rounding: float) -> np.ndarray:
    """The member as part of a patterned skin (A, B, C): only its sideways extent, as a
    wall of the pattern. The skin's own inner face limits it on the edge's nearest face,
    so the member and the skin share one inner face instead of two coincident ones
    (which the voxel grid turns into shards). The band on the other face is still limited
    by `depth`, measured along this face, as in member_field."""
    below = np.clip(-d, 0.0, None)
    along = np.sqrt(np.clip(to_edge**2 - below**2, 0.0, None))
    half = 0.5 * width
    on_this_face = along - half
    on_other_face = smooth_max(below - half, along - depth, rounding)
    # Where the bands on the two faces meet, the corner is rounded too: a sharp inside
    # corner along the whole edge is a crease the voxel grid turns into a saw-tooth row.
    return smooth_min(on_this_face, on_other_face, rounding)


MIN_ROUNDING_VOXELS = 2.0  # a rounding radius the voxel grid can draw without teeth


def edge_rounding_range(settings, h: float) -> tuple[float, float]:
    """(rounding used everywhere, rounding next to kept edges). With a radius under about
    two voxels the grid draws rounded edges (rib edges, hole rims, the grooves beside kept
    edges) as a saw-tooth row of shards, so the radius used is at least two voxels. A
    rounding of 0 (sharp) is left as set. Both values are the same since the minimum
    applies everywhere; the pair is kept for the callers that size margins."""
    rounding = settings.edge_rounding
    if rounding <= 0:
        return rounding, rounding
    used = max(rounding, MIN_ROUNDING_VOXELS * h)
    return used, used


def blend_used(settings, h: float) -> float:
    """Blend radius (strut joints, struts into the skin, rib joints): at least two voxels
    for the same reason as the rounding (edge_rounding_range). 0 stays 0."""
    blend = settings.blend_radius
    return max(blend, MIN_ROUNDING_VOXELS * h) if blend > 0 else blend


def rounding_near_edges(to_edge: np.ndarray, settings, h: float):
    """Rounding radius next to kept edges: the same as everywhere (edge_rounding_range)."""
    return edge_rounding_range(settings, h)[1]


def member_in_skin(surface_style: str) -> bool:
    """Styles whose skin draws the kept edges itself (see member_lateral)."""
    return surface_style in ("perforated", "rib_lattice", "embossed")


def member_depth(settings) -> float:
    """Patterned and closed skins: as deep as the skin (shell, or ribs for B). Bare struts
    (style None): as deep as wide."""
    if settings.surface_style in ("perforated", "embossed", "solid"):
        return settings.shell_thickness
    if settings.surface_style == "rib_lattice":
        return settings.rib_width
    return settings.keep_width


def member_reach(settings, h: float, rounding: float) -> float:
    """How far from a kept edge a point can be and still be affected (incl. blending)."""
    return 0.5 * settings.keep_width + member_depth(settings) + 2 * h + rounding + blend_used(settings, h)
