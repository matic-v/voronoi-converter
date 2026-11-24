"""Make interior struts end on skin material when the skin is open (styles A and B).

Surface and interior seeds are independent, so a strut heading for the skin would
often arrive at a hole and end as a loose stub. Here every strut that reaches the
skin is re-aimed: it keeps its inner end and its outer end is moved to the nearest
cell wall of the surface pattern - the middle of a wall between two holes (A) or of
a rib (B) - half-way into the skin.
"""

from __future__ import annotations

import numpy as np

from .body_field import BodyField
from .surface_pattern import SurfacePattern
from .parallel import kd_workers

ANCHOR_DEPTH = 0.6  # of the skin depth, measured from the surface
BISECTION_STEPS = 10
WALL_STEPS = 4


def nearest_wall_points(points: np.ndarray, body: BodyField, pattern: SurfacePattern):
    """For points near the surface: the nearest point of a cell wall on the surface,
    and the outward surface normal there."""
    _, on_surface, normal = body.surface.closest(points, k=8)
    for _ in range(WALL_STEPS):
        distance, index = pattern.tree.query(on_surface, k=2, workers=kd_workers())
        first, second = pattern.seeds[index[:, 0]], pattern.seeds[index[:, 1]]
        across = second - first
        gap = np.linalg.norm(across, axis=1)
        to_wall = (distance[:, 1] ** 2 - distance[:, 0] ** 2) / (2.0 * gap)
        # Step onto the plane half-way between the two nearest seeds, then back onto the surface.
        moved = on_surface + across * (to_wall / gap)[:, None]
        _, on_surface, normal = body.surface.closest(moved, k=8)
    return on_surface, normal


def attach_edges_to_skin(edges: np.ndarray, body: BodyField, pattern: SurfacePattern,
                         skin_depth) -> tuple[np.ndarray, int]:
    """Return (edges, number re-aimed).

    `skin_depth(points)` gives the skin depth at points. Struts wholly below the skin
    are kept; struts that cross into the skin are re-aimed at a wall; pieces that lie
    entirely in or beyond the skin are dropped (they have no inner end to hang on).
    """
    if len(edges) == 0:
        return edges, 0
    a, b = edges[:, 0], edges[:, 1]
    below_a = body.sample(a) < -skin_depth(a)
    below_b = body.sample(b) < -skin_depth(b)
    inner_only = below_a & below_b
    crossing = below_a ^ below_b
    inner = np.where(below_a[:, None], a, b)[crossing]
    outer = np.where(below_a[:, None], b, a)[crossing]
    if len(inner) == 0:
        return edges[inner_only], 0

    # Where the strut passes the inner face of the skin (bisection along the strut).
    low, high = np.zeros(len(inner)), np.ones(len(inner))
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (low + high)
        point = inner + (outer - inner) * middle[:, None]
        below = body.sample(point) < -skin_depth(point)
        low, high = np.where(below, middle, low), np.where(below, high, middle)
    entry = inner + (outer - inner) * low[:, None]

    wall, normal = nearest_wall_points(entry, body, pattern)
    anchor = wall - normal * (ANCHOR_DEPTH * skin_depth(wall))[:, None]
    # Going inwards along the normal can leave the wall plane near edges of the part:
    # put the anchor itself back on the plane between its two nearest seeds.
    distance, index = pattern.tree.query(anchor, k=2, workers=kd_workers())
    across = pattern.seeds[index[:, 1]] - pattern.seeds[index[:, 0]]
    gap = np.linalg.norm(across, axis=1)
    to_wall = (distance[:, 1] ** 2 - distance[:, 0] ** 2) / (2.0 * gap)
    anchor = anchor + across * (to_wall / gap)[:, None]
    attached = np.stack([inner, anchor], axis=1)
    return np.concatenate([edges[inner_only], attached]), len(attached)
