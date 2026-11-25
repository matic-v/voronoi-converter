"""Self-supporting struts: a lattice that prints on an FDM printer without supports.

Up is +Z (the part is already in its print orientation). With an overhang limit of
alpha (from vertical; 45 degrees by default):

1. Struts flatter than the limit are bridges. Bridges up to `max_bridge` long are
   kept (they print between their two ends); longer ones are removed.
2. Every node of the lattice must be held from below, or the printer would start it
   in mid-air (inside a closed part a support there could never be removed). A node
   is held when it lies on the bed, rests on the skin where the skin is a wall or a
   floor (not a ceiling), or has a strut down to a held node that is no flatter than
   the limit. Nodes are handled from the bottom up; a node that is not held gets a
   strut down to the nearest held node within reach (within the limit), or else a
   strut down to skin material (vertical, or leaning up to the limit) or the bed.
   With a skin, a node that still cannot be held (just over a gap in the floor) is
   removed with its struts; without one it is reported.

The cells themselves are stretched along Z beforehand (see lattice.interior_edges),
which turns most struts steep and leaves little to fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .body_field import BodyField

ON_BED_MM = 0.5


@dataclass
class SupportReport:
    removed_bridges: int = 0  # long flat struts, and (style None) struts leaving the part downwards
    added_struts: int = 0
    added_drops: int = 0
    removed_nodes: int = 0  # with a skin: nodes over a gap, too low to reach skin material
    unsupported: int = 0  # nodes that could not be held (style None, nothing below)
    notes: list[str] = field(default_factory=list)


def make_self_supporting(edges: np.ndarray, body: BodyField, *, overhang_deg: float,
                         max_bridge: float, reach: float, strut_radius: float,
                         skin_depth: float | None, bed_z: float, skin_value=None,
                         passes: int = 8) -> tuple[np.ndarray, SupportReport]:
    """Returns (edges, report). `skin_depth` is None when there is no skin (style None);
    `skin_value(points)` is the skin's field (negative in skin material), so that a node
    over a hole or a gap between ribs is not taken as resting on the skin.

    Repeated (at most `passes` times) until a pass changes nothing: a strut added in
    one pass can create the node another node needs in the next."""
    total = SupportReport()
    for _ in range(passes):
        edges, report = _one_pass(edges, body, overhang_deg=overhang_deg, max_bridge=max_bridge,
                                  reach=reach, strut_radius=strut_radius,
                                  skin_depth=skin_depth, bed_z=bed_z, skin_value=skin_value)
        total.removed_bridges += report.removed_bridges
        total.added_struts += report.added_struts
        total.added_drops += report.added_drops
        total.removed_nodes += report.removed_nodes
        total.unsupported = report.unsupported
        if not (report.removed_bridges or report.added_struts or report.added_drops
                or report.removed_nodes):
            break
    return edges, total


def _one_pass(edges: np.ndarray, body: BodyField, *, overhang_deg: float, max_bridge: float,
              reach: float, strut_radius: float, skin_depth: float | None,
              bed_z: float, skin_value=None) -> tuple[np.ndarray, SupportReport]:
    report = SupportReport()
    if len(edges) == 0:
        return edges, report
    steep = np.sin(np.radians(90.0 - overhang_deg))  # sine of the flattest allowed elevation

    # Nodes: strut ends that coincide are one node.
    ends = edges.reshape(-1, 3)
    nodes, index = np.unique(np.round(ends, 5), axis=0, return_inverse=True)
    pair = index.reshape(-1, 2)
    keep = pair[:, 0] != pair[:, 1]
    pair = pair[keep]
    vector = nodes[pair[:, 1]] - nodes[pair[:, 0]]
    length = np.linalg.norm(vector, axis=1)
    elevation = np.abs(vector[:, 2]) / np.maximum(length, 1e-12)

    # 1. Long bridges go.
    long_bridge = (elevation < steep) & (length > max_bridge)
    report.removed_bridges = int(long_bridge.sum())
    pair, length, elevation = pair[~long_bridge], length[~long_bridge], elevation[~long_bridge]

    # Which nodes are part of the printed body, and which rest on the bed or the skin.
    depth = -body.sample(nodes)
    inside = depth > -strut_radius
    if skin_depth is None:
        # No skin: a strut that leaves the part downwards is cut at the part's surface,
        # and that cut is its lowest point, hanging in mid-air. Remove it.
        z0, z1 = nodes[pair[:, 0], 2], nodes[pair[:, 1], 2]
        low = np.where(z0 <= z1, pair[:, 0], pair[:, 1])
        leaves_down = (depth[low] < 0.0) & (nodes[low, 2] > bed_z + ON_BED_MM)
        report.removed_bridges += int(leaves_down.sum())
        pair, length, elevation = pair[~leaves_down], length[~leaves_down], elevation[~leaves_down]
    on_bed = nodes[:, 2] <= bed_z + ON_BED_MM
    on_skin = np.zeros(len(nodes), dtype=bool)
    if skin_depth is not None:
        near = inside & (depth <= skin_depth + strut_radius)
        if near.any():
            _, _, normal = body.surface.closest(nodes[near], k=6)
            # The skin's inner face there points the other way from the outer surface;
            # it carries the node unless it is a ceiling (outer surface facing up).
            rests = normal[:, 2] <= np.cos(np.radians(overhang_deg))
            if skin_value is not None:  # only where there is skin material (not a hole)
                rests &= skin_value(nodes[near]) <= strut_radius
                # Inside solid skin that reaches straight down to the bed (a thin flange,
                # a solid floor) a node is held even if the nearest surface is above it.
                for i in np.flatnonzero(~rests):
                    rests[i] = _material_to_bed(nodes[near][i], skin_value, body.grid.h, bed_z)
            on_skin[np.flatnonzero(near)] = rests

    # Steep struts down from each node: neighbours below within the limit.
    lower = np.where(nodes[pair[:, 0], 2] <= nodes[pair[:, 1], 2], pair[:, 0], pair[:, 1])
    upper = np.where(lower == pair[:, 0], pair[:, 1], pair[:, 0])
    steep_edge = elevation >= steep
    down = {}
    for u, l in zip(upper[steep_edge].tolist(), lower[steep_edge].tolist()):
        down.setdefault(u, []).append(l)

    supported = np.zeros(len(nodes), dtype=bool)
    removed = np.zeros(len(nodes), dtype=bool)
    order = np.argsort(nodes[:, 2], kind="stable")
    tree = cKDTree(nodes)
    added = []
    for v in order.tolist():
        if not inside[v]:
            continue
        if on_bed[v] or on_skin[v] or any(supported[l] and not removed[l] for l in down.get(v, ())):
            supported[v] = True
            continue
        # 2a. A strut down to the nearest held node within reach and within the limit.
        p = nodes[v]
        best, best_distance = -1, np.inf
        for u in tree.query_ball_point(p, reach):
            if not supported[u] or nodes[u, 2] >= p[2]:
                continue
            if skin_depth is None and depth[u] < 0.0 and not on_bed[u]:
                continue  # without a skin, only nodes inside the part can carry a strut
            d = nodes[u] - p
            distance = float(np.linalg.norm(d))
            if distance > 1e-9 and -d[2] / distance >= steep and distance < best_distance:
                best, best_distance = u, distance
        if best >= 0:
            added.append((p, nodes[best]))
            report.added_struts += 1
            supported[v] = True
            continue
        # 2b. Straight down to the skin or the bed.
        end = _drop(p, body, skin_depth, strut_radius, bed_z, skin_value)
        if end is not None:
            added.append((p, end))
            report.added_drops += 1
            supported[v] = True
        elif skin_depth is not None:
            # 2c. With a skin this is a node just over a hole or a gap in the floor, too
            # low to reach skin material at the allowed angle: remove it and its struts
            # (the struts above it are handled in this same pass, higher up).
            removed[v] = True
            report.removed_nodes += 1
        else:
            report.unsupported += 1

    keep = ~(removed[pair[:, 0]] | removed[pair[:, 1]])
    pair = pair[keep]
    result = np.stack([nodes[pair[:, 0]], nodes[pair[:, 1]]], axis=1)
    if added:
        result = np.concatenate([result, np.array(added)])
    return result, report


def _drop(p: np.ndarray, body: BodyField, skin_depth: float | None, radius: float,
          bed_z: float, skin_value=None) -> np.ndarray | None:
    """End point of a vertical strut from p down to skin material or the bed, if any.
    The column must stay inside the part until it gets there."""
    h = body.grid.h
    steps = np.arange(h * 0.5, p[2] - bed_z + h, h * 0.5)
    if len(steps) == 0:
        return None
    column = np.repeat(p[None, :], len(steps), axis=0)
    column[:, 2] -= steps
    column[:, 2] = np.maximum(column[:, 2], bed_z)
    depth = -body.sample(column)
    near_bed = column[:, 2] <= bed_z + max(radius, h)
    leaves = np.flatnonzero((depth < 0.0) & ~near_bed)
    stop = leaves[0] if len(leaves) else len(column)  # the part ends there
    if skin_depth is not None:
        if skin_value is not None:
            solid = np.flatnonzero(skin_value(column[:stop]) < -0.25 * h)
        else:
            solid = np.flatnonzero(depth[:stop] < 0.5 * skin_depth)
        if len(solid):
            return column[solid[0]]
        if skin_value is not None:
            # The column only meets a hole or a gap between ribs: lean over to the
            # nearest skin material in the floor below, no flatter than 45 degrees.
            floor = np.flatnonzero(depth[:stop] < 0.5 * skin_depth)
            if len(floor):
                target = _lean_to_skin(p, column[floor[0]], skin_value, h)
                if target is not None:
                    return target
            return None
    # Otherwise only the bed can carry it.
    if stop == len(column) and column[-1, 2] <= bed_z + ON_BED_MM:
        return column[-1]
    return None


def _lean_to_skin(p: np.ndarray, below: np.ndarray, skin_value, h: float) -> np.ndarray | None:
    """Nearest point with skin material at the height of `below`, within 45 degrees of
    straight down from p (horizontal offset at most the drop height)."""
    reach = p[2] - below[2]
    if reach <= 0:
        return None
    offsets = np.arange(-reach, reach + 1e-9, 0.5 * h)
    gx, gy = np.meshgrid(offsets, offsets, indexing="ij")
    ring = np.hypot(gx, gy) <= reach
    points = np.column_stack([below[0] + gx[ring], below[1] + gy[ring], np.full(ring.sum(), below[2])])
    solid = skin_value(points) < -0.25 * h
    if not solid.any():
        return None
    candidates = points[solid]
    return candidates[np.argmin(np.hypot(candidates[:, 0] - p[0], candidates[:, 1] - p[1]))]


def _material_to_bed(p: np.ndarray, skin_value, h: float, bed_z: float) -> bool:
    """True if there is skin material all the way straight down from p to the bed."""
    if p[2] <= bed_z + ON_BED_MM:
        return True
    steps = np.arange(0.0, p[2] - bed_z + 1e-9, 0.5 * h)
    column = np.repeat(p[None, :], len(steps), axis=0)
    column[:, 2] -= steps
    return bool((skin_value(column) < 0.0).all())
