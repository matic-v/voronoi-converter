"""Edges of the input model the user can pick: sharp edges and curved (filleted) edges.

Sharp: the two triangles meeting at a mesh edge differ in direction by more than
SHARP_ANGLE_DEG. Sharp mesh edges are joined into one "model edge" through every
vertex where exactly two of them meet, so a model edge runs from corner to corner
(a cube has 12) or round a loop (a round hole rim is one).

Curved: a fillet or round has no sharp crease; it is a strip of small facets that
turns the surface smoothly. The surface is split into flat pieces (coplanar
triangles); where two neighbouring pieces meet at a slight angle, the local radius
is about (their width) / (angle). Pieces joined by a radius of at most
CURVED_RADIUS_MAX_MM form a strip, and every strip that turns the surface by at
least CURVED_TURN_MIN_DEG is one curved model edge. This finds fillets and rounds,
and the walls of small holes; large curved surfaces (a 60 mm sphere) are left out.

The numbering depends only on the mesh, so the same STL always gives the same ids,
which lets a saved settings file name the edges to keep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SHARP_ANGLE_DEG = 30.0
FLAT_ANGLE_DEG = 0.5  # triangles closer than this in direction are one flat piece
CURVED_RADIUS_MAX_MM = 8.0
CURVED_TURN_MIN_DEG = 30.0
MAX_SEGMENTS = 300_000  # more than this is noise, not edges anyone will click


@dataclass(frozen=True)
class ModelEdges:
    segments: np.ndarray  # (S, 2, 3) float64: every sharp mesh edge
    edge_of_segment: np.ndarray  # (S,) int64: the model edge each segment belongs to
    count: int  # number of model edges
    too_many: bool = False  # True when the mesh had more sharp edges than MAX_SEGMENTS
    curved_from: int | None = None  # ids from here on are curved edges (fillets, rounds)

    @property
    def sharp_count(self) -> int:
        return self.count if self.curved_from is None else self.curved_from

    @property
    def curved_count(self) -> int:
        return self.count - self.sharp_count

    def segments_of(self, ids) -> np.ndarray:
        ids = np.asarray(sorted(set(int(i) for i in ids)), dtype=np.int64)
        return self.segments[np.isin(self.edge_of_segment, ids)]

    def lengths(self) -> np.ndarray:
        length = np.linalg.norm(self.segments[:, 1] - self.segments[:, 0], axis=1)
        return np.bincount(self.edge_of_segment, weights=length, minlength=self.count)


def find_edges(vertices: np.ndarray, faces: np.ndarray,
               angle_deg: float = SHARP_ANGLE_DEG) -> ModelEdges:
    """Sharp and curved edges of a closed, manifold mesh. Sharp edges get the first
    numbers (so adding curved edges did not renumber them), curved edges follow."""
    mesh = _MeshEdges(vertices, faces)
    sharp_segments, sharp_ids, sharp_count = _sharp_edges(mesh, angle_deg)
    curved_segments, curved_ids, curved_count = _curved_edges(mesh, angle_deg)
    total = len(sharp_segments) + len(curved_segments)
    if total > MAX_SEGMENTS:
        return ModelEdges(np.empty((0, 2, 3)), np.empty(0, dtype=np.int64), 0, too_many=True)
    if total == 0:
        return ModelEdges(np.empty((0, 2, 3)), np.empty(0, dtype=np.int64), 0)
    return ModelEdges(
        np.concatenate([sharp_segments, curved_segments]),
        np.concatenate([sharp_ids, curved_ids + sharp_count]),
        sharp_count + curved_count,
        curved_from=sharp_count,
    )


class _MeshEdges:
    """Every mesh edge once, with its two triangles and the angle between them."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        tri = vertices[faces]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-300)[:, None]
        n = np.int64(len(vertices))
        a, b = faces.ravel(), faces[:, [1, 2, 0]].ravel()
        key = np.minimum(a, b) * n + np.maximum(a, b)
        order = np.argsort(key, kind="stable")
        key_sorted = key[order]
        # In a closed manifold mesh every edge key appears exactly twice, side by side.
        if not np.array_equal(key_sorted[0::2], key_sorted[1::2]):
            raise ValueError("mesh is not closed and manifold")
        self.vertices, self.faces, self.normal = vertices, faces, normal
        self.key = key_sorted[0::2]
        self.v0, self.v1 = self.key // n, self.key % n
        self.f0, self.f1 = order[0::2] // 3, order[1::2] // 3
        cos = np.clip(np.einsum("ij,ij->i", normal[self.f0], normal[self.f1]), -1.0, 1.0)
        self.angle = np.arccos(cos)  # radians, 0 = flat

    def segments(self, mask: np.ndarray) -> np.ndarray:
        return np.stack([self.vertices[self.v0[mask]], self.vertices[self.v1[mask]]], axis=1)


def _sharp_edges(mesh: _MeshEdges, angle_deg: float):
    sharp = mesh.angle > np.radians(angle_deg)
    ea, eb = mesh.v0[sharp], mesh.v1[sharp]
    if len(ea) == 0:
        return np.empty((0, 2, 3)), np.empty(0, dtype=np.int64), 0
    # Join two sharp segments through every vertex where exactly two of them meet.
    ends = np.concatenate([ea, eb])
    segment = np.concatenate([np.arange(len(ea)), np.arange(len(ea))])
    degree = np.bincount(ends, minlength=len(mesh.vertices))
    through = degree[ends] == 2
    by_vertex = np.argsort(ends[through], kind="stable")
    pair = segment[through][by_vertex].reshape(-1, 2)
    count, label = connected_components(
        coo_matrix((np.ones(len(pair), dtype=bool), (pair[:, 0], pair[:, 1])),
                   shape=(len(ea), len(ea))), directed=False)
    return mesh.segments(sharp), _number_in_order(label, count), int(count)


def _curved_edges(mesh: _MeshEdges, angle_deg: float):
    """Fillets and rounds: strips of narrow flat pieces that turn the surface tightly."""
    nothing = np.empty((0, 2, 3)), np.empty(0, dtype=np.int64), 0
    face_count = len(mesh.faces)
    flat = mesh.angle < np.radians(FLAT_ANGLE_DEG)
    pieces, piece = connected_components(
        coo_matrix((np.ones(int(flat.sum()), dtype=bool), (mesh.f0[flat], mesh.f1[flat])),
                   shape=(face_count, face_count)), directed=False)
    width = _piece_widths(mesh, piece, pieces)

    bend = (~flat) & (mesh.angle <= np.radians(angle_deg))
    radius = 0.5 * (width[piece[mesh.f0]] + width[piece[mesh.f1]]) / np.maximum(mesh.angle, 1e-12)
    curved = bend & (radius <= CURVED_RADIUS_MAX_MM)
    if not curved.any():
        return nothing
    p0, p1 = piece[mesh.f0[curved]], piece[mesh.f1[curved]]
    count, strip = connected_components(
        coo_matrix((np.ones(len(p0), dtype=bool), (p0, p1)), shape=(pieces, pieces)), directed=False)
    in_strip = np.zeros(pieces, dtype=bool)
    in_strip[p0] = in_strip[p1] = True

    # How far each strip turns the surface: the widest angle between its pieces' normals.
    piece_normal = np.zeros((pieces, 3))
    np.add.at(piece_normal, piece, mesh.normal)
    piece_normal /= np.maximum(np.linalg.norm(piece_normal, axis=1), 1e-300)[:, None]
    turn = np.zeros(count)
    members = np.nonzero(in_strip)[0]
    label = strip[members]
    first = np.full(count, -1)
    first[label[::-1]] = members[::-1]  # one piece of every strip
    for _ in range(2):  # from a piece to the one furthest from it, and back: about the widest angle
        cos = np.einsum("ij,ij->i", piece_normal[members], piece_normal[first[label]])
        far = np.full(count, 2.0)
        np.minimum.at(far, label, cos)
        turn = np.maximum(turn, np.degrees(np.arccos(np.clip(far, -1.0, 1.0))))
        furthest = np.full(count, -1)
        is_far = cos <= far[label] + 1e-12
        furthest[label[is_far]] = members[is_far]
        first = furthest
    keep_strip = turn >= CURVED_TURN_MIN_DEG
    edge_strip = strip[p0]
    chosen = keep_strip[edge_strip]
    if not chosen.any():
        return nothing
    number = np.full(count, -1)
    kept = np.nonzero(keep_strip)[0]
    # Number the strips in order of their first mesh edge (stable for a given mesh).
    first_edge = np.full(count, len(edge_strip))
    np.minimum.at(first_edge, edge_strip, np.arange(len(edge_strip)))
    number[kept[np.argsort(first_edge[kept], kind="stable")]] = np.arange(len(kept))
    curved_index = np.nonzero(curved)[0][chosen]
    segments = mesh.segments(np.isin(np.arange(len(mesh.key)), curved_index))
    return segments, number[edge_strip[chosen]], int(len(kept))


def _piece_widths(mesh: _MeshEdges, piece: np.ndarray, pieces: int) -> np.ndarray:
    """Narrow extent of every flat piece: for a long thin facet strip, its width."""
    corners = mesh.vertices[mesh.faces].reshape(-1, 3)
    owner = np.repeat(piece, 3)
    count = np.bincount(owner, minlength=pieces).astype(np.float64)
    mean = np.stack([np.bincount(owner, corners[:, k], pieces) for k in range(3)], axis=1) / count[:, None]
    centred = corners - mean[owner]
    cov = np.zeros((pieces, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            cov[:, i, j] = cov[:, j, i] = np.bincount(owner, centred[:, i] * centred[:, j], pieces) / count
    middle = np.linalg.eigvalsh(cov)[:, 1]  # in-plane short direction
    return 2.0 * np.sqrt(np.clip(middle, 0.0, None))


def _number_in_order(label: np.ndarray, count: int) -> np.ndarray:
    """Renumber components in order of their first member (stable for a given mesh)."""
    first = np.full(count, len(label))
    np.minimum.at(first, label, np.arange(len(label)))
    rank = np.empty(count, dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(count)
    return rank[label]
