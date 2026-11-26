"""Sharp edges of the part kept sharp in the result.

Marching cubes cannot put a vertex on a sharp edge of the part: next to the edge the
vertices lie on one face or the other, and the triangles between them cut the edge off
or poke out, a saw-tooth row along the edge. The same happens at the lines between the
flat facets of a fillet in the STL (each a crease of a few degrees), which gives rows of
teeth along every filleted edge. Smoothing moves vertices onto the exact
shape, which is sharp there, and makes the teeth more visible. Wherever material
reaches a sharp edge of the part (a rib crossing it, an edge strut along it) this shows.

Fix, as in feature-preserving meshing: every mesh edge whose two ends lie on different
faces of the part crosses a sharp edge; the end nearer to that edge is moved onto it.
The vertices on the edge then form a chain along it, and the triangles on either side
lie on their own face. A move that would fold a triangle over is undone.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .smoothing import face_normals, undo_folds

CREASE_DEG = 2.0  # input mesh edges bent more than this are creases (fillet facets too)


def crease_segments(vertices: np.ndarray, faces: np.ndarray,
                    min_angle_deg: float = CREASE_DEG) -> np.ndarray:
    """Every edge of the input mesh where its two triangles meet at more than
    `min_angle_deg`: sharp edges, and the lines between the flat facets of a fillet."""
    from .edges import _MeshEdges

    edges = _MeshEdges(np.asarray(vertices, np.float64), np.asarray(faces, np.int64))
    return edges.segments(edges.angle > np.radians(min_angle_deg))


def _closest_on_segments(points: np.ndarray, segments: np.ndarray, index: np.ndarray):
    a, b = segments[index, 0], segments[index, 1]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", points - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300), 0, 1)
    closest = a + t[:, None] * ab
    return closest, np.linalg.norm(points - closest, axis=1)


def snap_to_sharp_edges(vertices: np.ndarray, faces: np.ndarray, body, segments: np.ndarray,
                        h: float) -> tuple[np.ndarray, int]:
    """Move vertices next to the creases of the part onto them. `segments` (S, 2, 3) are
    the sharp edges of the input mesh. Returns (vertices, number of vertices moved)."""
    if len(segments) == 0:
        return vertices, 0
    # Vertices on the part's surface, with the normal of the face they lie on.
    near = np.nonzero(body.sample(vertices) > -h)[0]
    if len(near) == 0:
        return vertices, 0
    distance, _, normal = body.surface.closest(vertices[near], k=8)
    on = np.abs(distance) < 0.02 * h
    on_index = near[on]
    if len(on_index) == 0:
        return vertices, 0
    vertex_normal = np.zeros((len(vertices), 3))
    vertex_normal[on_index] = normal[on]
    on_surface = np.zeros(len(vertices), dtype=bool)
    on_surface[on_index] = True

    # Mesh edges between two vertices on the surface, on faces at a sharp angle.
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    a, b = edges[:, 0], edges[:, 1]
    crossing = on_surface[a] & on_surface[b]
    crossing[crossing] = np.einsum("ij,ij->i", vertex_normal[a[crossing]],
                                   vertex_normal[b[crossing]]) < np.cos(np.radians(0.5 * CREASE_DEG))
    a, b = a[crossing], b[crossing]
    if len(a) == 0:
        return vertices, 0

    # The sharp edge each crossing is next to: segment samples in a KD tree.
    spacing = 0.25 * h
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    count = np.maximum(np.ceil(lengths / spacing).astype(np.int64), 1) + 1
    owner = np.repeat(np.arange(len(segments)), count)
    t = np.concatenate([np.linspace(0, 1, c) for c in count])
    samples = segments[owner, 0] + t[:, None] * (segments[owner, 1] - segments[owner, 0])
    tree = cKDTree(samples)
    middle = 0.5 * (vertices[a] + vertices[b])
    gap, nearest = tree.query(middle, distance_upper_bound=2 * h)
    found = np.isfinite(gap)
    a, b, nearest = a[found], b[found], nearest[found]
    if len(a) == 0:
        return vertices, 0
    segment = owner[nearest]
    on_a, dist_a = _closest_on_segments(vertices[a], segments, segment)
    on_b, dist_b = _closest_on_segments(vertices[b], segments, segment)
    nearer_a = dist_a <= dist_b
    mover = np.where(nearer_a, a, b)
    target = np.where(nearer_a[:, None], on_a, on_b)
    reach = np.where(nearer_a, dist_a, dist_b)
    keep = reach < h  # never pull a vertex from further than a voxel
    mover, target, reach = mover[keep], target[keep], reach[keep]
    # A vertex in several crossings goes to its nearest target.
    order = np.lexsort((reach, mover))
    mover, target = mover[order], target[order]
    first = np.r_[True, mover[1:] != mover[:-1]]
    mover, target = mover[first], target[first]

    result = vertices.copy()
    result[mover] = target
    result = undo_folds(vertices, result, faces, reference=face_normals(vertices, faces))
    moved = int(np.any(result != vertices, axis=1).sum())
    return result, moved
