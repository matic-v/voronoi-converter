"""Field -> mesh: fragment removal, marching cubes, envelope clamp."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import measure

from .body_field import BodyField
from .grid import Grid

# The mesh may stick out of the input surface by at most this fraction of a voxel.
ENVELOPE_TOLERANCE_VOXELS = 0.02


def keep_largest_body(field: np.ndarray, h: float) -> int:
    """Remove all solid regions except the largest (in place). Returns how many were removed."""
    # A node exactly on the surface makes zero-area triangles: nudge it inside. This is
    # done before looking for fragments so that a nudged node cannot become one.
    eps = np.float32(1e-4 * h)
    field[np.abs(field) < eps] = -eps
    labels, count = ndimage.label(field < 0)
    if count <= 1:
        return 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    field[(labels > 0) & (labels != int(np.argmax(sizes)))] = h
    return count - 1


def extract_mesh(field: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Marching cubes at the zero level (call keep_largest_body first). Returns (vertices float64, faces int64),
    wound so that normals point out of the solid."""
    h = grid.h
    vertices, faces, _, _ = measure.marching_cubes(field, level=0.0, spacing=(h, h, h))
    vertices = vertices.astype(np.float64) + np.asarray(grid.origin)
    faces = faces.astype(np.int64)
    if signed_volume(vertices, faces) < 0:
        faces = faces[:, ::-1].copy()
    return vertices, faces


MIN_SPLIT_MM = 1e-4  # a split point closer than this to an end of the edge is not added


def fit_to_envelope(vertices: np.ndarray, faces: np.ndarray, body: BodyField,
                    passes: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Make sure the mesh does not stick out of the input surface.

    Marching cubes interpolates linearly, so near concave edges of the part it puts
    vertices a fraction of a voxel outside and bridges the corner with flat
    triangles. Vertices outside are moved onto the surface; edges whose middle is
    still outside are split, and the new vertex is placed on the crease.
    Returns the new (vertices, faces).
    """
    h = body.grid.h
    tolerance = ENVELOPE_TOLERANCE_VOXELS * h
    surface = body.surface

    near = body.sample(vertices) > -0.5 * h  # vertices on (or next to) the envelope
    normal = np.zeros_like(vertices)
    index = np.nonzero(near)[0]
    if len(index) == 0:
        return vertices, faces
    distance, closest, face_normal = surface.closest(vertices[index], k=8)
    outside = distance > 0
    vertices[index[outside]] = closest[outside]
    normal[index] = face_normal

    for _ in range(passes):
        n = np.int64(len(vertices))
        a, b = faces.ravel(), faces[:, [1, 2, 0]].ravel()
        candidate = near[a] & near[b]
        lo, hi = np.minimum(a, b)[candidate], np.maximum(a, b)[candidate]
        edges, inverse = np.unique(lo * n + hi, return_inverse=True)
        ea, eb = edges // n, edges % n
        middle = 0.5 * (vertices[ea] + vertices[eb])
        too_far = surface.signed_distance(middle, k=8) > tolerance
        if not too_far.any():
            break

        # Where the surfaces at the two ends meet (the crease), nearest to the middle.
        ea, eb, m = ea[too_far], eb[too_far], middle[too_far]
        n1, n2 = normal[ea], normal[eb]
        g = np.einsum("ij,ij->i", n1, n2)
        r1 = np.einsum("ij,ij->i", n1, vertices[ea] - m)
        r2 = np.einsum("ij,ij->i", n2, vertices[eb] - m)
        det = 1.0 - g * g
        usable = det > 1e-3
        det = np.where(usable, det, 1.0)
        shift = ((r1 - g * r2) / det)[:, None] * n1 + ((r2 - g * r1) / det)[:, None] * n2
        usable &= np.linalg.norm(shift, axis=1) <= h
        target = m + np.where(usable[:, None], shift, 0.0)
        _, new_points, new_normal = surface.closest(target, k=8)
        # At a sharp corner of the part several edges can find the same crease point,
        # or one a vertex already has: a second vertex there would merge with it in the
        # file and leave an edge shared by more than two triangles. Split only once.
        apart = _new_and_apart(new_points, vertices[near], MIN_SPLIT_MM)
        if not apart.all():
            split = np.nonzero(too_far)[0]
            too_far[split[~apart]] = False
            new_points, new_normal = new_points[apart], new_normal[apart]
            if not too_far.any():
                break

        new_id = np.full(len(edges), -1, dtype=np.int64)
        new_id[too_far] = n + np.arange(len(new_points))
        face_edge = np.full(len(a), -1, dtype=np.int64)
        face_edge[candidate] = new_id[inverse]
        faces = _split_edges(faces, face_edge.reshape(-1, 3))
        vertices = np.vstack([vertices, new_points])
        normal = np.vstack([normal, new_normal])
        near = np.concatenate([near, np.ones(len(new_points), dtype=bool)])
    # Where a split was not added because the crease point is already a vertex, the
    # edge across the corner is flipped to run to that vertex instead.
    faces = _flip_bridging_edges(vertices, faces, near, surface, tolerance)
    return vertices, faces


def _flip_bridging_edges(vertices, faces, near, surface, tolerance, rounds: int = 3):
    """Flip edges whose middle is outside the surface (by more than `tolerance`) when the
    other diagonal of their two triangles stays inside and the flip turns no triangle
    over. Each triangle takes part in at most one flip per round."""
    from .smoothing import face_normals

    for _ in range(rounds):
        n = np.int64(len(vertices))
        a, b = faces.ravel(), faces[:, [1, 2, 0]].ravel()
        opposite = faces[:, [2, 0, 1]].ravel()
        owner = np.repeat(np.arange(len(faces)), 3)
        candidate = np.nonzero(near[a] & near[b] & (a < b))[0]
        if len(candidate) == 0:
            break
        middle = 0.5 * (vertices[a[candidate]] + vertices[b[candidate]])
        out = candidate[surface.signed_distance(middle, k=8) > tolerance]
        if len(out) == 0:
            break
        # The twin half-edge (b -> a) of each outside edge.
        directed = a * n + b
        order = np.argsort(directed)
        twin_key = b[out] * n + a[out]
        found = np.minimum(np.searchsorted(directed[order], twin_key), len(order) - 1)
        has_twin = directed[order][found] == twin_key
        out, twin = out[has_twin], order[found][has_twin]
        c, d = opposite[out], opposite[twin]
        f1, f2 = owner[out], owner[twin]
        # The new diagonal must stay inside and must not exist yet.
        inside = surface.signed_distance(0.5 * (vertices[c] + vertices[d]), k=8) <= tolerance
        existing = np.isin(np.minimum(c, d) * n + np.maximum(c, d),
                           np.minimum(a, b) * n + np.maximum(a, b))
        new1 = np.column_stack([c, d, b[out]])
        new2 = np.column_stack([d, c, a[out]])
        before = face_normals(vertices, faces[f1]) + face_normals(vertices, faces[f2])
        keeps_side = (np.einsum("ij,ij->i", face_normals(vertices, new1), before) > 0) & \
                     (np.einsum("ij,ij->i", face_normals(vertices, new2), before) > 0)
        ok = np.nonzero(inside & ~existing & keeps_side & (c != d))[0]
        used = np.zeros(len(faces), dtype=bool)
        chosen = []
        for i in ok:  # few edges; a triangle in two flips at once would break the mesh
            if not used[f1[i]] and not used[f2[i]]:
                used[f1[i]] = used[f2[i]] = True
                chosen.append(i)
        if not chosen:
            break
        chosen = np.array(chosen)
        faces = faces.copy()
        faces[f1[chosen]] = new1[chosen]
        faces[f2[chosen]] = new2[chosen]
    return faces


def _new_and_apart(points: np.ndarray, existing: np.ndarray, distance: float) -> np.ndarray:
    """True for points further than `distance` from every existing point and from every
    earlier point in the list."""
    from scipy.spatial import cKDTree

    keep = np.ones(len(points), dtype=bool)
    if len(existing):
        gap, _ = cKDTree(existing).query(points, k=1, distance_upper_bound=distance)
        keep &= ~np.isfinite(gap)
    for i, j in sorted(cKDTree(points).query_pairs(distance)):
        if keep[i]:
            keep[j] = False
    return keep


def _split_edges(faces: np.ndarray, middle: np.ndarray) -> np.ndarray:
    """Split marked edges. middle[f, e] is the new vertex on edge e of face f
    (edge e joins corners e and e+1), or -1. Winding is preserved."""
    marked = middle >= 0
    count = marked.sum(axis=1)
    out = [faces[count == 0]]

    def rolled(rows: np.ndarray, start: np.ndarray):
        order = (start[:, None] + np.arange(3)) % 3
        f = faces[rows]
        v = np.take_along_axis(f, order, axis=1)
        mid = np.take_along_axis(middle[rows], order, axis=1)
        return v[:, 0], v[:, 1], v[:, 2], mid[:, 0], mid[:, 1], mid[:, 2]

    rows = np.nonzero(count == 1)[0]
    if len(rows):  # start at the split edge
        v0, v1, v2, m0, _, _ = rolled(rows, np.argmax(marked[rows], axis=1))
        out += [np.column_stack([v0, m0, v2]), np.column_stack([m0, v1, v2])]
    rows = np.nonzero(count == 2)[0]
    if len(rows):  # start just after the unsplit edge
        v0, v1, v2, m0, m1, _ = rolled(rows, (np.argmin(marked[rows], axis=1) + 1) % 3)
        out += [np.column_stack([m0, v1, m1]), np.column_stack([v0, m0, m1]),
                np.column_stack([v0, m1, v2])]
    rows = np.nonzero(count == 3)[0]
    if len(rows):
        v0, v1, v2, m0, m1, m2 = rolled(rows, np.zeros(len(rows), dtype=np.int64))
        out += [np.column_stack([v0, m0, m2]), np.column_stack([m0, v1, m1]),
                np.column_stack([m2, m1, v2]), np.column_stack([m0, m1, m2])]
    return np.vstack(out)


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    tri = vertices[faces]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
