"""Exact signed distance from points to a closed triangle mesh.

The surface is covered with sample points held in a KD-tree. For a query point the
triangles of its nearest samples are candidates; the exact closest point on each is
computed and the nearest wins. The sign comes from angle-weighted pseudo-normals
(Baerentzen & Aanaes), which is exact for watertight, consistently wound meshes -
exactly what input validation guarantees.

Convention: negative inside, positive outside.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from .parallel import kd_workers, run_chunks

CHUNK = 400_000


class SurfaceIndex:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, spacing: float):
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        tri = vertices[faces]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        double_area = np.linalg.norm(cross, axis=1)
        keep = double_area > 1e-12  # zero-area triangles carry no surface
        self.tri = tri[keep]
        faces = faces[keep]
        self.face_normal = cross[keep] / double_area[keep, None]
        self.vertex_normal = self._vertex_normals(vertices, faces)[faces]  # (F, 3, 3)
        self.edge_normal = self._edge_normals(len(vertices), faces)  # (F, 3, 3): AB, BC, CA
        self.samples, self.sample_face = sample_surface(self.tri, spacing)
        self.tree = cKDTree(self.samples)
        self.spacing = spacing

    # -- queries ------------------------------------------------------------

    def unsigned_distance_to_samples(self, points: np.ndarray) -> np.ndarray:
        """Distance to the nearest sample: a fast, close upper bound away from the surface."""
        distance, _ = self.tree.query(points, k=1, workers=kd_workers())
        return distance

    def signed_distance(self, points: np.ndarray, k: int = 4) -> np.ndarray:
        """Exact signed distance of each point."""
        return self.closest(points, k)[0]

    def closest(self, points: np.ndarray, k: int = 4):
        """Returns (signed distance, closest surface point, unit normal of the closest face)."""
        points = np.asarray(points, dtype=np.float64)
        k = min(k, len(self.samples))
        distance = np.empty(len(points))
        closest = np.empty((len(points), 3))
        normal = np.empty((len(points), 3))
        def work(start: int, stop: int) -> None:
            block = points[start:stop]
            _, nearest = self.tree.query(block, k=k, workers=kd_workers())
            candidates = self.sample_face[nearest.reshape(len(block), k)]
            d, c, f = self._best_of(block, candidates)
            distance[start:stop] = d
            closest[start:stop] = c
            normal[start:stop] = self.face_normal[f]

        run_chunks(work, len(points), CHUNK)
        return distance, closest, normal

    # -- internals ----------------------------------------------------------

    def _best_of(self, points: np.ndarray, candidates: np.ndarray):
        n = len(points)
        best_d2 = np.full(n, np.inf)
        best_point = np.zeros((n, 3))
        best_normal = np.zeros((n, 3))
        best_face = np.zeros(n, dtype=np.int64)
        for column in range(candidates.shape[1]):
            face = candidates[:, column]
            if column and np.array_equal(face, candidates[:, column - 1]):
                continue
            point, region = closest_point_on_triangle(points, self.tri[face])
            d2 = np.einsum("ij,ij->i", points - point, points - point)
            better = d2 < best_d2
            if not better.any():
                continue
            f, r = face[better], region[better]
            normal = self.face_normal[f].copy()
            for code, corner in ((0, 0), (1, 1), (3, 2)):  # vertex regions A, B, C
                m = r == code
                normal[m] = self.vertex_normal[f[m], corner]
            for code, edge in ((2, 0), (5, 1), (4, 2)):  # edge regions AB, BC, CA
                m = r == code
                normal[m] = self.edge_normal[f[m], edge]
            best_d2[better] = d2[better]
            best_point[better] = point[better]
            best_normal[better] = normal
            best_face[better] = f
        offset = points - best_point
        side = np.einsum("ij,ij->i", offset, best_normal)
        # Degenerate pseudo-normal: fall back to the face normal.
        fallback = side == 0
        if fallback.any():
            side[fallback] = np.einsum(
                "ij,ij->i", offset[fallback], self.face_normal[best_face[fallback]])
        return np.sqrt(best_d2) * np.where(side < 0, -1.0, 1.0), best_point, best_face

    def _vertex_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        normals = np.zeros_like(vertices)
        for corner in range(3):
            a = self.tri[:, corner]
            u = self.tri[:, (corner + 1) % 3] - a
            v = self.tri[:, (corner + 2) % 3] - a
            cos = np.einsum("ij,ij->i", u, v) / np.maximum(
                np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300)
            angle = np.arccos(np.clip(cos, -1.0, 1.0))
            np.add.at(normals, faces[:, corner], self.face_normal * angle[:, None])
        return normals

    def _edge_normals(self, vertex_count: int, faces: np.ndarray) -> np.ndarray:
        """Sum of the two face normals meeting at each face edge."""
        n = np.int64(vertex_count)
        a, b = faces.ravel(), faces[:, [1, 2, 0]].ravel()
        keys = a * n + b
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        opposite = b * n + a
        position = np.clip(np.searchsorted(sorted_keys, opposite), 0, len(keys) - 1)
        found = sorted_keys[position] == opposite
        own = np.repeat(self.face_normal, 3, axis=0)
        other_face = order[position] // 3
        normals = own + np.where(found[:, None], self.face_normal[other_face], 0.0)
        return normals.reshape(-1, 3, 3)


def closest_point_on_triangle(p: np.ndarray, tri: np.ndarray):
    """Closest point on each triangle to each point (Ericson, vectorised).

    Returns (points, region); region is 0/1/3 for vertex A/B/C, 2/5/4 for edge
    AB/BC/CA and 6 for the face interior.
    """
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    dot = lambda u, v: np.einsum("ij,ij->i", u, v)  # noqa: E731
    d1, d2 = dot(ab, ap), dot(ac, ap)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        v = np.where(denom != 0, vb / denom, 0.0)
        w = np.where(denom != 0, vc / denom, 0.0)
        out = a + ab * v[:, None] + ac * w[:, None]
        region = np.full(len(p), 6, dtype=np.int8)

        # Assigned in reverse order of precedence so earlier tests win.
        m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        t = np.where(m, (d4 - d3) / np.where(m, (d4 - d3) + (d5 - d6), 1.0), 0.0)
        out[m] = (b + (c - b) * np.nan_to_num(t)[:, None])[m]
        region[m] = 5
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        t = np.where(m, d2 / np.where(m, d2 - d6, 1.0), 0.0)
        out[m] = (a + ac * np.nan_to_num(t)[:, None])[m]
        region[m] = 4
        m = (d6 >= 0) & (d5 <= d6)
        out[m] = c[m]
        region[m] = 3
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        t = np.where(m, d1 / np.where(m, d1 - d3, 1.0), 0.0)
        out[m] = (a + ab * np.nan_to_num(t)[:, None])[m]
        region[m] = 2
        m = (d3 >= 0) & (d4 <= d3)
        out[m] = b[m]
        region[m] = 1
        m = (d1 <= 0) & (d2 <= 0)
        out[m] = a[m]
        region[m] = 0
    return out, region


def sample_surface(tri: np.ndarray, spacing: float):
    """Points strictly inside each triangle, no further than ~`spacing` apart.

    Stations are spaced along the longest edge; at each station points are spread
    across the triangle, perpendicular to that edge. Long thin triangles therefore
    get one line of samples that reaches into both sharp corners.
    Returns (points, triangle index of each point).
    """
    lengths = np.stack([
        np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
        np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
        np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
    ], axis=1)
    longest = np.argmax(lengths, axis=1)
    every = np.arange(len(tri))
    a = tri[every, longest]
    b = tri[every, (longest + 1) % 3]
    c = tri[every, (longest + 2) % 3]
    ab = b - a
    base = lengths[every, longest]
    height = np.linalg.norm(np.cross(ab, c - a), axis=1) / base
    # Where the apex C projects onto AB (0..1, because AB is the longest edge).
    apex = np.clip(np.einsum("ij,ij->i", c - a, ab) / base**2, 0.0, 1.0)

    station_count = np.maximum(1, np.ceil(base / spacing)).astype(np.int64)
    station_face = np.repeat(every, station_count)
    first = np.cumsum(station_count) - station_count
    t = (np.arange(len(station_face)) - np.repeat(first, station_count) + 0.5) \
        / station_count[station_face]
    tc = apex[station_face]
    rising = t <= tc
    # Fraction of the full height available at this station.
    fraction = np.where(rising, t / np.maximum(tc, 1e-300), (1.0 - t) / np.maximum(1.0 - tc, 1e-300))
    across_count = np.maximum(1, np.ceil(fraction * height[station_face] / spacing)).astype(np.int64)

    point_station = np.repeat(np.arange(len(station_face)), across_count)
    first = np.cumsum(across_count) - across_count
    w = (np.arange(len(point_station)) - np.repeat(first, across_count) + 0.5) \
        / across_count[point_station]
    face = station_face[point_station]
    on_base = a[face] + ab[face] * t[point_station, None]
    # Direction from the foot of the altitude to the apex, scaled to the local height.
    to_apex = c[face] - (a[face] + ab[face] * apex[face, None])
    points = on_base + to_apex * (fraction[point_station] * w)[:, None]
    return points, face
