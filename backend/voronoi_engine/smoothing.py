"""Smoothness: more triangles, placed on the exact shape.

Marching cubes gives a surface that is only as fine as the voxel grid, which shows
as facets on round struts and as saw-teeth along sharp edges. Halving the voxel
size would cost eight times the memory. Instead each smoothing level works on the
surface only: every triangle is split into four, and the vertices are moved onto
the zero level of the exact field (true capsules, true cell walls, true body
surface) rather than the grid's interpolation of it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .parallel import run_chunks
from .progress import Job

CHUNK = 1_000_000
MAX_TRIANGLES = 12_000_000
BYTES_PER_TRIANGLE = 260  # measured peak while smoothing
# A vertex may move at most this fraction of the current edge scale (voxel size
# halved per level): enough to reach the true surface, never enough to fold the mesh.
MAX_MOVE = 0.6
FOLD_COSINE = 0.25
SPIKE_RATIO = 1.0


def subdivide(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split every triangle into four with a new vertex in the middle of each edge.
    Edges shared by two triangles get one shared vertex, so a closed mesh stays closed."""
    n = np.int64(len(vertices))
    a, b = faces.ravel(), faces[:, [1, 2, 0]].ravel()
    key = np.minimum(a, b) * n + np.maximum(a, b)
    edges, inverse = np.unique(key, return_inverse=True)
    middle = 0.5 * (vertices[edges // n] + vertices[edges % n])
    m = (n + inverse).reshape(-1, 3)  # m[:, e] sits on the edge from corner e to corner e + 1
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    m01, m12, m20 = m[:, 0], m[:, 1], m[:, 2]
    new_faces = np.concatenate([
        np.column_stack([v0, m01, m20]), np.column_stack([v1, m12, m01]),
        np.column_stack([v2, m20, m12]), np.column_stack([m01, m12, m20])])
    return np.vstack([vertices, middle]), new_faces


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted outward unit normals."""
    tri = vertices[faces]
    face_normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal = np.empty_like(vertices)
    for k in range(3):
        normal[:, k] = sum(np.bincount(faces[:, corner], face_normal[:, k], len(vertices))
                           for corner in range(3))
    length = np.linalg.norm(normal, axis=1)
    return normal / np.maximum(length, 1e-300)[:, None]


def project(vertices: np.ndarray, normals: np.ndarray, field: Callable[[np.ndarray], np.ndarray],
            limit: float, job: Job | None = None) -> np.ndarray:
    """Move each vertex along its normal onto the zero level of `field` (negative
    inside): one step by the field value, then one secant step. No vertex moves
    further than `limit`, so it can never jump to another part of the surface, and a
    step that does not bring it closer is not taken."""
    job = job or Job()
    out = vertices.copy()

    def work(start: int, stop: int) -> None:
        p = vertices[start:stop]
        n = normals[start:stop]
        f0 = field(p)
        s1 = -np.clip(f0, -limit, limit)  # outside (positive) moves inwards
        f1 = field(p + n * s1[:, None])
        slope = f1 - f0
        usable = np.abs(slope) > 1e-9
        s2 = np.where(usable, s1 - f1 * s1 / np.where(usable, slope, 1.0), s1)
        # Fall back to the first step when the secant step overshoots; skip the vertex
        # when even that did not bring it closer.
        s = np.where(np.abs(s2) <= limit, s2, s1)
        s = np.where((np.abs(s) <= limit) & ((np.abs(f1) < np.abs(f0)) | (np.abs(s2) <= limit)), s, 0.0)
        out[start:stop] = p + n * s[:, None]

    run_chunks(work, len(vertices), CHUNK,
               done=lambda stop: job.report(stop / len(vertices), "Smoothing"))
    return out


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return normal / np.maximum(np.linalg.norm(normal, axis=1), 1e-300)[:, None]


def undo_folds(before: np.ndarray, after: np.ndarray, faces: np.ndarray,
               rounds: int = 4, reference: np.ndarray | None = None) -> np.ndarray:
    """Projection can push neighbouring vertices past each other at sharp edges, which
    folds a triangle over. Put the vertices of every triangle that turned by more than
    about 75 degrees back where they were, until none is left (or `rounds` is used up).
    `reference` = face_normals(before, faces), if already known."""
    if reference is None:
        reference = face_normals(before, faces)
    result = after.copy()
    check = np.arange(len(faces))  # after the first round, only faces next to changes
    for _ in range(rounds):
        turned = check[np.einsum("ij,ij->i", reference[check],
                                 face_normals(result, faces[check])) < FOLD_COSINE]
        if len(turned) == 0:
            break
        index = np.unique(faces[turned])
        result[index] = before[index]
        check = _faces_touching(index, faces, len(before))
    return result


def _faces_touching(vertex_index: np.ndarray, faces: np.ndarray, vertex_count: int) -> np.ndarray:
    mark = np.zeros(vertex_count, dtype=bool)
    mark[vertex_index] = True
    return np.nonzero(mark[faces].any(axis=1))[0]


def _neighbour_average(vertices: np.ndarray, faces: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Average position of each vertex's neighbours. In a closed mesh every edge lies in
    two triangles, so summing the other two corners of every triangle counts each
    neighbour exactly twice, and `count` (twice the number of neighbours) evens it out."""
    total = np.empty_like(vertices)
    n = len(vertices)
    for k in range(3):
        acc = np.zeros(n)
        for corner in range(3):
            others = vertices[faces[:, (corner + 1) % 3], k] + vertices[faces[:, (corner + 2) % 3], k]
            acc += np.bincount(faces[:, corner], others, minlength=n)
        total[:, k] = acc
    return total / count[:, None]


def max_triangles() -> int:
    """Largest result smoothing may produce: MAX_TRIANGLES, or less on a small memory
    budget (smoothing needs about 260 bytes per output triangle)."""
    from .memory import memory_budget_mb

    return int(min(MAX_TRIANGLES, memory_budget_mb() * 2**20 / BYTES_PER_TRIANGLE))


def relax_spikes(vertices: np.ndarray, faces: np.ndarray, reference: np.ndarray | None = None,
                 rounds: int = 4) -> np.ndarray:
    """At a sharp edge (edge rounding 0) only some vertices reach the true crease line,
    and they stand out as spikes. Move every vertex that is further from the average of
    its neighbours than the mesh's mean edge length to that average.

    With `reference` (one unit normal per face), a move is only kept if no triangle
    around the vertex turns by more than about 75 degrees from it, so relaxing can
    never fold the mesh over.
    """
    n = len(vertices)
    count = np.maximum(2 * np.bincount(faces.ravel(), minlength=n), 1)
    tri = vertices[faces]
    limit = SPIKE_RATIO * float(np.linalg.norm(tri - tri[:, [1, 2, 0]], axis=2).mean())
    result = vertices.copy()
    active_faces = faces  # after the first round, only the triangles around changes
    for _ in range(rounds):
        average = _neighbour_average(result, active_faces, count)
        active = np.zeros(n, dtype=bool)
        active[active_faces.ravel()] = True
        # (A vertex whose triangles are all in active_faces has its full average.)
        complete = np.bincount(active_faces.ravel(), minlength=n) * 2 == count
        spike = active & complete & (np.linalg.norm(result - average, axis=1) > limit)
        if not spike.any():
            break
        trial = result.copy()
        trial[spike] = average[spike]
        if reference is not None:
            near = _faces_touching(np.nonzero(spike)[0], faces, n)
            turned = near[np.einsum("ij,ij->i", reference[near],
                                    face_normals(trial, faces[near])) < FOLD_COSINE]
            if len(turned):  # keep the vertex where it was if relaxing it folds a triangle
                bad = np.zeros(n, dtype=bool)
                bad[faces[turned]] = True
                trial[spike & bad] = result[spike & bad]
        changed = np.nonzero(np.any(trial != result, axis=1))[0]
        if len(changed) == 0:
            break
        result = trial
        # Next round: vertices next to a change, with all of their triangles.
        ring = np.unique(faces[_faces_touching(changed, faces, n)])
        active_faces = faces[_faces_touching(ring, faces, n)]
    return result


def allowed_levels(triangles: int, wanted: int) -> int:
    """Levels that keep the result under the triangle limit (each level x 4)."""
    limit = max_triangles()
    levels = 0
    while levels < wanted and triangles * 4 ** (levels + 1) <= limit:
        levels += 1
    return levels


def smooth(vertices: np.ndarray, faces: np.ndarray, field: Callable[[np.ndarray], np.ndarray],
           levels: int, h: float, job: Job | None = None) -> tuple[np.ndarray, np.ndarray]:
    job = job or Job()
    for level in range(levels):
        vertices, faces = subdivide(vertices, faces)
        limit = MAX_MOVE * h / 2**level
        moved = project(vertices, vertex_normals(vertices, faces), field, limit,
                        job.sub(level / levels, (level + 1) / levels))
        reference = face_normals(vertices, faces)
        moved = undo_folds(vertices, moved, faces, reference=reference)
        # Relax spikes, but never in a way that turns a triangle over.
        vertices = relax_spikes(moved, faces, reference=reference)
    return vertices, faces
