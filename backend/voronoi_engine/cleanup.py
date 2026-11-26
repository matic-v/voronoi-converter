"""Clean-up of the final mesh: unfold fins, then simplify.

Marching cubes leaves thin triangles folded back on themselves (fins) where the shape
has a crease that the voxel grid cannot resolve: hole rims, rib and strut joints, the
part's own edges. Smoothing can add more. They are tiny but show as shards, and a
slicer has to handle each of them.

1. Unfold fins: every vertex of a triangle pair folded by more than the strength's
   limit is moved to the average of its neighbours. A move is kept only if it makes the
   folding around that vertex smaller, so the step can only straighten the mesh.
2. Simplify: quadric edge collapse (pyfqmr, "lossless" mode) removes vertices whose
   removal changes the shape by less than the strength's tolerance. Flat and gently
   curved areas lose most of their triangles; edges and small details stay.
3. Unfold once more (a collapse can leave a fin of its own).

Everything is deterministic: same input and strength give the same mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .progress import Job
from .smoothing import face_normals


@dataclass(frozen=True)
class Strength:
    fin_angle: float  # degrees between the normals of two triangles that counts as a fin
    epsilon: float  # pyfqmr lossless threshold (quadric error); 0 = no simplification
    tolerance_mm: float  # typical largest change, measured; shown in the UI and docs


STRENGTHS: dict[str, Strength] = {
    "light": Strength(fin_angle=150.0, epsilon=1e-6, tolerance_mm=0.005),
    "medium": Strength(fin_angle=120.0, epsilon=1e-5, tolerance_mm=0.01),
    "strong": Strength(fin_angle=100.0, epsilon=1e-4, tolerance_mm=0.02),
}


@dataclass
class CleanupReport:
    triangles_before: int = 0
    triangles_after: int = 0
    fins_before: int = 0
    fins_after: int = 0


def _gather(start: np.ndarray, data: np.ndarray, which: np.ndarray):
    """Rows `which` of a CSR list: (values, index into `which` of each value)."""
    counts = start[which + 1] - start[which]
    group = np.repeat(np.arange(len(which)), counts)
    offset = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    return data[np.repeat(start[which], counts) + offset], group


def _csr(keys: np.ndarray, values: np.ndarray, n: int):
    order = np.argsort(keys, kind="stable")
    return np.searchsorted(keys[order], np.arange(n + 1)), values[order]


class _Topology:
    """Edges of a closed mesh with the two faces on each, and the faces and neighbours
    of every vertex."""

    def __init__(self, faces: np.ndarray, vertex_count: int):
        corners = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        owner = np.tile(np.arange(len(faces)), 3)
        key = np.sort(corners, axis=1)
        order = np.lexsort((key[:, 1], key[:, 0]))
        sorted_key = key[order]
        # Closed and manifold: every edge in exactly two faces, so the sorted list pairs up.
        self.closed = len(order) % 2 == 0 and bool(
            (sorted_key[0::2] == sorted_key[1::2]).all()
            and (len(order) < 4 or (sorted_key[2::2] != sorted_key[1:-1:2]).any(axis=1).all()))
        self.face_a, self.face_b = owner[order][0::2], owner[order][1::2]
        edge_of_corner = np.empty(len(order), dtype=np.int64)
        edge_of_corner[order] = np.arange(len(order)) // 2
        self.face_edges = edge_of_corner.reshape(3, -1).T  # (faces, 3) edge numbers
        edges = sorted_key[0::2]
        self.neighbour_start, self.neighbours = _csr(
            np.concatenate([edges[:, 0], edges[:, 1]]),
            np.concatenate([edges[:, 1], edges[:, 0]]), vertex_count)
        self.face_start, self.faces_of = _csr(
            faces.ravel(), np.repeat(np.arange(len(faces)), 3), vertex_count)


def _angles(normal: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cos = np.einsum("ij,ij->i", normal[a], normal[b])
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def fold_angles(vertices: np.ndarray, faces: np.ndarray, topology: _Topology | None = None):
    """Angle (degrees) between the normals of the two faces on each edge: 0 flat,
    90 a right-angled edge, near 180 a triangle folded back onto its neighbour."""
    t = topology or _Topology(faces, len(vertices))
    return _angles(face_normals(vertices, faces), t.face_a, t.face_b)


def count_fins(vertices: np.ndarray, faces: np.ndarray, fin_angle: float) -> int:
    t = _Topology(faces, len(vertices))
    return int((fold_angles(vertices, faces, t) > fin_angle).sum()) if t.closed else -1


def unfold_fins(vertices: np.ndarray, faces: np.ndarray, fin_angle: float,
                rounds: int = 12) -> np.ndarray:
    """Move the corners of folded triangle pairs to their neighbours' average, keeping
    only moves that reduce the largest fold around the vertex. Only the triangles
    around the moved vertices are recomputed, so this is fast on big meshes."""
    t = _Topology(faces, len(vertices))
    result = vertices.copy()
    if not t.closed:  # not a closed mesh (a failed simplification): leave it to the checks
        return result
    normal = face_normals(result, faces)
    angle = _angles(normal, t.face_a, t.face_b)

    def worst(candidate, angles_now, star, group):
        per_face = angles_now[t.face_edges[star]].max(axis=1)
        out = np.zeros(len(candidate))
        np.maximum.at(out, group, per_face)
        return out

    def evaluate(points, star_faces, touched_edges):
        """Fold angles with `points`, recomputed only on the touched edges."""
        trial_normal = normal.copy()
        trial_normal[star_faces] = face_normals(points, faces[star_faces])
        trial_angle = angle.copy()
        trial_angle[touched_edges] = _angles(trial_normal, t.face_a[touched_edges],
                                             t.face_b[touched_edges])
        return trial_normal, trial_angle

    for _ in range(rounds):
        fin = np.nonzero(angle > fin_angle)[0]
        if len(fin) == 0:
            break
        candidate = np.unique(faces[np.concatenate([t.face_a[fin], t.face_b[fin]])])
        neighbour, group = _gather(t.neighbour_start, t.neighbours, candidate)
        average = np.zeros((len(candidate), 3))
        np.add.at(average, group, result[neighbour])
        average /= np.bincount(group, minlength=len(candidate))[:, None]
        star, star_group = _gather(t.face_start, t.faces_of, candidate)
        star_faces = np.unique(star)
        touched = np.unique(t.face_edges[star_faces].ravel())

        before = worst(candidate, angle, star, star_group)
        trial = result.copy()
        trial[candidate] = average
        _, trial_angle = evaluate(trial, star_faces, touched)
        keep = worst(candidate, trial_angle, star, star_group) < before
        if not keep.any():
            break
        trial[candidate[~keep]] = result[candidate[~keep]]
        # Neighbours moving together can still make a vertex worse: check once more.
        _, trial_angle = evaluate(trial, star_faces, touched)
        bad = keep & (worst(candidate, trial_angle, star, star_group) > before)
        trial[candidate[bad]] = result[candidate[bad]]
        if not (keep & ~bad).any():
            break
        result = trial
        normal, angle = evaluate(result, star_faces, touched)
    return result


def simplify(vertices: np.ndarray, faces: np.ndarray, epsilon: float):
    import pyfqmr

    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(np.ascontiguousarray(vertices, dtype=np.float64),
                       np.ascontiguousarray(faces, dtype=np.int32))
    simplifier.simplify_mesh_lossless(epsilon=epsilon, verbose=False)
    v, f, _ = simplifier.getMesh()
    return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)


def clean_up(vertices: np.ndarray, faces: np.ndarray, strength: str,
             job: Job | None = None, tolerance_scale: float = 1.0,
             ) -> tuple[np.ndarray, np.ndarray, CleanupReport]:
    """`tolerance_scale` < 1 simplifies less (used to retry when a collapse left a defect)."""
    job = job or Job()
    s = STRENGTHS[strength]
    report = CleanupReport(triangles_before=len(faces),
                           fins_before=count_fins(vertices, faces, s.fin_angle))
    job.report(0.0, "Cleaning up: unfolding fins")
    vertices = unfold_fins(vertices, faces, s.fin_angle)
    if s.epsilon > 0:
        job.report(0.4, "Cleaning up: simplifying")
        vertices, faces = simplify(vertices, faces, s.epsilon * tolerance_scale)
        job.report(0.8, "Cleaning up: unfolding fins")
        vertices = unfold_fins(vertices, faces, s.fin_angle)
    report.triangles_after = len(faces)
    report.fins_after = count_fins(vertices, faces, s.fin_angle)
    job.report(1.0, "Cleaned up")
    return vertices, faces, report
