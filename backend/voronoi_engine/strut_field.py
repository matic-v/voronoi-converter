"""Distance field of round beams (capsules) along a set of segments.

Values are (distance to the beam axis - beam radius), combined across beams with a
smooth minimum so that joints are filleted by the blend radius.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .grid import Grid
from .progress import Job
from .parallel import kd_workers, run_chunks

CHUNK = 250_000
FIRST_K = 16
MAX_K = 128


def smooth_min(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth minimum; the blend reaches about `k` from a sharp joint.
    `k` is a number, or an array with one radius per point (0 = sharp there)."""
    if np.ndim(k) == 0:
        if k <= 0:
            return np.minimum(a, b)
        weight = np.clip(k - np.abs(a - b), 0.0, None) / k
        return np.minimum(a, b) - weight * weight * (k / 4.0)
    k = np.asarray(k, dtype=np.result_type(a, b))
    safe = np.maximum(k, 1e-12)
    weight = np.clip(k - np.abs(a - b), 0.0, None) / safe
    return np.minimum(a, b) - weight * weight * (k / 4.0)


def smooth_max(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Smooth maximum (rounded intersection). Never below max(a, b), so it only ever
    removes material: an edge rounded with it stays inside the sharp version."""
    return -smooth_min(-a, -b, k)


def sample_segments(segments: np.ndarray, spacing: float):
    """Centres of equal pieces no longer than `spacing`. Returns (points, segment index)."""
    p, d = segments[:, 0], segments[:, 1] - segments[:, 0]
    count = np.maximum(1, np.ceil(np.linalg.norm(d, axis=1) / spacing)).astype(np.int64)
    index = np.repeat(np.arange(len(segments)), count)
    start = np.cumsum(count) - count
    t = (np.arange(len(index)) - np.repeat(start, count) + 0.5) / count[index]
    return p[index] + d[index] * t[:, None], index


def segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Returns (distance to the segment, position 0..1 of the closest point along it)."""
    ab = b - a
    ap = points - a
    t = np.einsum("ij,ij->i", ap, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300)
    t = np.clip(t, 0.0, 1.0)
    offset = ap - ab * t[:, None]
    return np.sqrt(np.einsum("ij,ij->i", offset, offset)), t


class StrutEvaluator:
    """Evaluates the smooth union of capsules at arbitrary points."""

    def __init__(self, segments: np.ndarray, radius: np.ndarray, blend: float, reach: float,
                 min_spacing: float):
        self.a, self.b = segments[:, 0], segments[:, 1]
        # Radius at the two ends of each beam; it varies linearly in between.
        radius = np.asarray(radius, dtype=np.float64)
        if radius.ndim == 1:
            radius = np.stack([radius, radius], axis=1)
        self.radius_a, self.radius_b = radius[:, 0], radius[:, 1]
        self.blend = blend
        self.reach = reach
        # Blending several beams at a joint lowers the value below what any single beam
        # gives, so a beam can matter a little beyond `reach`: up to one blend radius.
        self.influence = reach + blend
        # Coarse samples are only used to find which beams are near a point.
        self.spacing = max(0.75 * reach, min_spacing)
        points, self.sample_segment = sample_segments(segments, self.spacing)
        self.tree = cKDTree(points)
        self.sample_count = len(points)

    def near(self, points: np.ndarray) -> np.ndarray:
        """True where a beam may be within reach (elsewhere the field is just "far")."""
        limit = float(np.hypot(self.influence, 0.5 * self.spacing))
        distance, _ = self.tree.query(points, k=1, distance_upper_bound=limit, workers=kd_workers())
        return np.isfinite(distance)

    def __call__(self, points: np.ndarray, k: int = FIRST_K) -> np.ndarray:
        k = min(k, self.sample_count)
        distance, nearest = self.tree.query(points, k=k, workers=kd_workers())
        distance = distance.reshape(len(points), k)
        candidates = np.sort(self.sample_segment[nearest.reshape(len(points), k)], axis=1)
        fresh = np.ones(candidates.shape, dtype=bool)  # first occurrence of each beam in a row
        fresh[:, 1:] = candidates[:, 1:] != candidates[:, :-1]
        row, column = np.nonzero(fresh)
        beam = candidates[row, column]
        values = np.full(candidates.shape, np.inf)
        distance_to_axis, along = segment_distance(points[row], self.a[beam], self.b[beam])
        values[row, column] = distance_to_axis - (
            self.radius_a[beam] + (self.radius_b[beam] - self.radius_a[beam]) * along)
        values.sort(axis=1)

        # Folding from the smallest value makes the smooth minimum independent of beam order.
        result = values[:, 0].copy()
        if self.blend > 0:
            for j in range(1, k):
                gap = values[:, j] - result
                if not (gap < self.blend).any():
                    break
                weight = np.clip(self.blend - gap, 0.0, None) / self.blend
                result -= weight * weight * (self.blend / 4.0)

        # If even the furthest of the k samples is within range, beams may have been missed.
        limit = np.hypot(self.influence, 0.5 * self.spacing)
        crowded = distance[:, -1] <= limit
        if crowded.any() and k < min(MAX_K, self.sample_count):
            result[crowded] = self(points[crowded], k * 2)
        return result


def strut_margin(blend: float, h: float) -> float:
    """Value used where no beam is near. Values are needed up to one voxel above zero
    (for interpolation) and up to the blend radius (for the smooth union with the skin)."""
    return max(blend, h)


def make_evaluator(segments: np.ndarray, radius: np.ndarray, blend: float, h: float,
                   margin: float | None = None) -> StrutEvaluator:
    reach = float(np.max(radius)) + (strut_margin(blend, h) if margin is None else margin)
    return StrutEvaluator(segments, radius, blend, reach, min_spacing=h)


def strut_field(grid: Grid, body_sdf: np.ndarray, segments: np.ndarray, radius: np.ndarray,
                blend: float, job: Job | None = None,
                evaluator: StrutEvaluator | None = None,
                margin: float | None = None) -> np.ndarray:
    """Field on the grid, negative inside the beams. Only nodes in or next to the body
    and near a beam are evaluated; the rest hold a positive constant."""
    job = job or Job()
    h = grid.h
    # Values are needed up to one voxel above zero (for interpolation) and up to the
    # blend radius (for the smooth union with the skin).
    margin = strut_margin(blend, h) if margin is None else margin
    reach = float(radius.max()) + margin
    field = np.full(grid.shape, np.float32(margin), dtype=np.float32)
    if len(segments) == 0:
        return field

    # Voxels close enough to a beam to matter.
    fine, _ = sample_segments(segments, 0.9 * h)
    index = np.rint(grid.to_index(fine)).astype(np.int64)
    inside_grid = ((index >= 0) & (index < np.asarray(grid.shape))).all(axis=1)
    index = index[inside_grid]
    occupied = np.zeros(grid.shape, dtype=bool)
    occupied[index[:, 0], index[:, 1], index[:, 2]] = True
    near = ndimage.distance_transform_edt(~occupied) * h <= reach + blend + h
    del occupied
    nodes = np.nonzero(near & (body_sdf < h))
    del near
    job.report(0.2, "Building struts")

    evaluate = evaluator or StrutEvaluator(segments, radius, blend, reach, min_spacing=h)
    total = len(nodes[0])
    values = np.empty(total, dtype=np.float32)
    def work(start: int, stop: int) -> None:
        block = tuple(axis[start:stop] for axis in nodes)
        values[start:stop] = evaluate(grid.points(block))

    run_chunks(work, total, CHUNK,
               done=lambda stop: job.report(0.2 + 0.8 * stop / total, "Building struts"))
    field[nodes] = np.minimum(values, np.float32(margin))
    return field
