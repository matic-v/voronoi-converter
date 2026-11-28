"""Signed distance field of the input body on a voxel grid (negative inside).

Accuracy is spent where it matters:

* within ~3 voxels of the surface: exact distance and sign;
* inside the body down to a requested depth (see ``ensure_depth``): distance to the
  nearest surface sample, accurate to a small fraction of a voxel;
* everywhere else: a Euclidean distance transform, good to about one voxel.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .grid import Grid
from .progress import Job
from .surface import SurfaceIndex

SAMPLE_SPACING_VOXELS = 0.75
NEAR_BAND_VOXELS = 3.5
MAX_SURFACE_SAMPLES = 40_000_000



INNER_BLUR_VOXELS = 1.5  # Gaussian sigma of the blurred field for inner faces

class BodyField:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, grid: Grid, job: Job | None = None):
        job = job or Job()
        h = grid.h
        self.grid = grid
        self.surface = SurfaceIndex(vertices, faces, spacing=SAMPLE_SPACING_VOXELS * h)
        if len(self.surface.samples) > MAX_SURFACE_SAMPLES:
            raise MemoryError("surface is too large for this voxel size")
        job.report(0.15, "Measuring the body")

        # Rough unsigned distance everywhere: distance to the nearest voxel holding a sample.
        index = np.rint(grid.to_index(self.surface.samples)).astype(np.int64)
        index = np.clip(index, 0, np.asarray(grid.shape) - 1)
        occupied = np.zeros(grid.shape, dtype=bool)
        occupied[index[:, 0], index[:, 1], index[:, 2]] = True
        rough = ndimage.distance_transform_edt(~occupied).astype(np.float32) * np.float32(h)
        del occupied
        job.report(0.35, "Measuring the body")

        # Exact signed distance near the surface.
        near = rough <= NEAR_BAND_VOXELS * h
        near_index = np.nonzero(near)
        exact = self.surface.signed_distance(grid.points(near_index))
        sdf = rough
        sdf[near_index] = exact
        job.report(0.8, "Measuring the body")

        # The near band separates inside from outside, so every connected region
        # beyond it has one sign: copy it from a neighbouring near node.
        labels, count = ndimage.label(~near)
        sign = np.ones(count + 1, dtype=np.float32)
        full = slice(None)
        for axis in range(3):
            for far_side, near_side in ((slice(None, -1), slice(1, None)),
                                        (slice(1, None), slice(None, -1))):
                a = tuple(far_side if i == axis else full for i in range(3))
                b = tuple(near_side if i == axis else full for i in range(3))
                touching = (labels[a] > 0) & near[b]
                sign[labels[a][touching]] = np.where(sdf[b][touching] < 0, -1.0, 1.0)
        far = labels > 0
        sdf[far] *= sign[labels[far]]

        self.sdf: np.ndarray = sdf
        self._refined = near  # nodes whose value is accurate
        self.depth = (NEAR_BAND_VOXELS - 1.0) * h
        job.report(1.0, "Measuring the body")

    def ensure_depth(self, depth: float) -> None:
        """Make the field accurate inside the body down to `depth` mm below the surface."""
        if depth <= self.depth:
            return
        h = self.grid.h
        todo = (~self._refined) & (self.sdf < 0) & (self.sdf >= -(depth + 1.5 * h))
        index = np.nonzero(todo)
        if len(index[0]):
            distance = self.surface.unsigned_distance_to_samples(self.grid.points(index))
            self.sdf[index] = -distance.astype(np.float32)
            self._refined |= todo
        self.depth = depth

    @property
    def inner(self) -> np.ndarray:
        """The field blurred by INNER_BLUR_VOXELS: for faces inside the part (the skin's
        inner face, how deep a point is below the surface). The exact distance folds
        sharply wherever the nearest face changes (along every convex edge, in the middle
        of a thin wall); faces built on it end in creases the voxel grid cannot draw,
        and whatever meets them (openings, edge strips) comes out ragged. The blur rounds
        those folds. The part's own surface keeps the exact distance."""
        if getattr(self, "_inner", None) is None or self._inner_depth != self.depth:
            self._inner = ndimage.gaussian_filter(self.sdf, INNER_BLUR_VOXELS, mode="nearest")
            self._inner_depth = self.depth
        return self._inner

    def sample_inner(self, points: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of `inner` at world points."""
        return ndimage.map_coordinates(
            self.inner, self.grid.to_index(points).T, order=1, mode="nearest")

    def sample(self, points: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of the field at world points."""
        return ndimage.map_coordinates(
            self.sdf, self.grid.to_index(points).T, order=1, mode="nearest")
