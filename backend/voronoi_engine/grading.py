"""Grading: cell size and/or thickness vary smoothly through the part.

A position is mapped to u in 0..1:

* From skin: 0 at the surface, 1 at the deepest point of the part.
* X / Y / Z: 0 at the low end of the bounding box, 1 at the high end.

"Denser at start" makes u = 0 the dense end. The setting value (cell size, strut
diameter, ...) is what you get in the middle; at strength g the ratio between the
two ends is 1 + 3 g (so 4 : 1 at 100 %). Dense means smaller cells and/or thicker
struts, ribs and walls, depending on the grading target.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .body_field import BodyField
from .grid import Grid
from .settings import Settings
from .surface import sample_surface

AXES = {"x": 0, "y": 1, "z": 2}
DEPTH_GRID_DIVISIONS = 128  # depth table spacing = longest side / this
DEPTH_GRID_MARGIN = 3


def end_ratio(settings: Settings) -> float:
    """Sparse end : dense end."""
    if settings.grading_mode == "off":
        return 1.0
    return 1.0 + 3.0 * settings.grading_strength / 100.0


def scale_range(settings: Settings, target: str) -> tuple[float, float]:
    """(smallest, largest) multiplier applied to cell sizes or thicknesses."""
    ratio = end_ratio(settings)
    if ratio == 1.0 or settings.grading_target not in (target, "both"):
        return 1.0, 1.0
    return ratio ** -0.5, ratio ** 0.5


class Grading:
    def __init__(self, settings: Settings, lo: np.ndarray, hi: np.ndarray, body: BodyField):
        self.mode = settings.grading_mode
        self.ratio = end_ratio(settings)
        self.active = self.ratio > 1.0
        self.cells = self.active and settings.grading_target in ("cell_size", "both")
        self.thickness = self.active and settings.grading_target in ("thickness", "both")
        self.dense_at_start = settings.grading_direction == "denser_at_start"
        self.lo, self.hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        self.max_depth = 1.0
        self._depth = None
        if self.active and self.mode == "from_skin":
            # Depth below the skin is tabulated once on a coarse grid whose spacing is
            # fixed by the part size, from points sampled on the input triangles. It
            # never uses the voxel grid, so seeds come out the same for preview and final.
            size = float((self.hi - self.lo).max())
            self._step = size / DEPTH_GRID_DIVISIONS
            self._origin = self.lo - DEPTH_GRID_MARGIN * self._step
            shape = np.ceil((self.hi - self._origin) / self._step).astype(int) + DEPTH_GRID_MARGIN + 1
            points, _ = sample_surface(body.surface.tri, 0.7 * self._step)
            index = np.clip(np.rint((points - self._origin) / self._step).astype(np.int64), 0, shape - 1)
            occupied = np.zeros(shape, dtype=bool)
            occupied[index[:, 0], index[:, 1], index[:, 2]] = True
            self._depth = ndimage.distance_transform_edt(~occupied) * self._step
            # Deepest point of the part: probe every other node for being inside.
            probe = np.indices(self._depth[::2, ::2, ::2].shape).reshape(3, -1).T * 2
            inside = body.sample(self._origin + probe * self._step) < 0
            deepest = self._depth[::2, ::2, ::2].ravel()[inside]
            self.max_depth = float(deepest.max()) if len(deepest) else \
                0.25 * float((self.hi - self.lo).min())
            self.max_depth = max(self.max_depth, self._step)

    # -- position along the grading ---------------------------------------------

    def _density_coordinate(self, u: np.ndarray) -> np.ndarray:
        """0 at the dense end, 1 at the sparse end."""
        u = np.clip(u, 0.0, 1.0)
        return u if self.dense_at_start else 1.0 - u

    def _u(self, points: np.ndarray, depth: np.ndarray | None) -> np.ndarray:
        if self.mode == "from_skin":
            if depth is None:
                depth = ndimage.map_coordinates(
                    self._depth, ((points - self._origin) / self._step).T, order=1, mode="nearest")
            return np.asarray(depth) / self.max_depth
        axis = AXES[self.mode]
        return (points[:, axis] - self.lo[axis]) / max(self.hi[axis] - self.lo[axis], 1e-9)

    # -- multipliers ------------------------------------------------------------

    def cell_scale(self, points: np.ndarray, depth: np.ndarray | None = None):
        """Multiplier for cell sizes at points (1.0 when cell size is not graded)."""
        if not self.cells:
            return np.ones(len(points))
        return self.ratio ** (self._density_coordinate(self._u(points, depth)) - 0.5)

    def thickness_scale(self, points: np.ndarray, depth: np.ndarray | None = None):
        """Multiplier for thicknesses at points (dense = thicker)."""
        if not self.thickness:
            return np.ones(len(points))
        return self.ratio ** (0.5 - self._density_coordinate(self._u(points, depth)))

    def thickness_scale_on_grid(self, grid: Grid, d: np.ndarray):
        """Thickness multiplier at every grid node: 1.0, or an array that broadcasts
        against the grid."""
        if not self.thickness:
            return 1.0
        if self.mode == "from_skin":
            u = np.clip(-d, 0.0, None) / np.float32(self.max_depth)
        else:
            axis = AXES[self.mode]
            coordinate = grid.origin[axis] + np.arange(grid.shape[axis]) * grid.h
            u = (coordinate - self.lo[axis]) / max(self.hi[axis] - self.lo[axis], 1e-9)
            shape = [1, 1, 1]
            shape[axis] = -1
            u = u.reshape(shape)
        scale = self.ratio ** (0.5 - self._density_coordinate(u))
        return scale.astype(np.float32)
