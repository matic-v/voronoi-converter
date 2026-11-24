"""Voxel grid and the choice of voxel size."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .settings import Settings

Mode = Literal["preview", "final"]

PAD_VOXELS = 4
# The origin is shifted by a fraction of a voxel so that axis-aligned faces of the
# part (very common in CAD models) do not fall exactly on grid nodes.
ORIGIN_SHIFT = 0.37

# Voxels across the thinnest feature, and the grid-node budget, per quality.
VOXELS_ACROSS = {"draft": 3.0, "normal": 4.0, "fine": 6.0}
NODE_BUDGET = {"draft": 8_000_000, "normal": 20_000_000, "fine": 48_000_000}
PREVIEW_VOXELS_ACROSS = 2.0
PREVIEW_NODE_BUDGET = 1_200_000
PREVIEW_MIN_VOXELS_ACROSS = 1.5
PREVIEW_STRETCH_BUDGET = 3_000_000
MIN_VOXEL_MM = 0.05


@dataclass(frozen=True)
class Grid:
    origin: tuple[float, float, float]
    h: float
    shape: tuple[int, int, int]

    @staticmethod
    def for_bounds(lo: np.ndarray, hi: np.ndarray, h: float) -> "Grid":
        origin = np.asarray(lo, dtype=np.float64) - (PAD_VOXELS + ORIGIN_SHIFT) * h
        shape = np.ceil((np.asarray(hi) - origin) / h).astype(int) + PAD_VOXELS + 1
        return Grid(tuple(float(v) for v in origin), float(h), tuple(int(v) for v in shape))

    @property
    def node_count(self) -> int:
        return int(np.prod(self.shape))

    def points(self, index: tuple[np.ndarray, ...]) -> np.ndarray:
        """World coordinates of the nodes at integer indices (as from np.nonzero)."""
        return np.asarray(self.origin) + np.column_stack(index) * self.h

    def to_index(self, points: np.ndarray) -> np.ndarray:
        """Fractional node index of world points."""
        return (np.asarray(points) - np.asarray(self.origin)) / self.h


def thinnest_feature(settings: Settings) -> float:
    """Smallest thickness the current styles ask for (mm)."""
    from .grading import scale_range  # local import: grading needs Grid

    thin, _ = scale_range(settings, "thickness")
    small_cell, _ = scale_range(settings, "cell_size")
    sizes: list[float] = []
    if settings.surface_style in ("perforated", "embossed", "solid"):
        sizes.append(settings.shell_thickness * thin)
    if settings.surface_style in ("rib_lattice", "embossed"):
        sizes.append(settings.rib_width * thin)
    if settings.surface_style == "embossed":
        sizes.append(2.0 * settings.emboss_depth)  # a step needs half the voxels of a wall
    if settings.surface_style == "perforated":
        # The wall left between two holes.
        wall = (1.0 - settings.hole_size / 100.0) * settings.surface_cell_size * small_cell
        sizes.append(max(wall, 0.8))
    if settings.interior_style == "struts":
        sizes.append(settings.strut_diameter * thin)
    return min(sizes) if sizes else 2.0


@dataclass(frozen=True)
class Resolution:
    voxel_mm: float
    wanted_mm: float  # what the thinnest feature and the quality ask for
    feature_mm: float  # the thinnest feature
    limited_by: str | None  # None, "memory" or "quality" (the node limit of the quality level)

    @property
    def capped(self) -> bool:
        return self.limited_by is not None

    @property
    def voxels_across_feature(self) -> float:
        return self.feature_mm / self.voxel_mm


def choose_resolution(size_mm, settings: Settings, mode: Mode) -> Resolution:
    """Pick the voxel size from the part size and the thinnest feature.

    Final: Draft / Normal / Fine put 3 / 4 / 6 voxels across the thinnest feature, limited
    by the node limit of the quality level and by the memory budget.
    Preview: 2 voxels across the thinnest feature within a small node budget; the budget
    is stretched (up to PREVIEW_STRETCH_BUDGET) to keep at least 1.5 voxels across it,
    because thinner than that the feature breaks up.
    """
    from .memory import max_nodes  # local import keeps this module free of OS calls at import

    feature = thinnest_feature(settings)
    size = np.asarray(size_mm, dtype=np.float64)

    def nodes(h: float) -> float:
        return float(np.prod(size / h + 2 * PAD_VOXELS + 2))

    def fit(h: float, budget: float) -> float:
        while nodes(h) > budget:
            h *= max(1.01, (nodes(h) / budget) ** (1.0 / 3.0))
        return h

    memory_nodes = max_nodes()
    if mode == "preview":
        wanted = max(feature / PREVIEW_VOXELS_ACROSS, MIN_VOXEL_MM)
        h = fit(wanted, min(PREVIEW_NODE_BUDGET, memory_nodes))
        floor = feature / PREVIEW_MIN_VOXELS_ACROSS
        if h > floor:  # too coarse to show the thinnest feature: spend more nodes if allowed
            h = min(h, fit(floor, min(PREVIEW_STRETCH_BUDGET, memory_nodes)))
        limited = "quality" if h > wanted * 1.001 else None
    else:
        wanted = max(feature / VOXELS_ACROSS[settings.quality], MIN_VOXEL_MM)
        quality_nodes = NODE_BUDGET[settings.quality]
        h = fit(wanted, min(quality_nodes, memory_nodes))
        limited = None
        if h > wanted * 1.001:
            limited = "memory" if memory_nodes < quality_nodes else "quality"
    # Rounded so the value is stable and readable.
    return Resolution(float(np.round(h, 3)), float(wanted), float(feature), limited)


def choose_voxel_size(size_mm, settings: Settings, mode: Mode) -> tuple[float, bool]:
    """(voxel size in mm, capped) - short form of choose_resolution."""
    resolution = choose_resolution(size_mm, settings, mode)
    return resolution.voxel_mm, resolution.capped
