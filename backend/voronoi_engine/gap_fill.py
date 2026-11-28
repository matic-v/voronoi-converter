"""Closing gaps in the material that are too narrow for the voxel grid.

Where two pieces of material nearly touch (an edge strip and the next rib, the narrow
tip of an opening, two struts passing close by) the space between them can be narrower
than about two voxels. The grid cannot draw such a slot: it comes out as a torn, ragged
seam that no smoothing can repair. The slot is closed instead.

On the final grid, the material (field < 0) is closed morphologically along lines (the
axes and face diagonals) with a radius of one voxel; the nodes that adds are the narrow
gaps (material on opposite sides). Inside corners are not gaps and are left as they are.
The gap nodes are blurred into a smooth field `g`, and a filler shape where `g` is above
THRESHOLD is joined to the material with a blend of one voxel: it covers the gap and
nothing beside it, so the material next to a gap does not swell. `g` is stored on the
grid and interpolated, so the voxel construction and the exact field (smoothing) use
the same values.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

CLOSE_RADIUS_VOXELS = 1  # gaps up to twice this many voxels wide are closed
BLUR_VOXELS = 1.0  # Gaussian sigma that turns the filled nodes into a smooth field
THRESHOLD = 0.3  # the filler covers the nodes where the blurred gap field exceeds this


# Directions in which a gap is looked for: the axes and the face diagonals.
DIRECTIONS = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
              (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]


def _segment(direction, r: int) -> np.ndarray:
    """A line of 2r+1 nodes through the centre of a (2r+1)^3 structure."""
    out = np.zeros((2 * r + 1,) * 3, dtype=bool)
    for k in range(-r, r + 1):
        out[tuple(r + k * c for c in direction)] = True
    return out


def narrow_gaps(field: np.ndarray, inside: np.ndarray) -> np.ndarray | None:
    """Smooth gap field `g` on the grid (float32, 0..1), or None if there is no gap.
    `inside`: nodes inside the part (gaps outside it are not filled).

    A node is in a gap when material lies on both opposite sides of it within
    CLOSE_RADIUS_VOXELS along one of DIRECTIONS (a closing along that line). An inside
    corner, with material on two neighbouring sides only, is not a gap and stays sharp."""
    material = field < 0
    r = CLOSE_RADIUS_VOXELS
    padded = np.pad(material, r + 1)
    inner = tuple(slice(r + 1, -(r + 1)) for _ in range(3))
    gap = np.zeros_like(material)
    for direction in DIRECTIONS:
        gap |= ndimage.binary_closing(padded, structure=_segment(direction, r))[inner]
    gap &= ~material & inside
    if not gap.any():
        return None
    return ndimage.gaussian_filter(gap.astype(np.float32), BLUR_VOXELS, mode="constant")


def apply(field: np.ndarray, gaps: np.ndarray | None, h: float) -> np.ndarray:
    """The field with the narrow gaps filled (on the grid, or at points with `gaps`
    already sampled there)."""
    if gaps is None:
        return field
    from .strut_field import smooth_min

    filler = (np.float32(h / THRESHOLD) * (np.float32(THRESHOLD) - gaps)).astype(field.dtype)
    return smooth_min(field, filler, h)


def sample(gaps: np.ndarray | None, grid, points: np.ndarray):
    if gaps is None:
        return None
    return ndimage.map_coordinates(gaps, grid.to_index(points).T, order=1, mode="nearest")
