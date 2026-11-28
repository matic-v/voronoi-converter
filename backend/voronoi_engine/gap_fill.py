"""Closing gaps in the material that are too narrow for the voxel grid.

Where two pieces of material nearly touch (an edge strip and the next rib, the narrow
tip of an opening, two struts passing close by) the space between them can be narrower
than about two voxels. The grid cannot draw such a slot: it comes out as a torn, ragged
seam that no smoothing can repair. The slot is closed instead.

On the final grid, the material (field < 0) is closed morphologically with a radius of
one voxel; the nodes that closing adds are the narrow gaps. Their indicator, widened by
a voxel, is blurred into a smooth field `g` (0 away from gaps, 1 at full strength) and the material field becomes
`field - STRENGTH_VOXELS * h * g`, which fills the gap and rounds into the material
beside it. `g` is stored on the grid and interpolated, so the voxel construction and
the exact field (smoothing) use the same values.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

CLOSE_RADIUS_VOXELS = 1  # gaps up to about twice this are closed
BLUR_VOXELS = 1.0  # Gaussian sigma that turns the filled nodes into a smooth field
STRENGTH_VOXELS = 3.0  # how far the field is lowered at full strength (in voxels)


def narrow_gaps(field: np.ndarray, inside: np.ndarray) -> np.ndarray | None:
    """Smooth gap field `g` on the grid (float32, 0..1), or None if there is no gap.
    `inside`: nodes inside the part (gaps outside it are not filled)."""
    material = field < 0
    r = CLOSE_RADIUS_VOXELS
    structure = np.ones((2 * r + 1,) * 3, dtype=bool)
    padded = np.pad(material, r + 1)
    closed = ndimage.binary_closing(padded, structure=structure)[
        tuple(slice(r + 1, -(r + 1)) for _ in range(3))]
    gap = closed & ~material & inside
    if not gap.any():
        return None
    # Widened by a voxel before blurring: a short gap (small in every direction) would
    # otherwise blur to too little to close, and a closed slot would leave a thin neck.
    gap = ndimage.binary_dilation(gap, structure=structure) & inside
    return np.minimum(ndimage.gaussian_filter(gap.astype(np.float32), BLUR_VOXELS, mode="constant")
                      * 2.0, 1.0)


def apply(field: np.ndarray, gaps: np.ndarray | None, h: float) -> np.ndarray:
    """The field with the narrow gaps filled (on the grid, or at points with `gaps`
    already sampled there)."""
    if gaps is None:
        return field
    return field - np.float32(STRENGTH_VOXELS * h) * gaps


def sample(gaps: np.ndarray | None, grid, points: np.ndarray):
    if gaps is None:
        return None
    return ndimage.map_coordinates(gaps, grid.to_index(points).T, order=1, mode="nearest")
