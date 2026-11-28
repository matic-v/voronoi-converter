"""Filling gaps between the skins on the two sides of a thin wall.

The skin is the material between the part's surface and its inward offset by the skin
depth. Where the part is only a little thicker than twice the skin depth, the skins of
the two sides leave a gap in the middle that is narrower than the voxel grid can draw:
marching cubes turns it into a torn, ragged rim (seen through every opening of the
pattern). Gaps narrower than about two voxels are closed instead: the skin there is
made deeper until it meets the skin of the other side.

For each grid node near the skin's inner face, the material is followed inward (against
the gradient of the distance field) to find how far the part continues: `H`, the
largest depth along that path, is half the local wall thickness. With a gap of
`gap = H - depth` on each side of the middle, gaps up to 2 voxels are closed (the skins
overlap by a voxel), gaps from 2.5 voxels on are left, and in between the gap opens
steeply, so the closed and open parts meet without a step and the thin tip of the
closing gap is as short as possible. The extra depth is stored on the grid and interpolated, so the voxel construction
and the exact field (smoothing) use the same values.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .parallel import run_chunks

CLOSE_VOXELS = 2.0  # gaps up to this many voxels (on each side of the middle) are closed
OPEN_VOXELS = 2.5  # gaps from this many voxels on are left as they are
FILL_VOXELS = OPEN_VOXELS + 1.0  # the most the skin is made deeper


def extra_depth(body, depth, margin: float) -> np.ndarray | None:
    """Extra skin depth per grid node (float32 grid), or None where nothing is thin.
    `depth` is a number or a grid of skin depths; `margin` is how far beyond the inner
    face the skin is still evaluated (rounding)."""
    d, grid, h = body.sdf, body.grid, body.grid.h
    fill = OPEN_VOXELS * h
    depth_grid = np.broadcast_to(np.asarray(depth, dtype=np.float32), d.shape)
    near = (d < -(depth_grid - margin - h)) & (d > -(depth_grid + fill + margin + h))
    index = np.nonzero(near)
    if len(index[0]) == 0:
        return None
    start = grid.points(index)
    d0 = d[index].astype(np.float64)
    goal = depth_grid[index].astype(np.float64) + fill  # deeper than this: no fill needed
    half = np.maximum(-d0, 0.0)
    step = 0.5 * h

    def work(lo: int, hi: int) -> None:
        p = start[lo:hi]
        reached = half[lo:hi] >= goal[lo:hi]
        # Inward direction: against the gradient of the distance field.
        gradient = np.stack([body.sample(p + 0.5 * h * e) - body.sample(p - 0.5 * h * e)
                             for e in np.eye(3)], axis=1)
        inward = -gradient / np.maximum(np.linalg.norm(gradient, axis=1), 1e-12)[:, None]
        best = half[lo:hi].copy()
        travel = np.maximum(goal[lo:hi] + d0[lo:hi], 0.0) + h  # how far the goal can be
        active = np.nonzero(~reached)[0]
        t = step
        while len(active):
            q = p[active] + inward[active] * t
            best[active] = np.maximum(best[active], -body.sample(q))
            active = active[(best[active] < goal[lo:hi][active]) & (t + step <= travel[active])]
            t += step
        half[lo:hi] = best

    run_chunks(work, len(d0), 100_000)
    gap = half - depth_grid[index]
    # Half-width of the gap that is left: overlapping by a voxel (closed) up to CLOSE
    # voxels, the full gap from OPEN voxels on, steeply in between so that the narrow tip
    # of a closing gap covers as little of the part as possible.
    close, open_ = CLOSE_VOXELS * h, OPEN_VOXELS * h
    left = np.clip(-h + (gap - close) * (open_ + h) / (open_ - close), -h, gap)
    extra = np.clip(gap - left, 0.0, None).astype(np.float32)
    if not (extra > 0).any():
        return None
    out = np.zeros(d.shape, dtype=np.float32)
    out[index] = extra
    return out


def sample_extra(extra: np.ndarray | None, grid, points: np.ndarray) -> np.ndarray | float:
    """The extra depth at world points (trilinear, as the grid is sampled elsewhere)."""
    if extra is None:
        return 0.0
    return ndimage.map_coordinates(extra, grid.to_index(points).T, order=1, mode="nearest")
