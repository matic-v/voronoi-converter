"""Filling gaps between the skins on the two sides of a thin wall.

The skin is the material between the part's surface and its inward offset by the skin
depth. Where the part is only a little thicker than twice the skin depth, the skins of
the two sides leave a gap in the middle that is narrower than the voxel grid can draw:
marching cubes turns it into a torn, ragged rim (seen through every opening of the
pattern). Gaps narrower than about two voxels are closed instead: the skin there is
made deeper until it meets the skin of the other side.

For each grid node near the skin's inner face, the material is followed inward (against
the gradient of the distance field) until the path leaves the part on the other side:
half that wall thickness is `H`. (The largest depth along the path would not do: next to
a convex edge of the part it is the distance to the other face, which is small although
the part is thick there.) With a gap of
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
SPREAD_VOXELS = 2  # the extra depth also applies this many grid nodes around


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
    # Half the wall thickness up to which a gap may need closing; thicker: no fill.
    limit = depth_grid[index].astype(np.float64) + fill
    half = np.full(len(d0), np.inf)
    step = 0.5 * h

    def work(lo: int, hi: int) -> None:
        p = start[lo:hi]
        # Inward direction: against the gradient of the distance field.
        gradient = np.stack([body.sample(p + 0.5 * h * e) - body.sample(p - 0.5 * h * e)
                             for e in np.eye(3)], axis=1)
        inward = -gradient / np.maximum(np.linalg.norm(gradient, axis=1), 1e-12)[:, None]
        behind = np.maximum(-d0[lo:hi], 0.0)  # how far the surface behind the point is
        # The ray can stop once the wall would be thicker than 2 * limit.
        travel = np.maximum(2.0 * limit[lo:hi] - behind, 0.0) + step
        out = np.full(hi - lo, np.inf)
        previous = d0[lo:hi].copy()
        active = np.arange(hi - lo)
        t = step
        while len(active):
            value = body.sample(p[active] + inward[active] * t)
            left = value >= 0  # out through the far side: the wall's thickness is known
            if left.any():
                i = active[left]
                was = previous[i]
                fraction = np.clip(-was / np.maximum(value[left] - was, 1e-12), 0.0, 1.0)
                out[i] = behind[i] + t - step + fraction * step
            previous[active] = value
            active = active[~left & (t + step <= travel[active])]
            t += step
        half[lo:hi] = 0.5 * out

    run_chunks(work, len(d0), 100_000)
    thin = np.isfinite(half)
    gap = np.where(thin, half - depth_grid[index], 0.0)
    # Half-width of the gap that is left: overlapping by a voxel (closed) up to CLOSE
    # voxels, the full gap from OPEN voxels on, steeply in between so that the narrow tip
    # of a closing gap covers as little of the part as possible.
    close, open_ = CLOSE_VOXELS * h, OPEN_VOXELS * h
    left = np.clip(-h + (gap - close) * (open_ + h) / (open_ - close), -h, gap)
    extra = np.where(thin, np.clip(gap - left, 0.0, None), 0.0).astype(np.float32)
    if not (extra > 0).any():
        return None
    out = np.zeros(d.shape, dtype=np.float32)
    out[index] = extra
    # A point a little closer to a side face than to the wall's faces looks along the wall
    # and misses it, leaving a strip of gap along the wall's side; spreading the extra
    # depth by SPREAD_VOXELS closes that strip. Thick parts have nothing to spread.
    return ndimage.maximum_filter(out, size=2 * SPREAD_VOXELS + 1)


def sample_extra(extra: np.ndarray | None, grid, points: np.ndarray) -> np.ndarray | float:
    """The extra depth at world points (trilinear, as the grid is sampled elsewhere)."""
    if extra is None:
        return 0.0
    return ndimage.map_coordinates(extra, grid.to_index(points).T, order=1, mode="nearest")
