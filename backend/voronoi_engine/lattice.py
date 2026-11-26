"""Interior lattice: Poisson-disk seeds and the edges of their 3D Voronoi diagram."""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from .body_field import BodyField
from .surface import sample_surface
from .parallel import kd_workers

# Seeds are also placed in a layer this many cell sizes outside the body, so that
# every Voronoi edge that reaches the surface is finite and continues through it.
GHOST_LAYER_CELLS = 2.0
MAX_ROUNDS = 12
MAX_CANDIDATES_PER_ROUND = 1_500_000


def dart_throw(candidates, spacing: float, scale=None) -> np.ndarray:
    """Poisson-disk sampling by rounds of dart throwing.

    `candidates()` returns a fresh batch of random points in the region. A point is
    accepted unless an already accepted point is closer than its local spacing:
    `spacing`, times `scale(points)` when the spacing varies (grading). Within a
    batch, points are taken in order. Rounds stop when they hardly add anything.
    """
    accepted = np.empty((0, 3))
    for _ in range(MAX_ROUNDS):
        points = candidates()
        radius = np.full(len(points), spacing) if scale is None else spacing * scale(points)
        added = 0
        # Points with similar spacing are handled together, widest first, so the
        # neighbour search radius always fits the local density.
        for band in _spacing_bands(radius):
            batch, batch_radius = points[band], radius[band]
            if len(accepted) and len(batch):
                distance, _ = cKDTree(accepted).query(batch, k=1, workers=kd_workers())
                free = distance >= batch_radius
                batch, batch_radius = batch[free], batch_radius[free]
            if len(batch) == 0:
                continue
            chosen = batch[_first_come_independent_set(batch, batch_radius)]
            accepted = np.vstack([accepted, chosen])
            added += len(chosen)
        if added < 0.005 * max(len(accepted), 1):
            break
    return accepted


def _spacing_bands(radius: np.ndarray) -> list[np.ndarray]:
    """Boolean masks grouping points whose spacing is within a factor of two."""
    if len(radius) == 0:
        return []
    top, bottom = float(radius.max()), float(radius.min())
    if top <= 1.5 * bottom:
        return [np.ones(len(radius), dtype=bool)]
    bands = []
    while top > bottom * (1 - 1e-12):
        bands.append((radius <= top) & (radius > top / 2.0))
        top /= 2.0
    return bands


def surface_proxy(tri: np.ndarray, spacing: float) -> cKDTree:
    """KD-tree with one point per `spacing`-sized cell the surface passes through.
    Built from the input triangles only, so it is the same at every voxel size."""
    points, _ = sample_surface(tri, spacing)
    cells = np.floor(points / spacing).astype(np.int64)
    # Unique rows via one packed integer per cell (much faster than unique(axis=0)).
    low = cells.min(axis=0)
    span = cells.max(axis=0) - low + 1
    packed = np.unique(((cells[:, 0] - low[0]) * span[1] + (cells[:, 1] - low[1])) * span[2]
                       + (cells[:, 2] - low[2]))
    cells = np.stack([packed // (span[1] * span[2]), (packed // span[2]) % span[1],
                      packed % span[2]], axis=1) + low
    return cKDTree((cells + 0.5) * spacing)


def poisson_seeds(body: BodyField, lo: np.ndarray, hi: np.ndarray, spacing: float,
                  rng: np.random.Generator, scale=None,
                  scale_range: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Random points no closer than the local spacing to each other, filling the body
    and a ghost layer around it as densely as random dart throwing allows.

    `scale(points)` multiplies the spacing locally (grading); `scale_range` is its
    (min, max). The result does not depend on the voxel size, so preview and final
    get the same seeds: the edge of the region is decided by points sampled on the
    input triangles, never by the grid.
    """
    smallest = spacing * scale_range[0]
    margin = GHOST_LAYER_CELLS * spacing * scale_range[1]
    box_lo, box_hi = np.asarray(lo) - margin, np.asarray(hi) + margin
    volume = float(np.prod(box_hi - box_lo))
    per_round = int(np.clip(6.0 * volume / smallest**3, 2_000, MAX_CANDIDATES_PER_ROUND))
    near_surface = surface_proxy(body.surface.tri, spacing / 4.0)

    def candidates() -> np.ndarray:
        points = rng.uniform(box_lo, box_hi, size=(per_round, 3))
        if scale is not None:
            # Fewer candidates where cells are larger: density follows 1 / spacing^3.
            points = points[rng.random(len(points)) < scale(points) ** -3.0 * scale_range[0] ** 3.0]
        distance, _ = near_surface.query(points, k=1, distance_upper_bound=margin, workers=kd_workers())
        # Points deeper inside than the ghost layer are well away from the surface,
        # where the sign of the grid field is the same at every voxel size.
        return points[np.isfinite(distance) | (body.sample(points) < 0)]

    return dart_throw(candidates, spacing, scale)


def _first_come_independent_set(points: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Greedy selection in index order (a point is kept unless an earlier kept point
    is closer than its own radius), computed without a Python loop over points."""
    pairs = cKDTree(points).query_pairs(float(radius.max()), output_type="ndarray")  # i < j
    lower, upper = pairs[:, 0], pairs[:, 1]
    if len(pairs) and radius.min() < radius.max():
        close = np.linalg.norm(points[lower] - points[upper], axis=1) < radius[upper]
        lower, upper = lower[close], upper[close]
    undecided, kept = 0, 1
    state = np.zeros(len(points), dtype=np.int8)
    while (state == undecided).any():
        open_pair = (state[lower] == undecided) & (state[upper] == undecided)
        waiting = np.zeros(len(points), dtype=bool)
        waiting[upper[open_pair]] = True  # an earlier neighbour is still undecided
        newly = (state == undecided) & ~waiting
        state[newly] = kept
        state[upper[(state[lower] == kept) & (state[upper] == undecided)]] = 2
    return state == kept


def voronoi_edges(seeds: np.ndarray) -> np.ndarray:
    """Finite edges of the 3D Voronoi diagram, shape (E, 2, 3).

    A Voronoi edge joins the circumcentres of two Delaunay tetrahedra that share a face.
    """
    if len(seeds) < 5:
        return np.empty((0, 2, 3))
    delaunay = Delaunay(seeds)
    tets = seeds[delaunay.simplices]
    a = tets[:, 0]
    ba, ca, da = tets[:, 1] - a, tets[:, 2] - a, tets[:, 3] - a
    det = np.einsum("ij,ij->i", ba, np.cross(ca, da))
    scale = np.maximum(np.einsum("ij,ij->i", ba, ba), 1e-300) ** 1.5
    good = np.abs(det) > 1e-9 * scale  # flat tetrahedra have no usable circumcentre
    sq = lambda v: np.einsum("ij,ij->i", v, v)[:, None]  # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):
        centre = a + (sq(ba) * np.cross(ca, da) + sq(ca) * np.cross(da, ba)
                      + sq(da) * np.cross(ba, ca)) / (2.0 * det[:, None])

    first = np.repeat(np.arange(len(tets)), 4)
    second = delaunay.neighbors.ravel()
    pair = (second > first) & good[first] & good[np.maximum(second, 0)]
    edges = np.stack([centre[first[pair]], centre[second[pair]]], axis=1)
    length = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    return edges[length > 1e-6]


def clip_edges_to_box(edges: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Liang-Barsky clipping of segments to an axis-aligned box."""
    p, d = edges[:, 0], edges[:, 1] - edges[:, 0]
    t0, t1 = np.zeros(len(edges)), np.ones(len(edges))
    keep = np.ones(len(edges), dtype=bool)
    for axis in range(3):
        moving = d[:, axis] != 0
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (lo[axis] - p[:, axis]) / d[:, axis]
            tb = (hi[axis] - p[:, axis]) / d[:, axis]
        t0 = np.where(moving, np.maximum(t0, np.minimum(ta, tb)), t0)
        t1 = np.where(moving, np.minimum(t1, np.maximum(ta, tb)), t1)
        keep &= moving | ((p[:, axis] >= lo[axis]) & (p[:, axis] <= hi[axis]))
    keep &= t0 < t1
    p, d, t0, t1 = p[keep], d[keep], t0[keep], t1[keep]
    return np.stack([p + d * t0[:, None], p + d * t1[:, None]], axis=1)


def clip_edges_to_body(edges: np.ndarray, body: BodyField, reach: float) -> np.ndarray:
    """Keep the part of each edge that comes within `reach` of the body's inside."""
    if len(edges) == 0:
        return edges
    h = body.grid.h
    p, d = edges[:, 0], edges[:, 1] - edges[:, 0]
    count = np.ceil(np.linalg.norm(d, axis=1) / h).astype(np.int64) + 1
    edge = np.repeat(np.arange(len(edges)), count)
    start = np.cumsum(count) - count
    step = np.arange(len(edge)) - np.repeat(start, count)
    t = step / np.repeat(count - 1, count)
    close = body.sample(p[edge] + d[edge] * t[:, None]) < reach
    first = np.minimum.reduceat(np.where(close, step, np.iinfo(np.int64).max), start)
    last = np.maximum.reduceat(np.where(close, step, -1), start)
    keep = last >= 0
    t0 = np.clip((first - 1) / (count - 1), 0.0, 1.0)[keep]
    t1 = np.clip((last + 1) / (count - 1), 0.0, 1.0)[keep]
    p, d = p[keep], d[keep]
    return np.stack([p + d * t0[:, None], p + d * t1[:, None]], axis=1)


def stretched_voronoi_edges(seeds: np.ndarray, stretch: float = 1.0) -> np.ndarray:
    """Voronoi edges with the cells stretched along Z by `stretch`: computed in a space
    squashed along Z, then stretched back. The seeds stay where they are, so each cell
    keeps its volume but becomes taller and narrower, and its edges steeper."""
    if stretch == 1.0:
        return voronoi_edges(seeds)
    squashed = seeds.copy()
    squashed[:, 2] /= stretch
    edges = voronoi_edges(squashed)
    edges[:, :, 2] *= stretch
    return edges


def interior_edges(body: BodyField, lo: np.ndarray, hi: np.ndarray, cell_size: float,
                   reach: float, rng: np.random.Generator, scale=None,
                   scale_range: tuple[float, float] = (1.0, 1.0), stretch: float = 1.0) -> np.ndarray:
    """Voronoi edges of fresh Poisson-disk seeds, clipped to the body. Shape (E, 2, 3)."""
    seeds = poisson_seeds(body, lo, hi, cell_size, rng, scale, scale_range)
    edges = stretched_voronoi_edges(seeds, stretch)
    edges = clip_edges_to_box(edges, np.asarray(lo) - reach, np.asarray(hi) + reach)
    return clip_edges_to_body(edges, body, reach)
