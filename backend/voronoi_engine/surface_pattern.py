"""Voronoi pattern on the skin: surface seeds and the Worley-type fields built from them.

For a point x with nearest seed s1 and another seed sj, the distance from x to the
wall between their two cells (the bisector plane) is

    e_j = (|x - sj|^2 - |x - s1|^2) / (2 |s1 - sj|)

Near the middle of a wall this equals (F2 - F1) / 2 of the classic Worley field, but
unlike F2 - F1 it is a true distance, so ribs get a constant width.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .lattice import dart_throw
from .parallel import kd_workers, run_chunks

CANDIDATES_PER_SEED = 8.0
NEIGHBOURS = 6  # nearest seed + 5 others: enough around any cell corner
CHUNK = 400_000


def surface_seeds(tri: np.ndarray, spacing: float, rng: np.random.Generator, scale=None,
                  scale_range: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Poisson-disk points on the surface (straight-line distance >= local spacing).

    `scale(points)` multiplies the spacing locally (grading); `scale_range` is its (min, max).
    """
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    cumulative = np.cumsum(area)
    total = float(cumulative[-1])
    smallest = spacing * scale_range[0]
    per_round = int(np.clip(CANDIDATES_PER_SEED * total / smallest**2, 1_000, 2_000_000))

    def candidates() -> np.ndarray:
        face = np.searchsorted(cumulative, rng.random(per_round) * total)
        face = np.minimum(face, len(tri) - 1)
        r1, r2 = np.sqrt(rng.random(per_round)), rng.random(per_round)
        a, b, c = tri[face, 0], tri[face, 1], tri[face, 2]
        points = a * (1 - r1)[:, None] + b * (r1 * (1 - r2))[:, None] + c * (r1 * r2)[:, None]
        if scale is not None:
            # Fewer candidates where cells are larger: density follows 1 / spacing^2.
            points = points[rng.random(per_round) < scale(points) ** -2.0 * scale_range[0] ** 2.0]
        return points

    return dart_throw(candidates, spacing, scale)


class SurfacePattern:
    def __init__(self, seeds: np.ndarray):
        self.seeds = seeds
        self.tree = cKDTree(seeds)
        self.k = min(NEIGHBOURS, len(seeds))

    def wall_distance(self, points: np.ndarray, rounding, inset: float = 0.0) -> np.ndarray:
        """Distance to the nearest cell wall, with cell corners rounded by `rounding`
        (one value, or one per point).

        With `inset` (0..1) every wall is first moved towards the cell's seed by that
        fraction of the way, i.e. the value is measured from the cell scaled about its
        seed by (1 - inset). Positive inside the (scaled) cell, negative outside.
        """
        out = np.empty(len(points))

        def work(start: int, stop: int) -> None:
            block = points[start:stop]
            distance, index = self.tree.query(block, k=self.k, workers=kd_workers())
            own = self.seeds[index[:, 0]]
            walls = np.empty((len(block), self.k - 1))
            for j in range(1, self.k):
                gap = np.linalg.norm(self.seeds[index[:, j]] - own, axis=1)
                wall = (distance[:, j] ** 2 - distance[:, 0] ** 2) / (2.0 * gap)
                walls[:, j - 1] = wall - inset * 0.5 * gap
            walls.sort(axis=1)
            # Smooth minimum folded from the nearest wall: rounds the cell corners.
            value = walls[:, 0].copy()
            r = rounding[start:stop] if np.ndim(rounding) else rounding
            if np.max(r) > 0:
                r = np.maximum(r, 1e-9)
                for j in range(1, self.k - 1):
                    weight = np.clip(r - (walls[:, j] - value), 0.0, None) / r
                    value -= weight * weight * (r / 4.0)
            out[start:stop] = value
        run_chunks(work, len(points), CHUNK)
        return out


STEEP_WALL_NZ = 0.9  # surfaces flatter than this (|normal z| above it) get no pointed tops


def prepare_pointed_tops(pattern: SurfacePattern, surface_normals: np.ndarray) -> None:
    """Store, per seed, the outward normal of the surface there and the distance to the
    nearest other seed (needed by pointed_tops)."""
    pattern.normal = surface_normals
    distance, _ = pattern.tree.query(pattern.seeds, k=2)
    pattern.neighbour = distance[:, 1]


def pointed_tops(pattern: SurfacePattern, points: np.ndarray, opening: np.ndarray,
                 inradius: np.ndarray, overhang_deg: float) -> np.ndarray:
    """Give every opening (hole or gap between ribs) on a steep wall a pointed top.

    `opening` is positive inside the opening. Above each cell's seed, two lines at the
    flattest printable angle (90 - overhang_deg from horizontal) meet in a point; they
    touch the circle of radius `inradius[seed]` round the seed (a teardrop shape), so
    the opening keeps its size but has no flat ceiling. Returns the new opening field.
    """
    _, seed = pattern.tree.query(points, k=1, workers=kd_workers())
    n = pattern.normal[seed]
    up = np.array([0.0, 0.0, 1.0])
    v_dir = up - n[:, 2:3] * n  # "up" within the wall
    v_len = np.linalg.norm(v_dir, axis=1)
    steep = (np.abs(n[:, 2]) <= STEEP_WALL_NZ) & (v_len > 1e-9)
    v_dir = v_dir / np.maximum(v_len, 1e-12)[:, None]
    u_dir = np.cross(n, v_dir)  # horizontal within the wall
    rel = points - pattern.seeds[seed]
    u = np.einsum("ij,ij->i", rel, u_dir)
    v = np.einsum("ij,ij->i", rel, v_dir)
    beta = np.radians(90.0 - overhang_deg)  # roof angle from horizontal
    apex = inradius[seed] / np.cos(beta)
    roof = (apex - v - np.abs(u) * np.tan(beta)) * np.cos(beta)  # positive below the roof
    return np.where(steep, np.minimum(opening, roof), opening)
