"""The generated solid as a function of position (negative inside).

The grid pipeline in generate.py evaluates the same construction on voxel nodes;
this evaluates it at arbitrary points with the exact ingredients, for smoothing.
A test checks that the two agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .body_field import BodyField
from .edge_members import member_depth, member_field, member_reach
from .grading import Grading
from .settings import Settings
from .strut_field import StrutEvaluator, smooth_max, smooth_min
from .surface_pattern import SurfacePattern, pointed_tops

HOLE_ROUNDING = 0.3
RECESS_ROUNDING = 0.15


@dataclass
class ExactField:
    settings: Settings
    body: BodyField
    grading: Grading
    pattern: SurfacePattern | None  # for surface styles A, B, C
    struts: StrutEvaluator | None  # for interior style Struts
    strut_margin: float
    base_top: float | None  # Z of the top of the solid base
    exact_depth: float = 0.0  # use the exact distance to the input down to this depth
    keep: StrutEvaluator | None = None  # distance to the kept edges
    keep_rounding: float = 0.0

    def __call__(self, points: np.ndarray) -> np.ndarray:
        s, body, h = self.settings, self.body, self.body.grid.h
        surface, interior = s.surface_style, s.interior_style
        blend = s.blend_radius

        # Distance to the input surface: exact near it and through the skin (so the
        # skin's inner face is smoothed too), from the grid further in.
        d = body.sample(points).astype(np.float64)
        near = (d < 1.5 * h) & (d > -max(1.5 * h, self.exact_depth))
        if near.any():
            d[near] = body.surface.signed_distance(points[near], k=6)
        depth_below = np.clip(-d, 0.0, None)
        scale = self.grading.thickness_scale(points, depth=depth_below)

        skin, depth = None, 0.0
        if surface != "none":
            nominal = s.rib_width if surface == "rib_lattice" else s.shell_thickness
            depth = nominal * scale
            if not (surface == "solid" and interior == "solid"):
                shell = np.maximum(d, -(d + depth))
                skin = shell
                if surface != "solid":
                    rounding = s.edge_rounding
                    band = (d > -(depth + 2 * h + rounding)) & (d < 2 * h)
                    wall = np.zeros(len(points))
                    if band.any():
                        p = points[band]
                        local_cell = s.surface_cell_size * self.grading.cell_scale(
                            p, depth=depth_below[band])
                        pointed = s.pointed_holes and surface in ("perforated", "rib_lattice")
                        if surface == "perforated":
                            fraction = s.hole_size / 100.0
                            wall[band] = self.pattern.wall_distance(
                                p, HOLE_ROUNDING * fraction * local_cell, inset=1.0 - fraction)
                        elif surface == "rib_lattice":
                            wall[band] = self.pattern.wall_distance(p, blend) \
                                - 0.5 * s.rib_width * scale[band]
                        else:  # embossed
                            wall[band] = self.pattern.wall_distance(p, RECESS_ROUNDING * local_cell) \
                                - 0.5 * s.rib_width * scale[band]
                        if pointed:
                            from .generate import opening_inradius
                            wall[band] = pointed_tops(self.pattern, p, wall[band],
                                                      opening_inradius(self.pattern, s),
                                                      s.overhang_angle)
                    if surface == "embossed":
                        wall[band] = smooth_min(wall[band], d[band] + s.emboss_depth, rounding)
                    skin = shell.copy()
                    skin[band] = smooth_max(shell[band], wall[band], rounding)

        if interior == "solid":
            if skin is None or surface == "solid":
                solid = d.copy()
            else:
                solid = np.minimum(skin, d + depth)
                band = (d > -(depth + 2 * h + s.edge_rounding)) & (d < 2 * h)
                core = d[band] + (depth[band] if np.ndim(depth) else depth)
                solid[band] = smooth_min(skin[band], core, s.edge_rounding)
        elif interior == "struts":
            solid = np.full(len(points), self.strut_margin)
            if self.struts is not None:
                close = (d < h) & self.struts.near(points)
                if close.any():
                    solid[close] = np.minimum(self.struts(points[close]), self.strut_margin)
            if surface in ("perforated", "rib_lattice"):
                solid = np.maximum(solid, d + 0.5 * depth)
            elif surface == "embossed":
                solid = np.maximum(solid, d + s.emboss_depth + 0.5 * (depth - s.emboss_depth))
            if skin is not None:
                solid = smooth_min(skin, solid, blend)
        else:
            solid = skin

        if self.keep is not None:
            depth_e = member_depth(s)
            reach = member_reach(s, h, self.keep_rounding)
            close = (d > -(depth_e + 2 * h + self.keep_rounding + blend)) & (d < 2 * h)
            to_edge = np.full(len(points), reach)
            to_edge[close] = np.minimum(self.keep(points[close]), reach)
            close &= to_edge < reach
            if close.any():
                members = member_field(d[close], to_edge[close], s.keep_width, depth_e,
                                       self.keep_rounding)
                solid[close] = smooth_min(solid[close], members, blend)

        if self.base_top is not None:
            solid = np.minimum(solid, points[:, 2] - self.base_top)
        return np.maximum(solid, d)
