"""The generated solid as a function of position (negative inside).

The grid pipeline in generate.py evaluates the same construction on voxel nodes;
this evaluates it at arbitrary points with the exact ingredients, for smoothing.
A test checks that the two agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .body_field import BodyField
from .edge_members import (blend_used, edge_rounding_range, member_depth, member_field, member_in_skin,
                           member_reach, rounding_near_edges)
from .grading import Grading
from .settings import Settings
from .strut_field import StrutEvaluator, smooth_max, smooth_min
from .thin_core import sample_extra
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
    extra: np.ndarray | None = None  # extra skin depth in thin walls (thin_core)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        s, body, h = self.settings, self.body, self.body.grid.h
        surface, interior = s.surface_style, s.interior_style
        blend = blend_used(s, h)  # as in generate

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
                skin_depth = depth + sample_extra(self.extra, body.grid, points)
                inner = body.sample_inner(points)  # as in generate: blurred, for inner faces
                shell = np.maximum(d, -(inner + skin_depth))
                skin = shell
                if surface != "solid":
                    rounding = edge_rounding_range(s, h)[0]
                    in_skin = self.keep is not None and member_in_skin(surface)
                    widest = edge_rounding_range(s, h)[1] if in_skin else rounding
                    band = (d > -(skin_depth + 2 * h + widest)) & (d < 2 * h)
                    band_rounding = rounding  # a number, or one radius per band point
                    wall = np.zeros(len(points))
                    if band.any():
                        p = points[band]
                        local_cell = s.surface_cell_size * self.grading.cell_scale(
                            p, depth=depth_below[band])
                        pointed = s.pointed_holes and surface == "perforated"  # as in generate
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
                        if in_skin:
                            from .generate import with_kept_edges
                            reach = member_reach(s, h, edge_rounding_range(s, h)[1])
                            # Only points near a kept edge need the (slow) distance.
                            to_edge = np.full(len(p), reach)
                            close = self.keep.near(p)
                            if close.any():
                                to_edge[close] = np.minimum(self.keep(p[close]), reach)
                            band_rounding = rounding_near_edges(to_edge, s, h)
                            wall[band] = with_kept_edges(wall[band], inner[band], to_edge, s,
                                                         skin_depth[band] if np.ndim(skin_depth) else skin_depth,
                                                         reach, band_rounding, blend)
                    if surface == "embossed":
                        wall[band] = smooth_min(wall[band], d[band] + s.emboss_depth, band_rounding)
                    skin = shell.copy()
                    skin[band] = smooth_max(shell[band], wall[band], band_rounding)

        if interior == "solid":
            if skin is None or surface == "solid":
                solid = d.copy()
            else:
                inner = body.sample_inner(points)
                solid = np.minimum(skin, inner + depth)
                rounding = edge_rounding_range(s, h)[0]
                band = (d > -(depth + 2 * h + rounding)) & (d < 2 * h)
                core = inner[band] + (depth[band] if np.ndim(depth) else depth)
                solid[band] = smooth_min(skin[band], core, rounding)
        elif interior == "struts":
            solid = np.full(len(points), self.strut_margin)
            if self.struts is not None:
                close = (d < h) & self.struts.near(points)
                if close.any():
                    solid[close] = np.minimum(self.struts(points[close]), self.strut_margin)
            if surface in ("perforated", "rib_lattice"):
                solid = smooth_max(solid, body.sample_inner(points) + 0.5 * depth, blend)
            elif surface == "embossed":
                solid = smooth_max(solid, body.sample_inner(points) + s.emboss_depth
                                   + 0.5 * (depth - s.emboss_depth), blend)
            if skin is not None:
                solid = smooth_min(skin, solid, blend)
        else:
            solid = skin

        if self.keep is not None and not member_in_skin(surface):
            depth_e = member_depth(s)
            reach = member_reach(s, h, edge_rounding_range(s, h)[1])  # as in generate
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
