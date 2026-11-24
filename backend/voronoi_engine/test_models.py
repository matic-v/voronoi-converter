"""Test models generated in code (no binary assets in the repository).

All models are in millimetres, rest on the bed (min Z = 0) and are centred in X/Y.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import trimesh


def cube(size: float = 50.0) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=(size, size, size))
    return _place(mesh)


def sphere(diameter: float = 60.0, segments: int = 64, rings: int = 32) -> trimesh.Trimesh:
    # UV sphere with an equator ring and a segment count divisible by 4, so vertices
    # lie exactly on every axis and the bounding box is exactly the diameter.
    r = diameter / 2.0
    lat = np.pi * np.arange(1, rings) / rings  # polar angle of each inner ring
    lon = 2.0 * np.pi * np.arange(segments) / segments
    ring_xy = np.column_stack([np.cos(lon), np.sin(lon)])
    ring_xy[np.abs(ring_xy) < 1e-12] = 0.0
    vertices = [(0.0, 0.0, r)]
    for theta in lat:
        s, c = np.sin(theta), np.cos(theta)
        c = 0.0 if abs(c) < 1e-12 else c
        vertices += [(r * s * x, r * s * y, r * c) for x, y in ring_xy]
    vertices.append((0.0, 0.0, -r))
    south = len(vertices) - 1
    faces = []
    for j in range(segments):
        k = (j + 1) % segments
        faces.append((0, 1 + j, 1 + k))
        last = 1 + (rings - 2) * segments
        faces.append((south, last + k, last + j))
        for i in range(rings - 2):
            a, b = 1 + i * segments, 1 + (i + 1) * segments
            faces += [(a + j, b + j, b + k), (a + j, b + k, a + k)]
    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)
    return _place(mesh)


def bracket(length: float = 100.0, width: float = 40.0, height: float = 60.0,
            wall: float = 8.0, flange_length: float = 40.0,
            flange_thickness: float = 2.0) -> trimesh.Trimesh:
    """L-bracket: thick upright + thick base that steps down to a thin flange.

    Profile in the X/Z plane (extruded along Y)::

        z=height  +--+
                  |  |
                  |  |
        z=wall    |  +-----------+
                  |              +----------+  z=flange_thickness
        z=0       +-------------------------+
                  x=0  wall   length-flange  length
    """
    step_x = length - flange_length
    profile = np.array([
        (0.0, 0.0),
        (length, 0.0),
        (length, flange_thickness),
        (step_x, flange_thickness),
        (step_x, wall),
        (wall, wall),
        (wall, height),
        (0.0, height),
    ])
    return _place(_extrude_profile_xz(profile, width))


def filleted_bracket(width: float = 40.0, segments: int = 12) -> trimesh.Trimesh:
    """L-bracket 60 x 40 x 40 mm with an 8 mm round on the outer corner and a 5 mm
    fillet in the inner corner (each arc in `segments` facets): a model with curved edges."""

    def arc(cx: float, cz: float, r: float, start: float, end: float) -> list[tuple[float, float]]:
        angles = np.radians(np.linspace(start, end, segments + 1))
        return [(cx + r * np.cos(t), cz + r * np.sin(t)) for t in angles]

    profile = [(8.0, 0.0), (60.0, 0.0), (60.0, 8.0)]
    profile += arc(13.0, 13.0, 5.0, 270.0, 180.0)  # inner fillet, (13, 8) -> (8, 13)
    profile += [(8.0, 40.0), (0.0, 40.0)]
    profile += arc(8.0, 8.0, 8.0, 180.0, 270.0)[:-1]  # outer round, (0, 8) -> (8, 0)
    points = np.array(profile)
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9  # no repeated points
    return _place(_extrude_profile_xz(points[keep], width))


TEST_MODELS: dict[str, tuple[str, Callable[[], trimesh.Trimesh]]] = {
    "cube_50": ("cube 50", cube),
    "sphere_60": ("sphere 60", sphere),
    "bracket_100": ("bracket 100", bracket),
}


# --- deliberately broken models, to exercise input validation ---------------

def broken_open_cube(size: float = 50.0) -> trimesh.Trimesh:
    """Cube with its top face missing: 4 open edges."""
    mesh = cube(size)
    top = mesh.triangles_center[:, 2] > size - 1e-6
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[~top], process=False)


def broken_non_manifold(size: float = 30.0) -> trimesh.Trimesh:
    """Two cubes touching along one vertical edge: that edge is shared by 4 faces."""
    first = trimesh.creation.box(extents=(size, size, size))
    second = first.copy()
    second.apply_translation((size, size, 0.0))
    return _place(trimesh.util.concatenate([first, second]))


def broken_flipped_face(size: float = 50.0) -> trimesh.Trimesh:
    """Cube with one triangle wound the wrong way round."""
    mesh = cube(size)
    faces = mesh.faces.copy()
    faces[0] = faces[0][::-1]
    return trimesh.Trimesh(vertices=mesh.vertices, faces=faces, process=False)


# Models for trying out features; not part of every acceptance test.
FEATURE_MODELS: dict[str, tuple[str, Callable[[], trimesh.Trimesh]]] = {
    "filleted_bracket": ("filleted bracket", filleted_bracket),
}

BROKEN_MODELS: dict[str, tuple[str, Callable[[], trimesh.Trimesh]]] = {
    "broken_open_cube": ("open cube", broken_open_cube),
    "broken_non_manifold": ("non-manifold", broken_non_manifold),
    "broken_flipped_face": ("flipped face", broken_flipped_face),
}


def build(name: str) -> trimesh.Trimesh:
    for models in (TEST_MODELS, FEATURE_MODELS, BROKEN_MODELS):
        if name in models:
            return models[name][1]()
    raise KeyError(f"unknown test model '{name}'")


def to_binary_stl(mesh: trimesh.Trimesh) -> bytes:
    return mesh.export(file_type="stl")


# --- helpers ---------------------------------------------------------------

def _place(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    lo, hi = mesh.bounds
    mesh.apply_translation((-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]))
    return mesh


def _extrude_profile_xz(profile: np.ndarray, width: float) -> trimesh.Trimesh:
    """Extrude a simple (possibly concave) polygon given in X/Z along Y."""
    n = len(profile)
    triangles = _ear_clip(profile)
    front = np.column_stack([profile[:, 0], np.zeros(n), profile[:, 1]])
    back = front + (0.0, width, 0.0)
    vertices = np.vstack([front, back])
    faces = [tri for tri in triangles]
    faces += [(a + n, c + n, b + n) for a, b, c in triangles]
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j + n, j), (i, i + n, j + n)]
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _ear_clip(points: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangulate a simple polygon; returns index triples."""

    def cross(o, a, b) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    area = sum(points[i][0] * points[(i + 1) % len(points)][1]
               - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points)))
    order = list(range(len(points))) if area > 0 else list(range(len(points)))[::-1]
    triangles: list[tuple[int, int, int]] = []
    while len(order) > 3:
        for k in range(len(order)):
            ia, ib, ic = order[k - 1], order[k], order[(k + 1) % len(order)]
            a, b, c = points[ia], points[ib], points[ic]
            if cross(a, b, c) <= 1e-12:
                continue  # reflex or degenerate corner
            others = (points[i] for i in order if i not in (ia, ib, ic))
            if any(cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0
                   for p in others):
                continue  # another vertex lies inside this ear
            triangles.append((ia, ib, ic))
            del order[k]
            break
        else:
            raise ValueError("polygon is not simple; cannot triangulate")
    triangles.append((order[0], order[1], order[2]))
    return triangles
