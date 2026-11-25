"""Print orientation: turn the part so a chosen face lies on the print bed.

The bed face is given by its outward normal in the coordinates of the loaded model.
The part is turned (smallest rotation) so that this normal points straight down,
about the centre of its bounding box, and moved up or down so that its lowest point
is where the lowest point was before. Everything after that (voxel grid, grading
along Z, solid base, overhangs, export) works in this print orientation, with +Z up.

The default, [0, 0, -1], is the orientation as loaded and leaves the part untouched.
The vertex and triangle numbering does not change, so kept-edge numbers stay valid.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .mesh_io import LoadedMesh, MeshInfo

DOWN = np.array([0.0, 0.0, -1.0])


def is_as_loaded(bed_normal) -> bool:
    return bool(np.allclose(np.asarray(bed_normal, dtype=np.float64), DOWN, atol=1e-9))


def bed_rotation(bed_normal) -> np.ndarray:
    """Smallest rotation that turns `bed_normal` to point straight down (3 x 3)."""
    n = np.asarray(bed_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    c = float(n @ DOWN)
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:  # facing straight up: turn over about X
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(n, DOWN)
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


def orient(vertices: np.ndarray, bed_normal, bounds_min, bounds_max) -> np.ndarray:
    """Vertices turned into the print orientation."""
    if is_as_loaded(bed_normal):
        return vertices
    lo, hi = np.asarray(bounds_min, dtype=np.float64), np.asarray(bounds_max, dtype=np.float64)
    centre = 0.5 * (lo + hi)
    turned = (vertices - centre) @ bed_rotation(bed_normal).T + centre
    turned[:, 2] += lo[2] - turned[:, 2].min()
    return turned


def oriented(mesh: LoadedMesh, bed_normal) -> LoadedMesh:
    """The loaded model in the print orientation (the same object when as loaded)."""
    if is_as_loaded(bed_normal):
        return mesh
    vertices = orient(mesh.vertices, bed_normal, mesh.info.bounds_min, mesh.info.bounds_max)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    info = replace(mesh.info, size_mm=tuple(float(v) for v in hi - lo),
                   bounds_min=tuple(float(v) for v in lo), bounds_max=tuple(float(v) for v in hi))
    return LoadedMesh(name=mesh.name, vertices=vertices, faces=mesh.faces, info=info,
                      warnings=list(mesh.warnings))


# --- suggesting a bed face ---------------------------------------------------------------

FLAT_DEG = 0.5
MIN_FACE_SHARE = 0.005  # flat faces smaller than this share of the surface are not offered
MAX_CANDIDATES = 64
ON_BED_MM = 0.1


def suggest_bed_normal(mesh: LoadedMesh, overhang_deg: float = 45.0) -> dict:
    """Of the model's flat faces (and the orientation as loaded), the one to put on the bed
    so that the least of the outer surface overhangs; on a tie, the largest contact area.

    Returns {"normal", "overhang_mm2", "contact_mm2", "as_loaded_overhang_mm2"}.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from .edges import _MeshEdges

    v, f = mesh.vertices, mesh.faces
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normal = cross / np.maximum(2 * area, 1e-300)[:, None]

    # Flat faces: triangles joined across edges where they are (almost) coplanar.
    edges = _MeshEdges(v, f)
    flat = edges.angle < np.radians(FLAT_DEG)
    count, piece = connected_components(
        coo_matrix((np.ones(int(flat.sum()), dtype=bool), (edges.f0[flat], edges.f1[flat])),
                   shape=(len(f), len(f))), directed=False)
    piece_area = np.bincount(piece, area, count)
    piece_normal = np.zeros((count, 3))
    for k in range(3):
        piece_normal[:, k] = np.bincount(piece, normal[:, k] * area, count)
    piece_normal /= np.maximum(np.linalg.norm(piece_normal, axis=1), 1e-300)[:, None]
    big = np.flatnonzero(piece_area >= MIN_FACE_SHARE * area.sum())
    big = big[np.argsort(-piece_area[big], kind="stable")][:MAX_CANDIDATES]
    candidates = [DOWN] + [piece_normal[i] for i in big]

    cos_limit = np.cos(np.radians(overhang_deg))
    best = None
    as_loaded = None
    for n in candidates:
        rotation = bed_rotation(n)
        z = (v @ rotation.T)[:, 2]
        nz = normal @ rotation.T[:, 2]
        lowest = z[f].min(axis=1)
        on_bed = (nz < -0.999) & (lowest <= z.min() + ON_BED_MM)
        overhang = (nz < -cos_limit) & ~on_bed & (lowest > z.min() + ON_BED_MM)
        score = (round(float(area[overhang].sum()), 3), -round(float(area[on_bed].sum()), 3),
                 round(float(np.ptp(z)), 3))
        entry = {"normal": [round(float(x), 9) + 0.0 for x in n], "overhang_mm2": score[0],
                 "contact_mm2": -score[1]}
        if as_loaded is None:
            as_loaded = entry
        if entry["contact_mm2"] > 0 and (best is None or score < best[0]):
            best = (score, entry)
    chosen = best[1] if best else as_loaded
    return {**chosen, "as_loaded_overhang_mm2": as_loaded["overhang_mm2"]}
