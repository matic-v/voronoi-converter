"""STL loading and validation.

Broken input is never repaired: it is refused with a list of what is wrong.
The only processing done on load is welding bit-identical vertices, which is how
an STL (a "triangle soup") is turned into a connected mesh.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

MAX_RECOMMENDED_SIZE_MM = 250.0
MAX_RECOMMENDED_TRIANGLES = 500_000
MIN_SIZE_MM = 0.01  # below this on any axis the mesh is flat / zero-sized
SMALL_PART_MM = 5.0

REPAIR_HINT = (
    "This tool does not repair or split models. Fix the file in another tool "
    "(for example Meshmixer, MeshLab or Microsoft 3D Builder) and load it again."
)


class MeshValidationError(Exception):
    """The input cannot be used. ``problems`` lists every defect found."""

    def __init__(self, name: str, problems: list[str]):
        self.name = name
        self.problems = problems
        super().__init__(self.message)

    @property
    def message(self) -> str:
        lines = [f"Cannot use '{self.name}':"]
        lines += [f"  • {p}" for p in self.problems]
        lines += ["", REPAIR_HINT]
        return "\n".join(lines)


@dataclass(frozen=True)
class MeshInfo:
    size_mm: tuple[float, float, float]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    triangles: int
    volume_mm3: float
    bodies: int


@dataclass
class LoadedMesh:
    name: str
    vertices: np.ndarray  # (V, 3) float64, welded
    faces: np.ndarray  # (F, 3) int64
    info: MeshInfo
    warnings: list[str] = field(default_factory=list)

    _edges: object = field(default=None, repr=False, compare=False)

    @property
    def edges(self):
        """Sharp edges of the model (see edges.py), found on first use."""
        if self._edges is None:
            from .edges import find_edges
            self._edges = find_edges(self.vertices, self.faces)
        return self._edges

    def to_trimesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)


def load_stl(data: bytes, name: str = "model.stl") -> LoadedMesh:
    """Parse a binary or ASCII STL and validate it. Raises MeshValidationError."""
    vertices, faces = _parse(data, name)
    vertices, faces = _weld(vertices, faces)
    problems, volume, bodies = analyse_topology(vertices, faces)
    if problems:
        raise MeshValidationError(name, problems)

    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    size = hi - lo
    info = MeshInfo(
        size_mm=tuple(float(v) for v in size),
        bounds_min=tuple(float(v) for v in lo),
        bounds_max=tuple(float(v) for v in hi),
        triangles=int(len(faces)),
        volume_mm3=float(volume),
        bodies=bodies,
    )
    warnings: list[str] = []
    if size.max() > MAX_RECOMMENDED_SIZE_MM:
        warnings.append(
            f"Part is {size.max():.0f} mm long; larger than the {MAX_RECOMMENDED_SIZE_MM:.0f} mm "
            "this tool is designed for, so generation may be slow or coarse."
        )
    if size.max() < SMALL_PART_MM:
        warnings.append(
            f"Part is only {size.max():.2f} mm long. Units are read as millimetres - "
            "check the export units of your CAD tool."
        )
    if len(faces) > MAX_RECOMMENDED_TRIANGLES:
        warnings.append(
            f"Mesh has {len(faces):,} triangles; more than the recommended "
            f"{MAX_RECOMMENDED_TRIANGLES:,}, so loading and generation may be slow."
        )
    return LoadedMesh(name=name, vertices=vertices, faces=faces, info=info, warnings=warnings)


# --- internals -------------------------------------------------------------

def _parse(data: bytes, name: str) -> tuple[np.ndarray, np.ndarray]:
    if len(data) == 0:
        raise MeshValidationError(name, ["the file is empty"])
    try:
        loaded = trimesh.load(io.BytesIO(data), file_type="stl", process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.to_geometry()
        vertices = np.asarray(loaded.vertices, dtype=np.float64)
        faces = np.asarray(loaded.faces, dtype=np.int64)
    except Exception as exc:  # trimesh raises many different types on bad files
        raise MeshValidationError(
            name, [f"the file could not be read as a binary or ASCII STL ({exc})"]
        ) from exc
    if len(faces) == 0 or len(vertices) == 0:
        raise MeshValidationError(name, ["the file contains no triangles"])
    return vertices, faces


def _weld(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices with identical (float32) coordinates."""
    v32 = np.ascontiguousarray(vertices, dtype=np.float32) + np.float32(0.0)  # -0.0 -> 0.0
    keys = v32.view(np.dtype((np.void, v32.dtype.itemsize * 3))).ravel()
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    return v32[first].astype(np.float64), inverse.reshape(-1)[faces].astype(np.int64)


def analyse_topology(vertices: np.ndarray, faces: np.ndarray) -> tuple[list[str], float, int]:
    """Return (problems, signed volume, number of bodies)."""
    problems: list[str] = []
    if not np.isfinite(vertices).all():
        return ["the file contains invalid coordinates (NaN or infinity)"], 0.0, 0

    size = vertices.max(axis=0) - vertices.min(axis=0)
    if size.min() < MIN_SIZE_MM:
        problems.append(
            "mesh is flat or has no size "
            f"(bounding box {size[0]:.3f} × {size[1]:.3f} × {size[2]:.3f} mm)"
        )

    # Triangles that collapsed to a line or point when vertices were welded carry no surface.
    collapsed = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    solid_faces = faces[~collapsed]
    if len(solid_faces) == 0:
        return problems + ["mesh has no triangles with a non-zero area"], 0.0, 0

    n = np.int64(len(vertices))
    a = solid_faces.ravel()
    b = solid_faces[:, [1, 2, 0]].ravel()
    directed = a * n + b
    undirected = np.minimum(a, b) * n + np.maximum(a, b)
    _, counts = np.unique(undirected, return_counts=True)
    open_edges = int((counts == 1).sum())
    non_manifold = int((counts > 2).sum())
    if open_edges:
        problems.append(
            f"mesh has {open_edges:,} open edge{_s(open_edges)} (holes in the surface; not watertight)"
        )
    if non_manifold:
        problems.append(
            f"mesh has {non_manifold:,} non-manifold edge{_s(non_manifold)} "
            "(more than two faces share an edge)"
        )

    volume = 0.0
    if not open_edges and not non_manifold:
        # In a consistently wound closed mesh every directed edge occurs exactly once.
        ordered = np.sort(directed)
        flipped = int((ordered[1:] == ordered[:-1]).sum())
        if flipped:
            problems.append(
                f"normals are inconsistent: {flipped:,} edge{_s(flipped)} join faces that point "
                "in opposite directions (some faces are flipped)"
            )
        else:
            tri = vertices[solid_faces]
            volume = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
            if volume < 0:
                problems.append("mesh is inside-out (normals point inwards, volume is negative)")
            elif volume < 1e-9:
                problems.append("mesh encloses no volume")

    graph = coo_matrix((np.ones(len(a), dtype=bool), (a, b)), shape=(int(n), int(n)))
    count, labels = connected_components(graph, directed=False)
    used = np.zeros(count, dtype=bool)
    used[labels[a]] = True  # ignore vertices that no triangle uses
    bodies = int(used.sum())
    if not problems and bodies > 1:
        # A closed shell wound inwards is a cavity inside a body, not a body of its own.
        tri = vertices[solid_faces]
        shell_volume = np.bincount(
            labels[solid_faces[:, 0]],
            weights=np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])),
            minlength=count)
        bodies = int((shell_volume > 0).sum())
        if bodies > 1:
            problems.append(
                f"file contains {bodies} separate bodies; only one body can be loaded at a time "
                "(save each body as its own STL file)"
            )
    return problems, volume, bodies


def _s(count: int) -> str:
    return "" if count == 1 else "s"
