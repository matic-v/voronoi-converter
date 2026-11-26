"""Validation and automatic repair of the generated mesh.

The pipeline is built so that marching cubes gives a closed, consistently wound
surface, so normally there is nothing to repair. These repairs are the safety net
required before export; if a defect survives them, the result is marked invalid and
must not be exported.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

from .mesh_io import analyse_topology


def weld_as_exported(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Round to the 32-bit coordinates an STL stores and merge vertices that then
    coincide, exactly as any program reading the file will. Triangles that collapse
    are dropped. Returns (vertices, faces, vertices merged).

    Everything after this step (validation, repair, display, export) sees the same
    mesh the exported file describes.
    """
    v32 = np.ascontiguousarray(vertices, dtype=np.float32) + np.float32(0.0)  # -0.0 -> 0.0
    keys = v32.view(np.dtype((np.void, 12))).ravel()
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    merged = len(v32) - len(first)
    # Keep the original vertex order (unique sorts by bytes): stable output, same faces.
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    new_vertices = v32[first[order]].astype(np.float64)
    new_faces = rank[inverse.reshape(-1)][np.asarray(faces, dtype=np.int64)]
    if merged:
        collapsed = (new_faces[:, 0] == new_faces[:, 1]) | (new_faces[:, 1] == new_faces[:, 2]) \
            | (new_faces[:, 0] == new_faces[:, 2])
        new_faces = new_faces[~collapsed]
    return new_vertices, new_faces, merged


def validate(vertices: np.ndarray, faces: np.ndarray) -> list[str]:
    """Problems that make the mesh unfit for export (empty list = valid):
    open edges, non-manifold edges, inconsistent winding, non-positive volume,
    more than one body."""
    if len(faces) == 0:
        return ["the result has no triangles"]
    problems, _, _ = analyse_topology(vertices, faces)
    return [p.replace("file contains", "result contains")
             .replace("; only one body can be loaded at a time (save each body as its own STL file)", "")
            for p in problems]


def repair(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Try to fix the defects `validate` reports. Returns (vertices, faces, what was done)."""
    done: list[str] = []
    faces = np.asarray(faces, dtype=np.int64)

    # 1. Triangles that use a vertex twice carry no surface.
    collapsed = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    if collapsed.any():
        faces = faces[~collapsed]
        done.append(f"removed {int(collapsed.sum())} collapsed triangle(s)")

    # 2. Exact duplicate triangles.
    key = np.sort(faces, axis=1)
    order = np.lexsort(key.T[::-1])
    repeat = np.zeros(len(faces), dtype=bool)
    repeat[order[1:]] = (key[order[1:]] == key[order[:-1]]).all(axis=1)
    if repeat.any():
        done.append(f"removed {int(repeat.sum())} duplicate triangle(s)")
        faces = faces[~repeat]

    # 3. Small holes (only when there are open edges).
    if _open_edge_count(faces, len(vertices)):
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        before = len(mesh.faces)
        trimesh.repair.fill_holes(mesh)
        if len(mesh.faces) != before:
            done.append(f"closed hole(s) with {len(mesh.faces) - before} triangle(s)")
        vertices, faces = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)

    # 4. Winding: every triangle turned the same way as its neighbours.
    faces, flipped = fix_winding(faces, len(vertices))
    if flipped:
        done.append(f"made the winding consistent ({flipped} triangle(s) turned)")

    # 5. Stray pieces: keep the largest body and the cavities inside it.
    vertices, faces, removed = _keep_main_body(vertices, faces)
    if removed:
        done.append(f"removed {removed} stray piece(s)")
    return vertices, faces, done


def _open_edge_count(faces: np.ndarray, vertex_count: int) -> int:
    n = np.int64(vertex_count)
    a, b = faces.ravel(), faces[:, [1, 2, 0]].ravel()
    _, counts = np.unique(np.minimum(a, b) * n + np.maximum(a, b), return_counts=True)
    return int((counts == 1).sum())


def fix_winding(faces: np.ndarray, vertex_count: int) -> tuple[np.ndarray, int]:
    """Turn triangles so that neighbours agree (a shared edge is used in opposite
    directions). Within each connected piece the orientation most of its triangles
    already had is kept. Returns (faces, number of triangles turned).

    Works on whole arrays (no loop over triangles), so it handles millions of
    triangles in seconds: a tree of neighbouring triangles is grown from one root per
    piece, and whether each triangle must turn relative to its root is found by
    pointer jumping along the tree.
    """
    face_count = len(faces)
    if face_count == 0:
        return faces, 0
    n = np.int64(vertex_count)
    a, b = faces.ravel(), faces[:, [1, 2, 0]].ravel()
    key = np.minimum(a, b) * n + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    start = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
    length = np.diff(np.r_[start, len(sorted_key)])
    pair = start[length == 2]  # manifold edges only
    h1, h2 = order[pair], order[pair + 1]
    f1, f2 = h1 // 3, h2 // 3
    keep = f1 != f2
    f1, f2, h1, h2 = f1[keep], f2[keep], h1[keep], h2[keep]

    # Neighbour graph plus one extra node joined to every piece's first triangle.
    graph = coo_matrix((np.ones(len(f1), dtype=bool), (f1, f2)), shape=(face_count, face_count))
    pieces, label = connected_components(graph, directed=False)
    roots = np.full(pieces, face_count)
    np.minimum.at(roots, label, np.arange(face_count))
    hub = face_count
    rows = np.concatenate([f1, np.full(pieces, hub)])
    cols = np.concatenate([f2, roots])
    graph = coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)),
                       shape=(face_count + 1, face_count + 1)).tocsr()
    _, parent = breadth_first_order(graph, hub, directed=False, return_predecessors=True)
    parent = parent[:face_count].astype(np.int64)
    is_root = parent == hub

    # Must this triangle turn relative to its parent? Yes if they use their shared edge
    # in the same direction.
    child = np.flatnonzero(~is_root)
    mine = (faces[child] * n + faces[child][:, [1, 2, 0]])[:, :, None]
    theirs = (faces[parent[child]] * n + faces[parent[child]][:, [1, 2, 0]])[:, None, :]
    turn = np.zeros(face_count, dtype=bool)
    turn[child] = (mine == theirs).any(axis=(1, 2))

    # Pointer jumping: turn[f] becomes "turn relative to the root of its piece".
    up = np.where(is_root, np.arange(face_count), parent)
    while True:
        further = up[up]
        if np.array_equal(further, up):
            break
        turn = turn ^ turn[up]
        up = further
    turn = turn ^ turn[up]  # the last step to the root (roots have turn = False)

    # Keep the orientation that most triangles of each piece already had.
    turned_per_piece = np.bincount(label, turn.astype(np.int64), pieces)
    size = np.bincount(label, minlength=pieces)
    turn ^= (2 * turned_per_piece > size)[label]
    if not turn.any():
        return faces, 0
    faces = faces.copy()
    faces[turn] = faces[turn][:, ::-1]
    return faces, int(turn.sum())


def _keep_main_body(vertices: np.ndarray, faces: np.ndarray):
    n = len(vertices)
    a, b = faces.ravel(), faces[:, [1, 2, 0]].ravel()
    count, labels = connected_components(
        coo_matrix((np.ones(len(a), dtype=bool), (a, b)), shape=(n, n)), directed=False)
    face_label = labels[faces[:, 0]]
    tri = vertices[faces]
    volume = np.bincount(
        face_label, weights=np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0,
        minlength=count)
    used = np.bincount(face_label, minlength=count) > 0
    if used.sum() <= 1:
        return vertices, faces, 0
    if volume.max() <= 0:  # everything inside-out: turn it round
        faces, volume = faces[:, ::-1].copy(), -volume
    main = int(np.argmax(volume))
    lo = np.full((count, 3), np.inf)
    hi = np.full((count, 3), -np.inf)
    np.minimum.at(lo, labels, vertices)
    np.maximum.at(hi, labels, vertices)
    # A cavity is a shell wound inwards whose box lies inside the main body's box.
    cavity = (volume < 0) & (lo >= lo[main]).all(axis=1) & (hi <= hi[main]).all(axis=1)
    keep = cavity
    keep[main] = True
    removed = int((used & ~keep).sum())
    if removed == 0:
        return vertices, faces, 0
    faces = faces[keep[face_label]]
    index = np.unique(faces)
    remap = np.full(n, -1, dtype=np.int64)
    remap[index] = np.arange(len(index))
    return vertices[index], remap[faces], removed
