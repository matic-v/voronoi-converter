"""Export: binary STL and the settings document saved next to it."""

from __future__ import annotations

import re
import struct
import zlib
from pathlib import PurePath
from typing import Any

import numpy as np

from . import __version__
from .parallel import map_in_parallel
from .settings import SETTINGS_VERSION, Settings

STL_HEADER = b"Voronoi Body Generator - binary STL, units: mm".ljust(80, b" ")
STL_RECORD = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])


def to_binary_stl(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """Binary STL. The output depends only on the mesh: same mesh, same bytes."""
    tri = np.asarray(vertices, dtype="<f4")[np.asarray(faces)]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).astype(np.float64)
    length = np.linalg.norm(normal, axis=1)
    normal = np.divide(normal, length[:, None], out=np.zeros_like(normal), where=length[:, None] > 0)
    records = np.zeros(len(tri), dtype=STL_RECORD)
    records["normal"] = normal + 0.0  # no negative zeros
    records["vertices"] = tri
    return STL_HEADER + np.uint32(len(tri)).tobytes() + records.tobytes()


# --- 3MF --------------------------------------------------------------------------

CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
    '</Types>').encode()
RELATIONSHIPS = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
    'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
    '</Relationships>').encode()
BLOCK_VERTICES = 400_000
BLOCK_TRIANGLES = 600_000


def to_3mf(vertices: np.ndarray, faces: np.ndarray, title: str = "Voronoi body") -> bytes:
    """3MF package with one mesh, units millimetres. Same mesh, same bytes.

    3MF stores every vertex once and is zip-compressed, so it is several times smaller
    than a binary STL of the same mesh. Coordinates are written to 0.0001 mm.
    """
    title = re.sub(r'[<>&"]', "_", title)
    head = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
        f'<metadata name="Title">{title}</metadata>'
        f'<metadata name="Application">Voronoi Body Generator {__version__}</metadata>'
        '<resources><object id="1" type="model"><mesh><vertices>').encode()
    middle = b"</vertices><triangles>"
    tail = b'</triangles></mesh></object></resources><build><item objectid="1"/></build></model>'
    blocks = [head]
    for start in range(0, len(vertices), BLOCK_VERTICES):
        blocks.append(_vertex_text(vertices[start:start + BLOCK_VERTICES]))
    blocks.append(middle)
    for start in range(0, len(faces), BLOCK_TRIANGLES):
        blocks.append(_triangle_text(faces[start:start + BLOCK_TRIANGLES]))
    blocks.append(tail)
    model = _deflate_blocks(blocks)
    return _zip_package([
        ("[Content_Types].xml", *_deflate_one(CONTENT_TYPES)),
        ("_rels/.rels", *_deflate_one(RELATIONSHIPS)),
        ("3D/3dmodel.model", *model),
    ])


def _vertex_text(vertices: np.ndarray) -> bytes:
    # Integer formatting is several times faster than float formatting: coordinates are
    # written as <whole>.<4 digits> from integers in 1/10000 mm.
    q = np.rint(np.asarray(vertices, dtype=np.float64) * 10000.0).astype(np.int64)
    sign = np.where(q < 0, "-", "").tolist()
    magnitude = np.abs(q)
    whole, fraction = (magnitude // 10000).tolist(), (magnitude % 10000).tolist()
    return "".join(
        f'<vertex x="{s0}{w0}.{f0:04d}" y="{s1}{w1}.{f1:04d}" z="{s2}{w2}.{f2:04d}"/>'
        for (s0, s1, s2), (w0, w1, w2), (f0, f1, f2) in zip(sign, whole, fraction)).encode()


def _triangle_text(faces: np.ndarray) -> bytes:
    return "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>'
                   for a, b, c in np.asarray(faces).tolist()).encode()


def _deflate_one(data: bytes):
    packer = zlib.compressobj(6, zlib.DEFLATED, -15)
    return packer.compress(data) + packer.flush(), zlib.crc32(data), len(data)


def _deflate_blocks(blocks: list[bytes]):
    """Raw deflate of the blocks joined, compressed on all cores: every block is its own
    stream ended on a byte boundary (full flush), the last one finished; joined, they
    are one valid deflate stream (the method pigz uses)."""
    last = len(blocks) - 1

    def pack(item):
        index, data = item
        packer = zlib.compressobj(6, zlib.DEFLATED, -15)
        return packer.compress(data) + packer.flush(zlib.Z_FINISH if index == last else zlib.Z_FULL_FLUSH)

    compressed = b"".join(map_in_parallel(pack, list(enumerate(blocks))))
    crc = 0
    for data in blocks:
        crc = zlib.crc32(data, crc)
    return compressed, crc, sum(len(data) for data in blocks)


def _zip_package(entries) -> bytes:
    """A plain zip file (deflate, fixed date 1980-01-01 so equal input gives equal bytes)."""
    out, directory = [], []
    offset = 0
    dos_time, dos_date = 0, (0 << 9) | (1 << 5) | 1  # 1980-01-01 00:00
    for name, data, crc, size in entries:
        encoded = name.encode()
        if size >= 2**32 or len(data) >= 2**32 or offset >= 2**32:
            raise ValueError("model too large for a 3MF file (over 4 GB)")
        local = struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 0, 8, dos_time, dos_date,
                            crc, len(data), size, len(encoded), 0) + encoded
        directory.append(struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20, 0, 8, dos_time,
                                     dos_date, crc, len(data), size, len(encoded), 0, 0, 0, 0, 0,
                                     offset) + encoded)
        out += [local, data]
        offset += len(local) + len(data)
    central = b"".join(directory)
    end = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, len(entries), len(entries), len(central),
                      offset, 0)
    return b"".join(out) + central + end


def export_stem(input_name: str) -> str:
    """'<input name>_voronoi' with characters that are unsafe in file names replaced."""
    stem = PurePath(input_name).stem or "model"
    stem = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", stem).strip(" .") or "model"
    return f"{stem}_voronoi"


def settings_document(settings: Settings, input_name: str, input_info: dict[str, Any],
                      result_info: dict[str, Any]) -> dict[str, Any]:
    """Everything needed to reproduce an export: load the same STL, apply these settings."""
    keep = ("voxel_mm", "size_mm", "triangles", "volume_mm3", "fragments_removed", "quality_capped")
    return {
        "generator": "Voronoi Body Generator",
        "generator_version": __version__,
        "settings_version": SETTINGS_VERSION,
        "input": {"file": input_name, "size_mm": input_info["size_mm"],
                  "triangles": input_info["triangles"], "volume_mm3": input_info["volume_mm3"]},
        "settings": settings.model_dump(mode="json"),
        "result": {key: result_info[key] for key in keep if key in result_info},
    }


def settings_from_document(document: dict[str, Any]) -> Settings:
    """Read settings back from a saved document (or from a bare settings object)."""
    return Settings(**document.get("settings", document))
