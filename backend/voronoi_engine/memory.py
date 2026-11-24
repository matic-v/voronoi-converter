"""Memory budget for the voxel grid."""

from __future__ import annotations

import os

# Measured peak of the pipeline is about 88 bytes per grid node (several float32
# fields, labels and masks alive at once, plus the mesh).
BYTES_PER_NODE = 96
SHARE_OF_SYSTEM_MEMORY = 0.4
MAX_BUDGET_MB = 8192
FALLBACK_SYSTEM_MB = 8192


def system_memory_mb() -> float:
    """Installed RAM in MB (8 GB assumed when it cannot be read)."""
    try:
        if os.name == "nt":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong),
                            ("available_virtual", ctypes.c_ulonglong),
                            ("extended", ctypes.c_ulonglong)]

            status = Status()
            status.length = ctypes.sizeof(Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.total / 2**20
        else:
            return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (AttributeError, OSError, ValueError):
        pass
    return float(FALLBACK_SYSTEM_MB)


def memory_budget_mb() -> float:
    """Memory the generator may use: 40 % of the installed RAM, at most 8 GB.
    Set the environment variable VORONOI_MEMORY_MB to override."""
    override = os.environ.get("VORONOI_MEMORY_MB")
    if override:
        try:
            return max(64.0, float(override))
        except ValueError:
            pass
    return min(SHARE_OF_SYSTEM_MEMORY * system_memory_mb(), float(MAX_BUDGET_MB))


def max_nodes() -> int:
    return int(memory_budget_mb() * 2**20 / BYTES_PER_NODE)
