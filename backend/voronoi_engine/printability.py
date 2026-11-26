"""Non-blocking printability warnings, derived from the settings alone."""

from __future__ import annotations

from .grading import scale_range
from .settings import Settings

MIN_FEATURE_NOZZLES = 3.0
MIN_HOLE_NOZZLES = 5.0


def printability_warnings(settings: Settings) -> list[str]:
    nozzle = settings.nozzle_diameter
    min_feature = MIN_FEATURE_NOZZLES * nozzle
    min_hole = MIN_HOLE_NOZZLES * nozzle
    thin, _ = scale_range(settings, "thickness")
    small_cell, _ = scale_range(settings, "cell_size")
    graded = " at the thin end of the grading" if thin < 1.0 else ""
    graded_cell = " where cells are smallest" if small_cell < 1.0 else ""
    surface, interior = settings.surface_style, settings.interior_style
    warnings: list[str] = []

    def feature(name: str, size: float, note: str = "") -> None:
        if size < min_feature - 1e-9:
            warnings.append(
                f"{name} is {size:.2f} mm{note}: thinner than 3 × nozzle ({min_feature:.2f} mm).")

    def hole(name: str, size: float, note: str = "") -> None:
        if size < min_hole - 1e-9:
            warnings.append(
                f"{name} are about {max(size, 0.0):.2f} mm{note}: smaller than 5 × nozzle "
                f"({min_hole:.2f} mm).")

    if surface in ("perforated", "embossed", "solid"):
        feature("Shell wall", settings.shell_thickness * thin, graded)
    if surface in ("rib_lattice", "embossed"):
        feature("Rib width", settings.rib_width * thin, graded)
    if interior == "struts":
        feature("Strut diameter", settings.strut_diameter * thin, graded)

    cell = settings.surface_cell_size * small_cell
    if surface == "perforated":
        fraction = settings.hole_size / 100.0
        feature("Wall between holes", (1.0 - fraction) * cell, graded_cell)
        hole("Holes", fraction * cell, graded_cell)
    if surface == "rib_lattice":
        hole("Openings between ribs", cell - settings.rib_width, graded_cell)
    if surface == "embossed":
        floor = settings.shell_thickness * thin - settings.emboss_depth
        if interior != "solid" and floor > 1e-9:
            feature("Shell under the recesses", floor, graded)
    return warnings
