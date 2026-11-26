"""The single typed settings schema.

This model is the source of truth for the UI, the API and the settings JSON saved
next to an export. The TypeScript mirror in ``frontend/src/generated/settings.ts``
is generated from it by ``python -m voronoi_engine.export_schema``.

Each field carries UI metadata in ``json_schema_extra``:

* ``group``   - settings panel group (surface / interior / modifiers / print)
* ``label``   - label shown in the UI
* ``unit``    - "mm", "%" or ""
* ``step``    - slider step
* ``show_if`` - list of conditions; the field is visible when ANY condition holds.
                A condition maps field name -> allowed values and holds when ALL of
                its entries match. A missing ``show_if`` means "always visible".
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

SETTINGS_VERSION = 1


class SurfaceStyle(str, Enum):
    PERFORATED = "perforated"  # A - shell with rounded Voronoi holes
    RIB_LATTICE = "rib_lattice"  # B - only the cell edges remain as ribs
    EMBOSSED = "embossed"  # C - closed shell with recessed cells
    NONE = "none"  # no skin, interior cut by the envelope
    SOLID = "solid"  # plain solid shell


class InteriorStyle(str, Enum):
    STRUTS = "struts"
    SOLID = "solid"
    HOLLOW = "hollow"


class GradingMode(str, Enum):
    OFF = "off"
    FROM_SKIN = "from_skin"
    X = "x"
    Y = "y"
    Z = "z"


class GradingTarget(str, Enum):
    CELL_SIZE = "cell_size"
    THICKNESS = "thickness"
    BOTH = "both"


class GradingDirection(str, Enum):
    DENSER_AT_START = "denser_at_start"
    DENSER_AT_END = "denser_at_end"


class ExportFormat(str, Enum):
    THREE_MF = "3mf"
    STL = "stl"


class Smoothness(str, Enum):
    OFF = "off"
    SMOOTH = "smooth"  # one level: 4 x triangles
    EXTRA = "extra"  # two levels: 16 x triangles


SMOOTHING_LEVELS = {"off": 0, "smooth": 1, "extra": 2}


class Quality(str, Enum):
    DRAFT = "draft"
    NORMAL = "normal"
    FINE = "fine"


ENUM_LABELS: dict[str, dict[str, str]] = {
    "surface_style": {
        "perforated": "A · Perforated shell",
        "rib_lattice": "B · Open rib lattice",
        "embossed": "C · Embossed ribs",
        "none": "None",
        "solid": "Solid",
    },
    "interior_style": {"struts": "Struts", "solid": "Solid", "hollow": "Hollow"},
    "grading_mode": {
        "off": "Off",
        "from_skin": "From skin",
        "x": "Along X",
        "y": "Along Y",
        "z": "Along Z",
    },
    "grading_target": {"cell_size": "Cell size", "thickness": "Thickness", "both": "Both"},
    "grading_direction": {
        "denser_at_start": "Denser at start (near skin)",
        "denser_at_end": "Denser at end",
    },
    "quality": {"draft": "Draft", "normal": "Normal", "fine": "Fine"},
    "export_format": {"3mf": "3MF (compact, recommended)", "stl": "STL (binary)"},
    "smoothness": {"off": "Off", "smooth": "Smooth (4 × triangles)",
                   "extra": "Extra smooth (16 × triangles)"},
}

# Style combinations that describe an empty part. The UI hides the option that would
# complete one of these; the schema rejects them.
INVALID_COMBINATIONS: list[dict[str, str]] = [
    {"surface_style": "none", "interior_style": "hollow"},
]

_A, _B, _C = "perforated", "rib_lattice", "embossed"
_GRADING_ON = ["from_skin", "x", "y", "z"]
_PATTERNED: list[dict[str, list[str]]] = [
    {"surface_style": [_A, _B, _C]},
    {"interior_style": ["struts"]},
]
_PATTERNED_AND_GRADED = [{**cond, "grading_mode": _GRADING_ON} for cond in _PATTERNED]


def _ui(group: str, label: str, unit: str = "", step: float | None = None,
        show_if: list[dict[str, list[str]]] | None = None) -> dict[str, Any]:
    extra: dict[str, Any] = {"group": group, "label": label, "unit": unit}
    if step is not None:
        extra["step"] = step
    if show_if is not None:
        extra["show_if"] = show_if
    return extra


class Settings(BaseModel):
    """All user-tunable settings. Lengths are millimetres."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True, validate_default=True)

    # --- Surface -----------------------------------------------------------
    surface_style: SurfaceStyle = Field(
        SurfaceStyle.PERFORATED, json_schema_extra=_ui("surface", "Surface style"))
    shell_thickness: float = Field(
        2.0, ge=0.8, le=10,
        json_schema_extra=_ui("surface", "Shell thickness", "mm", 0.1,
                              [{"surface_style": [_A, _C, "solid"]}]))
    surface_cell_size: float = Field(
        10.0, ge=3, le=50,
        json_schema_extra=_ui("surface", "Surface cell size", "mm", 0.5,
                              [{"surface_style": [_A, _B, _C]}]))
    hole_size: float = Field(
        60.0, ge=10, le=90,
        json_schema_extra=_ui("surface", "Hole size (of cell)", "%", 1,
                              [{"surface_style": [_A]}]))
    rib_width: float = Field(
        2.0, ge=0.8, le=10,
        json_schema_extra=_ui("surface", "Rib width", "mm", 0.1,
                              [{"surface_style": [_B, _C]}]))
    emboss_depth: float = Field(
        0.8, ge=0.4, le=10,
        json_schema_extra=_ui("surface", "Emboss depth", "mm", 0.1,
                              [{"surface_style": [_C]}]) | {"max_field": "shell_thickness"})
    edge_rounding: float = Field(
        0.5, ge=0, le=2,
        json_schema_extra=_ui("surface", "Edge rounding", "mm", 0.1,
                              [{"surface_style": [_A, _B, _C]}]))

    # --- Interior ----------------------------------------------------------
    interior_style: InteriorStyle = Field(
        InteriorStyle.STRUTS, json_schema_extra=_ui("interior", "Interior style"))
    interior_cell_size: float = Field(
        12.0, ge=3, le=60,
        json_schema_extra=_ui("interior", "Interior cell size", "mm", 0.5,
                              [{"interior_style": ["struts"]}]))
    strut_diameter: float = Field(
        2.0, ge=0.8, le=10,
        json_schema_extra=_ui("interior", "Strut diameter", "mm", 0.1,
                              [{"interior_style": ["struts"]}]))

    # --- Modifiers ---------------------------------------------------------
    blend_radius: float = Field(
        0.5, ge=0, le=5,
        json_schema_extra=_ui("modifiers", "Blend radius", "mm", 0.1,
                              [{"interior_style": ["struts"]}, {"surface_style": [_B]}]))
    grading_mode: GradingMode = Field(
        GradingMode.OFF, json_schema_extra=_ui("modifiers", "Grading mode", show_if=_PATTERNED))
    grading_strength: float = Field(
        50.0, ge=0, le=100,
        json_schema_extra=_ui("modifiers", "Grading strength", "%", 1, _PATTERNED_AND_GRADED))
    grading_target: GradingTarget = Field(
        GradingTarget.CELL_SIZE,
        json_schema_extra=_ui("modifiers", "Grading target", show_if=_PATTERNED_AND_GRADED))
    grading_direction: GradingDirection = Field(
        GradingDirection.DENSER_AT_START,
        json_schema_extra=_ui("modifiers", "Grading direction", show_if=_PATTERNED_AND_GRADED))
    solid_base_thickness: float = Field(
        0.0, ge=0, le=5,
        json_schema_extra=_ui("modifiers", "Solid base thickness (0 = off)", "mm", 0.1))
    random_seed: int = Field(
        1, ge=0, le=2**31 - 1, json_schema_extra=_ui("modifiers", "Random seed", step=1))

    # --- Keep ----------------------------------------------------------------
    keep_edges: list[int] = Field(
        default_factory=list,
        json_schema_extra=_ui("keep", "Edges drawn by struts") | {"picked_in_viewport": True})
    keep_width: float = Field(
        2.0, ge=0.8, le=10,
        json_schema_extra=_ui("keep", "Edge strut width", "mm", 0.1))

    # --- Print -------------------------------------------------------------
    nozzle_diameter: float = Field(
        0.4, ge=0.2, le=1.0, json_schema_extra=_ui("print", "Nozzle diameter", "mm", 0.05))
    quality: Quality = Field(Quality.NORMAL, json_schema_extra=_ui("print", "Final quality"))
    smoothness: Smoothness = Field(
        Smoothness.OFF, json_schema_extra=_ui("print", "Smoothness (final only)"))
    export_format: ExportFormat = Field(
        ExportFormat.THREE_MF, json_schema_extra=_ui("print", "Export format"))

    @model_validator(mode="after")
    def _emboss_within_shell(self) -> "Settings":
        if self.surface_style == SurfaceStyle.EMBOSSED.value and self.emboss_depth > self.shell_thickness:
            raise ValueError(
                f"Emboss depth ({self.emboss_depth} mm) cannot exceed the shell thickness "
                f"({self.shell_thickness} mm)."
            )
        return self

    @model_validator(mode="after")
    def _not_an_empty_part(self) -> "Settings":
        for combination in INVALID_COMBINATIONS:
            if all(getattr(self, key) == value for key, value in combination.items()):
                raise ValueError(
                    "Surface style None with interior style Hollow leaves nothing to generate. "
                    "Choose a skin or an interior."
                )
        return self

    @model_validator(mode="after")
    def _kept_edges_sorted(self) -> "Settings":
        if any(i < 0 for i in self.keep_edges):
            raise ValueError("Kept edge numbers cannot be negative.")
        self.keep_edges = sorted(set(self.keep_edges))
        return self

    def to_saved_json(self) -> dict[str, Any]:
        """Payload written next to an export so a result can be reproduced."""
        return {"settings_version": SETTINGS_VERSION, "settings": self.model_dump(mode="json")}


def field_metadata() -> list[dict[str, Any]]:
    """Flat, ordered description of every setting for building the UI."""
    fields: list[dict[str, Any]] = []
    for name, info in Settings.model_fields.items():
        extra = dict(info.json_schema_extra or {})  # type: ignore[arg-type]
        annotation = info.annotation
        default = info.default_factory() if info.default_factory else info.default
        meta: dict[str, Any] = {"key": name, **extra, "default": _plain(default)}
        if getattr(annotation, "__origin__", None) is list:
            meta["kind"] = "list"
            meta["default"] = []
        elif isinstance(annotation, type) and issubclass(annotation, Enum):
            meta["kind"] = "enum"
            meta["options"] = [
                {"value": member.value, "label": ENUM_LABELS[name][member.value]}
                for member in annotation
            ]
        else:
            meta["kind"] = "int" if annotation is int else "float"
            for constraint in info.metadata:
                if getattr(constraint, "ge", None) is not None:
                    meta["min"] = constraint.ge
                if getattr(constraint, "le", None) is not None:
                    meta["max"] = constraint.le
        fields.append(meta)
    return fields


def _plain(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value
