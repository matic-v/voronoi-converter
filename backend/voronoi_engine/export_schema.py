"""Generate the TypeScript mirror of the settings schema.

Usage:  python -m voronoi_engine.export_schema [--check]
"""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path

from .settings import INVALID_COMBINATIONS, SETTINGS_VERSION, Settings, field_metadata

OUTPUT = Path(__file__).resolve().parents[2] / "frontend" / "src" / "generated" / "settings.ts"


def render() -> str:
    lines = [
        "// AUTO-GENERATED from backend/voronoi_engine/settings.py - do not edit.",
        "// Regenerate with: python -m voronoi_engine.export_schema",
        "",
        f"export const SETTINGS_VERSION = {SETTINGS_VERSION};",
        "",
    ]
    interface = ["export interface Settings {"]
    for name, info in Settings.model_fields.items():
        annotation = info.annotation
        if getattr(annotation, "__origin__", None) is list:
            interface.append(f"  {name}: number[];")
        elif isinstance(annotation, type) and issubclass(annotation, Enum):
            type_name = annotation.__name__
            union = " | ".join(json.dumps(member.value) for member in annotation)
            lines.append(f"export type {type_name} = {union};")
            interface.append(f"  {name}: {type_name};")
        else:
            interface.append(f"  {name}: number;")
    interface.append("}")
    defaults = Settings().model_dump(mode="json")
    lines += [
        "",
        *interface,
        "",
        "export type SettingKey = keyof Settings;",
        "",
        "export interface SettingField {",
        "  key: SettingKey;",
        '  kind: "enum" | "float" | "int" | "list";',
        '  group: "surface" | "interior" | "modifiers" | "keep" | "print";',
        "  label: string;",
        "  unit: string;",
        "  default: string | number | number[];",
        "  /** Set by clicking in the viewport, not by a slider. */",
        "  picked_in_viewport?: boolean;",
        "  min?: number;",
        "  max?: number;",
        "  step?: number;",
        "  /** Upper limit is additionally capped by the current value of this field. */",
        "  max_field?: SettingKey;",
        "  options?: { value: string; label: string }[];",
        "  /** Visible when ANY condition holds; a condition holds when ALL entries match. */",
        "  show_if?: Partial<Record<SettingKey, string[]>>[];",
        "}",
        "",
        f"export const DEFAULT_SETTINGS: Settings = {json.dumps(defaults, indent=2)};",
        "",
        f"export const SETTING_FIELDS: SettingField[] = {json.dumps(field_metadata(), indent=2, ensure_ascii=False)};",
        "",
        "/** Combinations of values that are not allowed together (they describe an empty part). */",
        "export const INVALID_COMBINATIONS: Partial<Record<SettingKey, string>>[] = "
        f"{json.dumps(INVALID_COMBINATIONS, indent=2)};",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    content = render()
    if "--check" in sys.argv:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != content:
            print(f"{OUTPUT} is out of date; run python -m voronoi_engine.export_schema")
            return 1
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
