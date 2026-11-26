import {
  DEFAULT_SETTINGS,
  INVALID_COMBINATIONS,
  SETTING_FIELDS,
  type SettingField,
  type SettingKey,
  type Settings,
} from "./generated/settings";

const GROUPS: { id: SettingField["group"]; title: string }[] = [
  { id: "surface", title: "Surface" },
  { id: "interior", title: "Interior" },
  { id: "modifiers", title: "Modifiers" },
  { id: "keep", title: "Keep edges" },
  { id: "printing", title: "Printing (orientation and supports)" },
  { id: "print", title: "Print" },
];

type Listener = (settings: Settings, changed: SettingKey | null) => void;

/** Settings panel built entirely from the generated schema metadata. */
export class SettingsPanel {
  private settings: Settings = { ...DEFAULT_SETTINGS };
  private readonly rows = new Map<SettingKey, HTMLElement>();
  private readonly syncers = new Map<SettingKey, () => void>();
  private readonly selects = new Map<SettingKey, HTMLSelectElement>();
  private readonly listeners: Listener[] = [];

  constructor(root: HTMLElement) {
    for (const group of GROUPS) {
      const heading = document.createElement("h2");
      heading.textContent = group.title;
      root.appendChild(heading);
      for (const field of SETTING_FIELDS.filter((f) => f.group === group.id)) {
        const row = this.buildField(field);
        this.rows.set(field.key, row);
        root.appendChild(row);
      }
    }
    this.refresh();
  }

  get value(): Settings {
    return { ...this.settings, keep_edges: [...this.settings.keep_edges] };
  }

  /** Bed face: the viewport picks it, the engine can suggest it. */
  private bedHandlers: { pick: () => void; suggest: () => void } = { pick: () => {}, suggest: () => {} };
  private bedNote = "";

  onBedFace(handlers: { pick: () => void; suggest: () => void }): void {
    this.bedHandlers = handlers;
  }

  setBedNormal(normal: number[], note = ""): void {
    this.bedNote = note;
    this.commit("bed_normal", normal);
  }

  setBedNote(note: string): void {
    this.bedNote = note;
    this.refresh();
  }

  private buildBedFace(field: SettingField, row: HTMLElement): HTMLElement {
    const text = document.createElement("div");
    text.className = "count";
    const controls = document.createElement("div");
    controls.className = "row";
    const button = (label: string, action: () => void) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = label;
      b.addEventListener("click", action);
      return b;
    };
    const reset = button("Reset", () => this.setBedNormal([0, 0, -1]));
    controls.append(
      button("Pick face", () => this.bedHandlers.pick()),
      button("Suggest", () => this.bedHandlers.suggest()),
      reset,
    );
    const note = document.createElement("div");
    note.className = "note";
    row.append(text, controls, note);
    this.syncers.set(field.key, () => {
      const [x, y, z] = this.settings.bed_normal;
      const asLoaded = Math.abs(x) < 1e-9 && Math.abs(y) < 1e-9 && z < 0;
      text.textContent = asLoaded ? "As loaded (its bottom on the bed)" : "Picked face on the bed";
      reset.disabled = asLoaded;
      note.textContent = this.bedNote;
    });
    return row;
  }

  /** Number of sharp edges in the loaded model (shown next to the kept count). */
  private edgeCount = 0;
  private edgeNote = "Load a model to see its sharp edges.";

  setEdgeCount(count: number, note?: string): void {
    this.edgeCount = count;
    this.edgeNote =
      note ??
      (count
        ? "Click an edge in the view to have the struts draw it (orange); click again to release it."
        : "This model has no sharp edges to keep.");
    this.refresh();
  }

  /** Replace the kept edges (from a click in the viewport or a button). */
  setKeptEdges(ids: number[]): void {
    this.commit("keep_edges", [...new Set(ids)].sort((a, b) => a - b));
  }

  toggleKeptEdge(id: number): void {
    const kept = new Set(this.settings.keep_edges);
    if (kept.has(id)) kept.delete(id);
    else kept.add(id);
    this.setKeptEdges([...kept]);
  }

  /** Called when a value is committed (slider released, number entered, option picked). */
  onChange(listener: Listener): void {
    this.listeners.push(listener);
  }

  reset(): void {
    this.settings = { ...DEFAULT_SETTINGS, keep_edges: [], bed_normal: [...DEFAULT_SETTINGS.bed_normal] };
    this.bedNote = "";
    this.refresh();
    this.emit(null);
  }

  /**
   * Apply values from a saved settings file. Unknown keys and values that are not
   * allowed are skipped; numbers are clamped to their range. Returns the keys skipped.
   */
  apply(values: Record<string, unknown>): string[] {
    const skipped: string[] = [];
    const known = new Map(SETTING_FIELDS.map((field) => [field.key as string, field]));
    for (const key of Object.keys(values)) if (!known.has(key)) skipped.push(key);
    const next = { ...DEFAULT_SETTINGS } as unknown as Record<string, string | number>;
    for (const field of SETTING_FIELDS) {
      const value = values[field.key];
      if (value === undefined) continue;
      if (field.widget === "bed_face") {
        const ok =
          Array.isArray(value) && value.length === 3 &&
          value.every((v) => typeof v === "number" && Number.isFinite(v)) &&
          Math.hypot(...(value as number[])) > 1e-9;
        if (ok) (next as Record<string, unknown>)[field.key] = value as number[];
        else skipped.push(field.key);
      } else if (field.kind === "bool") {
        if (typeof value === "boolean") (next as Record<string, unknown>)[field.key] = value;
        else skipped.push(field.key);
      } else if (field.kind === "list") {
        if (Array.isArray(value) && value.every((v) => Number.isInteger(v) && v >= 0)) {
          (next as Record<string, unknown>)[field.key] = [...new Set(value as number[])].sort((a, b) => a - b);
        } else skipped.push(field.key);
      } else if (field.kind === "enum") {
        if (field.options?.some((option) => option.value === value)) next[field.key] = value as string;
        else skipped.push(field.key);
      } else if (typeof value === "number" && Number.isFinite(value)) {
        const rounded = field.kind === "int" ? Math.round(value) : value;
        next[field.key] = Math.min(field.max!, Math.max(field.min!, rounded));
      } else {
        skipped.push(field.key);
      }
    }
    const candidate = next as unknown as Settings;
    this.bedNote = "";
    const invalid = INVALID_COMBINATIONS.some((combination) =>
      Object.entries(combination).every(([k, v]) => String(candidate[k as SettingKey]) === v),
    );
    if (invalid) return [...skipped, "surface_style + interior_style"];
    this.settings = candidate;
    for (const field of SETTING_FIELDS) {
      if (field.max_field) this.set(field.max_field, this.settings[field.max_field] as number);
    }
    this.refresh();
    this.emit(null);
    return skipped;
  }

  private set(key: SettingKey, value: string | number | boolean | number[]): void {
    (this.settings as unknown as Record<string, unknown>)[key] = value;
    // Keep dependent limits valid (emboss depth cannot exceed shell thickness).
    for (const field of SETTING_FIELDS) {
      if (field.max_field === key) {
        const cap = this.settings[key] as number;
        if ((this.settings[field.key] as number) > cap) this.set(field.key, cap);
      }
    }
  }

  private emit(changed: SettingKey | null): void {
    for (const listener of this.listeners) listener(this.value, changed);
  }

  private refresh(): void {
    for (const sync of this.syncers.values()) sync();
    for (const field of SETTING_FIELDS) {
      this.rows.get(field.key)!.hidden = !this.isVisible(field);
    }
    // Hide options that would complete a combination that is not allowed.
    for (const [key, select] of this.selects) {
      for (const option of select.options) {
        option.hidden = option.disabled = !this.isAllowed(key, option.value);
      }
    }
  }

  private isAllowed(key: SettingKey, value: string): boolean {
    const candidate = { ...this.settings, [key]: value } as Settings;
    return !INVALID_COMBINATIONS.some((combination) =>
      Object.entries(combination).every(([k, v]) => String(candidate[k as SettingKey]) === v),
    );
  }

  private isVisible(field: SettingField): boolean {
    if (!field.show_if) return true;
    return field.show_if.some((condition) =>
      Object.entries(condition).every(([key, allowed]) =>
        allowed!.includes(String(this.settings[key as SettingKey])),
      ),
    );
  }

  private buildField(field: SettingField): HTMLElement {
    const row = document.createElement("div");
    row.className = "field";
    const id = `setting-${field.key}`;
    const label = document.createElement("label");
    label.htmlFor = id;
    label.textContent = field.label;
    if (field.unit) {
      const unit = document.createElement("span");
      unit.className = "unit";
      unit.textContent = ` (${field.unit})`;
      label.appendChild(unit);
    }
    row.appendChild(label);

    if (field.widget === "bed_face") return this.buildBedFace(field, row);
    if (field.kind === "bool") {
      const box = document.createElement("input");
      box.type = "checkbox";
      box.id = id;
      box.addEventListener("change", () => this.commit(field.key, box.checked));
      this.syncers.set(field.key, () => (box.checked = Boolean(this.settings[field.key])));
      label.prepend(box);
      label.classList.add("check");
      return row;
    }
    if (field.kind === "list") {
      // Kept edges: picked in the viewport; here only the count and two buttons.
      const note = document.createElement("div");
      note.className = "note";
      const controls = document.createElement("div");
      controls.className = "row";
      const count = document.createElement("span");
      count.className = "count";
      const all = document.createElement("button");
      all.type = "button";
      all.textContent = "Select all";
      all.addEventListener("click", () =>
        this.setKeptEdges(Array.from({ length: this.edgeCount }, (_, i) => i)),
      );
      const none = document.createElement("button");
      none.type = "button";
      none.textContent = "Clear";
      none.addEventListener("click", () => this.setKeptEdges([]));
      controls.append(count, all, none);
      row.append(controls, note);
      this.syncers.set(field.key, () => {
        const kept = this.settings.keep_edges.length;
        count.textContent = `${kept} of ${this.edgeCount} edges kept`;
        all.disabled = this.edgeCount === 0 || kept === this.edgeCount;
        none.disabled = kept === 0;
        note.textContent = this.edgeNote;
      });
      return row;
    }

    if (field.kind === "enum") {
      const select = document.createElement("select");
      select.id = id;
      for (const option of field.options ?? []) {
        select.add(new Option(option.label, option.value));
      }
      select.addEventListener("change", () => this.commit(field.key, select.value));
      this.selects.set(field.key, select);
      this.syncers.set(field.key, () => (select.value = String(this.settings[field.key])));
      row.appendChild(select);
      return row;
    }

    const controls = document.createElement("div");
    controls.className = "row";
    const number = document.createElement("input");
    number.type = "number";
    number.id = id;
    number.min = String(field.min);
    number.max = String(field.max);
    number.step = String(field.step ?? 1);
    const maxNow = () =>
      field.max_field
        ? Math.min(field.max!, this.settings[field.max_field] as number)
        : field.max!;
    const clamp = (raw: number) => {
      let v = Number.isFinite(raw) ? raw : (field.default as number);
      if (field.kind === "int") v = Math.round(v);
      return Math.min(maxNow(), Math.max(field.min!, v));
    };
    number.addEventListener("change", () => this.commit(field.key, clamp(number.valueAsNumber)));

    if (field.key === "random_seed") {
      const randomize = document.createElement("button");
      randomize.type = "button";
      randomize.textContent = "Randomize";
      randomize.addEventListener("click", () =>
        this.commit(field.key, Math.floor(Math.random() * 1_000_000)),
      );
      number.style.width = "110px";
      controls.append(number, randomize);
      this.syncers.set(field.key, () => (number.value = String(this.settings[field.key])));
    } else {
      const slider = document.createElement("input");
      slider.type = "range";
      slider.min = String(field.min);
      slider.step = String(field.step ?? 1);
      slider.setAttribute("aria-label", field.label);
      // Dragging updates the number live; the value is committed on release.
      slider.addEventListener("input", () => (number.value = slider.value));
      slider.addEventListener("change", () => this.commit(field.key, clamp(slider.valueAsNumber)));
      controls.append(slider, number);
      this.syncers.set(field.key, () => {
        slider.max = number.max = String(maxNow());
        slider.value = number.value = String(this.settings[field.key]);
      });
    }
    row.appendChild(controls);
    return row;
  }

  private commit(key: SettingKey, value: string | number | boolean | number[]): void {
    this.set(key, value);
    this.refresh();
    this.emit(key);
  }
}
