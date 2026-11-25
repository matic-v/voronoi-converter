import "./style.css";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import {
  CancelledError, exportResult, fetchEdges, generate, getJson, suggestBed, uploadModel, UserError,
  type LoadedModel, type ResultInfo, type TestModel,
} from "./api";
import { SettingsPanel } from "./settingsPanel";
import { Viewport, type ClipAxis, type ViewMode, type ViewName } from "./viewport";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// --- errors: always shown in the UI, never only in the console ----------------
function showError(message: string): void {
  $("error-text").textContent = message;
  $("error-panel").hidden = false;
}
function hideError(): void {
  $("error-panel").hidden = true;
}
function report(error: unknown): void {
  if (error instanceof UserError) showError(error.message);
  else showError(`Unexpected error: ${(error as Error)?.message ?? error}`);
}
$("error-dismiss").addEventListener("click", hideError);
window.addEventListener("error", (e) => report(e.error ?? e.message));
window.addEventListener("unhandledrejection", (e) => report(e.reason));

// --- viewport -----------------------------------------------------------------
const viewport = new Viewport($("viewport"));
for (const button of document.querySelectorAll<HTMLButtonElement>("#view-buttons button")) {
  button.addEventListener("click", () => viewport.setView(button.dataset.view as ViewName));
}

const clipAxis = $<HTMLSelectElement>("clip-axis");
const clipPosition = $<HTMLInputElement>("clip-position");
const clipFlip = $<HTMLInputElement>("clip-flip");
function updateClip(): void {
  const off = clipAxis.value === "off";
  clipPosition.disabled = clipFlip.disabled = off;
  viewport.setClip(clipAxis.value as ClipAxis, clipPosition.valueAsNumber / 100, clipFlip.checked);
}
for (const control of [clipAxis, clipPosition, clipFlip]) control.addEventListener("input", updateClip);

const overhang = $<HTMLInputElement>("overhang");
overhang.addEventListener("change", () => viewport.setOverhangHighlight(overhang.checked));

const viewMode = $<HTMLSelectElement>("view-mode");
viewMode.addEventListener("change", () => viewport.setViewMode(viewMode.value as ViewMode));

// --- settings -----------------------------------------------------------------
const settings = new SettingsPanel($("settings-panel"));
$("btn-reset").addEventListener("click", () => settings.reset());

// --- loading ------------------------------------------------------------------
let current: LoadedModel | null = null;
let loadToken = 0;

let inputWarnings: string[] = [];
let resultWarnings: string[] = [];
function showWarnings(): void {
  const warnings = [...inputWarnings, ...resultWarnings];
  const holder = $("info-warnings");
  holder.textContent = warnings.length ? `⚠ ${warnings.join("  ⚠ ")}` : "";
  holder.title = warnings.join("\n");
}

function setInfo(text: string, warnings: string[] = []): void {
  $("info-text").textContent = $("info-text").title = text;
  inputWarnings = warnings;
  showWarnings();
}

function setResultInfo(text: string, busy = false, warnings: string[] = []): void {
  const holder = $("result-text");
  holder.textContent = holder.title = text;
  holder.classList.toggle("busy", busy);
  resultWarnings = warnings;
  showWarnings();
}

// --- generation ---------------------------------------------------------------
const PREVIEW_DELAY_MS = 300;
const generateButton = $<HTMLButtonElement>("btn-generate");
let pending: AbortController | null = null;
let previewTimer: number | undefined;

function describeResult(info: ResultInfo): string {
  return [
    info.mode === "final" ? "→ Final" : "→ Preview",
    `${info.size_mm.map((v) => v.toFixed(1)).join(" × ")} mm`,
    `${info.triangles.toLocaleString("en-US")} triangles`,
    `${info.seconds.toFixed(1)} s`,
    `voxel ${info.voxel_mm} mm`,
  ].join("  ·  ");
}

function resultWarningsOf(info: ResultInfo): string[] {
  const warnings = [...info.warnings];
  if (!info.valid) warnings.push(`Result mesh has defects: ${info.problems.join("; ")}`);
  return warnings;
}

// --- export -------------------------------------------------------------------
// Export is offered only for the final result on screen, and only if it passed validation.
const exportButton = $<HTMLButtonElement>("btn-export");
let exportJob: string | null = null;

function setExportable(jobId: string | null): void {
  exportJob = jobId;
  exportButton.disabled = jobId === null;
}

exportButton.addEventListener("click", async () => {
  if (!exportJob) return;
  hideError();
  try {
    const format = settings.value.export_format;
    setResultInfo(
      `Preparing the ${format.toUpperCase()} file…`, true, resultWarnings,
    );
    const [model, json] = await exportResult(exportJob, format);
    setResultInfo(`Exported ${model} and ${json} to your downloads folder`, false, resultWarnings);
  } catch (error) {
    setExportable(null);
    report(error);
  }
});

function dropResult(): void {
  setExportable(null);
  viewport.clearResult();
  viewMode.disabled = true;
  setResultInfo("");
}

const progress = $("progress");
const progressBar = $("progress-bar");
const progressText = $("progress-text");

function showProgress(mode: "preview" | "final", fraction: number, message: string): void {
  progress.hidden = false;
  progressBar.style.width = `${Math.round(fraction * 100)}%`;
  progressText.textContent = `${mode === "final" ? "Final" : "Preview"}: ${message} ${Math.round(fraction * 100)} %`;
}

async function runGeneration(mode: "preview" | "final"): Promise<void> {
  if (!current) return;
  window.clearTimeout(previewTimer);
  pending?.abort(); // the newest request wins; the engine cancels the older job
  const controller = (pending = new AbortController());
  hideError();
  setExportable(null); // whatever is on screen is about to be replaced
  showProgress(mode, 0, "Waiting");
  try {
    const mesh = await generate(
      current.id,
      settings.value,
      mode,
      (status) => showProgress(mode, status.progress, status.message),
      controller.signal,
    );
    if (controller.signal.aborted) return;
    viewport.setResult(mesh.positions, mesh.indices);
    viewMode.disabled = false;
    viewport.setViewMode(viewMode.value as ViewMode);
    setResultInfo(describeResult(mesh.info), false, resultWarningsOf(mesh.info));
    if (mode === "final") {
      if (mesh.info.valid) {
        setExportable(mesh.jobId);
      } else {
        showError(
          "The final result has defects that could not be repaired automatically, so it cannot " +
            `be exported:\n  • ${mesh.info.problems.join("\n  • ")}\n\n` +
            "Try a different random seed, thicker features or a higher quality.",
        );
      }
    }
  } catch (error) {
    if (pending !== controller) return; // replaced by a newer request
    if (error instanceof CancelledError) {
      // Cancelled by the user: the previous result stays on screen.
      setResultInfo(
        `${$("result-text").textContent?.replace(/  ·  cancelled.*$/, "") ?? ""}  ·  cancelled ${mode}`.replace(/^  ·  /, ""),
        false,
        resultWarnings,
      );
    } else {
      dropResult();
      report(error);
    }
  } finally {
    if (pending === controller) {
      pending = null;
      progress.hidden = true;
    }
  }
}

$("btn-cancel").addEventListener("click", () => pending?.abort());

function schedulePreview(): void {
  setExportable(null); // the result on screen no longer matches the settings
  window.clearTimeout(previewTimer);
  previewTimer = window.setTimeout(() => void runGeneration("preview"), PREVIEW_DELAY_MS);
}

// The export format does not change the shape: it needs no new preview, and a finished
// final can still be exported in the other format.
settings.onChange((_, changed) => {
  if (changed !== "export_format") schedulePreview();
});
const showExportFormat = () =>
  (exportButton.textContent = `Export ${settings.value.export_format.toUpperCase()}`);
settings.onChange(showExportFormat);
showExportFormat();
settings.onChange((value) => viewport.setKeptEdges(value.keep_edges));

// --- kept edges: shown in the viewport, picked by clicking -----------------------
const showEdges = $<HTMLInputElement>("show-edges");
showEdges.addEventListener("change", () => viewport.showEdges(showEdges.checked));
let pointerDown: { x: number; y: number } | null = null;
const canvas = $("viewport").querySelector("canvas")!;
canvas.addEventListener("pointerdown", (e) => {
  pointerDown = e.button === 0 ? { x: e.clientX, y: e.clientY } : null;
});
canvas.addEventListener("pointerup", (e) => {
  // A click, not the end of an orbit drag.
  if (!pointerDown || Math.hypot(e.clientX - pointerDown.x, e.clientY - pointerDown.y) > 4) return;
  if (pickingBed) {
    const picked = viewport.pickFace(e.clientX, e.clientY);
    if (picked) {
      stopPickingBed();
      settings.setBedNormal(picked.normal, "Picked in the view.");
      const above = viewport.heightAboveBed(picked.point);
      if (above > 0.2) {
        settings.setBedNote(
          `This face points down but is ${above.toFixed(1)} mm above the bed: other parts of the ` +
            "model are lower, so the part rests on those instead. Pick an outer face, or use Suggest.",
        );
      }
    }
    return;
  }
  const edge = viewport.pickEdge(e.clientX, e.clientY);
  if (edge !== null) settings.toggleKeptEdge(edge);
});

// --- bed face: picked in the view, or suggested by the engine -------------------------
let pickingBed = false;
const hint = $("viewport-hint");
let hintBefore = "";
function stopPickingBed(): void {
  if (!pickingBed) return;
  pickingBed = false;
  hint.textContent = hintBefore;
  hint.classList.remove("picking");
}
settings.onBedFace({
  pick: () => {
    if (!current) return;
    pickingBed = true;
    hintBefore = hint.textContent ?? "";
    hint.textContent = "Click the face of the part that should lie on the bed (Esc to cancel)";
    hint.classList.add("picking");
    viewMode.value = "original";
    viewport.setViewMode("original");
  },
  suggest: async () => {
    if (!current) return;
    try {
      const s = await suggestBed(current.id, settings.value.overhang_angle);
      const note =
        `Suggested: ${s.overhang_mm2.toFixed(0)} mm² of the outer surface overhangs ` +
        `(as loaded: ${s.as_loaded_overhang_mm2.toFixed(0)} mm²), ${s.contact_mm2.toFixed(0)} mm² on the bed.`;
      settings.setBedNormal(s.normal, note);
    } catch (error) {
      report(error);
    }
  },
});
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") stopPickingBed();
});
settings.onChange((value, changed) => {
  if (changed === "bed_normal" || changed === null) viewport.setOrientation(value.bed_normal);
});

async function showModelEdges(model: LoadedModel): Promise<void> {
  if (model.edges.too_many) {
    settings.setEdgeCount(0, "This model has too many sharp edges to pick from.");
    return;
  }
  const edges = await fetchEdges(model.id);
  if (current?.id !== model.id) return;
  viewport.setEdges(edges.positions, edges.edgeOfSegment, settings.value.keep_edges);
  viewport.showEdges(showEdges.checked);
  const kinds = model.edges.curved
    ? ` (${model.edges.sharp} sharp, ${model.edges.curved} curved: fillets, rounds and small holes, ` +
      "drawn as bands of lines)"
    : "";
  settings.setEdgeCount(
    edges.count,
    edges.count
      ? `Click an edge in the view to have the struts draw it (orange); click again to release it.${kinds}`
      : undefined,
  );
}
generateButton.addEventListener("click", () => void runGeneration("final"));

function describe(model: LoadedModel, seconds: number): string {
  const { size_mm: s, triangles, volume_mm3 } = model.info;
  return [
    model.name,
    `${s.map((v) => v.toFixed(1)).join(" × ")} mm`,
    `${triangles.toLocaleString("en-US")} triangles`,
    `${(volume_mm3 / 1000).toFixed(1)} cm³`,
    `loaded in ${seconds.toFixed(1)} s`,
  ].join("  ·  ");
}

/** Apply a settings file saved next to an export (the model itself is loaded separately). */
async function loadSettingsFile(file: File): Promise<void> {
  let document: unknown;
  try {
    document = JSON.parse(await file.text());
  } catch {
    showError(`'${file.name}' is not a valid JSON file.`);
    return;
  }
  const root = document as { settings?: unknown; input?: { file?: string } } | null;
  const values = root?.settings ?? root;
  if (!values || typeof values !== "object" || Array.isArray(values)) {
    showError(`'${file.name}' does not contain Voronoi Body Generator settings.`);
    return;
  }
  const skipped = settings.apply(values as Record<string, unknown>);
  if (skipped.includes("surface_style + interior_style")) {
    showError(`The settings in '${file.name}' combine styles that are not allowed; nothing was changed.`);
    return;
  }
  const notes = [`Settings loaded from ${file.name}.`];
  if (root?.input?.file && root.input.file !== current?.name) {
    notes.push(`They were saved for '${root.input.file}': load that STL to reproduce the result.`);
  }
  if (skipped.length) notes.push(`Ignored: ${skipped.join(", ")}.`);
  if (!current) setInfo(notes.join(" "));
  else if (notes.length > 1) showError(notes.join("\n"));
}

async function loadFile(file: File): Promise<void> {
  const token = ++loadToken;
  const started = performance.now();
  hideError();
  if (file.name.toLowerCase().endsWith(".json")) {
    await loadSettingsFile(file);
    return;
  }
  if (!file.name.toLowerCase().endsWith(".stl")) {
    showError(
      `'${file.name}' cannot be loaded. Load an .stl model (binary or ASCII), or a settings ` +
        ".json saved by Export.",
    );
    return;
  }
  setInfo(`Loading ${file.name}…`);
  try {
    // The engine validates the mesh; the same bytes are parsed here for display.
    const [model, buffer] = await Promise.all([uploadModel(file), file.arrayBuffer()]);
    if (token !== loadToken) return; // a newer load replaced this one
    pending?.abort();
    viewport.setModel(new STLLoader().parse(buffer));
    current = model;
    // Edge numbers belong to one model: a new model starts with none kept.
    if (settings.value.keep_edges.length) settings.setKeptEdges([]);
    // The bed face belongs to one model too: a new model starts as loaded.
    stopPickingBed();
    const [bx, by, bz] = settings.value.bed_normal;
    if (Math.abs(bx) > 1e-9 || Math.abs(by) > 1e-9 || bz > 0) settings.setBedNormal([0, 0, -1]);
    await showModelEdges(model);
    dropResult();
    generateButton.disabled = false;
    updateClip();
    $("viewport-hint").textContent = "Drag: orbit · right-drag: pan · scroll: zoom";
    setInfo(describe(model, (performance.now() - started) / 1000), model.warnings);
    void runGeneration("preview");
  } catch (error) {
    if (token !== loadToken) return;
    // A refused file leaves the previously loaded model in place.
    setInfo(current ? describe(current, current.load_seconds) : "No model loaded.", current?.warnings);
    report(error);
  }
}

const picker = $<HTMLInputElement>("file-input");
$("btn-load").addEventListener("click", () => picker.click());
picker.addEventListener("change", () => {
  const file = picker.files?.[0];
  picker.value = ""; // allow re-loading the same file
  if (file) void loadFile(file);
});

// Drag and drop anywhere on the window.
const dropOverlay = $("drop-overlay");
let dragDepth = 0;
const hasFiles = (e: DragEvent) => e.dataTransfer?.types.includes("Files") ?? false;
window.addEventListener("dragenter", (e) => {
  if (!hasFiles(e)) return;
  dragDepth++;
  dropOverlay.hidden = false;
});
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) dropOverlay.hidden = true;
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  dropOverlay.hidden = true;
  const file = e.dataTransfer?.files[0];
  if (file) void loadFile(file);
});

// --- backend ------------------------------------------------------------------
async function loadTestModel(model: TestModel): Promise<void> {
  const response = await fetch(`/api/test-models/${model.filename}`);
  if (!response.ok) throw new UserError(`Could not fetch test model '${model.name}'.`);
  await loadFile(new File([await response.blob()], model.filename));
}

async function connect(): Promise<void> {
  const status = $("backend-status");
  try {
    const health = await getJson<{ version: string }>("/api/health");
    status.textContent = `Engine v${health.version} connected`;
    status.className = "status ok";
    $<HTMLButtonElement>("btn-load").disabled = false;
    const models = await getJson<TestModel[]>("/api/test-models");
    const holder = $("test-models");
    const addGroup = (title: string, group: TestModel[]) => {
      holder.append(title);
      group.forEach((model, i) => {
        const link = document.createElement("a");
        link.href = "#";
        link.textContent = model.label;
        link.addEventListener("click", (e) => {
          e.preventDefault();
          loadTestModel(model).catch(report);
        });
        holder.append(i ? " · " : "", link);
      });
    };
    addGroup("Test: ", models.filter((m) => !m.broken));
    addGroup("   Broken: ", models.filter((m) => m.broken));
  } catch (error) {
    status.textContent = "Engine not reachable";
    status.className = "status bad";
    report(error);
  }
}
setInfo("No model loaded. Use Load STL or drop a file onto the window.");
void connect();
