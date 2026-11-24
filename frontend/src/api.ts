import type { Settings } from "./generated/settings";

export interface MeshInfo {
  size_mm: [number, number, number];
  bounds_min: [number, number, number];
  bounds_max: [number, number, number];
  triangles: number;
  volume_mm3: number;
  bodies: number;
}

export interface LoadedModel {
  id: string;
  name: string;
  /** Sharp edges found in the model; `too_many` when it has too many to pick from. */
  edges: { count: number; sharp: number; curved: number; too_many: boolean };
  info: MeshInfo;
  warnings: string[];
  load_seconds: number;
}

export interface TestModel {
  name: string;
  label: string;
  filename: string;
  broken: boolean;
}

export interface ResultInfo {
  mode: "preview" | "final";
  voxel_mm: number;
  size_mm: [number, number, number];
  triangles: number;
  vertices: number;
  volume_mm3: number;
  seconds: number;
  fragments_removed: number;
  quality_capped: boolean;
  valid: boolean;
  problems: string[];
  warnings: string[];
}

export interface GeneratedMesh {
  /** The engine job that produced it; needed to export. */
  jobId: string;
  info: ResultInfo;
  positions: Float32Array;
  indices: Uint32Array;
}

/** An error whose message is already written for the user. */
export class UserError extends Error {}

async function readJson(response: Response): Promise<any> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function unreachable(): UserError {
  return new UserError(
    "Cannot reach the geometry engine. Make sure the app was started with start.bat " +
      "(or ./start.sh) and that its window is still open.",
  );
}

export async function getJson<T>(url: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url);
  } catch {
    throw unreachable();
  }
  if (!response.ok) throw new UserError(`The geometry engine returned an error (HTTP ${response.status}).`);
  return response.json();
}

/** Send an STL to the engine for validation. Rejects with a UserError if it is refused. */
export async function uploadModel(file: File): Promise<LoadedModel> {
  const form = new FormData();
  form.append("file", file, file.name);
  let response: Response;
  try {
    response = await fetch("/api/models", { method: "POST", body: form });
  } catch {
    throw unreachable();
  }
  const body = await readJson(response);
  if (response.ok && body) return body as LoadedModel;
  throw new UserError(
    body?.message ?? `The geometry engine could not load the file (HTTP ${response.status}).`,
  );
}

export interface JobStatus {
  id: string;
  mode: "preview" | "final";
  state: "queued" | "running" | "done" | "failed" | "cancelled";
  progress: number;
  message: string;
  error: string | null;
}

/** Thrown when a job was cancelled (by the user or by a newer job). */
export class CancelledError extends Error {}

const POLL_MS = 150;
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function request(url: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch {
    throw unreachable();
  }
}

function parseMesh(buffer: ArrayBuffer, jobId: string): GeneratedMesh {
  // Layout: see pack_mesh in the server.
  const headerLength = new DataView(buffer).getUint32(0, true);
  const info: ResultInfo = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)),
  );
  const start = 4 + headerLength;
  return {
    jobId,
    info,
    positions: new Float32Array(buffer, start, info.vertices * 3),
    indices: new Uint32Array(buffer, start + info.vertices * 12, info.triangles * 3),
  };
}

/**
 * Run a generation job in the engine and wait for it. `onProgress` is called while it
 * runs. Aborting `signal` cancels the job in the engine; the promise then rejects with
 * CancelledError.
 */
export async function generate(
  modelId: string,
  settings: Settings,
  mode: "preview" | "final",
  onProgress: (status: JobStatus) => void,
  signal: AbortSignal,
): Promise<GeneratedMesh> {
  const response = await request(`/api/models/${modelId}/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ settings, mode }),
  });
  if (!response.ok) {
    const body = await readJson(response);
    const detail = Array.isArray(body?.detail)
      ? body.detail.map((d: { msg: string }) => d.msg.replace(/^Value error, /, "")).join("\n")
      : null;
    throw new UserError(
      body?.message ?? detail ?? `The engine refused the request (HTTP ${response.status}).`,
    );
  }
  let status: JobStatus = await response.json();
  const cancel = () => void fetch(`/api/jobs/${status.id}/cancel`, { method: "POST" }).catch(() => {});
  if (signal.aborted) cancel();
  signal.addEventListener("abort", cancel, { once: true });
  try {
    while (status.state === "queued" || status.state === "running") {
      onProgress(status);
      await sleep(POLL_MS);
      if (signal.aborted) throw new CancelledError();
      const poll = await request(`/api/jobs/${status.id}`);
      if (!poll.ok) throw new UserError("The engine lost track of the job. Try again.");
      status = await poll.json();
    }
    if (status.state === "cancelled") throw new CancelledError();
    if (status.state === "failed") throw new UserError(status.error ?? "Generation failed.");
    onProgress(status);
    const result = await request(`/api/jobs/${status.id}/result`);
    if (!result.ok) {
      const body = await readJson(result);
      throw new UserError(body?.message ?? "The result could not be fetched from the engine.");
    }
    return parseMesh(await result.arrayBuffer(), status.id);
  } finally {
    signal.removeEventListener("abort", cancel);
  }
}

function saveAs(url: string, filename?: string): void {
  const link = document.createElement("a");
  link.href = url;
  if (filename) link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

/**
 * Download a validated final result (3MF or STL) and, next to it, the settings used.
 * Returns the two file names. The engine names them <input name>_voronoi.3mf|stl / .json.
 */
export async function exportResult(jobId: string, format: "3mf" | "stl"): Promise<[string, string]> {
  // The small settings file is fetched first: if the engine refuses the export
  // (not final, not valid, result gone) the reason arrives here as a message.
  const response = await request(`/api/jobs/${jobId}/export.json?format=${format}`);
  if (!response.ok) {
    const body = await readJson(response);
    throw new UserError(body?.message ?? "This result is no longer available. Generate final again.");
  }
  const jsonName =
    /filename="([^"]+)"/.exec(response.headers.get("Content-Disposition") ?? "")?.[1] ??
    "model_voronoi.json";
  const modelName = jsonName.replace(/\.json$/, `.${format}`);
  saveAs(`/api/jobs/${jobId}/export.${format}`); // large: streamed straight to disk by the browser
  const url = URL.createObjectURL(await response.blob());
  await sleep(400); // browsers drop a second download started in the same instant
  saveAs(url, jsonName);
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
  return [modelName, jsonName];
}

export interface ModelEdges {
  count: number;
  /** Two points (xyz xyz) per segment. */
  positions: Float32Array;
  /** The model edge each segment belongs to. */
  edgeOfSegment: Uint32Array;
}

/** Sharp edges of a loaded model, for display and picking. Layout: see model_edges in the server. */
export async function fetchEdges(modelId: string): Promise<ModelEdges> {
  const response = await request(`/api/models/${modelId}/edges`);
  if (!response.ok) throw new UserError("Could not fetch the edges of the model from the engine.");
  const buffer = await response.arrayBuffer();
  const head = new Uint32Array(buffer, 0, 2);
  const segments = head[0];
  return {
    count: head[1],
    positions: new Float32Array(buffer, 8, segments * 6),
    edgeOfSegment: new Uint32Array(buffer, 8 + segments * 24, segments),
  };
}
