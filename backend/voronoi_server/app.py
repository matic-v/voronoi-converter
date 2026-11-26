"""Local web API. Serves the built front end from frontend/dist when present."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from voronoi_engine import __version__, test_models
from voronoi_engine.body_field import BodyField
from voronoi_engine.export import export_stem, settings_document, to_3mf, to_binary_stl
from voronoi_engine.generate import field_cache_key, generate
from voronoi_engine.mesh_io import LoadedMesh, MeshValidationError, load_stl
from voronoi_engine.orientation import suggest_bed_normal
from voronoi_engine.progress import Job
from voronoi_engine.settings import SETTINGS_VERSION, Settings, field_metadata

from .jobs import JobManager, JobRecord

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

app = FastAPI(title="Voronoi Body Generator", version=__version__)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/api/settings/schema")
def settings_schema() -> dict:
    return {
        "settings_version": SETTINGS_VERSION,
        "defaults": Settings().model_dump(mode="json"),
        "fields": field_metadata(),
        "json_schema": Settings.model_json_schema(),
    }


ALL_TEST_MODELS = {**test_models.TEST_MODELS, **test_models.FEATURE_MODELS,
                   **test_models.BROKEN_MODELS}


@lru_cache(maxsize=None)
def _test_model_stl(name: str) -> bytes:
    return test_models.to_binary_stl(test_models.build(name))


@app.get("/api/test-models")
def list_test_models() -> list[dict]:
    return [
        {"name": name, "label": label, "filename": f"{name}.stl",
         "broken": name in test_models.BROKEN_MODELS}
        for name, (label, _) in ALL_TEST_MODELS.items()
    ]


@app.get("/api/test-models/{filename}")
def get_test_model(filename: str) -> Response:
    name = filename.removesuffix(".stl")
    if name not in ALL_TEST_MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown test model '{name}'.")
    return Response(
        content=_test_model_stl(name),
        media_type="model/stl",
        headers={"Content-Disposition": f'attachment; filename="{name}.stl"'},
    )


# --- input models ------------------------------------------------------------

MAX_UPLOAD_BYTES = 300 * 1024 * 1024
MAX_STORED_MODELS = 3
MODELS: "OrderedDict[str, LoadedMesh]" = OrderedDict()


@app.post("/api/models")
def upload_model(file: UploadFile) -> JSONResponse:
    """Load and validate an STL. Broken input is refused with HTTP 422."""
    started = time.perf_counter()
    name = Path(file.filename or "model.stl").name
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    try:
        if len(data) > MAX_UPLOAD_BYTES:
            raise MeshValidationError(
                name, [f"the file is larger than {MAX_UPLOAD_BYTES // 2**20} MB"])
        mesh = load_stl(data, name)
    except MeshValidationError as error:
        return JSONResponse(
            status_code=422, content={"message": error.message, "problems": error.problems})
    model_id = hashlib.sha1(data).hexdigest()[:16]
    MODELS[model_id] = mesh
    MODELS.move_to_end(model_id)
    while len(MODELS) > MAX_STORED_MODELS:
        MODELS.popitem(last=False)
    edges = mesh.edges
    return JSONResponse({
        "id": model_id,
        "name": name,
        "edges": {"count": edges.count, "sharp": edges.sharp_count,
                  "curved": edges.curved_count, "too_many": edges.too_many},
        "info": asdict(mesh.info),
        "warnings": mesh.warnings,
        "load_seconds": round(time.perf_counter() - started, 3),
    })


@app.get("/api/models/{model_id}/suggest-bed")
def suggest_bed(model_id: str, overhang_angle: float = 45.0) -> dict:
    """The flat face to put on the bed for the least overhanging outer surface."""
    mesh = MODELS.get(model_id)
    if mesh is None:
        raise HTTPException(status_code=404, detail="Unknown model.")
    return suggest_bed_normal(mesh, min(max(overhang_angle, 30.0), 60.0))


@app.get("/api/models/{model_id}/edges")
def model_edges(model_id: str) -> Response:
    """Sharp edges for display and picking:
    [uint32 segment count][uint32 edge count][float32 xyz xyz per segment][uint32 edge id per segment]."""
    mesh = MODELS.get(model_id)
    if mesh is None:
        raise HTTPException(status_code=404, detail="Unknown model.")
    edges = mesh.edges
    return Response(content=b"".join([
        np.uint32(len(edges.segments)).tobytes(), np.uint32(edges.count).tobytes(),
        np.ascontiguousarray(edges.segments, dtype="<f4").tobytes(),
        np.ascontiguousarray(edges.edge_of_segment, dtype="<u4").tobytes(),
    ]), media_type="application/octet-stream")


# --- generation ----------------------------------------------------------------

class GenerateRequest(BaseModel):
    settings: Settings
    mode: Literal["preview", "final"] = "preview"


# Body distance fields are reused while only settings change: per model, the most
# recent preview and final grids.
FIELD_CACHE: dict[str, "OrderedDict[float, BodyField]"] = {}
MAX_FIELDS_PER_MODEL = 2
GENERATE_LOCK = threading.Lock()


def pack_mesh(info: dict, vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """[uint32 header length][JSON header, space padded to 4 bytes][float32 xyz][uint32 faces]."""
    header = json.dumps(info).encode()
    header += b" " * (-len(header) % 4)
    return b"".join([
        np.uint32(len(header)).tobytes(), header,
        np.ascontiguousarray(vertices, dtype="<f4").tobytes(),
        np.ascontiguousarray(faces, dtype="<u4").tobytes(),
    ])


JOBS = JobManager()


@app.post("/api/models/{model_id}/jobs", status_code=202)
def start_job(model_id: str, request: GenerateRequest) -> JSONResponse:
    """Start a preview or final generation in the background. Any older unfinished job is cancelled."""
    mesh = MODELS.get(model_id)
    if mesh is None:
        return JSONResponse(status_code=404, content={
            "message": "The model is no longer loaded in the engine. Load the STL again."})

    def work(job: Job):
        with GENERATE_LOCK:
            cache = FIELD_CACHE.setdefault(model_id, OrderedDict())
            for stale in [key for key in FIELD_CACHE if key not in MODELS]:
                del FIELD_CACHE[stale]
            result = generate(mesh, request.settings, request.mode, cache, job)
            cache.move_to_end(field_cache_key(result.voxel_mm, request.settings))
            while len(cache) > MAX_FIELDS_PER_MODEL:
                cache.popitem(last=False)
        return result, pack_mesh(result.info(), result.vertices, result.faces)

    record = JOBS.submit(model_id, request.mode, work, meta={
        "settings": request.settings, "input_name": mesh.name, "input_info": asdict(mesh.info)})
    return JSONResponse(status_code=202, content=record.status())


def _job_or_404(job_id: str) -> JobRecord:
    record = JOBS.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    return record


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return _job_or_404(job_id).status()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    _job_or_404(job_id)
    return JOBS.cancel(job_id).status()


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str) -> Response:
    record = _job_or_404(job_id)
    if record.result is None:
        return JSONResponse(status_code=409, content={
            "message": "This result is not available (the job is not finished, or a newer "
                       "result replaced it)."})
    return Response(content=record.result, media_type="application/octet-stream")


# --- export --------------------------------------------------------------------

def _exportable(job_id: str) -> JobRecord | JSONResponse:
    """The job's record if it may be exported, else the refusal to send."""
    record = _job_or_404(job_id)
    info = record.meta.get("result_info")
    if record.result is None or info is None:
        return JSONResponse(status_code=409, content={
            "message": "There is no finished result to export. Generate final first."})
    if record.mode != "final":
        return JSONResponse(status_code=409, content={
            "message": "Only a final result can be exported. Click Generate final first."})
    if not info["valid"]:
        return JSONResponse(status_code=409, content={
            "message": "This result did not pass validation and cannot be exported: "
                       + "; ".join(info["problems"]) + "."})
    return record


def unpack_mesh(packed: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of pack_mesh (vertices, faces)."""
    length = int(np.frombuffer(packed, dtype="<u4", count=1)[0])
    info = json.loads(packed[4:4 + length])
    start = 4 + length
    vertices = np.frombuffer(packed, dtype="<f4", count=info["vertices"] * 3, offset=start)
    faces = np.frombuffer(packed, dtype="<u4", count=info["triangles"] * 3,
                          offset=start + info["vertices"] * 12)
    return vertices.reshape(-1, 3), faces.reshape(-1, 3)


@app.get("/api/jobs/{job_id}/export.stl")
def export_stl(job_id: str) -> Response:
    """Binary STL of a validated final result, named <input name>_voronoi.stl."""
    record = _exportable(job_id)
    if isinstance(record, JSONResponse):
        return record
    name = export_stem(record.meta["input_name"]) + ".stl"
    return Response(
        content=to_binary_stl(*unpack_mesh(record.result)), media_type="model/stl",
        headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/jobs/{job_id}/export.3mf")
def export_3mf(job_id: str) -> Response:
    """3MF of a validated final result, named <input name>_voronoi.3mf."""
    record = _exportable(job_id)
    if isinstance(record, JSONResponse):
        return record
    stem = export_stem(record.meta["input_name"])
    return Response(
        content=to_3mf(*unpack_mesh(record.result), title=stem),
        media_type="model/3mf",
        headers={"Content-Disposition": f'attachment; filename="{stem}.3mf"'})


@app.get("/api/jobs/{job_id}/export.json")
def export_settings(job_id: str, format: Literal["3mf", "stl"] | None = None) -> Response:
    """The settings used for that result, to be saved next to the exported model.
    `format` records the export format chosen when exporting."""
    record = _exportable(job_id)
    if isinstance(record, JSONResponse):
        return record
    settings = record.meta["settings"]
    if format is not None:
        settings = settings.model_copy(update={"export_format": format})
    document = settings_document(settings, record.meta["input_name"],
                                 record.meta["input_info"], record.meta["result_info"])
    name = export_stem(record.meta["input_name"]) + ".json"
    return Response(
        content=json.dumps(document, indent=2), media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{name}"'})


if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
