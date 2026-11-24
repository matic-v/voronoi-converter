"""Step 1 tests: test models, settings schema, API scaffold."""

from __future__ import annotations

import io

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from pydantic import ValidationError

from voronoi_engine import export_schema, test_models
from voronoi_engine.settings import Settings, field_metadata
from voronoi_server.app import app

EXPECTED_SIZE = {
    "cube_50": (50.0, 50.0, 50.0),
    "sphere_60": (60.0, 60.0, 60.0),
    "bracket_100": (100.0, 40.0, 60.0),
}


@pytest.mark.parametrize("name", list(test_models.TEST_MODELS))
def test_model_is_valid_solid(name: str) -> None:
    mesh = test_models.build(name)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0
    assert len(mesh.split(only_watertight=False)) == 1
    np.testing.assert_allclose(mesh.extents, EXPECTED_SIZE[name], atol=1e-6)
    assert mesh.bounds[0][2] == pytest.approx(0.0, abs=1e-9)  # rests on the bed


def test_model_volumes() -> None:
    assert test_models.cube().volume == pytest.approx(50.0**3)
    assert test_models.sphere().volume == pytest.approx(4 / 3 * np.pi * 30**3, rel=0.01)
    # upright 8x60 + base 52x8 + flange 40x2, extruded 40 mm
    assert test_models.bracket().volume == pytest.approx((8 * 60 + 52 * 8 + 40 * 2) * 40)


def test_bracket_has_thin_flange() -> None:
    mesh = test_models.bracket()
    far_end = mesh.vertices[mesh.vertices[:, 0] > mesh.bounds[1][0] - 1e-6]
    assert np.ptp(far_end[:, 2]) == pytest.approx(2.0)


def test_models_are_deterministic_and_round_trip_stl() -> None:
    for name in test_models.TEST_MODELS:
        data = test_models.to_binary_stl(test_models.build(name))
        assert data == test_models.to_binary_stl(test_models.build(name))
        loaded = trimesh.load(io.BytesIO(data), file_type="stl")
        assert loaded.is_watertight


def test_settings_defaults_match_spec() -> None:
    s = Settings()
    assert s.surface_style == "perforated" and s.interior_style == "struts"
    assert (s.shell_thickness, s.surface_cell_size, s.hole_size) == (2.0, 10.0, 60.0)
    assert (s.rib_width, s.emboss_depth) == (2.0, 0.8)
    assert (s.interior_cell_size, s.strut_diameter, s.blend_radius) == (12.0, 2.0, 0.5)
    assert (s.grading_mode, s.grading_strength) == ("off", 50.0)
    assert (s.grading_target, s.grading_direction) == ("cell_size", "denser_at_start")
    assert (s.solid_base_thickness, s.random_seed, s.nozzle_diameter) == (0.0, 1, 0.4)
    assert s.quality == "normal"


def test_settings_validation() -> None:
    with pytest.raises(ValidationError):
        Settings(strut_diameter=0.5)
    with pytest.raises(ValidationError):
        Settings(surface_style="embossed", shell_thickness=1.0, emboss_depth=1.5)
    with pytest.raises(ValidationError):
        Settings(unknown_setting=1)


def test_settings_json_round_trip() -> None:
    s = Settings(surface_style="rib_lattice", random_seed=42)
    saved = s.to_saved_json()
    assert Settings(**saved["settings"]) == s


def test_field_metadata_covers_every_setting() -> None:
    meta = field_metadata()
    assert [m["key"] for m in meta] == list(Settings.model_fields)
    assert all(m["group"] in {"surface", "interior", "modifiers", "keep", "print"} for m in meta)


def test_generated_typescript_is_up_to_date() -> None:
    assert export_schema.OUTPUT.read_text(encoding="utf-8") == export_schema.render()


def test_api_scaffold() -> None:
    client = TestClient(app)
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/settings/schema").json()["defaults"] == Settings().model_dump(mode="json")
    models = client.get("/api/test-models").json()
    assert {m["name"] for m in models if not m["broken"]} == (
        set(test_models.TEST_MODELS) | set(test_models.FEATURE_MODELS))
    stl = client.get("/api/test-models/cube_50.stl")
    assert stl.status_code == 200 and len(stl.content) == 84 + 12 * 50
    assert client.get("/api/test-models/nope.stl").status_code == 404
