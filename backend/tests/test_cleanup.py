"""Clean-up: fins unfolded, mesh simplified within its tolerance, always a valid result."""

from __future__ import annotations

import numpy as np
import pytest

import voronoi_engine.cleanup as cleanup_module
from voronoi_engine import test_models
from voronoi_engine.cleanup import STRENGTHS, clean_up, count_fins, unfold_fins
from voronoi_engine.generate import generate
from voronoi_engine.mesh_io import load_stl
from voronoi_engine.repair import validate
from voronoi_engine.settings import Settings, field_metadata
from voronoi_engine.surface import SurfaceIndex


@pytest.fixture(scope="module")
def bracket():
    return load_stl(test_models.to_binary_stl(test_models.bracket()))


@pytest.fixture(scope="module")
def raw(bracket):
    """A final result without clean-up (perforated shell, smoothed)."""
    result = generate(bracket, Settings(surface_style="perforated", smoothness="smooth",
                                        cleanup="off"), "final")
    assert result.valid
    return result.vertices, result.faces


def test_setting() -> None:
    assert Settings().cleanup == "medium"
    meta = {m["key"]: m for m in field_metadata()}["cleanup"]
    assert meta["group"] == "print" and [o["value"] for o in meta["options"]] == [
        "off", "light", "medium", "strong"]


def test_unfolding_only_straightens(raw) -> None:
    v, f = raw
    before = count_fins(v, f, 120)
    after_v = unfold_fins(v, f, 120)
    assert before > 20  # the raw mesh has fins to remove
    assert count_fins(after_v, f, 120) < 0.2 * before
    assert count_fins(after_v, f, 150) <= count_fins(v, f, 150)
    assert not validate(after_v, f)


@pytest.mark.parametrize("strength", ["light", "medium", "strong"])
def test_strengths_simplify_within_tolerance(raw, strength: str) -> None:
    v0, f0 = raw
    v, f, report = clean_up(v0, f0, strength)
    assert not validate(v, f)
    assert report.triangles_after == len(f) < 0.7 * len(f0)
    assert report.fins_after < 0.25 * report.fins_before
    # Typical change (99 % of the points) within the strength's tolerance; the largest
    # changes are the fin tips moving back, a few tenths of a millimetre.
    there = np.abs(SurfaceIndex(v0, f0, 0.5).signed_distance(v))
    back = np.abs(SurfaceIndex(v, f, 0.5).signed_distance(v0))
    assert np.percentile(np.r_[there, back], 99) <= STRENGTHS[strength].tolerance_mm
    assert max(there.max(), back.max()) < 0.5


def test_stronger_means_fewer_triangles(raw) -> None:
    counts = [len(clean_up(*raw, s)[1]) for s in ("light", "medium", "strong")]
    assert counts[0] > counts[1] > counts[2]


def test_deterministic(raw) -> None:
    a = clean_up(*raw, "medium")
    b = clean_up(*raw, "medium")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_pipeline_uses_it_for_final_only(bracket) -> None:
    on = generate(bracket, Settings(), "final")
    off = generate(bracket, Settings(cleanup="off"), "final")
    assert on.valid and off.valid and len(on.faces) < len(off.faces)
    assert count_fins(on.vertices, on.faces, 120) < 0.25 * count_fins(off.vertices, off.faces, 120)
    preview_on = generate(bracket, Settings(), "preview")
    preview_off = generate(bracket, Settings(cleanup="off"), "preview")
    assert np.array_equal(preview_on.faces, preview_off.faces)


def test_falls_back_when_simplifying_breaks_the_mesh(bracket, monkeypatch) -> None:
    def broken(vertices, faces, epsilon):
        # A third triangle on an edge, to a new point: the kind of defect repair cannot fix.
        a, b, c = faces[0]
        tip = vertices[[a, b, c]].mean(axis=0) + 0.3
        return np.vstack([vertices, tip]), np.vstack([faces, [b, a, len(vertices)]])

    monkeypatch.setattr(cleanup_module, "simplify", broken)
    result = generate(bracket, Settings(surface_style="rib_lattice"), "final")
    assert result.valid
    assert any("only the folded triangles were straightened" in w for w in result.warnings)
