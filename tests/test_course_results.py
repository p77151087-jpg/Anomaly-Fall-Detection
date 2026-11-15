import copy

import numpy as np
import pytest

from fall_detect.evaluation import course_report
from fall_detect.evaluation.metrics import sequence_metrics
from fall_detect.io_utils import ContractError, sha256, write_json
from fall_detect.visualization.presentation import display_values, draw_presentation, resolve_font


def test_unknown_denominator_and_tampered_metrics_rejected():
    rows = [
        {"sequence_id": "normal", "label": 0, "prediction": 0},
        {"sequence_id": "miss", "label": 1, "prediction": 0},
        {"sequence_id": "unavailable", "label": 1, "prediction": None},
    ]
    metrics = sequence_metrics(rows)
    assert metrics["sequence_accuracy"] == 0.5
    assert metrics["operational_accuracy"] == pytest.approx(1 / 3)
    course_report.verify_metrics(rows, metrics)
    for key in ("n_unknown", "n_evaluable", "n_total", "operational_accuracy"):
        tampered = {**metrics, key: 999}
        with pytest.raises(ContractError, match="Metric/denominator"):
            course_report.verify_metrics(rows, tampered)


def test_notebook_fails_closed_on_changed_or_missing_input(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence.csv"
    evidence.write_text("original", encoding="utf8")
    manifest = tmp_path / "docs/validation/course-report-inputs.json"
    write_json(manifest, {"files": {"evidence.csv": sha256(evidence)}})
    monkeypatch.setattr(course_report, "project_path", lambda p: tmp_path / p)
    evidence.write_text("tampered", encoding="utf8")
    with pytest.raises(ContractError, match="input missing or changed"):
        course_report.load_course_results()
    evidence.unlink()
    with pytest.raises(ContractError, match="input missing or changed"):
        course_report.load_course_results()


def test_unknown_never_displays_previous_score():
    stale = {"state": "unknown", "raw_score": 9.9, "event_score": 7.7, "available_at_ms": 1200}
    assert display_values(stale) == ("—", "—", "分數暫停更新")
    fresh = {"state": "normal", "raw_score": 0.1, "event_score": 0.08, "available_at_ms": 1200}
    assert display_values(fresh)[2] == "分數更新於 1.20 秒"


@pytest.mark.parametrize("mode", ["presentation", "simple"])
def test_renderer_preserves_inference_inputs(skeleton, mode):
    from fall_detect.contracts import PoseRecord

    path = None
    if mode == "presentation":
        try:
            path = resolve_font()
        except ContractError:
            pytest.skip("Windows Chinese font unavailable; real visual QA is separate")
    pixels = np.zeros((480, 640, 3), dtype=np.uint8)
    pose = PoseRecord(
        "fixture", 1, 1, 0, 0, np.array([20, 30, 400, 400]), skeleton[0].copy(), (640, 480), 0.9
    )
    states = {
        0: {
            "state": "suspected_fall",
            "raw_score": 0.7,
            "event_score": 0.6,
            "available_at_ms": 1300,
        }
    }
    original = copy.deepcopy(states)
    points = pose.keypoints.copy()
    if mode == "presentation":
        output = draw_presentation(
            pixels, [pose], states, 1333, 0.5, source_kind="synthetic", font_path=path
        )
        assert output.shape == (568, 960, 3)
    else:
        from fall_detect.visualization.simple import draw_simple

        output = draw_simple(pixels, [pose], states, 1333, 0.5, source_kind="synthetic")
        assert output.shape == (544, 640, 3)
    assert states == original and np.array_equal(points, pose.keypoints) and not pixels.any()
