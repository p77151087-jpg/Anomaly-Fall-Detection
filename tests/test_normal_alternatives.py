from dataclasses import replace

import numpy as np
import pytest
import torch

from fall_detect.contracts import FeatureCache, FeatureRecord
from fall_detect.detection.events import score_sequence
from fall_detect.io_utils import ContractError
from fall_detect.models.normal_alternatives import (
    TemporalAutoencoder,
    calibrate_normal_scores,
    candidate_rank,
    fit_svm,
    svm_scores,
    temporal_inputs,
)
from fall_detect.training.scaler import FeatureScaler


def cache(n=12):
    values = np.arange(n * 512, dtype=np.float32).reshape(n, 512) / 1000
    records = [
        FeatureRecord(
            "adl-01", 1, i, i + 29, i * 100 / 3, (i + 29) * 100 / 3, (i + 29) * 100 / 3, "train", i
        )
        for i in range(n)
    ]
    return FeatureCache(values, records, {"sequence_id": "adl-01", "stride": 1, "end_ms": 2000})


def test_svm_score_direction_and_normal_only_fit():
    rng = np.random.default_rng(42)
    train = rng.normal(size=(80, 3)) * 0.1
    scaler = FeatureScaler.fit(train)
    model = fit_svm(scaler.transform(train), 0.05, "scale")
    values = scaler.transform(np.array([[0, 0, 0], [20, 20, 20]], dtype=np.float32))
    scores = svm_scores(model, values)
    np.testing.assert_array_equal(scores, -model.decision_function(values.astype(np.float64)))
    assert scores[1] > scores[0]
    # Neither held-out labels nor values participate in either fit.
    other = fit_svm(scaler.transform(train), 0.05, "scale")
    np.testing.assert_array_equal(model.support_vectors_, other.support_vectors_)
    np.testing.assert_array_equal(model.dual_coef_, other.dual_coef_)


@pytest.mark.parametrize("bad", [np.array([[np.nan]]), np.empty((0, 2)), np.array([1, 2])])
def test_svm_refuses_invalid_training(bad):
    with pytest.raises(ContractError):
        fit_svm(bad, 0.01, "scale")


def test_temporal_preserves_denominator_and_causality():
    c = cache()
    inputs, adjusted, history = temporal_inputs(c, 8)
    assert inputs.shape == (5, 8, 512)
    assert len(adjusted.records) == len(c.records)
    np.testing.assert_array_equal(inputs[0], c.features[:8])
    assert history[0]["input_sample_indices"] == list(range(29, 37))
    assert history[0]["target_sample_index"] == 36
    future = cache()
    future.features[8:] = 100000
    np.testing.assert_array_equal(temporal_inputs(future, 8)[0][0], inputs[0])
    row, windows, _ = score_sequence(adjusted, np.ones(5), 0.5)
    assert row["n_scheduled_endpoints"] == 12
    assert row["n_valid_endpoints"] == 5
    assert row["first_alarm_ms"] == c.records[9].available_at_ms
    assert windows[0]["state"] == "unknown"


@pytest.mark.parametrize("failure", ["invalid", "gap", "track"])
def test_temporal_resets_and_never_pads(failure):
    c = cache(18)
    if failure == "invalid":
        c.records[8] = replace(c.records[8], row_index=None, quality="low_confidence")
    elif failure == "gap":
        c.records = [r for i, r in enumerate(c.records) if i != 8]
    else:
        c.records[8] = replace(c.records[8], track_id=2)
    _, _, history = temporal_inputs(c, 8)
    assert all(37 not in h["input_sample_indices"] for h in history)
    assert all(
        not (min(h["input_sample_indices"]) < 37 < max(h["input_sample_indices"])) for h in history
    )
    assert any(min(h["input_sample_indices"]) == 38 for h in history)


@pytest.mark.parametrize("failure", ["video", "future", "duplicate", "time"])
def test_temporal_rejects_invalid_timeline(failure):
    c = cache()
    if failure == "video":
        c.records[1] = replace(c.records[1], sequence_id="fall-19")
    elif failure == "future":
        c.records[1] = replace(c.records[1], available_at_ms=c.records[1].end_ms - 1)
    elif failure == "duplicate":
        c.records.append(c.records[0])
    else:
        c.records[1] = replace(
            c.records[1], available_at_ms=c.records[0].available_at_ms, end_ms=c.records[0].end_ms
        )
    with pytest.raises(ContractError):
        temporal_inputs(c, 8)


def test_temporal_gru_order_finite_and_strict_reload():
    torch.manual_seed(42)
    model = TemporalAutoencoder(16, 8, 4, 8).eval()
    x = torch.randn(2, 8, 16)
    output = model(x)
    assert output.shape == x.shape and torch.isfinite(output).all()
    assert not torch.allclose(output, model(x.flip(1)))
    second = TemporalAutoencoder(16, 8, 4, 8).eval()
    second.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(output, second(x), atol=0, rtol=0)


def test_threshold_negative_direction_strict_and_unknown_denominator():
    rows = [
        {"sequence_id": "adl", "label": 0, "sequence_score": -2.0, "prediction": 0},
        {"sequence_id": "fall", "label": 1, "sequence_score": -1.0, "prediction": 1},
        {"sequence_id": "unknown", "label": 1, "sequence_score": None, "prediction": None},
    ]
    chosen, _ = calibrate_normal_scores(rows)
    assert chosen["threshold"] == -2.0
    assert chosen["metrics"]["operational_accuracy"] == 2 / 3
    assert chosen["metrics"]["sequence_accuracy"] == 1.0
    assert chosen["metrics"]["n_total"] == 3 and chosen["metrics"]["n_unknown"] == 1
    assert candidate_rank(chosen["metrics"], 0) > candidate_rank(chosen["metrics"], 1)


def test_k3_is_not_replaced_by_single_peak():
    c = cache(6)
    scores = np.array([10.0, 0.0, 10.0, 0.0, 10.0, 0.0])
    result, _, events = score_sequence(c, scores, 1.0)
    assert result["sequence_score"] == 0 and result["prediction"] == 0 and not events


def test_calibration_cannot_silently_drop_a_class():
    with pytest.raises(ContractError):
        calibrate_normal_scores([{"label": 0, "sequence_score": 1.0}])
