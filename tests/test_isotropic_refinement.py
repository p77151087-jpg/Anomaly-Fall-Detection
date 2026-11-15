"""The diagnostic/refinement workflow must never consume Test features."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from fall_detect.io_utils import ContractError
from fall_detect.training import experiment

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/human_isotropic_refinement.py"
spec = importlib.util.spec_from_file_location("isotropic_refinement", SCRIPT)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def test_guard_blocks_test_before_loader_and_records_allowed_access(monkeypatch):
    index = {
        "split_manifest": {
            "sequences": [
                {"sequence_id": "a", "split": "train"},
                {"sequence_id": "b", "split": "validation"},
                {"sequence_id": "c", "split": "test"},
            ]
        }
    }
    called = []
    monkeypatch.setattr(workflow, "ORIGINAL_CACHE_FOR", lambda i, s: called.append(s))
    accesses = []
    with workflow.train_validation_only(accesses):
        experiment.cache_for(index, "a")
        experiment.cache_for(index, "b")
        with pytest.raises(ContractError, match="forbids Test"):
            experiment.cache_for(index, "c")
    assert called == ["a", "b"]
    assert [r["split"] for r in accesses] == ["train", "validation"]


def test_diagnostic_distances_match_independent_euclidean_calculation():
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(5)
    train = rng.normal(size=(83, 6)).astype(np.float32)
    query = rng.normal(size=(71, 6)).astype(np.float32)
    owners = np.asarray([f"adl-{i}" for i in range(len(train))])
    expected = cdist(query, train, "sqeuclidean") / train.shape[1]
    distance, nearest = workflow.nearest_train(query, train, owners)
    np.testing.assert_allclose(distance, expected.min(axis=1), atol=1e-12)
    assert nearest == owners[expected.argmin(axis=1)].tolist()


def test_completed_evidence_cannot_be_overwritten(tmp_path):
    path = tmp_path / "complete.json"
    workflow.save_new(path, {"original": True})
    before = path.read_bytes()
    with pytest.raises(ContractError, match="Preserve"):
        workflow.save_new(path, {"original": False})
    assert path.read_bytes() == before


def test_actual_training_and_calibration_under_test_guard(tmp_path, config, monkeypatch):
    from fall_detect.smoke import synthetic_dataset

    config["autoencoder"].update(max_epochs=2, patience=2)
    index_path = synthetic_dataset(tmp_path / "features", config)
    accesses = []
    with workflow.train_validation_only(accesses):
        run = experiment.train(config, index_path, tmp_path / "runs/candidate", synthetic=True)
        experiment.calibrate(run)
        sys.path.insert(0, str(SCRIPT.parent))
        try:
            monkeypatch.setattr(workflow, "ROOT", tmp_path)
            # Real scoring and calibration use synthetic caches here, without any Test load.
            result = workflow.validation_result(
                "candidate",
                {
                    "coarse_intervals": {},
                    "reference_fn": [],
                    "reference_metrics": {
                        "operational_accuracy": 0,
                        "sequence_fall_f1": 0,
                        "FP": 100,
                        "n_unknown": 100,
                    },
                },
                export=False,
            )
        finally:
            sys.path.pop(0)
    assert result["metrics"]["n_total"] == 2
    assert result["eligible"] is False  # No fabricated timing evidence.
    assert {a["split"] for a in accesses} == {"train", "validation"}
