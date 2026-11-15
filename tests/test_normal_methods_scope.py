"""Risk tests for research execution boundaries, not recognition evidence."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fall_detect.contracts import FeatureCache, FeatureRecord
from fall_detect.io_utils import ContractError, sha256
from fall_detect.training.scaler import FeatureScaler

SPEC = importlib.util.spec_from_file_location(
    "normal_method_workflow", Path(__file__).parents[1] / "scripts/ur_normal_methods.py"
)
WORKFLOW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKFLOW)


def index():
    sources = [{"sequence_id": f"adl-{i:02d}", "label": 0, "split": "train"} for i in range(24)]
    sources += [
        {"sequence_id": f"val-{i:02d}", "label": int(i >= 8), "split": "validation"}
        for i in range(23)
    ]
    sources += [{"sequence_id": "test-private", "label": 1, "split": "test"}]
    return {
        "source_kind": "real",
        "preparation_scope": ["train", "validation"],
        "split_manifest": {"sequences": sources},
        "sequences": {
            r["sequence_id"]: {"status": "not_prepared" if r["split"] == "test" else "prepared"}
            for r in sources
        },
    }


@pytest.mark.parametrize("failure", ["fall_train", "missing_validation", "unscoped", "incomplete"])
def test_scope_cannot_silently_expand_or_shrink(monkeypatch, failure):
    data = index()
    if failure == "fall_train":
        data["split_manifest"]["sequences"][0]["label"] = 1
    elif failure == "missing_validation":
        data["split_manifest"]["sequences"] = [
            r for r in data["split_manifest"]["sequences"] if r["sequence_id"] != "val-00"
        ]
    elif failure == "unscoped":
        data.pop("preparation_scope")
    else:
        data["sequences"]["adl-00"]["status"] = "not_prepared"
    monkeypatch.setattr(WORKFLOW, "load_index", lambda _: data)
    with pytest.raises(ContractError):
        WORKFLOW.validated_index()


@pytest.mark.parametrize("kind", ["svm", "temporal"])
def test_test_paths_never_requested_by_input_builder(monkeypatch, kind):
    data = index()
    calls = []

    def read_cache(_, sid):
        assert sid != "test-private"
        calls.append(sid)
        records = [
            FeatureRecord(sid, 1, i, i + 29, i, i + 29, i + 29, "train", i) for i in range(10)
        ]
        return FeatureCache(
            np.ones((10, 512), dtype=np.float32),
            records,
            {"sequence_id": sid, "stride": 1, "end_ms": 100},
        )

    monkeypatch.setattr(WORKFLOW, "cache_for", read_cache)
    scaler = FeatureScaler.fit(np.zeros((2, 512)))
    values, _, _ = WORKFLOW.inputs_for(data, {"kind": kind, "steps": 8}, scaler)
    assert len(calls) == 47 and len(values) == 47 and "test-private" not in values


def test_immutable_evidence_refuses_overwrite(tmp_path):
    file = tmp_path / "result.json"
    WORKFLOW.save(file, {"original": True})
    before = sha256(file)
    with pytest.raises(ContractError):
        WORKFLOW.save(file, {"original": False})
    assert sha256(file) == before
