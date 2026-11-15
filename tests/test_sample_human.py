"""Sample integration contracts; synthetic geometry is not recognition evidence."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fall_detect.config import load_config, project_path, recipe_fingerprint, validate
from fall_detect.io_utils import ContractError, read_json
from fall_detect.models.stgcn import HumanFeatureExtractor
from fall_detect.models.weights import select_device
from fall_detect.pose.human import add_track_proposals


def test_track_crop_proposals_are_bounded_and_do_not_change_observations():
    measured = torch.tensor([[10, 20, 100, 200, 0.9, 1, 0]], dtype=torch.float32)
    original = measured.clone()
    track = lambda box, age=1: SimpleNamespace(to_tlbr=lambda: np.array(box), time_since_update=age)
    tracks = [
        track([-20, -10, 120, 220]),
        track([0, 0, 100, 200], 31),
        track([np.nan, 0, 1, 2]),
        track([500, 0, 600, 100]),
    ]
    result = add_track_proposals(measured, tracks, (400, 240), 30)
    assert result.shape == (2, 7)
    torch.testing.assert_close(result[1, :4], torch.tensor([0.0, 0.0, 120.0, 220.0]))
    torch.testing.assert_close(measured, original)
    assert add_track_proposals(None, [], (400, 240), 30) is None
    assert add_track_proposals(None, tracks, (400, 240), 30).shape == (1, 7)


def test_track_proposals_are_opt_in_and_change_cache_identity():
    config = load_config()
    historical = deepcopy(config)
    historical["pose"].pop("use_track_proposals")
    assert recipe_fingerprint(config, {}) == recipe_fingerprint(historical, {})
    config["pose"]["use_track_proposals"] = True
    assert recipe_fingerprint(config, {}) != recipe_fingerprint(historical, {})
    other = load_config(project_path("configs/ur-ae-decay-001.yaml"))
    other["pose"]["use_track_proposals"] = True
    with pytest.raises(ContractError, match="Human AlphaPose"):
        validate(other)


def test_real_sample_weights_extract_preclassifier_features(skeleton):
    # Genuine downloaded parameters with synthetic coordinates: interface evidence only.
    filename = project_path("weights/sample-human-manifest.json")
    if not filename.exists():
        pytest.skip("Real sample resources unavailable")
    resources = {r["role"]: r for r in read_json(filename)["resources"]}
    if any(not project_path(r["local_path"]).exists() for r in resources.values()):
        pytest.skip("Real sample resources incomplete")
    config = load_config(project_path("configs/ur-sample-human-s5-001.yaml"))
    extractor = HumanFeatureExtractor(config, resources, select_device("cpu"))
    features = extractor.extract_features(skeleton, (640, 480))
    assert features.shape == (1, 512) and np.isfinite(features).all()
    assert extractor.forward_verified and extractor.load_report["strict"]
    assert all(not p.requires_grad for p in extractor.model.parameters())
    # The seven-class head cannot influence AE features.
    with torch.no_grad():
        extractor.model.fcn.weight.zero_()
        extractor.model.fcn.bias.fill_(20)
    np.testing.assert_array_equal(features, extractor.extract_features(skeleton, (640, 480)))
