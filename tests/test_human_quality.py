from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fall_detect.config import load_config, project_path, recipe_fingerprint, validate
from fall_detect.io_utils import ContractError, read_json
from fall_detect.pose.human import COCO17_MIRROR, pose_heatmaps
from fall_detect.training.patterns import pattern_sampler


def test_flip_mapping_and_legacy_single_forward():
    class Model:
        def __init__(self):
            self.inputs = []

        def __call__(self, x):
            self.inputs.append(x.clone())
            return torch.arange(17 * 3 * 4).reshape(1, 17, 3, 4).float() + len(self.inputs)

    crops = torch.randn(1, 3, 12, 16)
    model = Model()
    legacy = pose_heatmaps(model, crops)
    assert len(model.inputs) == 1
    model = Model()
    actual = pose_heatmaps(model, crops, True)
    assert len(model.inputs) == 2
    torch.testing.assert_close(model.inputs[1], crops.flip(-1), rtol=0, atol=0)
    expected = (legacy + (legacy + 1)[:, COCO17_MIRROR].flip(-1)) / 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert tuple(COCO17_MIRROR[i] for i in COCO17_MIRROR) == tuple(range(17))


def test_flip_matches_pinned_upstream_and_default_recipe(config):
    from fall_detect.models.weights import import_upstream

    import_upstream(config)
    from SPPE.src.utils.img import flip, shuffleLR

    heatmaps = torch.randn(2, 17, 64, 48)
    dataset = SimpleNamespace(flipRef=tuple((i, i + 1) for i in range(2, 18, 2)))
    expected = flip(shuffleLR(heatmaps.clone(), dataset))
    torch.testing.assert_close(heatmaps[:, COCO17_MIRROR].flip(-1), expected, rtol=0, atol=0)
    run = project_path("runs/ur-sample-human-s1-001")
    if run.exists():
        old = load_config(run / "resolved-config.yaml")
        index = read_json(run / "dataset-index.json")
        hashes = {k: v["sha256"] for k, v in index["weights"].items()}
        assert recipe_fingerprint(old, hashes) == index["recipe_hash"]
        old["pose"]["flip_tta"] = True
        assert recipe_fingerprint(old, hashes) != index["recipe_hash"]
    config["pose"]["flip_tta"] = "yes"
    with pytest.raises(ContractError, match="boolean"):
        validate(config)


def test_pattern_mass_reproducible_and_not_labels():
    values = np.random.default_rng(12).normal(size=(60, 12)).astype(np.float32)
    counts = {"a": 10, "empty": 0, "b": 50}
    a, profile = pattern_sampler(values, counts, torch.Generator().manual_seed(42))
    b, other = pattern_sampler(values, counts, torch.Generator().manual_seed(42))
    assert profile == other and list(a) == list(b)
    weights = np.asarray(profile["sample_weights"])
    labels = np.asarray(profile["assignments"])
    for start, stop in ((0, 10), (10, 60)):
        assert weights[start:stop].sum() == pytest.approx(1)
        groups = np.unique(labels[start:stop])
        for g in groups:
            assert weights[start:stop][labels[start:stop] == g].sum() == pytest.approx(
                1 / len(groups)
            )
    assert profile["fit_split"] == "train" and not profile["validation_used"]


@pytest.mark.parametrize(
    "values, counts",
    [(np.zeros((2, 3)), {"a": 3}), (np.full((2, 3), np.nan), {"a": 2}), (np.zeros((0, 3)), {})],
)
def test_pattern_invalid_inputs_rejected(values, counts):
    with pytest.raises(ContractError):
        pattern_sampler(values, counts, torch.Generator())


def test_pattern_training_fits_only_train(tmp_path, config, monkeypatch):
    from fall_detect.smoke import synthetic_dataset
    from fall_detect.training import experiment, patterns
    from fall_detect.training.scaler import FeatureScaler

    config["autoencoder"].update(sampling="sequence_patterns", max_epochs=2)
    index_path = synthetic_dataset(tmp_path / "features", config)
    index = read_json(index_path)
    train_x, _ = experiment.normal_matrix(index, "train")
    expected = FeatureScaler.fit(train_x).transform(train_x)
    original = patterns.pattern_sampler
    calls = []

    def checked(values, counts, generator, seed):
        np.testing.assert_array_equal(values, expected)
        assert all("train" in sid for sid in counts)
        calls.append(len(values))
        return original(values, counts, generator, seed)

    monkeypatch.setattr(patterns, "pattern_sampler", checked)
    original_cache = experiment.cache_for

    def no_test(index, sid):
        assert "test" not in sid
        return original_cache(index, sid)

    monkeypatch.setattr(experiment, "cache_for", no_test)
    run = experiment.train(config, index_path, tmp_path / "run", synthetic=True)
    experiment.calibrate(run)
    assert calls == [len(expected)]
    experiment.verify_artifacts(run)
    info = read_json(run / "sources.json")["training_sampling"]["normal_patterns"]
    assert len(info["assignments"]) == len(expected)
