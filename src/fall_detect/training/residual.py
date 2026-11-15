"""Normal-Train residual scales for frozen AE experiments, never classifier probabilities."""

import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from ..config import ANOMALY_SCORE_MODES, validate
from ..io_utils import ContractError, atomic_bytes, read_json, sha256, write_json


def weights_from_means(channel_means, mode):
    means = np.asarray(channel_means, dtype=np.float64)
    if means.ndim != 1 or not len(means) or not np.isfinite(means).all() or (means < 0).any():
        raise ContractError("Residual scales require finite nonnegative channel means")
    positive = means[means > 0]
    floor = max(1e-12, 0.1 * float(np.median(positive))) if len(positive) else 1e-12
    if mode == "train_channel_mse":
        scales = means
    elif mode == "train_stream_mse" and len(means) == 512:
        scales = np.repeat([means[:256].mean(), means[256:].mean()], 256)
    else:
        raise ContractError("Unsupported residual mode or Human stream dimension")
    weights = 1.0 / np.maximum(scales, floor)
    weights /= weights.mean()
    return weights.astype(np.float32), floor


@torch.inference_mode()
def fit_residual_profile(scorer, mode):
    from .experiment import normal_matrix

    # This is the sole data access in fitting: normal_matrix filters split AND label.
    features, hashes, counts = normal_matrix(scorer.index, "train", return_counts=True)
    normalized = scorer.scaler.transform(features)
    total = np.zeros(normalized.shape[1], dtype=np.float64)
    for value in normalized:
        batch = torch.as_tensor(value[None], dtype=torch.float32, device=scorer.device)
        total += ((batch - scorer.model(batch)) ** 2).cpu().numpy()[0]
    means = total / len(normalized)
    weights, floor = weights_from_means(means, mode)
    return {
        "schema_version": 1,
        "mode": mode,
        "fit_split": "train",
        "fit_label": 0,
        "input_dim": len(means),
        "recipe_hash": scorer.index["recipe_hash"],
        "split_hash": scorer.index["split_manifest"]["split_hash"],
        "model_sha256": sha256(scorer.run / "autoencoder.pt"),
        "scaler_sha256": sha256(scorer.run / "scaler.npz"),
        "feature_hashes": hashes,
        "sequence_counts": counts,
        "n_windows": len(normalized),
        "floor": floor,
        "channel_means": means.tolist(),
        "weights": weights.tolist(),
        "weight_rule": "inverse Train mean squared residual, fixed floor; mean weight=1",
        "test_used": False,
        "validation_used_for_scales": False,
    }


def load_residual_weights(scorer):
    mode = scorer.config["anomaly_score"]["mode"]
    if mode == "mse":
        return None
    profile = read_json(scorer.run / "residual-profile.json")
    train_ids = {
        r["sequence_id"]
        for r in scorer.index["split_manifest"]["sequences"]
        if r["split"] == "train" and r["label"] == 0
    }
    if (
        profile.get("schema_version") != 1
        or profile.get("mode") != mode
        or profile.get("fit_split") != "train"
        or profile.get("fit_label") != 0
        or profile.get("input_dim") != scorer.model.input_dim
        or profile.get("recipe_hash") != scorer.index["recipe_hash"]
        or profile.get("split_hash") != scorer.index["split_manifest"]["split_hash"]
        or profile.get("model_sha256") != sha256(scorer.run / "autoencoder.pt")
        or profile.get("scaler_sha256") != sha256(scorer.run / "scaler.npz")
        or set(profile.get("feature_hashes", {})) != train_ids
        or profile.get("feature_hashes")
        != read_json(scorer.run / "sources.json").get("train_feature_hashes")
        or set(profile.get("sequence_counts", {})) != train_ids
        or profile.get("test_used") is not False
        or profile.get("validation_used_for_scales") is not False
    ):
        raise ContractError("Residual profile provenance mismatch")
    weights, floor = weights_from_means(profile["channel_means"], mode)
    if (
        len(weights) != scorer.model.input_dim
        or not np.array_equal(weights, np.asarray(profile["weights"], dtype=np.float32))
        or floor != profile["floor"]
        or profile["n_windows"] != sum(profile["sequence_counts"].values())
        or profile["n_windows"] <= 0
    ):
        raise ContractError("Residual profile weights/counts mismatch")
    return weights


def derive_residual_run(source, destination, mode):
    """Reuse one frozen model; record a new Train-only score profile in a new run."""
    from .experiment import ARTIFACTS, Scorer, artifact_hashes, verify_artifacts

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ContractError("Residual output run exists; choose a new directory")
    if mode not in ANOMALY_SCORE_MODES or mode == "mse":
        raise ContractError("Choose an explicit Train residual profile")
    before = verify_artifacts(source)
    scorer = Scorer(source)
    if scorer.residual_weights is not None:
        raise ContractError("Residual profiles must derive directly from an MSE baseline")
    config = scorer.config
    config["anomaly_score"]["mode"] = mode
    validate(config)
    profile = fit_residual_profile(scorer, mode)
    destination.mkdir(parents=True)
    for name in ARTIFACTS:
        shutil.copyfile(source / name, destination / name)
    atomic_bytes(
        destination / "resolved-config.yaml",
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False).encode("utf-8"),
    )
    write_json(destination / "residual-profile.json", profile)
    provenance = {
        "source_run": str(source),
        "source_frozen_manifest_sha256": sha256(source / "frozen-artifacts.json"),
        "training_repeated": False,
        "additional_epochs": 0,
        "score_mode": mode,
        "residual_fit_split": "train",
        "selection_split": "validation",
        "test_used": False,
    }
    sources = read_json(destination / "sources.json")
    write_json(destination / "sources.json", {**sources, "residual_derivation": provenance})
    status = read_json(source / "status.json")
    write_json(
        destination / "status.json",
        {
            **status,
            "calibration": "not_run",
            "training_reused_from": source.name,
            "additional_epochs": 0,
            "residual_fit": "complete",
        },
    )
    write_json(destination / "training-artifacts.json", artifact_hashes(destination, training=True))
    if verify_artifacts(source) != before:
        raise ContractError("Source run changed during residual derivation")
    return destination
