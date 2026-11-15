import io
import random
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from ..config import CALIBRATION_OBJECTIVES, load_config, project_path, recipe_fingerprint
from ..data.cache import load_feature_cache
from ..data.splits import CAUCA_PROTOCOL, EXTENDED_PROTOCOL, PROTOCOL, validate_split
from ..detection.events import above_threshold, score_sequence
from ..evaluation.metrics import sequence_metrics
from ..io_utils import (
    ContractError,
    atomic_bytes,
    fingerprint,
    read_json,
    sha256,
    write_csv,
    write_json,
)
from ..models.autoencoder import Autoencoder, reconstruction_scores
from ..models.weights import select_device
from .scaler import FeatureScaler


def load_index(path):
    index = read_json(path)
    if index.get("schema_version") != 1 or index.get("source_kind") not in ("real", "synthetic"):
        raise ContractError("Invalid dataset index")
    validate_split(index["split_manifest"])
    if index["source_kind"] == "real" and index["split_manifest"]["protocol_id"] not in (
        PROTOCOL,
        EXTENDED_PROTOCOL,
        CAUCA_PROTOCOL,
    ):
        raise ContractError("Formal runs require a supported, validated sequence protocol")
    entries = index["split_manifest"]["sequences"]
    expected = {r["sequence_id"]: r for r in entries}
    if set(index["sequences"]) != set(expected):
        raise ContractError(
            "Dataset index must retain every planned sequence, including missing ones"
        )
    if "preparation_scope" in index:
        if index["preparation_scope"] != ["train", "validation"]:
            raise ContractError("Unsupported explicit preparation scope")
        for sid, source in expected.items():
            if source["split"] == "test" and (
                index["sequences"][sid].get("status") != "not_prepared"
                or any(k in index["sequences"][sid] for k in ("feature_path", "pose_path"))
            ):
                raise ContractError("Scoped Test must remain unprepared without cache paths")
    return index


def cache_for(index, sid):
    row = index["sequences"][sid]
    source = next(s for s in index["split_manifest"]["sequences"] if s["sequence_id"] == sid)
    scope = index.get("preparation_scope")
    if scope is not None and source["split"] not in scope:
        raise ContractError(f"{sid}: split outside explicit preparation scope")
    if row["status"] != "prepared":
        raise ContractError(f"{sid}: {row.get('reason', row['status'])}; run prepare")
    return load_feature_cache(
        project_path(row["feature_path"]),
        {
            "split_hash": index["split_manifest"]["split_hash"],
            "recipe_hash": index["recipe_hash"],
            "source_kind": index["source_kind"],
            "source_hash": row["source_hash"],
            "sequence_id": sid,
            "split": source["split"],
        },
    )


def unavailable_sequence(index, sid, config, reason):
    sample_count = index["sequences"][sid].get("n_samples")
    count = (
        len(
            range(config["skeleton"]["window_size"] - 1, sample_count, config["skeleton"]["stride"])
        )
        if sample_count is not None
        else None
    )
    return {
        "sequence_id": sid,
        "sequence_score": None,
        "prediction": None,
        "unknown_reason": str(reason),
        "n_scheduled_endpoints": count,
        "n_valid_endpoints": 0,
        "window_coverage": 0.0 if count else None,
        "first_alarm_ms": None,
    }


def normal_matrix(index, split, return_counts=False):
    arrays, hashes, counts = [], {}, {}
    for source in index["split_manifest"]["sequences"]:
        if source["split"] != split or source["label"] != 0:
            continue
        cache = cache_for(index, source["sequence_id"])
        arrays.append(cache.features)
        hashes[source["sequence_id"]] = cache.metadata["array_sha256"]
        counts[source["sequence_id"]] = len(cache.features)
    if not arrays or sum(len(a) for a in arrays) == 0:
        raise ContractError(f"No valid {split} ADL features")
    result = (np.concatenate(arrays), hashes)
    return (*result, counts) if return_counts else result


def training_sampler(counts, strategy, generator):
    """Give each usable Train ADL sequence equal expected mass, without inventing features."""
    if strategy == "windows":
        return None
    if strategy != "sequences":
        raise ContractError("Unsupported training sampling strategy")
    if not counts or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ContractError("Training sequence counts must be nonnegative integers")
    nonempty = [n for n in counts.values() if n]
    if not nonempty:
        raise ContractError("Sequence sampling requires actual normal training features")
    weights = torch.cat([torch.full((n,), 1.0 / n, dtype=torch.float64) for n in nonempty])
    return torch.utils.data.WeightedRandomSampler(
        weights, num_samples=sum(nonempty), replacement=True, generator=generator
    )


def save_torch(path, value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    atomic_bytes(path, buffer.getvalue())


def training_reconstruction_loss(model, clean, noise_std=0.0, generator=None):
    """Corrupt only a Train input; always reconstruct its original normal feature."""
    observed = clean
    if noise_std:
        noise = torch.randn(
            clean.shape, dtype=clean.dtype, device=clean.device, generator=generator
        )
        observed = clean + noise_std * noise
    return torch.nn.functional.mse_loss(model(observed), clean)


def train(config, index_path, run, synthetic=False, *, train_validation_only=False):
    if config["anomaly_score"]["mode"] != "mse":
        raise ContractError("Train the baseline MSE AE first, then derive a residual profile")
    run = Path(run)
    if run.exists():
        raise ContractError(f"Run already exists; choose a new run ID: {run}")
    try:
        return _train(config, index_path, run, synthetic, train_validation_only)
    except BaseException as exc:
        # Never turn an interrupted run into a completed model or leave it marked running.
        status_path = run / "status.json"
        if status_path.is_file():
            try:
                status = read_json(status_path)
                if status.get("training") == "running":
                    write_json(
                        status_path,
                        {
                            **status,
                            "training": "interrupted"
                            if isinstance(exc, KeyboardInterrupt)
                            else "failed",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                        },
                    )
            except (OSError, ValueError):
                pass  # Preserve the original exception if the filesystem remains unavailable.
        raise


def _train(config, index_path, run, synthetic=False, train_validation_only=False):
    from ..diagnostics import environment_report
    from ..pipeline import source_code_hash
    from ..visualization.reports import training_curve

    index = load_index(index_path)
    if index.get("preparation_scope") is not None and not train_validation_only:
        raise ContractError("Scoped index requires explicit Train/Validation-only training")
    if (index["source_kind"] == "synthetic") != synthetic:
        raise ContractError("Synthetic features require an explicitly synthetic run")
    if not synthetic:
        if config["data"]["protocol_id"] != index["split_manifest"]["protocol_id"]:
            raise ContractError("Training config and dataset protocol differ")
        weight_hashes = {role: entry["sha256"] for role, entry in index["weights"].items()}
        if recipe_fingerprint(config, weight_hashes) != index["recipe_hash"]:
            raise ContractError(
                "Training config differs from the prepared pose/feature recipe; rerun prepare"
            )
    if train_validation_only and index.get("preparation_scope") != ["train", "validation"]:
        raise ContractError("Train/Validation-only requires an explicitly scoped prepared index")
    required = [
        index["sequences"][source["sequence_id"]]
        for source in index["split_manifest"]["sequences"]
        if not train_validation_only or source["split"] in ("train", "validation")
    ]
    if (not synthetic or train_validation_only) and not all(
        row["status"] == "prepared" for row in required
    ):
        raise ContractError(
            "Formal training requires all planned sequences in the requested scope prepared"
        )
    run = Path(run)
    if run.exists():
        raise ContractError(f"Run already exists; choose a new run ID: {run}")
    train_x, train_hashes, train_counts = normal_matrix(index, "train", return_counts=True)
    val_x, val_hashes = normal_matrix(index, "validation")
    if train_x.shape[1] != config["features"]["expected_dim"] or val_x.shape[1] != train_x.shape[1]:
        raise ContractError("AE input_dim disagrees with feature recipe")
    seed = config["project"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    device = select_device(config["runtime"]["device"], config["runtime"]["torch_threads"])
    options = config["autoencoder"]
    scaler = FeatureScaler.fit(train_x)
    train_z, val_z = scaler.transform(train_x), scaler.transform(val_x)
    model = Autoencoder(train_x.shape[1], options["hidden_dim"], options["latent_dim"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=options["learning_rate"],
        weight_decay=options.get("weight_decay", 0.0),
    )
    generator = torch.Generator().manual_seed(seed)
    noise_std = options.get("input_noise_std", 0.0)
    noise_generator = torch.Generator(device=device).manual_seed(seed) if noise_std else None
    sampling = options.get("sampling", "windows")
    pattern_profile = None
    if sampling == "sequence_patterns":
        from .patterns import pattern_sampler

        sampler, pattern_profile = pattern_sampler(train_z, train_counts, generator, seed)
    else:
        sampler = training_sampler(train_counts, sampling, generator)
    loader = torch.utils.data.DataLoader(
        torch.from_numpy(train_z),
        batch_size=options["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator,
        num_workers=config["runtime"]["num_workers"],
    )
    run.mkdir(parents=True)
    write_json(
        run / "status.json",
        {"source_kind": index["source_kind"], "training": "running", "calibration": "not_run"},
    )
    atomic_bytes(
        run / "resolved-config.yaml", yaml.safe_dump(config, sort_keys=False).encode("utf-8")
    )
    write_json(run / "split-manifest.json", index["split_manifest"])
    write_json(run / "dataset-index.json", index)
    write_json(
        run / "environment.json",
        {
            **environment_report(),
            "training_device": str(device),
            "deterministic_algorithms": "warn_only",
            "cudnn_benchmark": False,
        },
    )
    write_json(
        run / "sources.json",
        {
            "recipe_hash": index["recipe_hash"],
            "source_kind": index["source_kind"],
            "weights": index.get("weights", {}),
            "train_feature_hashes": train_hashes,
            "val_normal_feature_hashes": val_hashes,
            "index_sha256": sha256(index_path),
            "pipeline_version": "0.1.0",
            "source_provenance": index.get("provenance", {}),
            "training_source_code_hash": source_code_hash(),
            "inference_scoring_batch_size": 1,
            "execution_scope": "train_validation_only" if train_validation_only else "all_splits",
            "training_regularization": {
                "weight_decay": options.get("weight_decay", 0.0),
                "input_noise_std": noise_std,
                "noise_space": "Train-standardized ST-GCN features",
                "noise_scope": "Train input only; clean targets, Validation and inference",
                "noise_seed": seed if noise_std else None,
            },
            "training_sampling": {
                **({"normal_patterns": pattern_profile} if pattern_profile is not None else {}),
                "strategy": sampling,
                "replacement": sampler is not None,
                "samples_per_epoch": len(train_z),
                "train_normal_feature_counts": train_counts,
                "validation_loss": "unweighted mean over all valid normal Validation windows",
                "scaler_fit": "all valid Train ADL windows, unchanged",
            },
        },
    )
    scaler.save(run / "scaler.npz")
    best_loss, stopping_loss, best_state, stale, history = float("inf"), float("inf"), None, 0, []
    for epoch in range(1, options["max_epochs"] + 1):
        started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = training_reconstruction_loss(model, batch, noise_std, noise_generator)
            if not torch.isfinite(loss):
                raise ContractError("AE training diverged to non-finite loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        val_loss = float(reconstruction_scores(model, val_z, device).mean())
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / len(train_z),
                "val_normal_loss": val_loss,
                "seconds": time.perf_counter() - started,
            }
        )
        # Publish one atomic snapshot after each completed epoch for the local viewer.
        write_json(
            run / "training-progress.json",
            {"source_kind": index["source_kind"], "training": "running", "history": history},
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if val_loss < stopping_loss - options["min_delta"]:
            stopping_loss, stale = val_loss, 0
        else:
            stale += 1
        print(
            f"epoch {epoch}: train={history[-1]['train_loss']:.6f}, val_normal={val_loss:.6f}",
            flush=True,
        )
        if stale >= options["patience"]:
            break
    if best_state is None:
        raise ContractError("No finite validation checkpoint")
    save_torch(
        run / "autoencoder.pt",
        {
            "schema_version": 1,
            "architecture": "mlp-ae-v1",
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "latent_dim": model.latent_dim,
            "recipe_hash": index["recipe_hash"],
            "state_dict": best_state,
        },
    )
    write_csv(run / "training-history.csv", history)
    training_curve(run / "training-curve.png", history, index["source_kind"])
    write_json(
        run / "status.json",
        {
            "source_kind": index["source_kind"],
            "training": "complete",
            "calibration": "not_run",
            "epochs": len(history),
            "best_val_normal_loss": best_loss,
        },
    )
    write_json(run / "training-artifacts.json", artifact_hashes(run, training=True))
    return run


ARTIFACTS = (
    "autoencoder.pt",
    "scaler.npz",
    "resolved-config.yaml",
    "split-manifest.json",
    "sources.json",
    "dataset-index.json",
    "environment.json",
    "training-history.csv",
)


def artifact_hashes(run, training=False):
    names = ARTIFACTS if training else ARTIFACTS + ("threshold.json", "calibration.json")
    if load_config(Path(run) / "resolved-config.yaml")["anomaly_score"]["mode"] != "mse":
        names += ("residual-profile.json",)
    return {name: sha256(Path(run) / name) for name in names}


def verify_artifacts(run, calibrated=True):
    record = "frozen-artifacts.json" if calibrated else "training-artifacts.json"
    if not (Path(run) / record).exists():
        raise ContractError(
            "Run is not calibrated/frozen" if calibrated else "Training has not completed"
        )
    expected = read_json(Path(run) / record)
    if expected != artifact_hashes(run, training=not calibrated):
        raise ContractError(
            "Frozen run artifact hash mismatch; do not mutate an existing experiment"
        )
    return expected


class Scorer:
    def __init__(self, run, device=None, calibrated=True):
        self.run = Path(run)
        verify_artifacts(run, calibrated)
        self.config = load_config(self.run / "resolved-config.yaml")
        self.index = load_index(self.run / "dataset-index.json")
        self.device = select_device(
            device or self.config["runtime"]["device"], self.config["runtime"]["torch_threads"]
        )
        self.scaler = FeatureScaler.load(self.run / "scaler.npz")
        checkpoint = torch.load(self.run / "autoencoder.pt", map_location="cpu", weights_only=True)
        if (
            checkpoint.get("schema_version") != 1
            or checkpoint.get("architecture") != "mlp-ae-v1"
            or checkpoint["recipe_hash"] != self.index["recipe_hash"]
            or checkpoint["input_dim"] != self.config["features"]["expected_dim"]
            or checkpoint["input_dim"] != len(self.scaler.mean)
        ):
            raise ContractError("AE architecture/dimension/recipe mismatch")
        self.model = Autoencoder(
            checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["latent_dim"]
        )
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        from .residual import load_residual_weights

        self.residual_weights = load_residual_weights(self)
        self.threshold = read_json(self.run / "threshold.json") if calibrated else None
        if self.threshold is not None:
            if (
                self.threshold["recipe_hash"] != self.index["recipe_hash"]
                or self.threshold["split_hash"] != self.index["split_manifest"]["split_hash"]
                or self.threshold["comparison"] != "gt"
                or self.threshold["consecutive_windows"] != 3
            ):
                raise ContractError("Threshold contract mismatch")
            above_threshold(0.0, self.threshold["threshold"])
            if self.residual_weights is not None and self.threshold.get(
                "residual_profile_sha256"
            ) != sha256(self.run / "residual-profile.json"):
                raise ContractError("Threshold residual profile mismatch")

    def score(self, features):
        # Calibration/evaluation use the same B=1 GEMM shape as streaming demo.
        # Tiny batch-size roundoff differences otherwise flip strict > at a chosen threshold.
        return reconstruction_scores(
            self.model,
            self.scaler.transform(features),
            self.device,
            batch_size=1,
            residual_weights=self.residual_weights,
        )


def choose_threshold(rows, objective="sequence_fall_f1"):
    if objective not in CALIBRATION_OBJECTIVES:
        raise ContractError(f"Unsupported calibration objective: {objective}")
    for label in (0, 1):
        if not any(r["label"] == label and r["sequence_score"] is not None for r in rows):
            raise ContractError(
                "Calibration requires at least one evaluable validation sequence of each class"
            )
    values = np.unique(
        np.array(
            [float(r["sequence_score"]) for r in rows if r["sequence_score"] is not None],
            dtype=np.float64,
        )
    )
    below = float(np.nextafter(values[0], -np.inf))
    if not np.isfinite(below) or not np.isfinite(values).all():
        raise ContractError("Cannot create finite calibration candidates")
    candidates = []
    for threshold in [below] + values.tolist():
        predictions = [
            {
                **r,
                "prediction": int(above_threshold(r["sequence_score"], threshold))
                if r["sequence_score"] is not None
                else None,
            }
            for r in rows
        ]
        metrics = sequence_metrics(predictions)
        # Include unknown sequences in each class denominator; never reward abstention.
        metrics["balanced_sequence_accuracy"] = 0.5 * (
            metrics["TP"] / metrics["by_class"]["Fall"]["n_total"]
            + metrics["TN"] / metrics["by_class"]["ADL"]["n_total"]
        )
        candidates.append({"threshold": float(threshold), "metrics": metrics})
    chosen = max(
        candidates,
        key=lambda r: (r["metrics"][objective], -r["metrics"]["FP"], r["threshold"]),
    )
    return chosen, candidates


def derive_calibration_run(source, destination, objective):
    """Copy a frozen trained model for an explicit alternative Validation-only calibration."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if objective not in CALIBRATION_OBJECTIVES:
        raise ContractError(f"Unsupported calibration objective: {objective}")
    before = verify_artifacts(source)
    config = load_config(source / "resolved-config.yaml")
    if config["calibration"]["objective"] == objective:
        raise ContractError("Alternative calibration must use a different objective")
    if destination.exists():
        raise ContractError("Calibration output run exists; choose a new directory")
    config["calibration"]["objective"] = objective
    destination.mkdir(parents=True)
    for name in ARTIFACTS:
        shutil.copyfile(source / name, destination / name)
    if config["anomaly_score"]["mode"] != "mse":
        shutil.copyfile(source / "residual-profile.json", destination / "residual-profile.json")
    atomic_bytes(
        destination / "resolved-config.yaml",
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False).encode("utf-8"),
    )
    provenance = {
        "source_run": str(source),
        "source_frozen_manifest_sha256": sha256(source / "frozen-artifacts.json"),
        "training_repeated": False,
        "objective": objective,
        "selection_split": "validation",
        "test_used": False,
    }
    sources = read_json(destination / "sources.json")
    write_json(destination / "sources.json", {**sources, "calibration_derivation": provenance})
    status = read_json(source / "status.json")
    write_json(
        destination / "status.json",
        {**status, "calibration": "not_run", "training_reused_from": source.name},
    )
    write_json(destination / "training-artifacts.json", artifact_hashes(destination, training=True))
    if verify_artifacts(source) != before:
        raise ContractError("Source run changed during calibration derivation")
    return destination


def calibrate(run):
    run = Path(run)
    if (run / "threshold.json").exists():
        raise ContractError("Run already calibrated; use a new run for changes")
    scorer = Scorer(run, calibrated=False)
    rows, feature_hashes = [], {}
    for source in scorer.index["split_manifest"]["sequences"]:
        if source["split"] != "validation":
            continue
        try:
            cache = cache_for(scorer.index, source["sequence_id"])
            result, _, _ = score_sequence(cache, scorer.score(cache.features), 0.0)
            feature_hashes[source["sequence_id"]] = cache.metadata["array_sha256"]
        except (ContractError, OSError) as exc:
            result = unavailable_sequence(scorer.index, source["sequence_id"], scorer.config, exc)
        rows.append({**result, "label": source["label"]})
    objective = scorer.config["calibration"]["objective"]
    chosen, candidates = choose_threshold(rows, objective)
    protocol = {
        "schema_version": 1,
        "threshold": float(chosen["threshold"]),
        "comparison": "gt",
        "consecutive_windows": 3,
        "split": "validation",
        "objective": objective,
        "protocol_id": scorer.index["split_manifest"]["protocol_id"],
        "split_hash": scorer.index["split_manifest"]["split_hash"],
        "recipe_hash": scorer.index["recipe_hash"],
        "feature_hashes": feature_hashes,
        "model_sha256": sha256(run / "autoencoder.pt"),
        "scaler_sha256": sha256(run / "scaler.npz"),
    }
    if scorer.residual_weights is not None:
        protocol["residual_profile_sha256"] = sha256(run / "residual-profile.json")
    write_json(
        run / "calibration.json",
        {
            "source_kind": scorer.index["source_kind"],
            "sequences": rows,
            "candidates": candidates,
            "chosen": chosen,
            "objective": objective,
            "selection_rule": f"max {objective}, fewer FP, higher threshold",
            "balanced_accuracy_definition": "0.5 * (TP / all Fall + TN / all ADL); unknown retained",
        },
    )
    write_json(run / "threshold.json", protocol)
    status = read_json(run / "status.json")
    write_json(run / "status.json", {**status, "calibration": "complete"})
    write_json(run / "frozen-artifacts.json", artifact_hashes(run))
    return protocol


def evaluate(run, split="test", output=None):
    from ..visualization.reports import evaluation_report

    if split != "test":
        raise ContractError("v1 evaluate accepts only frozen test evaluation")
    scorer = Scorer(run)
    if scorer.index.get("preparation_scope") is not None:
        raise ContractError("Test evaluation unavailable for a Train/Validation-only run")
    before = artifact_hashes(run)
    rows, windows, events = [], [], []
    for source in scorer.index["split_manifest"]["sequences"]:
        if source["split"] != "test":
            continue
        try:
            cache = cache_for(scorer.index, source["sequence_id"])
            result, sequence_windows, sequence_events = score_sequence(
                cache, scorer.score(cache.features), scorer.threshold["threshold"]
            )
            windows.extend(sequence_windows)
            events.extend(sequence_events)
        except (ContractError, OSError) as exc:
            result = unavailable_sequence(scorer.index, source["sequence_id"], scorer.config, exc)
        rows.append({**result, "label": source["label"]})
    metrics = {
        **sequence_metrics(rows),
        "protocol_id": scorer.index["split_manifest"]["protocol_id"],
        "split_hash": scorer.index["split_manifest"]["split_hash"],
        "recipe_hash": scorer.index["recipe_hash"],
        "source_kind": scorer.index["source_kind"],
        "artifact_hashes": before,
        "config_hash": fingerprint(scorer.config),
    }
    if scorer.index["split_manifest"]["protocol_id"] in (EXTENDED_PROTOCOL, CAUCA_PROTOCOL):
        assignments = {
            r["sequence_id"]: r["dataset"] for r in scorer.index["split_manifest"]["sequences"]
        }
        metrics["by_dataset"] = {
            name: sequence_metrics([r for r in rows if assignments[r["sequence_id"]] == name])
            for name in sorted(set(assignments.values()))
        }
    destination = (
        Path(output)
        if output
        else Path(run) / "evaluations" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    if destination.exists():
        raise ContractError("Evaluation output exists; choose a new directory")
    destination.mkdir(parents=True)
    evaluation_report(destination, metrics, rows, windows, events)
    if before != artifact_hashes(run):
        raise ContractError("Evaluation mutated frozen artifacts")
    return destination, metrics
