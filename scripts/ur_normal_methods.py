"""Bounded UR-only normal-data comparison. No Test entry point."""

import argparse
import copy
import io
import random
import shutil
from datetime import UTC, datetime

import joblib
import numpy as np
import torch

from fall_detect.config import project_path
from fall_detect.data.cache import load_pose_cache
from fall_detect.detection.events import score_sequence
from fall_detect.evaluation.metrics import sequence_metrics
from fall_detect.io_utils import (
    ContractError,
    atomic_bytes,
    fingerprint,
    read_json,
    sha256,
    write_json,
    write_npz,
)
from fall_detect.models.autoencoder import Autoencoder
from fall_detect.models.normal_alternatives import (
    TemporalAutoencoder,
    calibrate_normal_scores,
    candidate_rank,
    fit_svm,
    svm_scores,
    temporal_inputs,
)
from fall_detect.models.stgcn import HumanFeatureExtractor
from fall_detect.skeleton.windows import CausalResampler, WindowBuilder
from fall_detect.training.experiment import (
    Scorer,
    cache_for,
    choose_threshold,
    load_index,
    normal_matrix,
    training_sampler,
)
from fall_detect.training.scaler import FeatureScaler
from fall_detect.visualization.reports import evaluation_report, training_curve

ROOT = project_path(".")
TAG = "ur-normal-methods-20261012-001"
E = ROOT / "docs/validation" / TAG
O = ROOT / "outputs" / TAG
C = ROOT / "data/cache" / TAG
BASE = "ur-isotropic-wide-20261012-001"
IMPLEMENTATIONS = ["scripts/ur_normal_methods.py", "src/fall_detect/models/normal_alternatives.py"]


def save(path, value):
    if path.exists():
        raise ContractError(f"Preserve existing evidence: {path}")
    write_json(path, value)


def verify_lock():
    record = read_json(E / "svm-lock.json")
    for name, digest in record["implementation_hashes"].items():
        if sha256(ROOT / name) != digest:
            raise ContractError(f"Locked implementation changed: {name}")
    for name, digest in record["input_hashes"].items():
        if sha256(ROOT / name) != digest:
            raise ContractError(f"Locked input changed: {name}")
    return record


def validated_index():
    index = load_index(C / "dataset-index.json")
    if index.get("preparation_scope") != ["train", "validation"] or index["source_kind"] != "real":
        raise ContractError("Requires real, explicitly Train/Validation-only index")
    rows = index["split_manifest"]["sequences"]
    if len([r for r in rows if r["split"] == "train" and r["label"] == 0]) != 24:
        raise ContractError("Exactly 24 original Train ADL required")
    if any(r["label"] != 0 for r in rows if r["split"] == "train"):
        raise ContractError("Only normal Train permitted")
    if len([r for r in rows if r["split"] == "validation"]) != 23:
        raise ContractError("All 23 Validation sequences required")
    if any(
        index["sequences"][r["sequence_id"]]["status"] != "prepared"
        for r in rows
        if r["split"] != "test"
    ):
        raise ContractError("Incomplete Train/Validation preparation")
    return index


def phases_for(rows, windows, events, lock):
    phases = []
    for sid, interval in lock["coarse_intervals"].items():
        lo, hi = interval["fall_phase_ms"]
        selected = [w for w in windows if w["sequence_id"] == sid]
        good = {tuple(pair) for pair in lock["current_good_pose_keys"][sid]}
        phases.append(
            {
                "sequence_id": sid,
                "interval": interval,
                "phase_event_starts": [
                    e["confirmed_start_ms"]
                    for e in events
                    if e["sequence_id"] == sid and lo <= e["confirmed_start_ms"] <= hi
                ],
                "early_event_starts": [
                    e["confirmed_start_ms"]
                    for e in events
                    if e["sequence_id"] == sid
                    and e["confirmed_start_ms"] < interval["normal_until_ms"]
                ],
                "visible_phase_alert_ms": [
                    w["available_at_ms"]
                    for w in selected
                    if w["state"] == "suspected_fall"
                    and lo <= w["available_at_ms"] <= hi
                    and (w["sample_index"], w["track_id"]) in good
                ],
                "phase_valid_windows": sum(
                    w["raw_score"] is not None for w in selected if lo <= w["available_at_ms"] <= hi
                ),
                "phase_legal_k3": sum(
                    w["event_score"] is not None
                    for w in selected
                    if lo <= w["available_at_ms"] <= hi
                ),
            }
        )
    return phases


def freeze():
    if (E / "svm-lock.json").exists() or C.exists() or O.exists():
        raise ContractError("Comparison already prepared; no overwrite")
    read_json(E / "before.json")
    reference = Scorer(ROOT / "runs" / BASE)
    index = copy.deepcopy(reference.index)
    C.mkdir(parents=True)
    O.mkdir(parents=True)
    copied = {}
    for source in index["split_manifest"]["sequences"]:
        sid = source["sequence_id"]
        if source["split"] == "test":
            index["sequences"][sid] = {
                "status": "not_prepared",
                "source_hash": index["sequences"][sid]["source_hash"],
                "reason": "No new Test in this comparison",
            }
            continue
        entry = index["sequences"][sid]
        for key in ["feature_path", "pose_path"]:
            src = project_path(entry[key])
            dest = C / ("features" if key == "feature_path" else "poses") / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            for original in [src, src.with_suffix(".json")]:
                target = dest.with_suffix(original.suffix)
                shutil.copyfile(original, target)
                if sha256(target) != sha256(original):
                    raise ContractError("Copied cache differs")
                copied[target.relative_to(ROOT).as_posix()] = sha256(target)
            entry[key] = dest.relative_to(ROOT).as_posix()
    index["preparation_scope"] = ["train", "validation"]
    write_json(C / "dataset-index.json", index)
    index = validated_index()
    train, train_hashes, counts = normal_matrix(index, "train", True)
    val, val_hashes = normal_matrix(index, "validation")
    scaler = FeatureScaler.fit(train)
    np.testing.assert_array_equal(scaler.mean, reference.scaler.mean)
    np.testing.assert_array_equal(scaler.scale, reference.scaler.scale)
    source = read_json(ROOT / "docs/validation/ur-short20-isotropic-20261012-lock.json")
    intervals = source["coarse_intervals"]
    from human_residual import visible_pose_keys

    good = {sid: sorted(visible_pose_keys(reference, sid)) for sid in intervals}
    # Real strict load and actual original30 ST-GCN recomputation on fixed samples.
    extractor = HumanFeatureExtractor(reference.config, index["weights"], torch.device("cuda"))
    parity = []
    for sid in ["adl-05", "adl-06", "fall-19"]:
        observations, _ = load_pose_cache(
            project_path(index["sequences"][sid]["pose_path"]),
            {
                "sequence_id": sid,
                "recipe_hash": index["recipe_hash"],
                "source_hash": index["sequences"][sid]["source_hash"],
            },
        )
        cache = cache_for(index, sid)
        targets = {r.sample_index: r for r in cache.records if r.row_index is not None}
        resampler = CausalResampler(30, reference.config["skeleton"]["resampling_hold_policy"])
        builder = WindowBuilder(reference.config)
        for frame, poses in observations:
            for sample in resampler.push(frame, poses):
                for window in builder.push(sample):
                    if window.quality == "valid" and window.sample_index in targets:
                        actual = extractor.extract_features(window.keypoints, window.frame_size_wh)[
                            0
                        ]
                        expected = cache.features[targets[window.sample_index].row_index]
                        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
                        parity.append(
                            {
                                "sequence_id": sid,
                                "sample_index": window.sample_index,
                                "max_abs": float(np.max(np.abs(actual - expected))),
                            }
                        )
                        break
                if parity and parity[-1]["sequence_id"] == sid:
                    break
            if parity and parity[-1]["sequence_id"] == sid:
                break
    if len(parity) != 3 or not extractor.forward_verified:
        raise ContractError("Real feature smoke incomplete")
    svm = [
        {
            "id": f"ur-ocsvm-{i + 1:02d}-20261012-001",
            "kind": "svm",
            "nu": nu,
            "gamma": gamma,
            "order": i,
        }
        for i, (nu, gamma) in enumerate(
            (n, g) for n in [0.01, 0.05, 0.10] for g in ["scale", 0.001]
        )
    ]
    lock = {
        "locked_at": datetime.now(UTC).isoformat(),
        "candidates": svm,
        "feature_recipe_hash": index["recipe_hash"],
        "source_index_sha256": sha256(ROOT / "runs" / BASE / "dataset-index.json"),
        "scoped_index_sha256": sha256(C / "dataset-index.json"),
        "train_feature_hashes": train_hashes,
        "val_normal_feature_hashes": val_hashes,
        "train_counts": counts,
        "train_shape": list(train.shape),
        "val_normal_shape": list(val.shape),
        "coarse_intervals": intervals,
        "current_good_pose_keys": good,
        "reference_metrics": read_json(
            ROOT / "docs/validation/ur-short20-isotropic-20261012-validation.json"
        )["reference"]["metrics"],
        "reference_threshold": reference.threshold["threshold"],
        "implementation_hashes": {n: sha256(ROOT / n) for n in IMPLEMENTATIONS},
        "input_hashes": {
            **copied,
            (C / "dataset-index.json").relative_to(ROOT).as_posix(): sha256(
                C / "dataset-index.json"
            ),
            "docs/ur-normal-methods-20261012-plan.md": sha256(
                ROOT / "docs/ur-normal-methods-20261012-plan.md"
            ),
        },
        "selection_rule": "operational Accuracy, Fall F1, all-Fall Recall, fewer FP, fixed order",
        "threshold_tie": "higher threshold",
        "quality_and_k3_unchanged": True,
        "test_evaluations": 0,
        "test_features_read": 0,
        "real_feature_smoke": {
            "strict_load": extractor.load_report,
            "head_parity": extractor.forward_verified,
            "recomputed_samples": parity,
        },
    }
    save(E / "svm-lock.json", lock)
    for spec in svm:
        save(
            ROOT / "configs" / f"{spec['id']}.json",
            {
                **spec,
                "source_feature_recipe_hash": index["recipe_hash"],
                "quality": reference.config["quality"],
                "k": 3,
                "score": "negative_decision_function",
                "threshold_rule": lock["selection_rule"],
            },
        )
    print({"locked": svm, "smoke": parity, "train": train.shape}, flush=True)


def inputs_for(index, spec, scaler):
    arrays, scoring_caches, provenance = {}, {}, {}
    for source in index["split_manifest"]["sequences"]:
        if source["split"] == "test":
            continue
        sid = source["sequence_id"]
        cache = cache_for(index, sid)
        if spec["kind"] == "temporal":
            raw, adjusted, histories = temporal_inputs(cache, spec["steps"])
            z = scaler.transform(raw.reshape(-1, 512)).reshape(raw.shape) if len(raw) else raw
            arrays[sid], scoring_caches[sid], provenance[sid] = z, adjusted, histories
        else:
            arrays[sid], scoring_caches[sid] = scaler.transform(cache.features), cache
    return arrays, scoring_caches, provenance


def score_ae(model, values, kind, device):
    scores = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), 1):
            batch = torch.as_tensor(values[start : start + 1], device=device)
            error = (batch - model(batch)) ** 2
            score = error[:, -1].mean(-1) if kind == "temporal" else error.mean(-1)
            scores.extend(score.cpu().tolist())
    result = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ContractError("Non-finite AE scores")
    return result


def train_ae(run, spec, arrays, index):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normal = {
        split: [
            r["sequence_id"]
            for r in index["split_manifest"]["sequences"]
            if r["split"] == split and r["label"] == 0
        ]
        for split in ["train", "validation"]
    }
    if any(len(arrays[sid]) == 0 for sid in normal["train"] + normal["validation"]):
        raise ContractError("No usable normal input for required sequence")
    train = np.concatenate([arrays[sid] for sid in normal["train"]])
    val = np.concatenate([arrays[sid] for sid in normal["validation"]])
    model = (
        TemporalAutoencoder(512, 128, 64, spec["steps"])
        if spec["kind"] == "temporal"
        else Autoencoder(512, 256, 64)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    generator = torch.Generator().manual_seed(42)
    sampler = training_sampler(
        {sid: len(arrays[sid]) for sid in normal["train"]}, "sequences", generator
    )
    loader = torch.utils.data.DataLoader(
        torch.from_numpy(train), batch_size=32, sampler=sampler, generator=generator
    )
    noise_generator = torch.Generator(device=device).manual_seed(42)
    best, stale, history, best_epoch = float("inf"), 0, [], None
    for epoch in range(1, 101):
        model.train()
        total = 0.0
        for clean in loader:
            clean = clean.to(device)
            noisy = (
                clean
                + torch.randn(clean.shape, generator=noise_generator, device=device)
                * spec["noise_std"]
                if spec["noise_std"]
                else clean
            )
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(noisy) - clean) ** 2).mean()
            if not torch.isfinite(loss):
                raise ContractError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(clean)
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for start in range(0, len(val), 32):
                batch = torch.as_tensor(val[start : start + 32], device=device)
                val_loss += float(((model(batch) - batch) ** 2).mean()) * len(batch)
        val_loss /= len(val)
        if not np.isfinite(val_loss):
            raise ContractError("Non-finite normal Validation loss")
        history.append(
            {"epoch": epoch, "train_loss": total / len(train), "val_normal_loss": val_loss}
        )
        improved = val_loss < best - 1e-6
        if improved:
            best, stale, best_epoch = val_loss, 0, epoch
            checkpoint = {
                "architecture": spec["kind"],
                "spec": spec,
                "state_dict": model.state_dict(),
                "best_epoch": epoch,
                "val_normal_loss": best,
                "method_recipe_hash": read_json(run / "method-config.json")["method_recipe_hash"],
            }
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            atomic_bytes(run / "model.pt", buffer.getvalue())
        else:
            stale += 1
        write_json(run / "training-history.json", history)
        print(
            f"{spec['id']} epoch {epoch}: Train {total / len(train):.6f}; normal Val {val_loss:.6f}; best {best_epoch}",
            flush=True,
        )
        if stale >= 10:
            break
    checkpoint = torch.load(run / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    training_curve(run / "loss.png", history, "real_feature_smoke")
    write_json(
        run / "training.json",
        {
            "epochs": len(history),
            "best_epoch": best_epoch,
            "best_val_normal_loss": best,
            "parameters": sum(p.numel() for p in model.parameters()),
            "train_sequences": normal["train"],
            "val_normal_sequences": normal["validation"],
            "train_input_counts": {sid: len(arrays[sid]) for sid in normal["train"]},
            "val_input_counts": {sid: len(arrays[sid]) for sid in normal["validation"]},
            "seed": 42,
            "augmentation_split": "train_only",
            "augmentation": "on-the-fly Gaussian input noise; clean target"
            if spec["noise_std"]
            else "none",
            "normal_validation_clean": True,
            "determinism": "warn_only",
            "device": str(device),
        },
    )
    return {sid: score_ae(model, x, spec["kind"], device) for sid, x in arrays.items()}


def candidate(spec):
    lock = verify_lock()
    run = ROOT / "runs" / spec["id"]
    if run.exists():
        raise ContractError(f"Candidate already exists: {run}")
    run.mkdir(parents=True)
    write_json(run / "status.json", {"training": "running", "calibration": "not_run"})
    try:
        index = validated_index()
        train, train_hashes = normal_matrix(index, "train")
        _, val_hashes = normal_matrix(index, "validation")
        scaler = FeatureScaler.fit(train)
        scaler.save(run / "scaler.npz")
        arrays, caches, provenance = inputs_for(index, spec, scaler)
        method_recipe = fingerprint(
            {
                "spec": spec,
                "features": index["recipe_hash"],
                "calibration": lock["selection_rule"],
                "k": 3,
            }
        )
        save(
            run / "method-config.json",
            {
                **spec,
                "method_recipe_hash": method_recipe,
                "feature_recipe_hash": index["recipe_hash"],
            },
        )
        save(run / "dataset-index.json", index)
        save(
            run / "sources.json",
            {
                "train_feature_hashes": train_hashes,
                "val_normal_feature_hashes": val_hashes,
                "feature_recipe_hash": index["recipe_hash"],
                "method_recipe_hash": method_recipe,
                "weights": index["weights"],
                "lock_sha256": sha256(
                    E / ("svm-lock.json" if spec["kind"] == "svm" else "ae-lock.json")
                ),
                "execution_scope": "train_validation_only",
                "test_unknown": True,
            },
        )
        if spec["kind"] == "svm":
            model = fit_svm(
                np.concatenate(
                    [
                        arrays[r["sequence_id"]]
                        for r in index["split_manifest"]["sequences"]
                        if r["split"] == "train"
                    ]
                ),
                spec["nu"],
                spec["gamma"],
            )
            joblib.dump(model, run / "model.joblib")
            scores = {sid: svm_scores(model, x) for sid, x in arrays.items()}
            save(
                run / "training.json",
                {
                    "train_sequences": list(train_hashes),
                    "train_rows": len(train),
                    "support_vectors": len(model.support_),
                    "effective_gamma": float(model._gamma),
                    "epochs": 0,
                },
            )
        else:
            if spec["kind"] == "temporal":
                derived = C / spec["id"]
                derived.mkdir()
                for sid, x in arrays.items():
                    write_npz(derived / f"{sid}.npz", inputs=x)
                save(
                    derived / "provenance.json",
                    {
                        "method_recipe_hash": method_recipe,
                        "histories": provenance,
                        "array_hashes": {p.name: sha256(p) for p in derived.glob("*.npz")},
                        "scaler_sha256": sha256(run / "scaler.npz"),
                    },
                )
            scores = train_ae(run, spec, arrays, index)
        # Scores for every Train/Validation input are immutable diagnostic evidence.
        write_npz(run / "scores.npz", **scores)
        rows = []
        for src in index["split_manifest"]["sequences"]:
            if src["split"] == "validation":
                sid = src["sequence_id"]
                result, _, _ = score_sequence(caches[sid], scores[sid], 0.0)
                rows.append({**result, "label": src["label"]})
        chosen, grid = calibrate_normal_scores(rows)
        balanced, _ = choose_threshold(rows, "balanced_sequence_accuracy")
        save(
            run / "calibration.json",
            {
                "chosen": chosen,
                "candidates": grid,
                "selection_rule": lock["selection_rule"],
                "diagnostic_historical_balanced_calibration": balanced,
            },
        )
        threshold = chosen["threshold"]
        save(
            run / "threshold.json",
            {
                "threshold": threshold,
                "comparison": "gt",
                "consecutive_windows": 3,
                "split": "validation",
                "method_recipe_hash": method_recipe,
                "feature_recipe_hash": index["recipe_hash"],
                "split_hash": index["split_manifest"]["split_hash"],
                "score_direction": "larger_is_more_anomalous",
                "scaler_sha256": sha256(run / "scaler.npz"),
                "model_sha256": sha256(
                    run / ("model.joblib" if spec["kind"] == "svm" else "model.pt")
                ),
            },
        )
        rows, windows, events = [], [], []
        for src in index["split_manifest"]["sequences"]:
            if src["split"] == "validation":
                sid = src["sequence_id"]
                row, ws, es = score_sequence(caches[sid], scores[sid], threshold)
                rows.append({**row, "label": src["label"]})
                windows.extend(ws)
                events.extend(es)
        metrics = sequence_metrics(rows)
        assert all(metrics[k] == v for k, v in chosen["metrics"].items())
        phases = phases_for(rows, windows, events, lock)
        destination = O / spec["id"]
        destination.mkdir()
        evaluation_report(
            destination,
            {
                **metrics,
                "source_kind": "real_feature_smoke",
                "protocol_id": index["split_manifest"]["protocol_id"],
            },
            rows,
            windows,
            events,
        )
        recovered = [
            r["sequence_id"] for r in rows if r["sequence_id"] == "fall-19" and r["prediction"] == 1
        ]
        display = {
            "phase_hits_at_least_four": sum(bool(p["phase_event_starts"]) for p in phases) >= 4,
            "no_coarse_early_events": not any(p["early_event_starts"] for p in phases),
            "recovered_fn_visible": all(
                any(p["sequence_id"] == sid and p["visible_phase_alert_ms"] for p in phases)
                for sid in recovered
            ),
        }
        result = {
            "spec": spec,
            "metrics": metrics,
            "threshold": threshold,
            "phases": phases,
            "display_checks": display,
            "recognition_better_than_wide": candidate_rank(metrics, 0)[:-1]
            > candidate_rank(lock["reference_metrics"], 0)[:-1],
            "false_negatives": [
                r["sequence_id"] for r in rows if r["label"] == 1 and r["prediction"] == 0
            ],
            "false_positives": [
                r["sequence_id"] for r in rows if r["label"] == 0 and r["prediction"] == 1
            ],
            "test": "unknown_not_evaluated",
            "evidence_tier": "real_feature_smoke",
            "real_end_to_end": "not_run",
        }
        save(run / "result.json", result)
        write_json(
            run / "status.json",
            {"training": "complete", "calibration": "complete", "validation": "complete"},
        )
        save(
            run / "frozen-artifacts.json", {p.name: sha256(p) for p in run.iterdir() if p.is_file()}
        )
        print(spec["id"], metrics, flush=True)
        return result
    except BaseException as exc:
        write_json(
            run / "status.json", {"training": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        )
        raise


def svm():
    lock = verify_lock()
    if (E / "svm-results.json").exists():
        raise ContractError("SVM stage complete; grid closed")
    if any((ROOT / "runs" / s["id"]).exists() for s in lock["candidates"]):
        raise ContractError("Existing candidate: inspect, never silently retrain")
    results = [candidate(spec) for spec in lock["candidates"]]
    save(
        E / "svm-results.json",
        {
            "candidates": results,
            "selected": max(
                results, key=lambda r: candidate_rank(r["metrics"], r["spec"]["order"])
            )["spec"]["id"],
            "grid_closed": True,
        },
    )


def ae_lock():
    lock = verify_lock()
    previous = read_json(E / "svm-results.json")
    diagnostic = read_json(E / "diagnosis-stage1.json")
    specs = [
        {
            "id": "ur-denoise-wide-methods-20261012-001",
            "kind": "denoise",
            "order": 6,
            "hidden": 256,
            "latent": 64,
            "noise_std": 0.1,
        },
        {
            "id": "ur-temporal-gru-methods-20261012-001",
            "kind": "temporal",
            "order": 7,
            "hidden": 128,
            "latent": 64,
            "steps": 8,
            "noise_std": 0.0,
        },
    ]
    save(
        E / "ae-lock.json",
        {
            "locked_at": datetime.now(UTC).isoformat(),
            "stage1_sha256": sha256(E / "svm-results.json"),
            "diagnosis_sha256": sha256(E / "diagnosis-stage1.json"),
            "candidates": specs,
            "why": "Contrast normal manifold robustness (denoising) with retained cross-window order (GRU), on unchanged real512D features; controlled hypotheses, not asserted causes.",
            "architecture": "denoise:512-256-64-256-512 ReLU, linear output; GRU:512->128 ReLU->GRU64, context64+position->GRU128->Linear512",
            "training_loss": "clean-target MSE; temporal averages all8x512, anomaly score uses current last-step512 MSE",
            "sampling": "sequence-balanced Train, original24 videos",
            "seed": 42,
            "max_epochs": 100,
            "patience": 10,
            "min_delta": 1e-6,
            "batch": 32,
            "optimizer": "Adam lr.001 wd.001",
            "noise": "denoise .1 Gaussian standardized features, Train only, on-the-fly dedicated seed42 generator; no Val augmentation",
            "temporal": "8 consecutive original legal same-track windows, stride1, reset on invalid/gap; no padding/future; original endpoint denominator retained",
            "calibration": lock["selection_rule"],
            "stage1_best": previous["selected"],
            "diagnostic_summary": diagnostic["summary"],
            "stop": "exactly two candidates; no noise/history/capacity/seed extension",
        },
    )
    for spec in specs:
        save(ROOT / "configs" / f"{spec['id']}.json", spec)
    print(specs, flush=True)


def ae():
    verify_lock()
    specs = read_json(E / "ae-lock.json")["candidates"]
    if (E / "ae-results.json").exists() or any((ROOT / "runs" / s["id"]).exists() for s in specs):
        raise ContractError("AE candidate exists; no overwrite or retraining")
    results = [candidate(spec) for spec in specs]
    save(E / "ae-results.json", {"candidates": results, "grid_closed": True})


def status():
    for name in [
        "before",
        "svm-lock",
        "svm-results",
        "diagnosis-stage1",
        "ae-lock",
        "ae-results",
        "diagnosis-stage2",
        "verification",
        "selection",
    ]:
        print(name, "saved" if (E / f"{name}.json").exists() else "not_run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "svm", "ae-lock", "ae", "status"])
    args = parser.parse_args()
    {"freeze": freeze, "svm": svm, "ae-lock": ae_lock, "ae": ae, "status": status}[args.stage]()
