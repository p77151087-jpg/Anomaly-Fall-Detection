from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .config import project_path
from .contracts import FeatureCache, FeatureRecord
from .data.cache import save_feature_cache
from .data.splits import ensure_split, smoke_ids
from .io_utils import ContractError, fingerprint, write_json


def synthetic_dataset(directory, config):
    """Handcrafted feature values for plumbing tests. These are never ST-GCN outputs."""
    directory = Path(directory)
    sources = [
        {"sequence_id": sid, "split": split, "label": label}
        for sid, split, label in [
            ("synthetic-train-a", "train", 0),
            ("synthetic-train-b", "train", 0),
            ("synthetic-val-normal", "validation", 0),
            ("synthetic-val-fall", "validation", 1),
            ("synthetic-test-normal", "test", 0),
            ("synthetic-test-fall", "test", 1),
            ("synthetic-test-unknown", "test", 1),
        ]
    ]
    manifest = {
        "schema_version": 1,
        "protocol_id": "synthetic-sequence-v1",
        "seed": 42,
        "numpy_version": np.__version__,
        "sequences": sources,
    }
    manifest["split_hash"] = fingerprint(manifest)
    recipe = fingerprint(
        {
            "synthetic": True,
            "features": "handcrafted trigonometric fixture",
            "dim": config["features"]["expected_dim"],
            "recipe_id": config["skeleton"]["recipe_id"],
        }
    )
    index = {
        "schema_version": 1,
        "source_kind": "synthetic",
        "recipe_hash": recipe,
        "split_manifest": manifest,
        "sequences": {},
    }
    for number, source in enumerate(sources):
        unknown = source["sequence_id"].endswith("unknown")
        rows, values = [], []
        for j in range(12):
            sample_index = 29 + j * 5
            rows.append(
                FeatureRecord(
                    source["sequence_id"],
                    None if unknown else 1,
                    j,
                    sample_index,
                    (sample_index - 29) * 1000 / 30,
                    sample_index * 1000 / 30,
                    sample_index * 1000 / 30,
                    source["split"],
                    None if unknown else j,
                    quality="no_person" if unknown else "valid",
                    feature_dim=config["features"]["expected_dim"],
                    recipe_id=config["skeleton"]["recipe_id"],
                )
            )
            if not unknown:
                values.append(
                    np.sin(
                        np.arange(config["features"]["expected_dim"]) * 0.07
                        + j * 0.08
                        + number * 0.015
                    )
                    + source["label"] * 3.0
                )
        meta = {
            "source_kind": "synthetic",
            "recipe_id": config["skeleton"]["recipe_id"],
            "recipe_hash": recipe,
            "feature_dim": config["features"]["expected_dim"],
            "split_hash": manifest["split_hash"],
            "source_hash": fingerprint(source),
            "sequence_id": source["sequence_id"],
            "split": source["split"],
            "window_size": 30,
            "stride": 5,
            "n_samples": 85,
            "end_ms": 84 * 1000 / 30,
        }
        path = directory / (source["sequence_id"] + ".npz")
        save_feature_cache(
            path,
            FeatureCache(
                np.asarray(values, dtype=np.float32).reshape(
                    -1, config["features"]["expected_dim"]
                ),
                rows,
                meta,
            ),
        )
        index["sequences"][source["sequence_id"]] = {
            "status": "prepared",
            "feature_path": str(path.resolve()),
            "source_hash": meta["source_hash"],
        }
    path = directory / "dataset-index.json"
    write_json(path, index)
    return path


def synthetic_smoke(config, output=None):
    from .contracts import PoseRecord
    from .detection.events import score_sequence
    from .io_utils import write_csv
    from .training.experiment import Scorer, cache_for, calibrate, evaluate, train
    from .visualization.video import VideoOutput, draw_overlay

    directory = (
        Path(output)
        if output
        else project_path("outputs/synthetic") / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    if "synthetic" not in str(directory).lower():
        raise ContractError("Synthetic output path must contain 'synthetic'")
    config = deepcopy(config)
    config["runtime"]["device"] = "cpu"
    config["autoencoder"]["max_epochs"] = 6
    config["autoencoder"]["patience"] = 3
    index = synthetic_dataset(directory / "features", config)
    run = train(config, index, directory / "run", synthetic=True)
    calibrate(run)
    evaluation_dir, metrics = evaluate(run)
    scorer = Scorer(run, device="cpu")
    cache = cache_for(scorer.index, "synthetic-test-fall")
    _, windows, _ = score_sequence(
        cache, scorer.score(cache.features), scorer.threshold["threshold"]
    )
    video = VideoOutput(directory / "synthetic-demo.mp4", 30, (640, 480))
    keypoints = np.array(
        [
            [320, 80],
            [270, 145],
            [370, 145],
            [250, 220],
            [390, 220],
            [230, 290],
            [410, 290],
            [285, 270],
            [355, 270],
            [280, 345],
            [360, 345],
            [270, 435],
            [370, 435],
        ],
        dtype=np.float32,
    )
    keypoints = np.c_[keypoints, np.ones(13)].astype(np.float32)
    pose = PoseRecord(
        "synthetic-demo", 0, 0, 0, 1, np.array([220, 70, 420, 445]), keypoints, (640, 480), 1.0
    )
    latest, logs = {"state": "warming_up", "raw_score": None}, []
    by_index = {w["sample_index"]: w for w in windows}
    for index in range(85):
        latest = by_index.get(index, latest)
        frame = draw_overlay(
            np.full((480, 640, 3), 40, np.uint8),
            [pose],
            {1: latest},
            index * 1000 / 30,
            scorer.threshold["threshold"],
            source_kind="SYNTHETIC software test",
        )
        video.write(frame)
        logs.append(
            {
                "sample_index": index,
                "timestamp_ms": index * 1000 / 30,
                "state": latest["state"],
                "raw_score": latest["raw_score"],
            }
        )
    info = video.close()
    write_csv(directory / "synthetic-demo.csv", logs)
    report = {
        "mode": "synthetic",
        "passed": True,
        "formal_urfall_metrics": None,
        "run": str(run),
        "evaluation": str(evaluation_dir),
        "video": info,
        "fixture_metrics": metrics,
        "meaning": "software plumbing only; handcrafted features",
    }
    write_json(directory / "smoke-report.json", report)
    return directory, report


def real_smoke(config, output=None):
    from .pipeline import prepare, raw_dataset_report

    directory = Path(output) if output else project_path("outputs/real_feature_smoke")
    manifest = ensure_split(project_path(config["data"]["manifest_dir"]))
    ids = smoke_ids(manifest)
    report = {
        "mode": "real_feature_smoke",
        "recipe_id": config["skeleton"]["recipe_id"],
        "feature_dim": config["features"]["expected_dim"],
        "sequence_ids": ids,
        "classification": "not_performed",
        "passed": False,
        "dataset_report": raw_dataset_report(config, manifest, ids),
    }
    try:
        path, index = prepare(
            config,
            selected_ids=ids,
            limit_frames=120,
            output_index=directory / "dataset-index.json",
            refresh=True,
        )
        report["dataset_index"] = str(path)
        report["model_load_reports"] = index["model_load_reports"]
        report["verification"] = index["verification"]
        report["sequences"] = {sid: index["sequences"][sid] for sid in ids}
        report["passed"] = (
            index["verification"]["pose_forward"]
            and index["verification"]["feature_forward"]
            and all(
                r["status"] == "prepared" and r["valid_features"] > 0
                for r in report["sequences"].values()
            )
        )
        report["status"] = "passed" if report["passed"] else "failed_quality_or_model"
    except (ContractError, OSError, ImportError) as exc:
        report.update(status="blocked_external_asset", error=str(exc))
    write_json(directory / "smoke-report.json", report)
    return directory, report
