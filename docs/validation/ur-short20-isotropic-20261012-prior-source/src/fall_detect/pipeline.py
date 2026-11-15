"""Single shared skeleton-to-feature path for cached preparation and streaming demo."""

from collections import Counter
from pathlib import Path

import numpy as np

from .assets import ensure_weight_manifest, require_weights
from .backends import create_backends
from .config import ROOT, project_path, recipe_fingerprint
from .contracts import FeatureCache, FeatureRecord
from .data.cache import load_feature_cache, load_pose_cache, save_feature_cache, save_pose_cache
from .data.readers import image_inventory, read_source, read_timestamps, source_identity
from .data.splits import dataset_split
from .io_utils import ContractError, fingerprint, read_json, sha256, write_json
from .models.weights import select_device
from .recipes import upstream_commits
from .skeleton.windows import CausalResampler, WindowBuilder


def features_from_observations(observations, extractor, config, metadata):
    builder = WindowBuilder(config)
    resampler = CausalResampler(
        config["skeleton"]["target_fps"],
        config["skeleton"]["resampling_hold_policy"],
        config["skeleton"]["timestamp_tolerance_ms"],
    )
    records, arrays, n_samples, repeated, missing, end_ms = [], [], 0, 0, 0, 0.0
    for frame, poses in observations:
        end_ms = frame.timestamp_ms
        for sample in resampler.push(frame, poses):
            n_samples += 1
            repeated += int(sample.repeated)
            missing += int(sample.missing)
            for window in builder.push(sample):
                row_index = None
                intervals = metadata.get("annotation_intervals_ms")
                if intervals is not None and not any(
                    start <= window.start_ms + 1e-7 and window.end_ms <= end + 1e-7
                    for start, end in intervals
                ):
                    window.quality = "unlabelled_interval"
                if window.quality == "valid":
                    feature = extractor.extract_features(window.keypoints, window.frame_size_wh)
                    if (
                        feature.shape != (1, extractor.feature_dim)
                        or not np.isfinite(feature).all()
                    ):
                        raise ContractError("Extractor returned invalid features")
                    row_index = len(arrays)
                    arrays.append(feature[0])
                records.append(
                    FeatureRecord(
                        window.sequence_id,
                        window.track_id,
                        (window.sample_index - config["skeleton"]["window_size"] + 1)
                        // config["skeleton"]["stride"],
                        window.sample_index,
                        window.start_ms,
                        window.end_ms,
                        window.end_ms,
                        metadata["split"],
                        row_index,
                        extractor.feature_dim,
                        window.quality,
                        recipe_id=config["skeleton"]["recipe_id"],
                    )
                )
    features = np.asarray(arrays, dtype=np.float32).reshape(-1, extractor.feature_dim)
    return FeatureCache(
        features,
        records,
        {
            **metadata,
            "schema_version": 1,
            "kind": "feature_cache",
            "recipe_id": config["skeleton"]["recipe_id"],
            "feature_dim": extractor.feature_dim,
            "window_size": config["skeleton"]["window_size"],
            "stride": config["skeleton"]["stride"],
            "n_samples": n_samples,
            "end_ms": end_ms,
            "repeated_samples": repeated,
            "missing_samples": missing,
            "quality_counts": dict(Counter(r.quality for r in records)),
        },
    )


def raw_dataset_report(config, manifest, ids=None):
    root = project_path(config["data"]["raw_dir"])
    report = {"schema_version": 1, "split_hash": manifest["split_hash"], "sequences": []}
    for source in manifest["sequences"]:
        sid = source["sequence_id"]
        if ids and sid not in ids:
            continue
        directory = project_path(source["source_path"]) if "source_path" in source else root / sid
        try:
            if directory.is_file():
                if "source_sha256" in source and sha256(directory) != source["source_sha256"]:
                    raise ContractError("Video changed since the split manifest was created")
                count, estimated = 0, 0
                for frame in read_source(directory, sequence_id=sid):
                    count += 1
                    estimated += frame.timestamp_estimated
                report["sequences"].append(
                    {
                        **source,
                        "status": "validated",
                        "decoded_images": count,
                        "timestamp_source": "video_pts_or_fps",
                        "estimated_timestamps": estimated,
                    }
                )
                print(f"Validated {sid}: video decoded", flush=True)
                continue
            paths, detail = image_inventory(directory)
            sync = directory / f"{sid}-data.csv"
            times = read_timestamps(sync) if sync.exists() else None
            from .data.readers import numeric_frame_id

            missing_alignment = [
                numeric_frame_id(p)
                for p in paths
                if times is not None and numeric_frame_id(p) not in times
            ]
            if missing_alignment and not (
                config["data"]["allow_missing_timestamp_tail"]
                and times
                and all(frame_id > max(times) for frame_id in missing_alignment)
            ):
                raise ContractError(
                    f"No source timestamp for RGB frame IDs {missing_alignment[:10]}"
                )
            # Actually decode every selected image; this is read-only data validation.
            decoded = sum(
                1
                for frame in read_source(
                    directory,
                    sequence_id=sid,
                    sync_path=sync,
                    fallback_fps=config["data"]["fallback_fps"],
                    allow_missing_timestamp_tail=config["data"]["allow_missing_timestamp_tail"],
                )
                if not frame.source_missing
            )
            report["sequences"].append(
                {
                    **source,
                    **detail,
                    "status": "validated",
                    "decoded_images": decoded,
                    "timestamp_source": (
                        "source_with_estimated_tail" if missing_alignment else "source_frame_id"
                    )
                    if times
                    else "estimated_fps",
                    "estimated_rgb_timestamp_ids": missing_alignment
                    if times
                    else [numeric_frame_id(p) for p in paths],
                    "timestamp_fallback_fps": config["data"]["fallback_fps"]
                    if missing_alignment or not times
                    else None,
                    "posture_labels": "not_used",
                    "event_annotations": "not_available",
                }
            )
            print(
                f"Validated {sid}: {decoded} images; estimated timestamps={len(missing_alignment) if times else decoded}",
                flush=True,
            )
        except (ContractError, OSError) as exc:
            report["sequences"].append(
                {**source, "status": "missing_or_invalid", "reason": str(exc)}
            )
    report["n_planned"] = len(report["sequences"])
    report["n_validated"] = sum(row["status"] == "validated" for row in report["sequences"])
    name = "dataset-report-smoke.json" if ids else "dataset-report.json"
    write_json(project_path(config["data"]["manifest_dir"]) / name, report)
    return report


def reuse_prepared_sequence(index, sid, metadata, feature_path, pose_path, config=None):
    """Re-key identical observations/features under a new split, with full checks."""
    row = index["sequences"].get(sid, {})
    if row.get("status") != "prepared":
        return None
    expected = {
        **metadata,
        "split_hash": index["split_manifest"]["split_hash"],
        "recipe_hash": index["recipe_hash"],
        "annotation_intervals_ms": metadata.get("annotation_intervals_ms"),
    }
    old_feature = project_path(row["feature_path"])
    old_pose = project_path(row["pose_path"])
    cache = load_feature_cache(old_feature, expected)
    observations, old_meta = load_pose_cache(old_pose, expected)
    changed_recipe = index["recipe_hash"] != metadata["recipe_hash"]
    if changed_recipe:
        from .data.track_reuse import identical_tracking_resets, validate_tracking_reuse

        if config is None:
            raise ContractError("Recipe mismatch: changed tracking needs compatibility validation")
        validate_tracking_reuse(index, config)
        if not identical_tracking_resets(observations, index["preparation_config"], config):
            return None  # Re-infer this source; never carry over altered track identities.
    provenance = {
        "feature_array_sha256": sha256(old_feature),
        "pose_array_sha256": sha256(old_pose),
        "old_split_hash": index["split_manifest"]["split_hash"],
        "old_recipe_hash": index["recipe_hash"],
        "tracking_reset_equivalence_verified": changed_recipe,
    }
    cache.metadata.update(metadata)
    cache.metadata["reused_from"] = provenance
    save_feature_cache(feature_path, cache)
    save_pose_cache(pose_path, observations, {**old_meta, **metadata, "reused_from": provenance})
    return cache


def prepare(
    config,
    selected_ids=None,
    limit_frames=None,
    output_index=None,
    refresh=False,
    reuse_index=None,
    reuse_poses_index=None,
):
    manifest = dataset_split(config)
    resources = require_weights(config)
    device = select_device(config["runtime"]["device"], config["runtime"]["torch_threads"])
    hashes = {role: entry["sha256"] for role, entry in resources.items()}
    recipe_hash = recipe_fingerprint(config, hashes)
    pose_reusable = read_json(reuse_poses_index) if reuse_poses_index else None
    if pose_reusable:
        from .data.pose_reuse import validate_pose_reuse

        if reuse_index or refresh or limit_frames:
            raise ContractError("Pose-only reuse is incompatible with other reuse/refresh/limits")
        validate_pose_reuse(pose_reusable, config, manifest, hashes)
    reusable = read_json(reuse_index) if reuse_index else None
    if reusable:
        from .data.splits import validate_split

        validate_split(reusable["split_manifest"])
        if reusable["source_kind"] != "real":
            raise ContractError("Reuse index must contain the exact same real pose/feature recipe")
        if reusable["recipe_hash"] != recipe_hash:
            from .data.track_reuse import validate_tracking_reuse

            if {role: entry["sha256"] for role, entry in reusable["weights"].items()} != hashes:
                raise ContractError("Reuse weights differ")
            validate_tracking_reuse(reusable, config)
        if limit_frames or refresh:
            raise ContractError("Reuse is incompatible with refresh or frame-limited smoke")
    raw_root = project_path(config["data"]["raw_dir"])
    cache_root = project_path(config["data"]["cache_dir"])
    if selected_ids or limit_frames:
        cache_root /= "real_feature_smoke"
    backend, extractor = create_backends(config, resources, device)
    index = {
        "schema_version": 1,
        "source_kind": "real",
        "recipe_hash": recipe_hash,
        "weights": resources,
        "split_manifest": manifest,
        "sequences": {},
        "provenance": {
            "upstream_commits": upstream_commits(config),
            "source_code_hash": source_code_hash(),
            **({"pose_reuse_index_sha256": sha256(reuse_poses_index)} if pose_reusable else {}),
        },
        "recipe_id": config["skeleton"]["recipe_id"],
        "preparation_config": config,
        "model_load_reports": {
            "pose": getattr(backend, "load_report", None),
            "stgcn": extractor.load_report,
        },
    }
    for source in manifest["sequences"]:
        sid = source["sequence_id"]
        index["sequences"][sid] = {"status": "not_selected", "reason": "not selected for smoke"}
        if selected_ids and sid not in selected_ids:
            continue
        try:
            directory = (
                project_path(source["source_path"]) if "source_path" in source else raw_root / sid
            )
            sync = directory / f"{sid}-data.csv"
            identity = source_identity(directory, sync)
            if (
                "source_sha256" in source
                and identity["files"][0]["sha256"] != source["source_sha256"]
            ):
                raise ContractError("Video changed since the split manifest was created")
            metadata = {
                "source_hash": identity["source_hash"],
                "split_hash": manifest["split_hash"],
                "recipe_hash": recipe_hash,
                "recipe_id": config["skeleton"]["recipe_id"],
                "source_kind": "real",
                "sequence_id": sid,
                "split": source["split"],
                "limit_frames": limit_frames,
            }
            if "annotation_intervals_ms" in source:
                metadata["annotation_intervals_ms"] = source["annotation_intervals_ms"]
            key = fingerprint(metadata)
            feature_path = cache_root / "features" / f"{sid}-{key}.npz"
            pose_path = cache_root / "poses" / f"{sid}-{key}.npz"
            if not refresh and feature_path.exists() and feature_path.with_suffix(".json").exists():
                cache = load_feature_cache(feature_path, metadata)
                print(f"Reusing feature cache: {sid}", flush=True)
            else:
                reused = (
                    reuse_prepared_sequence(
                        reusable, sid, metadata, feature_path, pose_path, config
                    )
                    if reusable
                    else None
                )
                if reused is not None:
                    print(f"Verified reuse from source index: {sid}", flush=True)
                if not refresh and pose_path.exists() and pose_path.with_suffix(".json").exists():
                    observations, _ = load_pose_cache(pose_path, metadata)
                elif pose_reusable:
                    from .data.pose_reuse import reuse_pose_observations

                    observations = reuse_pose_observations(
                        pose_reusable, sid, metadata, directory, pose_path, config
                    )
                    print(f"Rebuilding features from verified observed poses: {sid}", flush=True)
                else:
                    backend.reset()
                    observations = []
                    for frame in read_source(
                        directory,
                        sequence_id=sid,
                        sync_path=sync,
                        fallback_fps=config["data"]["fallback_fps"],
                        allow_missing_timestamp_tail=config["data"]["allow_missing_timestamp_tail"],
                        limit=limit_frames,
                    ):
                        poses = backend.predict(frame)
                        frame.image_bgr = None  # Keep only compact metadata/skeletons in memory.
                        observations.append((frame, poses))
                        if frame.sample_index % 30 == 0:
                            print(
                                f"{sid}: source frame {frame.source_frame_id}, tracks={len(poses)}",
                                flush=True,
                            )
                    save_pose_cache(
                        pose_path, observations, {**metadata, "source_identity": identity}
                    )
                if reused is not None:
                    cache = reused
                else:
                    cache = features_from_observations(observations, extractor, config, metadata)
                    save_feature_cache(feature_path, cache)
            index["sequences"][sid] = {
                "status": "prepared",
                "source_hash": identity["source_hash"],
                "feature_path": str(feature_path.relative_to(ROOT)),
                "pose_path": str(pose_path.relative_to(ROOT)),
                "valid_features": len(cache.features),
                "n_samples": cache.metadata["n_samples"],
                "quality_counts": cache.metadata["quality_counts"],
            }
        except (ContractError, OSError) as exc:
            index["sequences"][sid] = {"status": "failed", "reason": str(exc)}
            print(f"{sid}: {exc}", flush=True)
        destination = Path(output_index) if output_index else cache_root / "dataset-index.json"
        # Each completed sequence is resumable even if a later sequence fails.
        partial = {
            **index,
            "sequences": {
                s["sequence_id"]: index["sequences"].get(
                    s["sequence_id"], {"status": "pending", "reason": "not processed yet"}
                )
                for s in manifest["sequences"]
            },
        }
        write_json(destination, partial)
    destination = Path(output_index) if output_index else cache_root / "dataset-index.json"
    index["verification"] = {
        "pose_forward": backend.forward_verified,
        "pose_observations_reused": pose_reusable is not None,
        "feature_forward": extractor.forward_verified,
        "device": str(device),
        "refreshed": refresh,
    }
    write_json(destination, index)
    if backend.forward_verified and extractor.forward_verified:
        weights_manifest = ensure_weight_manifest(config)
        for entry in weights_manifest["resources"]:
            entry["status"] = "usable"
            previous_devices = entry.get("verification", {}).get(
                "devices", [entry.get("verification", {}).get("device")]
            )
            entry["verification"] = {
                "strict_load": entry["role"] != "detector_config",
                "real_forward": True,
                "device": str(device),
                "devices": sorted({d for d in previous_devices if d} | {str(device)}),
                "evidence": str(destination),
            }
        write_json(project_path(config["pose"]["weights_manifest"]), weights_manifest)
    return destination, index


def source_code_hash():
    from .io_utils import sha256

    files = sorted((ROOT / "src/fall_detect").rglob("*.py"))
    return fingerprint({str(p.relative_to(ROOT)): sha256(p) for p in files})
