"""Controlled observed-skeleton versus pooled-feature neighborhood diagnostic."""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from ur_normal_methods import BASE, ROOT, C, E, save, validated_index, verify_lock

from fall_detect.config import project_path
from fall_detect.data.cache import load_pose_cache
from fall_detect.io_utils import ContractError, write_npz
from fall_detect.skeleton.preprocessing import human_preprocess
from fall_detect.skeleton.windows import CausalResampler, WindowBuilder
from fall_detect.training.experiment import Scorer, cache_for
from fall_detect.training.scaler import FeatureScaler


def main():
    lock = verify_lock()
    if (E / "geometry-diagnosis.json").exists():
        raise ContractError("Preserve completed geometry diagnosis")
    save(
        E / "geometry-rule.json",
        {
            "purpose": "Support or reject a bounded representation hypothesis, not prove joint GT or causality",
            "raw_representation": "Original legal30 per_frame_isotropic XY tensor 30x14x2, no confidence channel; Train-only dimension standardization, exact nearest Train window distance/sqrt840",
            "support_rule": "At least3 consecutive fall19 coarse-phase endpoints with current good pose, raw-skeleton nearest distance above normalVal p95 while pooled512 nearest distance <= normalVal p95",
            "labels_used_to_train": False,
            "test_used": False,
            "if_not_supported": "Stop self-supervised stage; no blind fine tuning",
        },
    )
    index = validated_index()
    config = Scorer(ROOT / "runs" / BASE).config
    raw, pooled, rows = {}, {}, {}
    for src in index["split_manifest"]["sequences"]:
        if src["split"] == "test":
            continue
        sid = src["sequence_id"]
        entry = index["sequences"][sid]
        cache = cache_for(index, sid)
        observations, _ = load_pose_cache(
            project_path(entry["pose_path"]),
            {
                "sequence_id": sid,
                "recipe_hash": index["recipe_hash"],
                "source_hash": entry["source_hash"],
            },
        )
        original = {
            (r.sample_index, r.track_id): r for r in cache.records if r.row_index is not None
        }
        values, diagnostics = {}, {}
        resampler = CausalResampler(30, config["skeleton"]["resampling_hold_policy"])
        builder = WindowBuilder(config)
        for frame, poses in observations:
            for sample in resampler.push(frame, poses):
                for window in builder.push(sample):
                    key = (window.sample_index, window.track_id)
                    if window.quality == "valid" and key in original:
                        rec = original[key]
                        pts, _ = human_preprocess(
                            window.keypoints, window.frame_size_wh, 1e-8, 30, "per_frame_isotropic"
                        )
                        values[rec.row_index] = pts[0, :2].transpose(1, 2, 0).reshape(-1)
                        xy = window.keypoints[..., :2]
                        confidence = window.keypoints[..., 2]
                        width, height = window.frame_size_wh
                        reliable = confidence >= 0.2
                        outside = (
                            (xy[..., 0] < 0)
                            | (xy[..., 0] >= width)
                            | (xy[..., 1] < 0)
                            | (xy[..., 1] >= height)
                        )
                        diagnostics[rec.row_index] = {
                            "sample_index": rec.sample_index,
                            "track_id": rec.track_id,
                            "available_at_ms": rec.available_at_ms,
                            "reliable_joint_fraction": float(np.mean(reliable)),
                            "reliable_outside_fraction": float(
                                np.sum(outside & reliable) / max(1, np.sum(reliable))
                            ),
                        }
        if set(values) != set(range(len(cache.features))):
            raise ContractError("Observed skeletons do not match cached legal feature rows")
        raw[sid] = np.stack([values[i] for i in range(len(values))])
        pooled[sid] = cache.features
        rows[sid] = [diagnostics[i] for i in range(len(values))]
    train_ids = [
        r["sequence_id"] for r in index["split_manifest"]["sequences"] if r["split"] == "train"
    ]
    normal_ids = [
        r["sequence_id"]
        for r in index["split_manifest"]["sequences"]
        if r["split"] == "validation" and r["label"] == 0
    ]
    distances, percentiles, scalers = {}, {}, {}
    for kind, arrays in [("raw", raw), ("pooled", pooled)]:
        train = np.concatenate([arrays[sid] for sid in train_ids])
        scaler = FeatureScaler.fit(train)
        scalers[kind] = scaler
        model = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=4).fit(
            scaler.transform(train)
        )
        d = {
            sid: model.kneighbors(scaler.transform(x))[0][:, 0] / np.sqrt(x.shape[1])
            for sid, x in arrays.items()
            if sid not in train_ids
        }
        distances[kind] = d
        normal = np.concatenate([d[sid] for sid in normal_ids])
        percentiles[kind] = {
            "p50": float(np.quantile(normal, 0.5)),
            "p95": float(np.quantile(normal, 0.95)),
        }
    sid = "fall-19"
    lo, hi = lock["coarse_intervals"][sid]["fall_phase_ms"]
    good = {tuple(x) for x in lock["current_good_pose_keys"][sid]}
    phase = []
    for i, row in enumerate(rows[sid]):
        if lo <= row["available_at_ms"] <= hi:
            raw_d, pooled_d = float(distances["raw"][sid][i]), float(distances["pooled"][sid][i])
            phase.append(
                {
                    **row,
                    "raw_nearest_distance": raw_d,
                    "pooled_nearest_distance": pooled_d,
                    "current_good_pose": (row["sample_index"], row["track_id"]) in good,
                    "support_endpoint": (row["sample_index"], row["track_id"]) in good
                    and raw_d > percentiles["raw"]["p95"]
                    and pooled_d <= percentiles["pooled"]["p95"],
                }
            )
    longest, length, previous = 0, 0, None
    for row in phase:
        key = (row["sample_index"], row["track_id"])
        adjacent = previous is not None and key[0] == previous[0] + 1 and key[1] == previous[1]
        length = (length + 1 if adjacent else 1) if row["support_endpoint"] else 0
        longest = max(longest, length)
        previous = key
    directory = C / "geometry-diagnostic"
    directory.mkdir()
    for kind, scaler in scalers.items():
        scaler.save(directory / f"{kind}-train-only-scaler.npz")
    write_npz(directory / "fall19-observed-raw.npz", normalized_xy=raw[sid])
    save(
        E / "geometry-diagnosis.json",
        {
            "normal_val_distance_percentiles": percentiles,
            "fall19_phase": phase,
            "longest_supporting_run": longest,
            "self_supervised_supported": longest >= 3,
            "data_kind": "Observed inferred pose, not manually verified skeleton GT",
            "limitation": "Neighborhood contrast is diagnostic evidence; quality confidence and AI coarse times cannot establish pose accuracy or causal feature loss",
            "test_feature_reads": 0,
            "test_evaluations": 0,
        },
    )
    print(
        {
            "percentiles": percentiles,
            "longest_supporting_run": longest,
            "self_supervised_supported": longest >= 3,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
