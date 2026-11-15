"""Strict, project-relative configuration; experiment semantics are fingerprinted."""

import math
from copy import deepcopy
from pathlib import Path

import yaml

from .contracts import RECIPE_ID
from .io_utils import ContractError, fingerprint
from .recipes import OPENPOSE_RECIPE, RECIPES, YOLO_RECIPE, upstream_commits

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_OBJECTIVES = ("sequence_fall_f1", "balanced_sequence_accuracy")
DEFAULTS = {
    "project": {"name": "fall-detection-anomaly", "seed": 42},
    "runtime": {"device": "auto", "num_workers": 0, "torch_threads": 4},
    "data": {
        "dataset": "urfall",
        "raw_dir": "data/raw/urfall",
        "manifest_dir": "data/manifests",
        "cache_dir": "data/cache",
        "camera": 0,
        "modality": "rgb",
        "protocol_id": "urfall-cam0-rgb-sequence-v1",
        "split_seed": 42,
        "adl_counts": [24, 8, 8],
        "fall_counts": [0, 15, 15],
        "timestamp_policy": "prefer_source",
        "fallback_fps": 30,
        "allow_missing_timestamp_tail": False,
    },
    "pose": {
        "backend": "human_alphapose",
        "weights_manifest": "weights/manifest.json",
        "upstream_dir": "third_party/human",
        "backbone": "resnet50",
        "input_height": 256,
        "input_width": 192,
        "detector_size": 384,
        "detector_confidence": 0.45,
        "fallback_detector_confidence": 0.0,
        "nms_threshold": 0.2,
        "track_max_age": 30,
        "track_n_init": 3,
        "use_track_proposals": False,
        "track_similarity_joints": 3,
        "track_gap_tolerance_ms": 0.0,
        "refinement_stages": 3,
        "inference_refinement_stage": 1,
        "rotation_angles": [0],
        "rotation_policy": "always",
    },
    "skeleton": {
        "recipe_id": RECIPE_ID,
        "target_fps": 30,
        "window_size": 30,
        "stride": 5,
        "missing_frame_policy": "invalidate_window",
        "span_tolerance": 1e-8,
        "resampling_hold_policy": "target_interval",
        "timestamp_tolerance_ms": 0.0,
    },
    "quality": {"min_joint_confidence": 0.2, "min_valid_joints": 9, "min_good_frame_ratio": 0.8},
    "features": {
        "backend": "human_tsstg",
        "expected_dim": 512,
        "frozen": True,
        "upstream_dir": "third_party/stgcn",
    },
    "autoencoder": {
        "sampling": "windows",
        "hidden_dim": 128,
        "latent_dim": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "input_noise_std": 0.0,
        "batch_size": 32,
        "max_epochs": 100,
        "patience": 10,
        "min_delta": 1e-6,
    },
    "calibration": {
        "split": "validation",
        "objective": "sequence_fall_f1",
        "comparison": "gt",
        "consecutive_windows": 3,
    },
    "evaluation": {"split": "test", "metric_unit": "sequence"},
    "demo": {"display": False, "output_dir": "outputs/demo", "score_hold_stride_multiplier": 2},
}


def _merge(base, changes, prefix=""):
    if not isinstance(changes, dict):
        raise ContractError(f"{prefix or 'config'} must be a mapping")
    for key, value in changes.items():
        name = prefix + key
        if key not in base:
            raise ContractError(f"Unknown config field {name}; check spelling")
        if isinstance(base[key], dict):
            _merge(base[key], value, name + ".")
        elif isinstance(base[key], bool):
            if type(value) is not bool:
                raise ContractError(f"{name} must be boolean")
            base[key] = value
        else:
            base[key] = value


def validate(config):
    timing_tolerance = config["skeleton"]["timestamp_tolerance_ms"]
    if (
        type(timing_tolerance) not in (int, float)
        or not math.isfinite(timing_tolerance)
        or not 0 <= timing_tolerance <= 1.0
    ):
        raise ContractError("skeleton.timestamp_tolerance_ms must be between 0 and 1 ms")
    tolerance = config["pose"]["track_gap_tolerance_ms"]
    if (
        type(tolerance) not in (int, float)
        or not math.isfinite(tolerance)
        or not 0 <= tolerance <= 0.001
    ):
        raise ContractError("Tracking time tolerance must be between 0 and 0.001 ms")
    for group, defaults in DEFAULTS.items():
        for key, expected in defaults.items():
            value = config[group][key]
            field = f"{group}.{key}"
            if type(expected) in (int, float):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ContractError(f"{field} must be a finite number")
                if type(expected) is int and type(value) is not int:
                    raise ContractError(f"{field} must be an integer")
                if value < 0 or (expected > 0 and value == 0):
                    raise ContractError(
                        f"{field} must be {'positive' if expected > 0 else 'nonnegative'}"
                    )
            elif isinstance(expected, str) and (not isinstance(value, str) or not value):
                raise ContractError(f"{field} must be a nonempty string")
    if config["runtime"]["device"] not in {"cpu", "cuda", "auto"}:
        raise ContractError("runtime.device must be auto, cpu or cuda")
    if config["calibration"]["objective"] not in CALIBRATION_OBJECTIVES:
        raise ContractError(f"calibration.objective must be one of {CALIBRATION_OBJECTIVES}")
    if config["autoencoder"]["sampling"] not in {"windows", "sequences"}:
        raise ContractError("autoencoder.sampling must be windows or sequences")
    if config["skeleton"]["resampling_hold_policy"] not in {
        "target_interval",
        "source_interval",
        "bounded_jitter",
    }:
        raise ContractError(
            "skeleton.resampling_hold_policy must be target_interval, source_interval or bounded_jitter"
        )
    for group, key in [
        ("quality", "min_joint_confidence"),
        ("quality", "min_good_frame_ratio"),
        ("pose", "detector_confidence"),
        ("pose", "nms_threshold"),
    ]:
        if not 0 < config[group][key] <= 1:
            raise ContractError(f"{group}.{key} must be in (0,1]")
    recipe_id = config["skeleton"]["recipe_id"]
    if recipe_id not in RECIPES:
        raise ContractError(f"Unknown recipe: {recipe_id}")
    fallback = config["pose"]["fallback_detector_confidence"]
    if fallback and (
        recipe_id != YOLO_RECIPE or not 0 < fallback < config["pose"]["detector_confidence"]
    ):
        raise ContractError(
            "Pose confidence fallback requires YOLO and a lower positive confidence"
        )
    real_joints = RECIPES[recipe_id]["joints"] - len(RECIPES[recipe_id].get("derived_joints", ()))
    if config["pose"]["use_track_proposals"] and recipe_id != RECIPE_ID:
        raise ContractError("Track crop proposals require the Human AlphaPose recipe")
    if config["quality"]["min_valid_joints"] > real_joints:
        raise ContractError("quality.min_valid_joints exceeds recipe joint count")
    required = {
        ("data", "camera"): 0,
        ("data", "modality"): "rgb",
        ("data", "adl_counts"): [24, 8, 8],
        ("data", "fall_counts"): [0, 15, 15],
        ("data", "split_seed"): 42,
        ("data", "timestamp_policy"): "prefer_source",
        ("pose", "backend"): "human_alphapose",
        ("pose", "backbone"): "resnet50",
        ("features", "backend"): "human_tsstg",
        ("features", "frozen"): True,
        ("features", "expected_dim"): 512,
        ("skeleton", "recipe_id"): RECIPE_ID,
        ("skeleton", "window_size"): 30,
        ("skeleton", "missing_frame_policy"): "invalidate_window",
        ("calibration", "split"): "validation",
        ("calibration", "comparison"): "gt",
        ("calibration", "consecutive_windows"): 3,
        ("evaluation", "split"): "test",
        ("evaluation", "metric_unit"): "sequence",
    }
    if recipe_id in (OPENPOSE_RECIPE, YOLO_RECIPE):
        required.update(
            {
                ("pose", "backend"): "lightweight_openpose",
                ("pose", "backbone"): "mobilenet_v1",
                ("pose", "refinement_stages"): 3,
                ("features", "backend"): "stgcn_kinetics",
                ("features", "expected_dim"): 256,
                ("skeleton", "recipe_id"): recipe_id,
            }
        )
    if recipe_id == YOLO_RECIPE:
        required.update({("pose", "backend"): "yolo11_pose", ("pose", "backbone"): "yolo11m"})
    for (group, key), expected in required.items():
        if config[group][key] != expected:
            raise ContractError(f"{group}.{key}: v1 recipe requires {expected!r}")
    from .data.splits import CAUCA_PROTOCOL, EXTENDED_PROTOCOL, PROTOCOL

    protocols = {
        "urfall": PROTOCOL,
        "urfall_le2i": EXTENDED_PROTOCOL,
        "urfall_le2i_caucafall": CAUCA_PROTOCOL,
    }
    if config["data"]["dataset"] not in protocols:
        raise ContractError("Unsupported data.dataset")
    expected_protocol = protocols[config["data"]["dataset"]]
    if config["data"]["protocol_id"] != expected_protocol:
        raise ContractError("data.protocol_id disagrees with data.dataset")
    if recipe_id == OPENPOSE_RECIPE and config["pose"]["inference_refinement_stage"] not in (
        1,
        2,
        3,
    ):
        raise ContractError("pose.inference_refinement_stage must be 1, 2 or 3")
    angles = config["pose"]["rotation_angles"]
    if (
        not isinstance(angles, list)
        or not angles
        or angles[0] != 0
        or any(type(a) is not int or a not in (0, -90, 90) for a in angles)
        or len(angles) != len(set(angles))
    ):
        raise ContractError(
            "pose.rotation_angles must start with 0 and contain unique 0/-90/90 values"
        )
    if recipe_id not in (OPENPOSE_RECIPE, YOLO_RECIPE) and angles != [0]:
        raise ContractError("Rotation inference requires an OpenPose-layout backend")
    if config["pose"]["rotation_policy"] not in {"always", "low_quality"}:
        raise ContractError("pose.rotation_policy must be always or low_quality")
    for key in ("input_height", "input_width", "detector_size"):
        if config["pose"][key] % 32:
            raise ContractError(f"pose.{key} must be divisible by 32")
    return config


def load_config(path=None, device=None):
    config = deepcopy(DEFAULTS)
    if path:
        _merge(config, yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")))
    if device:
        config["runtime"]["device"] = device
    return validate(config)


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def recipe_fingerprint(config, weight_hashes):
    semantic = {k: deepcopy(config[k]) for k in ("pose", "skeleton", "quality", "features")}
    if not semantic["pose"].get("use_track_proposals", False):
        semantic["pose"].pop("use_track_proposals", None)
    if semantic["skeleton"].get("timestamp_tolerance_ms", 0.0) == 0.0:
        semantic["skeleton"].pop("timestamp_tolerance_ms", None)
    if semantic["pose"].get("track_gap_tolerance_ms", 0.0) == 0.0:
        semantic["pose"].pop("track_gap_tolerance_ms", None)
    if semantic["pose"].get("fallback_detector_confidence", 0.0) == 0.0:
        semantic["pose"].pop("fallback_detector_confidence", None)
    if semantic["skeleton"].get("resampling_hold_policy") == "target_interval":
        semantic["skeleton"].pop("resampling_hold_policy")
    # Default [0] is the historical algorithm; preserve old frozen run fingerprints.
    if semantic["pose"].get("rotation_angles") == [0]:
        semantic["pose"].pop("rotation_angles")
        semantic["pose"].pop("rotation_policy", None)
    elif semantic["pose"].get("rotation_policy") == "always":
        semantic["pose"].pop("rotation_policy")
    temporal = {
        k: config["data"][k]
        for k in (
            "timestamp_policy",
            "fallback_fps",
            "camera",
            "modality",
            "allow_missing_timestamp_tail",
        )
    }
    return fingerprint(
        {
            "schema_version": 1,
            "preprocessing_version": 1,
            "upstream_commits": upstream_commits(config),
            "config": semantic,
            "temporal": temporal,
            "weight_hashes": weight_hashes,
        }
    )
