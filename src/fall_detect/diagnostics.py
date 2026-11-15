import importlib.metadata
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import ROOT, project_path
from .io_utils import ContractError, write_json


def environment_report():
    import torch

    packages = {}
    for package in (
        "numpy",
        "torch",
        "torchvision",
        "opencv-python-headless",
        "PyYAML",
        "scikit-learn",
        "matplotlib",
        "scipy",
        "pytest",
        "gdown",
        "truststore",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        gpu = None
    return {
        "os": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
        "gpu_driver_info": gpu,
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
    }


def codec_probe():
    import cv2
    import numpy as np

    result = {}
    with tempfile.TemporaryDirectory(prefix="fall-codec-") as directory:
        for codec, suffix in [("mp4v", ".mp4"), ("MJPG", ".avi")]:
            path = Path(directory) / (codec + suffix)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), 30, (160, 120))
            if writer.isOpened():
                for index in range(6):
                    writer.write(np.full((120, 160, 3), index * 20, dtype=np.uint8))
            writer.release()
            cap = cv2.VideoCapture(str(path))
            count = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape != (120, 160, 3):
                    break
                count += 1
            cap.release()
            result[codec] = {
                "passed": count == 6 and path.exists() and path.stat().st_size > 0,
                "decoded_frames": count,
            }
    return result


def doctor(config, mode="core", output=None):
    from .assets import register_weights
    from .data.splits import dataset_split
    from .models.weights import select_device

    report = {"mode": mode, "checks": {}, "environment": environment_report(), "errors": []}
    import torch

    device = None
    try:
        device = select_device(config["runtime"]["device"], config["runtime"]["torch_threads"])
        # This is a device arithmetic check, not model inference.
        value = torch.ones((2, 3), device=device)
        torch.testing.assert_close((value @ value.T).cpu(), torch.full((2, 2), 3.0))
        report["checks"]["device"] = str(device)
    except (ContractError, RuntimeError, AssertionError) as exc:
        report["errors"].append(str(exc))
    report["checks"]["codecs"] = codec_probe()
    if not any(item["passed"] for item in report["checks"]["codecs"].values()):
        report["errors"].append("No verified VideoWriter codec")
    try:
        weights = register_weights(config)
        report["weights"] = weights["resources"]
        if mode == "full":
            missing = [
                e["role"]
                for e in weights["resources"]
                if e["status"] != "usable"
                or str(device)
                not in e.get("verification", {}).get(
                    "devices", [e.get("verification", {}).get("device")]
                )
            ]
            if missing:
                report["errors"].append(
                    "Weights not real-forward verified: "
                    + ", ".join(missing)
                    + "; fetch-assets --models, then smoke-test --mode real"
                )
            try:
                import importlib

                from .models.weights import import_upstream
                from .recipes import OPENPOSE_RECIPE, POSE_COMMIT, STGCN_COMMIT, YOLO_RECIPE

                if config["skeleton"]["recipe_id"] in (OPENPOSE_RECIPE, YOLO_RECIPE):
                    from .models.weights import import_pinned

                    import_pinned(
                        config["pose"]["upstream_dir"], POSE_COMMIT, "models.with_mobilenet"
                    )
                    import_pinned(config["features"]["upstream_dir"], STGCN_COMMIT, "net.st_gcn")
                    if config["skeleton"]["recipe_id"] == YOLO_RECIPE:
                        from .pose.yolo import yolo_api

                        yolo_api()
                else:
                    import_upstream(config)
                    for module in (
                        "torchvision",
                        "SPPE.src.models.FastPose",
                        "Detection.Models",
                        "Track.Tracker",
                    ):
                        importlib.import_module(module)
                report["checks"]["pose_imports"] = "passed"
            except (ImportError, ContractError) as exc:
                report["errors"].append(str(exc))
    except ContractError as exc:
        report["errors"].append(str(exc))
    split = dataset_split(config)
    root = project_path(config["data"]["raw_dir"])
    present = [
        row["sequence_id"]
        for row in split["sequences"]
        if (
            project_path(row["source_path"]).exists()
            if "source_path" in row
            else (root / row["sequence_id"] / "source.json").exists()
        )
    ]
    planned = len(split["sequences"])
    report["dataset"] = {
        "n_planned": planned,
        "n_downloaded": len(present),
        "downloaded_ids": present,
    }
    if mode == "full" and len(present) != planned:
        report["errors"].append(f"Full dataset requires {planned} sequences; found {len(present)}")
    report["passed"] = not report["errors"]
    destination = Path(output) if output else project_path(f"outputs/doctor-{mode}.json")
    write_json(destination, report)
    return report
