"""Pinned COCO17 pose model, explicit OpenPose18 mapping, frozen strict weights.

Neck is the shoulder midpoint only when both shoulders are observed. Its confidence
is the smaller shoulder confidence; it is excluded from the real-joint quality count.
No missing measured joint is interpolated or copied from another frame.
"""

import os
from importlib.metadata import version

import numpy as np
import torch

from ..config import project_path
from ..io_utils import ContractError, sha256, write_json
from ..models.weights import import_pinned, strict_load
from ..recipes import POSE_COMMIT, YOLO_SOURCE_SHA, YOLO_URL, YOLO_VERSION
from ..skeleton.preprocessing import good_frame
from .lightweight import LightweightPoseBackend

COCO_TO_OPENPOSE = (0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3)


def coco17_to_openpose18(points):
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (17, 3) or not np.isfinite(points).all():
        raise ContractError("YOLO pose must be finite [17,3]")
    if np.any((points[:, 2] < 0) | (points[:, 2] > 1)):
        raise ContractError("YOLO keypoint confidence must be in [0,1]")
    result = np.zeros((18, 3), dtype=np.float32)
    for destination, source in enumerate(COCO_TO_OPENPOSE):
        if source >= 0:
            result[destination] = points[source]
    if min(points[5, 2], points[6, 2]) > 0:
        result[1, :2] = (points[5, :2] + points[6, :2]) / 2
        result[1, 2] = min(points[5, 2], points[6, 2])
    result[result[:, 2] == 0, :2] = 0
    return result


def yolo_api():
    if version("ultralytics") != YOLO_VERSION:
        raise ContractError(f"This recipe requires ultralytics=={YOLO_VERSION}")
    settings = project_path("outputs/ultralytics-settings")
    settings.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings)
    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ["YOLO_OFFLINE"] = "True"
    from ultralytics.nn.tasks import PoseModel

    from ultralytics import YOLO

    return YOLO, PoseModel


def new_yolo():
    YOLO, PoseModel = yolo_api()
    wrapper = YOLO("yolo11m-pose.yaml", task="pose")
    # The packaged YAML has nc=80; the official COCO pose checkpoint has nc=1.
    wrapper.model = PoseModel(wrapper.model.yaml, nc=1, verbose=False)
    wrapper.model.names = {0: "person"}
    wrapper.model.kpt_shape = wrapper.model.model[-1].kpt_shape
    wrapper.model.task = "pose"
    return wrapper


def fetch_yolo_weights(config, http):
    from ..assets import download, ensure_weight_manifest

    manifest = ensure_weight_manifest(config)
    source = project_path("weights/pretrained/yolo11/yolo11m-pose.pt")
    download(http, YOLO_URL, source)
    if sha256(source) != YOLO_SOURCE_SHA:
        raise ContractError("Official YOLO source checksum mismatch")
    for entry in manifest["resources"]:
        destination = project_path(entry["local_path"])
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if entry["role"] == "stgcn":
            import gdown

            temporary = destination.with_suffix(".part")
            if not gdown.download(entry["download_url"], str(temporary), quiet=False):
                raise ContractError("ST-GCN download failed")
            temporary.replace(destination)
        else:
            YOLO, _ = yolo_api()
            # Trusted, exact-hash official object checkpoint is deserialized once.
            # Runtime inference uses the resulting tensor-only checkpoint instead.
            original = YOLO(str(source))
            state = original.model.float().state_dict()
            new_yolo().model.load_state_dict(state, strict=True)
            from pathlib import Path
            from tempfile import TemporaryDirectory

            # Preserve the checkpoint archive's basename for deterministic re-fetch.
            with TemporaryDirectory(dir=destination.parent) as temporary_dir:
                temporary = Path(temporary_dir) / destination.name
                torch.save(state, temporary)
                temporary.replace(destination)
            write_json(
                destination.with_suffix(".source.json"),
                {
                    "source_url": YOLO_URL,
                    "source_sha256": sha256(source),
                    "state_sha256": sha256(destination),
                    "package_version": YOLO_VERSION,
                    "conversion": "float32 state_dict only; strict nc=1 model; no training",
                },
            )


class YoloPoseBackend(LightweightPoseBackend):
    def __init__(self, config, resources, device):
        self.wrapper = new_yolo()
        self.model = self.wrapper.model
        entry = resources["pose"]
        self.load_report = strict_load(
            self.model, project_path(entry["local_path"]), device, entry["sha256"]
        )
        self.load_report.update(
            package_version=YOLO_VERSION, state_keys=len(self.model.state_dict())
        )
        self.pose_module = import_pinned(
            config["pose"]["upstream_dir"], POSE_COMMIT, "modules.pose"
        )
        self.config, self.device = config, device
        self.forward_verified = False
        self.reset()

    @torch.inference_mode()
    def _decode(self, image):
        # Ultralytics changes this process-wide variable, including setting -1 on CPU.
        # Restore it so a CPU diagnostic cannot disable later explicit CUDA work.
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            candidates = self._predict_candidates(image, self.config["pose"]["detector_confidence"])
            fallback = self.config["pose"].get("fallback_detector_confidence", 0.0)
            if fallback and not any(
                good_frame(p, self.config["quality"], 18, (1,)) for p, _ in candidates
            ):
                # Retry only the same observed image. Quality rules and missing-joint
                # semantics are unchanged; never synthesize a person from history.
                candidates = self._predict_candidates(image, fallback)
            return candidates
        finally:
            if visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    def _predict_candidates(self, image, confidence):
        result = self.wrapper.predict(
            source=image,
            imgsz=self.config["pose"]["input_height"],
            conf=confidence,
            iou=self.config["pose"]["nms_threshold"],
            classes=[0],
            # torch.device bypasses upstream string selection and its global
            # CUDA_VISIBLE_DEVICES mutation before CUDA has initialized.
            device=self.device,
            half=False,
            verbose=False,
            save=False,
        )[0]
        points = result.keypoints.data.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        if points.shape != (len(scores), 17, 3) or not np.isfinite(points).all():
            raise ContractError("Invalid real YOLO pose output")
        self.forward_verified = True
        return [(coco17_to_openpose18(p), float(s)) for p, s in zip(points, scores)]
