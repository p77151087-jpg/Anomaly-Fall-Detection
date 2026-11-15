"""Pinned Lightweight OpenPose, full 18-joint output; no synthetic joint filling.

Image normalization/resize/decode follows upstream demo.py and val.py (Apache-2.0).
All three training refinement stages are strictly loaded. Inference uses stage one,
matching the author's published one-refinement demo. Missing joints retain score 0.
"""

import math
from contextlib import nullcontext

import cv2
import numpy as np
import torch

from ..config import project_path
from ..contracts import PoseRecord
from ..io_utils import ContractError
from ..models.weights import import_pinned, strict_load
from ..recipes import POSE_COMMIT, recipe_spec
from ..skeleton.preprocessing import good_frame


def prepare_image(image, input_height):
    scale = input_height / image.shape[0]
    resized = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) - np.array([128, 128, 128], np.float32)) * np.float32(
        1 / 256
    )
    height, width = normalized.shape[:2]
    padded_height = math.ceil(input_height / 8) * 8
    padded_width = math.ceil(max(width, input_height) / 8) * 8
    top, left = (padded_height - height) // 2, (padded_width - width) // 2
    pad = [top, left, padded_height - height - top, padded_width - width - left]
    padded = cv2.copyMakeBorder(
        normalized, pad[0], pad[2], pad[1], pad[3], cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return torch.from_numpy(padded).permute(2, 0, 1)[None].float(), scale, pad


def restore_rotation(points, angle, width, height):
    """Map real rotated-image observations back; zero-score joints stay missing."""
    result = points.copy()
    valid = result[:, 2] > 0
    x, y = points[valid, 0], points[valid, 1]
    if angle == 90:  # cv2 counterclockwise
        result[valid, 0], result[valid, 1] = width - 1 - y, x
    elif angle == -90:
        result[valid, 0], result[valid, 1] = y, height - 1 - x
    elif angle != 0:
        raise ContractError("Unsupported image rotation")
    result[~valid, :2] = 0
    return result


def suppress_duplicate_poses(candidates):
    """Keep a complete observation, never average or invent joints across views."""
    selected = []
    for points, score in sorted(
        candidates,
        key=lambda item: (np.count_nonzero(item[0][:, 2] >= 0.2), float(item[0][:, 2].sum())),
        reverse=True,
    ):
        valid = points[:, 2] > 0
        duplicate = False
        for old, _ in selected:
            common = valid & (old[:, 2] > 0)
            if np.count_nonzero(common) < 3:
                continue
            extent = np.ptp(old[old[:, 2] > 0, :2], axis=0)
            scale = max(float(np.linalg.norm(extent)), 1.0)
            distances = np.linalg.norm(points[common, :2] - old[common, :2], axis=1)
            if np.median(distances) / scale < 0.1:
                duplicate = True
                break
        if not duplicate:
            selected.append((points, score))
    return selected


def tracking_gap_exceeded(previous_time, timestamp_ms, config):
    tolerance = config["pose"].get("track_gap_tolerance_ms", 0.0)
    return previous_time is not None and timestamp_ms - previous_time > (
        1500 / config["skeleton"]["target_fps"] + tolerance
    )


class LightweightPoseBackend:
    def __init__(self, config, resources, device):
        path = config["pose"]["upstream_dir"]
        module = import_pinned(path, POSE_COMMIT, "models.with_mobilenet")
        self.keypoint_module = import_pinned(path, POSE_COMMIT, "modules.keypoints")
        self.pose_module = import_pinned(path, POSE_COMMIT, "modules.pose")
        self.model = module.PoseEstimationWithMobileNet(num_refinement_stages=3)
        entry = resources["pose"]
        self.load_report = strict_load(
            self.model, project_path(entry["local_path"]), device, entry["sha256"]
        )
        self.config, self.device = config, device
        self.forward_verified = False
        self.reset()

    def reset(self):
        self.previous = []
        self.last_id = -1
        self.last_time = None

    @torch.inference_mode()
    def predict(self, frame):
        # Native CPU kernels avoid the very slow Windows oneDNN depthwise path.
        # This changes neither device nor model parameters; GPU uses its usual backend.
        context = (
            torch.backends.mkldnn.flags(enabled=False)
            if self.device.type == "cpu"
            else nullcontext()
        )
        with context:
            return self._predict(frame)

    def _predict(self, frame):
        if frame.source_missing or frame.image_bgr is None:
            self.previous = []
            self.last_time = frame.timestamp_ms
            return []
        if tracking_gap_exceeded(self.last_time, frame.timestamp_ms, self.config):
            self.previous = []
        self.last_time = frame.timestamp_ms
        candidates = []
        angles = self.config["pose"]["rotation_angles"]
        primary_good = False
        for angle in angles:
            if angle and primary_good and self.config["pose"]["rotation_policy"] == "low_quality":
                break
            image = frame.image_bgr
            if angle:
                image = cv2.rotate(
                    image,
                    cv2.ROTATE_90_COUNTERCLOCKWISE if angle == 90 else cv2.ROTATE_90_CLOCKWISE,
                )
            for points, score in self._decode(image):
                candidates.append(
                    (restore_rotation(points, angle, frame.width, frame.height), score)
                )
            if angle == 0:
                quality = self.config["quality"]
                primary_good = any(
                    good_frame(
                        p, quality, len(p), recipe_spec(self.config).get("derived_joints", ())
                    )
                    for p, _ in candidates
                )
        if len(angles) > 1:
            candidates = suppress_duplicate_poses(candidates)
        current = []
        for points, score in candidates:
            tracking_points = np.full((18, 2), -1, np.int32)
            valid = points[:, 2] > 0
            tracking_points[valid] = points[valid, :2].astype(np.int32)
            current.append(self.pose_module.Pose(tracking_points, score))
        self.pose_module.Pose.last_id = self.last_id
        self.pose_module.track_poses(
            self.previous,
            current,
            threshold=self.config["pose"]["track_similarity_joints"],
            smooth=False,
        )
        self.last_id = self.pose_module.Pose.last_id
        self.previous = current
        result = []
        for pose, (points, _) in zip(current, candidates):
            x, y, w, h = pose.bbox
            result.append(
                PoseRecord(
                    frame.sequence_id,
                    frame.source_frame_id,
                    frame.sample_index,
                    frame.timestamp_ms,
                    pose.id,
                    np.array([x, y, x + w, y + h], np.float32),
                    points,
                    (frame.width, frame.height),
                    pose.confidence,
                    layout_id="openpose18",
                    recipe_id=self.config["skeleton"]["recipe_id"],
                )
            )
        return result

    def _decode(self, image):
        tensor, scale, pad = prepare_image(image, self.config["pose"]["input_height"])
        self.model.eval()
        # All heads stay strictly loaded/frozen; selection is part of the recipe hash.
        backbone = self.model.cpm(self.model.model(tensor.to(self.device)))
        initial = self.model.initial_stage(backbone)
        heat, paf = initial
        stage = self.config["pose"]["inference_refinement_stage"]
        for refinement in self.model.refinement_stages[:stage]:
            heat, paf = refinement(torch.cat([backbone, heat, paf], dim=1))
        if not torch.isfinite(heat).all() or not torch.isfinite(paf).all():
            raise ContractError("Non-finite pose network output")
        if not self.forward_verified:
            original = self.model(tensor.to(self.device))
            torch.testing.assert_close(heat, original[stage * 2], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(paf, original[stage * 2 + 1], atol=1e-6, rtol=1e-5)
            self.forward_verified = True
        maps = [
            cv2.resize(
                x[0].permute(1, 2, 0).cpu().numpy(),
                (0, 0),
                fx=4,
                fy=4,
                interpolation=cv2.INTER_CUBIC,
            )
            for x in (heat, paf)
        ]
        all_by_type, total = [], 0
        for joint in range(18):
            total += self.keypoint_module.extract_keypoints(
                maps[0][:, :, joint], all_by_type, total
            )
        entries, all_points = self.keypoint_module.group_keypoints(all_by_type, maps[1])
        if not len(all_points):
            return []
        all_points[:, 0] = (all_points[:, 0] * 2 - pad[1]) / scale
        all_points[:, 1] = (all_points[:, 1] * 2 - pad[0]) / scale
        result = []
        for entry in entries:
            points = np.zeros((18, 3), np.float32)
            for joint in range(18):
                if entry[joint] >= 0:
                    points[joint] = all_points[int(entry[joint]), :3]
            result.append((points, float(entry[18])))
        return result
