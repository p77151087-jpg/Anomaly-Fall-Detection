"""Use the pinned networks and crop/heatmap operations with explicit device handling."""

import cv2
import numpy as np
import torch

from ..config import project_path
from ..contracts import PoseRecord
from ..io_utils import ContractError
from ..models.weights import import_upstream, strict_load
from ..skeleton.preprocessing import COCO17_KEEP, good_frame


def pose_nms_compatible(boxes, scores, points, confidence):
    """Pinned Human pose-NMS arithmetic; preserve tensor types across deletion.

    Upstream mixes np.delete with torch tensors. NumPy 2 returns ndarrays, which
    subsequent torch reductions reject. Selection/merging follows the same algorithm.
    """
    from pPose_nms import PCK_match, get_parametric_distance, p_merge_fast

    confidence = confidence.clone()
    confidence[confidence == 0] = 1e-5
    reference = 0.1 * torch.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    ids = torch.arange(len(boxes))
    original_points, original_conf = points.clone(), confidence.clone()
    result = []
    while len(ids):
        local = int(confidence.mean(dim=1).argmax())
        chosen = int(ids[local])
        distance = reference[chosen]
        similarity = get_parametric_distance(local, points, confidence, distance)
        matches = PCK_match(points[local], points, distance)
        mask = (similarity > 22.48) | (matches >= 5)
        if not mask.any():
            mask[local] = True
        merged_ids = ids[mask]
        if original_conf[chosen].max() >= 0.3:
            merged_pose, merged_conf = p_merge_fast(
                original_points[chosen],
                original_points[merged_ids],
                original_conf[merged_ids],
                distance,
            )
            if merged_conf.max() >= 0.3:
                result.append(
                    {
                        "bbox": boxes[chosen],
                        "bbox_score": scores[chosen],
                        "keypoints": merged_pose - 0.3,
                        "kp_score": merged_conf,
                    }
                )
        ids, points, confidence = ids[~mask], points[~mask], confidence[~mask]
    return result


def add_track_proposals(boxes, tracks, frame_size_wh, max_age):
    """Sample main.py's Kalman crop proposals, followed by fresh image pose inference.

    A proposed box is not an observed skeleton. It cannot fill missing frames or
    directly enter the feature window; normal pose quality rules still apply.
    """
    width, height = frame_size_wh
    proposals = []
    for track in tracks:
        if track.time_since_update > max_age:
            continue
        box = np.asarray(track.to_tlbr(), dtype=np.float32).copy()
        if not np.isfinite(box).all():
            continue
        box[[0, 2]] = box[[0, 2]].clip(0, width - 1)
        box[[1, 3]] = box[[1, 3]].clip(0, height - 1)
        if box[2] > box[0] + 1 and box[3] > box[1] + 1:
            proposals.append([*box, 0.5, 1.0, 0.0])
    if not proposals:
        return boxes
    added = torch.tensor(proposals, dtype=torch.float32)
    return added if boxes is None else torch.cat((boxes, added), dim=0)


class HumanPoseBackend:
    def __init__(self, config, resources, device):
        import_upstream(config)
        from Detection.Models import Darknet
        from Detection.Utils import ResizePadding, non_max_suppression
        from SPPE.src.models.FastPose import FastPose
        from SPPE.src.utils.eval import getPrediction
        from SPPE.src.utils.img import crop_dets
        from Track.Tracker import Detection, Tracker

        self.config, self.device = config, device
        self.options = config["pose"]
        self.size = self.options["detector_size"]
        self.detector = Darknet(str(project_path(resources["detector_config"]["local_path"])))
        self.pose = FastPose("resnet50", 17)
        self.load_reports = {}
        for role, model in [("detector", self.detector), ("pose", self.pose)]:
            entry = resources[role]
            self.load_reports[role] = strict_load(
                model, project_path(entry["local_path"]), device, entry["sha256"]
            )
        self.resize = ResizePadding(self.size, self.size)
        self.nms, self.crop, self.decode = non_max_suppression, crop_dets, getPrediction
        self.Tracker, self.Detection = Tracker, Detection
        self.reset()

    def reset(self):
        self.tracker = self.Tracker(
            max_age=self.options["track_max_age"], n_init=self.options["track_n_init"]
        )
        self.sequence_id = None
        self.forward_verified = False

    @torch.inference_mode()
    def predict(self, frame):
        if self.sequence_id is None:
            self.sequence_id = frame.sequence_id
        if frame.sequence_id != self.sequence_id:
            raise ContractError("Reset pose tracker before processing a different sequence")
        if frame.source_missing:
            self.tracker.predict()
            self.tracker.update([])
            return []
        rgb = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2RGB)
        scaled = self.resize(rgb)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(scaled.transpose(2, 0, 1)))
            .float()
            .div(255)[None]
            .to(self.device)
        )
        predicted = self.detector(tensor)
        boxes = self.nms(
            predicted, self.options["detector_confidence"], self.options["nms_threshold"]
        )[0]
        detections = []
        detector_scores = {}
        self.tracker.predict()
        if boxes is not None:
            boxes = boxes.detach().cpu()
            scale = min(self.size / frame.width, self.size / frame.height)
            boxes[:, [0, 2]] -= (self.size - scale * frame.width) / 2
            boxes[:, [1, 3]] -= (self.size - scale * frame.height) / 2
            boxes[:, :4] /= scale
            boxes[:, :2] -= 5
            boxes[:, 2:4] += 5
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, frame.width - 1)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, frame.height - 1)
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes = boxes[keep]
        if self.options.get("use_track_proposals", False):
            boxes = add_track_proposals(
                boxes,
                self.tracker.tracks,
                (frame.width, frame.height),
                self.options["track_max_age"],
            )
        if boxes is not None and len(boxes):
            height, width = self.options["input_height"], self.options["input_width"]
            crops, corner1, corner2 = self.crop(rgb, boxes[:, :4], height, width)
            heatmaps = self.pose(crops.to(self.device)).cpu()[:, COCO17_KEEP, ...]
            if not torch.isfinite(heatmaps).all():
                raise ContractError("Non-finite AlphaPose heatmap")
            _, xy, confidence = self.decode(
                heatmaps,
                corner1,
                corner2,
                height,
                width,
                heatmaps.shape[-2],
                heatmaps.shape[-1],
            )
            poses = pose_nms_compatible(boxes[:, :4], boxes[:, 4], xy, confidence)
            self.forward_verified = True
            for pose in poses:
                keypoints = torch.cat((pose["keypoints"], pose["kp_score"]), dim=1).numpy()
                xy = keypoints[:, :2]
                box = np.r_[xy.min(axis=0) - 20, xy.max(axis=0) + 20]
                detections.append(self.Detection(box, keypoints, float(keypoints[:, 2].mean())))
                detector_scores[keypoints.tobytes()] = float(pose["bbox_score"])
        self.tracker.update(detections)
        results = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update != 0 or not track.keypoints_list:
                continue
            keypoints = np.asarray(track.keypoints_list[-1], dtype=np.float32)
            results.append(
                PoseRecord(
                    frame.sequence_id,
                    frame.source_frame_id,
                    frame.sample_index,
                    frame.timestamp_ms,
                    track.track_id,
                    track.to_tlbr(),
                    keypoints,
                    (frame.width, frame.height),
                    detector_scores.get(keypoints.tobytes(), 0.0),
                    "valid" if good_frame(keypoints, self.config["quality"]) else "low_confidence",
                )
            )
        return results
