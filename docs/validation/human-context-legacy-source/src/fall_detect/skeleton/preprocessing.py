import numpy as np

from ..io_utils import ContractError

COCO17_KEEP = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
HUMAN13_NAMES = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def coco17_to_human13(points):
    points = np.asarray(points)
    if points.shape[-2:] != (17, 3):
        raise ContractError("COCO17 input must end in [17,3]")
    return points[..., COCO17_KEEP, :].copy()


def good_frame(points, quality, joints=13, derived_joints=()):
    points = np.asarray(points)
    measured = np.ones(joints, dtype=bool)
    measured[list(derived_joints)] = False
    return (
        points.shape == (joints, 3)
        and np.isfinite(points).all()
        and np.count_nonzero(points[measured, 2] >= quality["min_joint_confidence"])
        >= quality["min_valid_joints"]
    )


def human_preprocess(points, frame_size_wh, tolerance=1e-8):
    """Numeric parity with pinned ActionsEstLoader/pose_utils; preserves input dtype operations."""
    points = np.asarray(points)
    if points.shape != (30, 13, 3) or not np.issubdtype(points.dtype, np.floating):
        raise ContractError("Human TSSTG requires floating [30,13,3] skeletons")
    if not np.isfinite(points).all():
        raise ContractError("degenerate_pose: non-finite coordinates/confidence")
    width, height = frame_size_wh
    if width <= 0 or height <= 0:
        raise ContractError("frame_size_wh must contain positive width, height")
    pts = points.copy()
    pts[:, :, 0] /= width
    pts[:, :, 1] /= height
    low = pts[:, :, :2].min(axis=1)
    high = pts[:, :, :2].max(axis=1)
    if np.any(high - low <= tolerance):
        raise ContractError("degenerate_pose: zero/tiny per-frame coordinate span")
    for index in range(len(pts)):
        pts[index, :, :2] = ((pts[index, :, :2] - low[index]) / (high[index] - low[index])) * 2 - 1
    pts = np.concatenate((pts, ((pts[:, 1, :] + pts[:, 2, :]) / 2)[:, None, :]), axis=1)
    points_tensor = pts.astype(np.float32).transpose(2, 0, 1)[None].copy()
    motion_tensor = points_tensor[:, :2, 1:, :] - points_tensor[:, :2, :-1, :]
    return points_tensor, motion_tensor


OPENPOSE18_NAMES = (
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
)
OPENPOSE18_EDGES = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]


def openpose_preprocess(points, frame_size_wh):
    """Kinetics feeder: x/width-.5, y/height-.5, zero coordinates where score=0.

    One actual person, no fake second person, no per-frame min/max, no padding to 300.
    The 30-frame causal short window is an explicit recipe deviation from pretraining.
    """
    points = np.asarray(points)
    if points.shape != (30, 18, 3) or not np.issubdtype(points.dtype, np.floating):
        raise ContractError("OpenPose ST-GCN requires floating [30,18,3] skeletons")
    if not np.isfinite(points).all() or np.any(points[:, :, 2] < 0):
        raise ContractError("degenerate_pose: invalid OpenPose coordinates/confidence")
    width, height = frame_size_wh
    if width <= 0 or height <= 0:
        raise ContractError("frame_size_wh must contain positive width, height")
    data = points.astype(np.float32).copy()
    data[:, :, 0] = data[:, :, 0] / width - 0.5
    data[:, :, 1] = data[:, :, 1] / height - 0.5
    data[data[:, :, 2] == 0, :2] = 0
    return data.transpose(2, 0, 1)[None, :, :, :, None].copy()
