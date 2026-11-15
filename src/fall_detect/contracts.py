"""Versioned records shared by readers, preprocessing, caches and scoring."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

SCHEMA_VERSION = 1
RECIPE_ID = "human_tsstg_coco_cut_v1"
Split = Literal["train", "validation", "test", "inference"]


@dataclass
class FrameRecord:
    sequence_id: str
    source_frame_id: int
    sample_index: int
    timestamp_ms: float
    image_bgr: np.ndarray
    width: int
    height: int
    schema_version: int = SCHEMA_VERSION
    timestamp_estimated: bool = False
    source_missing: bool = False
    source_fps: float | None = None
    source_fps_estimated: bool = False


@dataclass
class PoseRecord:
    sequence_id: str
    source_frame_id: int
    sample_index: int
    timestamp_ms: float
    track_id: int
    bbox_xyxy: np.ndarray
    keypoints: np.ndarray
    frame_size_wh: tuple[int, int]
    detection_confidence: float
    quality: str = "valid"
    layout_id: str = "human13"
    schema_version: int = SCHEMA_VERSION
    recipe_id: str = RECIPE_ID


@dataclass
class FeatureRecord:
    sequence_id: str
    track_id: int | None
    window_id: int
    sample_index: int
    start_ms: float
    end_ms: float
    available_at_ms: float
    split: str
    row_index: int | None
    feature_dim: int = 512
    quality: str = "valid"
    recipe_id: str = RECIPE_ID
    schema_version: int = SCHEMA_VERSION


@dataclass
class Window:
    sequence_id: str
    track_id: int | None
    sample_index: int
    start_ms: float
    end_ms: float
    keypoints: np.ndarray | None
    frame_size_wh: tuple[int, int]
    quality: str
    good_frame_ratio: float = 0.0


@dataclass
class Sample:
    sequence_id: str
    sample_index: int
    timestamp_ms: float
    source_frame_id: int | None
    poses: list[PoseRecord]
    frame_size_wh: tuple[int, int]
    repeated: bool = False
    missing: bool = False
    schema_version: int = SCHEMA_VERSION


@dataclass
class FeatureCache:
    features: np.ndarray
    records: list[FeatureRecord]
    metadata: dict = field(default_factory=dict)
