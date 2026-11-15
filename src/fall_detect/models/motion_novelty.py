"""A normal-only side branch retaining motion removed by per-frame centering."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from ..io_utils import ContractError
from ..training.scaler import FeatureScaler


def motion_trajectory(points):
    """Ordered 30-frame center displacement and log-span, anchored to frame 0."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (30, 13, 3) or not np.isfinite(points).all():
        raise ContractError("Motion requires a finite original [30,13,3] legal window")
    low, high = points[..., :2].min(1), points[..., :2].max(1)
    if np.any(high - low <= 1e-8):
        raise ContractError("Degenerate original motion window")
    center, span = (high + low) / 2, (high - low).max(1)
    result = np.column_stack(((center - center[0]) / span[0], np.log(span / span[0])))
    if not np.isfinite(result).all():
        raise ContractError("Nonfinite motion representation")
    return result.reshape(90)


@dataclass
class MotionNovelty:
    scaler: FeatureScaler
    bank: np.ndarray
    groups: np.ndarray
    neighbors: int = 5

    @classmethod
    def fit(cls, values, groups, labels, neighbors=5):
        values, groups, labels = np.asarray(values), np.asarray(groups), np.asarray(labels)
        if (
            values.ndim != 2
            or values.shape[1] != 90
            or len(values) != len(groups)
            or labels.shape != (len(values),)
            or np.any(labels != 0)
            or type(neighbors) is not int
            or neighbors < 1
            or len(values) < neighbors
            or len(np.unique(groups)) < 2
        ):
            raise ContractError("Fit needs only normal Train windows with video identities")
        if min(np.sum(groups != g) for g in np.unique(groups)) < neighbors:
            raise ContractError("Too few windows after leaving out a complete video")
        scaler = FeatureScaler.fit(values)
        return cls(scaler, scaler.transform(values).astype(np.float64), groups.copy(), neighbors)

    def score(self, values, exclude_groups=None):
        values = self.scaler.transform(values).astype(np.float64)
        if exclude_groups is not None:
            exclude_groups = np.asarray(exclude_groups)
            if len(exclude_groups) != len(values):
                raise ContractError("Exclude-group count mismatch")
        output = []
        for start in range(0, len(values), 128):
            distances = cdist(values[start : start + 128], self.bank, metric="euclidean")
            if exclude_groups is not None:
                distances[exclude_groups[start : start + 128, None] == self.groups[None]] = np.inf
            chosen = np.partition(distances, self.neighbors - 1, axis=1)[:, : self.neighbors]
            output.extend(chosen.mean(1) / np.sqrt(90))
        output = np.asarray(output, dtype=np.float64)
        if not np.isfinite(output).all():
            raise ContractError("No finite motion-neighbor scores")
        return output


def fusion_scores(ae_scores, motion_scores, ae_p95, motion_p95):
    ae, motion = np.asarray(ae_scores), np.asarray(motion_scores)
    if (
        ae.shape != motion.shape
        or ae.ndim != 1
        or not np.isfinite(ae).all()
        or not np.isfinite(motion).all()
        or not np.isfinite([ae_p95, motion_p95]).all()
        or min(ae_p95, motion_p95) <= 0
        or np.any(ae < 0)
        or np.any(motion < 0)
    ):
        raise ContractError("Fusion needs nonnegative finite scores and positive Train scales")
    return np.maximum(ae / ae_p95, motion / motion_p95)
