import numpy as np
import pytest
from conftest import observation, sample

from fall_detect.io_utils import ContractError
from fall_detect.models.weights import import_upstream
from fall_detect.skeleton.preprocessing import COCO17_KEEP, coco17_to_human13, human_preprocess
from fall_detect.skeleton.windows import CausalResampler, WindowBuilder


def test_layout_shoulders_and_upstream_numeric_parity(config, skeleton):
    import_upstream(config)
    from pose_utils import normalize_points_with_size, scale_pose

    raw = np.arange(17 * 3).reshape(17, 3)
    np.testing.assert_array_equal(coco17_to_human13(raw), raw[list(COCO17_KEEP)])
    original = skeleton.copy()
    baseline = skeleton.copy()
    baseline[:, :, :2] = normalize_points_with_size(baseline[:, :, :2], 640, 360)
    baseline[:, :, :2] = scale_pose(baseline[:, :, :2])
    baseline = np.concatenate((baseline, ((baseline[:, 1] + baseline[:, 2]) / 2)[:, None]), axis=1)
    points, motion = human_preprocess(skeleton, (640, 360))
    assert points.shape == (1, 3, 30, 14)
    assert motion.shape == (1, 2, 29, 14)
    np.testing.assert_allclose(points, baseline.transpose(2, 0, 1)[None], rtol=0, atol=0)
    np.testing.assert_array_equal(skeleton, original)
    np.testing.assert_allclose(points[0, :, :, 13], (points[0, :, :, 1] + points[0, :, :, 2]) / 2)


@pytest.mark.parametrize("kind", ["zero_span", "nan", "inf"])
def test_degenerate_not_formal_features(skeleton, kind):
    if kind == "zero_span":
        skeleton[:, :, 0] = 7
    else:
        skeleton[10, 1, 0] = np.nan if kind == "nan" else np.inf
    with pytest.raises(ContractError, match="degenerate_pose"):
        human_preprocess(skeleton, (640, 480))


def test_quality_ratio_missing_and_global_endpoints(config, skeleton):
    def windows(missing=False, bad_frames=6):
        builder = WindowBuilder(config)
        result = []
        for i in range(45):
            points = skeleton[i % 30].copy()
            if i < bad_frames:
                points[:, 2] = 0.1
            result.extend(builder.push(sample(i, None if missing and i == 10 else points)))
        return result

    assert [w.sample_index for w in windows()] == [29, 34, 39, 44]
    assert windows()[0].quality == "valid"  # exactly 80% good frames
    assert windows(bad_frames=7)[0].quality == "low_confidence"
    assert windows(missing=True, bad_frames=0)[0].quality == "track_gap"
    empty = WindowBuilder(config)
    result = [w for i in range(45) for w in empty.push(sample(i, None))]
    assert len(result) == 4 and all(w.quality == "no_person" for w in result)


def test_track_and_sequence_boundaries(config, skeleton):
    builder = WindowBuilder(config)
    windows = [
        w for i in range(30) for w in builder.push(sample(i, skeleton[i], track=1 if i < 15 else 2))
    ]
    assert len(windows) == 2 and all(w.quality == "track_gap" for w in windows)
    with pytest.raises(ContractError):
        builder.push(sample(30, skeleton[0], sid="other"))


def test_resampling_no_future_and_gap(skeleton):
    resampler = CausalResampler(30)
    values = []
    for index, timestamp in enumerate([0, 20, 40, 150]):
        frame, poses = observation(index, skeleton[index], timestamp=timestamp)
        values.extend(resampler.push(frame, poses))
    assert [v.timestamp_ms for v in values] == pytest.approx(
        [0, 1000 / 30, 2000 / 30, 100, 4000 / 30]
    )
    assert values[1].source_frame_id == 2  # timestamp 20, never timestamp 40
    assert values[-1].missing and not values[-1].poses
    for value in values:
        assert all(p.timestamp_ms <= value.timestamp_ms + 1e-7 for p in value.poses)


def test_future_changes_do_not_change_earlier_windows(config, skeleton):
    def process(change):
        builder = WindowBuilder(config)
        result = []
        for i in range(60):
            points = skeleton[i % 30].copy()
            if change and i > 40:
                points[:, :2] *= 3
            result.extend(builder.push(sample(i, points)))
        return result

    a, b = process(False), process(True)
    for x, y in zip(a, b):
        if x.sample_index <= 40:
            np.testing.assert_array_equal(x.keypoints, y.keypoints)
