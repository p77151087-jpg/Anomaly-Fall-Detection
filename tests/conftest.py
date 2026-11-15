import numpy as np
import pytest

from fall_detect.config import load_config
from fall_detect.contracts import FrameRecord, PoseRecord, Sample


@pytest.fixture
def config():
    return load_config(device="cpu")


@pytest.fixture
def skeleton():
    time = np.arange(30)[:, None]
    joint = np.arange(13)[None, :]
    x = 100 + joint * 8 + np.sin(time * 0.1 + joint) * 2
    y = 50 + joint * 13 + np.cos(time * 0.15 + joint) * 3
    return np.stack((x, y, np.full((30, 13), 0.9)), axis=2).astype(np.float32)


def observation(index, points, sid="synthetic", track=1, timestamp=None, width=640, height=480):
    timestamp = index * 1000 / 30 if timestamp is None else timestamp
    frame = FrameRecord(sid, index + 1, index, timestamp, None, width, height)
    poses = (
        []
        if points is None
        else [
            PoseRecord(
                sid,
                index + 1,
                index,
                timestamp,
                track,
                np.array([50, 30, 400, 470]),
                points.copy(),
                (width, height),
                0.9,
            )
        ]
    )
    return frame, poses


def sample(index, points, sid="synthetic", track=1):
    frame, poses = observation(index, points, sid, track)
    return Sample(sid, index, frame.timestamp_ms, frame.source_frame_id, poses, (640, 480))
