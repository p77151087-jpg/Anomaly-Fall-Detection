"""Frozen real RGB/pose panels for AI visual diagnosis; no new human pose truth."""

import argparse

import cv2
from PIL import Image, ImageDraw

from fall_detect.config import project_path
from fall_detect.data.cache import load_pose_cache
from fall_detect.io_utils import ContractError, read_json, sha256, write_json

ROOT = project_path(".")
E = ROOT / "docs/validation/ur-pose-resolution-20261013-001"
PILOT = ROOT / "data/cache/ur-pose-resolution-20261013-001/pilot-index.json"
BASE = ROOT / "runs/ur-isotropic-wide-20261012-001/dataset-index.json"
IDS = [33, 43, 44, 47, 51, 54, 62, 90]
EDGES = [
    (1, 2),
    (1, 3),
    (3, 5),
    (2, 4),
    (4, 6),
    (1, 7),
    (2, 8),
    (7, 8),
    (7, 9),
    (9, 11),
    (8, 10),
    (10, 12),
]


def render():
    target = E / "pilot-pose-panels"
    if target.exists():
        raise ContractError("Preserve existing paired visual evidence")
    target.mkdir()
    observation_maps = []
    for path in [BASE, PILOT]:
        index = read_json(path)
        obs, _ = load_pose_cache(project_path(index["sequences"]["fall-19"]["pose_path"]), {})
        observation_maps.append({f.source_frame_id: (f, p) for f, p in obs})
    sheet = Image.new("RGB", (768, 4 * 340), "white")
    inputs, panels = {}, []
    for n, fid in enumerate(IDS):
        raw = ROOT / "data/raw/urfall/fall-19/fall-19-cam0-rgb" / f"fall-19-cam0-rgb-{fid:03d}.png"
        inputs[raw.relative_to(ROOT).as_posix()] = sha256(raw)
        image = cv2.imread(str(raw))
        if image is None:
            raise ContractError("Real RGB decode required")
        pair = Image.new("RGB", (768, 340), "white")
        draw = ImageDraw.Draw(pair)
        for column, poses in enumerate(observation_maps):
            frame, records = poses[fid]
            rendered = image.copy()
            for p in records:
                k = p.keypoints
                for a, b in EDGES:
                    if min(k[a, 2], k[b, 2]) >= 0.2:
                        cv2.line(
                            rendered,
                            tuple(k[a, :2].astype(int)),
                            tuple(k[b, :2].astype(int)),
                            (0, 255, 0),
                            2,
                        )
                for j, point in enumerate(k):
                    color = (0, 255, 0) if point[2] >= 0.2 else (0, 0, 255)
                    xy = tuple(point[:2].astype(int))
                    cv2.circle(rendered, xy, 3, color, -1)
                    cv2.putText(rendered, str(j), xy, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            tile = Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)).resize((384, 288))
            pair.paste(tile, (column * 384, 45))
            draw.text(
                (column * 384 + 8, 6),
                f"f{fid} / {frame.timestamp_ms:.0f}ms / {'old256x192' if column == 0 else 'new512x384'}",
                fill="black",
            )
            draw.text(
                (column * 384 + 8, 23),
                "green confidence>=.2; red low; no human new-pose GT",
                fill="black",
            )
        path = target / f"frame-{fid:03d}.png"
        pair.save(path)
        panels.append(path.relative_to(ROOT).as_posix())
        if n < 4:
            sheet.paste(pair, (0, n * 340))
    sheet.save(target / "transition-pairs.png")
    write_json(
        target / "manifest.json",
        {
            "source_sha256": inputs,
            "panels": panels,
            "human_new_pose_confirmed": False,
            "origin": "AI visual diagnosis only",
        },
    )
    print(target)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    render()
