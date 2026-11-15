"""Build/serve one four-clip human review page using accepted Validation demos only."""

import argparse
import csv
import hashlib
import json
import shutil
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.visualization.media_server import RangeRequestHandler

ROOT = Path(__file__).resolve().parents[1]
TAG = "ur-shoulder-display-review-20261112-001"
OUT = ROOT / "outputs" / TAG
QA = ROOT / "docs/validation" / TAG
DEMOS = ROOT / "outputs/ur-shoulder-demo-20261112-001"
TEMPLATE = ROOT / "scripts/templates/shoulder-display-review.html"
CASES = {"fall-19": [1.1, 2.1], "fall-09": [4.2, 5.1], "fall-18": [1.0, 1.8], "fall-30": [1.2, 2.1]}
CHOICES = {
    "timing_relation": {"during_fall", "before_fall", "after_fall", "uncertain"},
    "subject_pose": {
        "same_person_plausible",
        "same_person_misaligned",
        "wrong_person",
        "uncertain",
    },
    "display_clarity": {"clear", "slow_only", "unclear", "uncertain"},
}


def require(ok, message):
    if not ok:
        raise ContractError(message)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_review(annotation, bundle):
    require(isinstance(annotation, dict), "Review must be an object")
    require(
        annotation.get("schema_version") == 1
        and annotation.get("bundle_sha256") == bundle["bundle_sha256"],
        "Review bundle/schema mismatch",
    )
    require(
        annotation.get("review_scope") == "shoulder_demo_display_only"
        and annotation.get("annotation_origin") == "human_self_reported"
        and annotation.get("human_attested") is True
        and annotation.get("new_temporal_ground_truth") is False
        and not annotation.get("draft"),
        "Completed human self-report is required",
    )
    require(
        isinstance(annotation.get("reviewer"), str) and annotation["reviewer"].strip(),
        "Reviewer alias required",
    )
    actual = annotation.get("clips", [])
    require(
        isinstance(actual, list)
        and len(actual) == 4
        and all(isinstance(r, dict) for r in actual)
        and {r.get("sequence_id") for r in actual} == set(CASES),
        "Exactly four different selected clips required",
    )
    expected = {r["sequence_id"]: r for r in bundle["clips"]}
    inherited = annotation.get("inherited_annotations")
    require(
        isinstance(inherited, list)
        and len(inherited) == 4
        and all(isinstance(r, dict) for r in inherited)
        and {r.get("sequence_id") for r in inherited} == set(CASES),
        "Four inherited references required",
    )
    for reference in inherited:
        clip = expected[reference["sequence_id"]]
        require(
            reference.get("annotation") == clip["inherited_annotation"]
            and reference.get("reference_sha256") == clip["reference_sha256"],
            "Inherited annotation changed",
        )
    for row in actual:
        clip = expected[row["sequence_id"]]
        require(row.get("reviewed") is True, "Each clip must be reviewed")
        for field, choices in CHOICES.items():
            require(row.get(field) in choices, f"Explicit {field} judgment required")
        require(
            row.get("video_sha256") == clip["video_sha256"]
            and row.get("reference_sha256") == clip["reference_sha256"],
            "Review source/reference changed",
        )
        require(
            isinstance(row.get("notes"), str) and len(row["notes"]) <= 2000,
            "Notes must be text, at most 2000 characters",
        )
    return {
        "valid": True,
        "clips": 4,
        "scope": "display review; existing annotations retained",
        "changes_model_or_metrics": False,
        "new_temporal_ground_truth": False,
    }


def check_hashes(files):
    for name, digest in files.items():
        require(sha256(ROOT / name) == digest, f"Protected source changed: {name}")


def build():
    require(
        not OUT.exists() and not QA.exists(), "Review output or QA already exists; use verify/serve"
    )
    acceptance = read_json(DEMOS / "validation-acceptance.json")
    require(
        acceptance["passed"] and len(acceptance["sequences"]) == 23,
        "Completed RGB acceptance required",
    )
    accepted = {r["sequence_id"]: r for r in acceptance["sequences"]}
    human_path = ROOT / "outputs/ur-validation-human-timing-20261105-001/comparisons.json"
    humans = {r["sequence_id"]: r for r in read_json(human_path)}
    phase_path = ROOT / "outputs/ur-fall19-human-amendment-20261013-001/human-frame-diagnosis.csv"
    previous_lock = read_json(ROOT / "outputs/ur-fall19-shoulder-descent-20261107-001/lock.json")
    for path in (human_path, phase_path):
        require(
            sha256(path) == previous_lock["source_sha256"][path.relative_to(ROOT).as_posix()],
            "Previously verified human annotations changed",
        )
    phases = rows(phase_path)
    protected = {p for p in DEMOS.rglob("*") if p.is_file()}
    protected.update(
        (
            human_path,
            phase_path,
            TEMPLATE,
            Path(__file__).resolve(),
            ROOT / "src/fall_detect/visualization/media_server.py",
        )
    )
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(protected)}
    clips = []
    for sid, span in CASES.items():
        result = accepted[sid]
        folder = DEMOS / ("pilot" if sid == "fall-19" else "validation")
        video = DEMOS / "gallery" / f"{sid}.mp4"
        expected_video = read_json(DEMOS / "gallery/verification.json")["clips"][sid]["h264_sha256"]
        require(sha256(video) == expected_video, "Accepted H264 changed")
        timeline = rows(folder / f"{sid}.csv")
        event_rows = rows(folder / f"{sid}-events.csv")
        first = float(result["first_alarm_ms"])
        at_alarm = next(
            r
            for r in timeline
            if float(r["timestamp_ms"]) == first and r["state"] == "suspected_fall"
        )
        if sid == "fall-19":
            first_frame = int(at_alarm["source_frame_id"])
            label = next(r for r in phases if int(r["source_frame_id"]) == first_frame)
            ranges = []
            for phase in dict.fromkeys(r["phase"] for r in phases):
                group = [r for r in phases if r["phase"] == phase]
                ranges.append(
                    {
                        "phase": phase,
                        "frame_ids": [
                            int(group[0]["source_frame_id"]),
                            int(group[-1]["source_frame_id"]),
                        ],
                        "time_ms": [
                            float(group[0]["timestamp_ms"]),
                            float(group[-1]["timestamp_ms"]),
                        ],
                    }
                )
            inherited = {
                "kind": "human_pose_phases_not_instantaneous_bounds",
                "phase_ranges": ranges,
                "first_alert_source_frame": first_frame,
                "phase_at_first_alert": label["phase"],
                "pose_at_first_alert": label["pose_check"],
                "note": "整段姿態階段；原骨架錯位標註保留，不改稱已修好",
                "source_path": str(phase_path.relative_to(ROOT)),
                "source_sha256": sha256(phase_path),
            }
        else:
            h = humans[sid]
            inherited = {
                "kind": "human_instantaneous_uncertainty_bounds",
                "onset_bounds_ms": h["onset_bounds_ms"],
                "impact_bounds_ms": h["impact_bounds_ms"],
                "skeleton_quality": h["human_skeleton_quality"],
                "notes": h["human_notes"],
                "note": "開始／落地的最早至最晚可能時刻，不是整段姿態階段",
                "source_path": str(human_path.relative_to(ROOT)),
                "source_sha256": sha256(human_path),
            }
        clips.append(
            {
                "sequence_id": sid,
                "split": "validation",
                "video": f"media/{sid}.mp4",
                "video_sha256": expected_video,
                "fps": 30,
                "focus_seconds": span,
                "first_alarm_ms": first,
                "current_alert_frames": result["current_alert_frames"],
                "events": [
                    {
                        k: (r[k] if k == "end_reason" else float(r[k]))
                        for k in ("confirmed_start_ms", "last_active_ms", "end_ms", "end_reason")
                    }
                    for r in event_rows
                ],
                "timeline": [
                    {
                        "sample_index": int(r["sample_index"]),
                        "timestamp_ms": float(r["timestamp_ms"]),
                        "source_frame_id": int(r["source_frame_id"])
                        if r["source_frame_id"]
                        else None,
                        "state": r["state"],
                    }
                    for r in timeline
                ],
                "inherited_annotation": inherited,
                "reference_sha256": fingerprint(inherited),
            }
        )
    bundle = {
        "schema_version": 1,
        "id": TAG,
        "scope": "Validation display review only",
        "baseline": "ur-isotropic-wide-20261012-001",
        "shoulder_candidate": "ur-fall19-shoulder-descent-20261107-001",
        "new_temporal_ground_truth": False,
        "clips": clips,
    }
    bundle["bundle_sha256"] = fingerprint(bundle)
    OUT.mkdir(parents=True)
    QA.mkdir(parents=True)
    (OUT / "media").mkdir()
    for sid in CASES:
        shutil.copyfile(DEMOS / "gallery" / f"{sid}.mp4", OUT / "media" / f"{sid}.mp4")
    data = json.dumps(bundle, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    page = TEMPLATE.read_text(encoding="utf-8").replace("__BUNDLE__", data)
    (OUT / "index.html").write_text(page, encoding="utf-8")
    write_json(OUT / "review-bundle.json", bundle)
    write_json(
        OUT / "sources.json",
        {"files": hashes, "training": False, "inference": False, "test": False},
    )
    for name in ("docs/current-progress.md", "docs/acceptance-results.md"):
        destination = QA / "prior-source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    write_json(
        QA / "prior-documents.json",
        {
            name: sha256(ROOT / name)
            for name in ("docs/current-progress.md", "docs/acceptance-results.md")
        },
    )
    write_json(
        OUT / "artifact-manifest.json",
        {
            "files": {
                str(p.relative_to(OUT)): sha256(p) for p in sorted(OUT.rglob("*")) if p.is_file()
            }
        },
    )
    verify()


def verify():
    sources = read_json(OUT / "sources.json")
    check_hashes(sources["files"])
    bundle = read_json(OUT / "review-bundle.json")
    body = {k: v for k, v in bundle.items() if k != "bundle_sha256"}
    require(fingerprint(body) == bundle["bundle_sha256"], "Bundle hash differs")
    require([c["sequence_id"] for c in bundle["clips"]] == list(CASES), "Selected clips differ")
    require(
        all(
            c["split"] == "validation"
            and fingerprint(c["inherited_annotation"]) == c["reference_sha256"]
            and sha256(OUT / c["video"]) == c["video_sha256"]
            for c in bundle["clips"]
        ),
        "Clip/reference differs",
    )
    for name, digest in read_json(OUT / "artifact-manifest.json")["files"].items():
        require(sha256(OUT / name) == digest, f"Review artifact changed: {name}")
    print(
        json.dumps(
            {
                "verified": True,
                "clips": 4,
                "source_hashes": len(sources["files"]),
                "human_review": "pending; no human choices prefilled",
                "no_model_execution": True,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "verify", "serve", "validate-review"])
    parser.add_argument("--port", type=int, default=8850)
    parser.add_argument("--annotation", type=Path)
    args = parser.parse_args()
    if args.stage == "build":
        build()
    elif args.stage == "verify":
        verify()
    elif args.stage == "validate-review":
        require(args.annotation is not None, "--annotation required")
        verify()
        print(
            json.dumps(
                validate_review(read_json(args.annotation), read_json(OUT / "review-bundle.json")),
                ensure_ascii=False,
            )
        )
    else:
        verify()
        handler = partial(RangeRequestHandler, directory=str(OUT))
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            print(f"http://127.0.0.1:{args.port}/index.html", flush=True)
            server.serve_forever()


if __name__ == "__main__":
    main()
