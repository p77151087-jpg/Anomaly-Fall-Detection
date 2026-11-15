"""Independent Decimal/CSV audit of human timing; no feature or model execution."""

import argparse
import bisect
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "ur-validation-human-timing-20261105-001"
OUT = ROOT / "outputs" / TAG
QA = ROOT / "docs/validation" / TAG
OLD = ROOT / "outputs/ur-validation-alert-quality-20261013-001"
WIDE = ROOT / "outputs/isotropic-refinement-20261012/ur-isotropic-wide-20261012-001"
SELECTED = ("fall-18", "fall-09", "fall-30")


def load(path):
    return json.loads(path.read_text("utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def equal_numeric(actual, expected):
    assert math.isclose(float(actual), float(expected), abs_tol=1e-9, rel_tol=0), (actual, expected)


def independent_audit():
    annotation = load(OUT / "human-review-original.json")
    inputs = load(QA / "inputs.json")
    assert (
        annotation["annotation_origin"] == "human_self_reported"
        and annotation["human_attested"] is True
    )
    assert len(annotation["clips"]) == 3 and {r["sequence_id"] for r in annotation["clips"]} == set(
        SELECTED
    )
    assert digest(OUT / "human-review-original.json") == inputs["annotation_sha256"]
    assert digest(Path(inputs["annotation_external_path"])) == inputs["annotation_sha256"]
    for name, expected in inputs["files"].items():
        assert digest(ROOT / name) == expected, name
    preserved = load(QA / "preservation.json")["files"]
    for name, expected in preserved.items():
        normalized = name.replace("\\", "/")
        archived = QA / "prior-contracts" / normalized
        original = (
            archived
            if normalized in {"docs/current-progress.md", "docs/acceptance-results.md"}
            else ROOT / normalized
        )
        assert digest(original) == expected, name
    formal, overview = (
        rows(WIDE / "sequences.csv"),
        rows(OUT / "validation-23-human-supplement.csv"),
    )
    base_overview = {r["sequence_id"]: r for r in rows(OLD / "validation-23.csv")}
    assert len(formal) == len(overview) == len(base_overview) == 23
    assert {r["sequence_id"] for r in formal} == set(base_overview)
    counts = Counter((int(r["label"]), r["prediction"]) for r in formal)
    matrix = [
        [counts[0, "0"], counts[0, "1"], counts[0, ""]],
        [counts[1, "0"], counts[1, "1"], counts[1, ""]],
    ]
    tn, fp, un = matrix[0]
    fn, tp, uf = matrix[1]
    summary = load(OUT / "summary.json")
    metrics = summary["formal_validation_metrics_unchanged"]
    assert matrix == [[8, 0, 0], [1, 14, 0]] == metrics["outcomes_with_unknown"]
    assert metrics["operational_accuracy"] == (tp + tn) / 23
    assert metrics["sequence_accuracy"] == (tp + tn) / (tp + tn + fp + fn)
    assert metrics["sequence_fall_f1"] == 2 * tp / (2 * tp + fp + fn)
    assert metrics["sequence_fall_precision"] == tp / (tp + fp)
    assert metrics["end_to_end_fall_recall"] == tp / (tp + fn + uf)
    assert metrics["n_unknown"] == un + uf
    for r in overview:
        for key, value in base_overview[r["sequence_id"]].items():
            assert r[key] == value, (r["sequence_id"], key)
        assert (r["human_onset_impact_review"] == "self_reported_validated") == (
            r["sequence_id"] in SELECTED
        )
    originals = rows(WIDE / "windows.csv")
    enriched = {
        (r["sequence_id"], r["sample_index"], r["track_id"]): r
        for r in rows(OLD / "windows-with-quality.csv")
    }
    all_events = rows(WIDE / "events.csv")
    active = rows(OUT / "active-endpoints.csv")
    selected_saved = rows(OUT / "selected-windows-with-human-time.csv")
    comparisons = load(OUT / "comparisons.json")
    assert len(comparisons) == 3 and len(active) == 10 and len(selected_saved) == 233
    independent_rows, evaluated = [], []
    threshold = Decimal("0.5341113805770874")
    for reviewed in annotation["clips"]:
        sid = reviewed["sequence_id"]
        assert reviewed["reviewed"] is True
        name = next(
            name
            for name in inputs["files"]
            if "poses" in name and f"{sid}-" in Path(name).name and name.endswith(".json")
        )
        meta = load(ROOT / name)
        assert meta["split"] == "validation" and meta["sequence_id"] == sid
        source = {f["source_frame_id"]: Decimal(str(f["timestamp_ms"])) for f in meta["frames"]}
        onset = [source[i] for i in reviewed["onset"]["bounds_frame_ids"]]
        impact = [source[i] for i in reviewed["impact"]["bounds_frame_ids"]]
        assert onset[0] <= onset[1] < impact[0] <= impact[1]
        c = next(r for r in comparisons if r["sequence_id"] == sid)
        assert list(map(Decimal, map(str, c["onset_bounds_ms"]))) == onset
        assert list(map(Decimal, map(str, c["impact_bounds_ms"]))) == impact
        ws = [w for w in originals if w["sequence_id"] == sid]
        tracks = defaultdict(list)
        for w in ws:
            tracks[w["track_id"]].append(w)
        expected_active = []
        for tw in tracks.values():
            for i, w in enumerate(tw):
                triple = tw[max(0, i - 2) : i + 1]
                legal = (
                    len(triple) == 3
                    and all(t["raw_score"] for t in triple)
                    and all(
                        int(triple[j + 1]["sample_index"]) == int(triple[j]["sample_index"]) + 1
                        for j in range(2)
                    )
                )
                score = min(Decimal(t["raw_score"]) for t in triple) if legal else None
                state = (
                    "unknown"
                    if not w["raw_score"]
                    else "warming_up"
                    if score is None
                    else "suspected_fall"
                    if score > threshold
                    else "normal"
                )
                assert w["state"] == state
                assert (
                    w["event_score"] == "" if score is None else Decimal(w["event_score"]) == score
                )
                if state == "suspected_fall":
                    expected_active.append(w)
                evaluated.append(w)
        expected_active.sort(key=lambda w: (Decimal(w["available_at_ms"]), w["track_id"]))
        new_active = [w for w in active if w["sequence_id"] == sid]
        assert len(expected_active) == len(new_active) == c["active_endpoint_count"]
        good_count = in_possible = in_core = after_impact = good_possible = 0
        for old, new in zip(expected_active, new_active, strict=True):
            t = Decimal(old["available_at_ms"])
            assert Decimal(new["available_at_ms"]) == t
            assert new["track_id"] == old["track_id"] and new["sample_index"] == old["sample_index"]
            possible, core = onset[0] <= t <= impact[1], onset[1] <= t <= impact[0]
            category = (
                "before_onset"
                if t < onset[0]
                else "after_impact"
                if t > impact[1]
                else "onset_uncertainty"
                if t < onset[1]
                else "impact_uncertainty"
                if t >= impact[0]
                else "guaranteed_fall_interval"
            )
            assert new["relation"] == category
            assert new["inside_possible_fall_envelope"] == str(possible)
            assert new["inside_guaranteed_fall_interval"] == str(core)
            assert new["definitely_after_impact"] == str(t > impact[1])
            good = (
                enriched[(sid, old["sample_index"], old["track_id"])][
                    "current_same_track_confidence_good"
                ]
                == "True"
            )
            assert new["current_same_track_confidence_good"] == str(good)
            times = [source[f["source_frame_id"]] for f in meta["frames"]]
            nearest = bisect.bisect_right(times, t) - 1
            fid = meta["frames"][nearest]["source_frame_id"]
            assert int(new["causal_source_frame_id"]) == fid
            assert Decimal(new["causal_source_timestamp_ms"]) == source[fid] <= t
            good_count += good
            in_possible += possible
            in_core += core
            after_impact += t > impact[1]
            good_possible += good and possible
        assert (good_count, in_possible, in_core, after_impact, good_possible) == (
            c["good_active_endpoint_count"],
            c["possible_fall_active_endpoints"],
            c["guaranteed_fall_active_endpoints"],
            c["after_latest_impact_active_endpoints"],
            c["possible_fall_good_active_endpoints"],
        )
        event = next(e for e in all_events if e["sequence_id"] == sid)
        first = Decimal(event["confirmed_start_ms"])
        assert first == Decimal(expected_active[0]["available_at_ms"])
        assert Decimal(event["last_active_ms"]) == Decimal(expected_active[-1]["available_at_ms"])
        equal_numeric(c["first_alarm_ms"], first)
        equal_numeric(c["event_end_ms"], Decimal(event["end_ms"]))
        for actual, expected in zip(
            c["delay_from_onset_bounds_ms"], [first - onset[1], first - onset[0]], strict=True
        ):
            equal_numeric(actual, expected)
        for actual, expected in zip(
            c["delay_from_impact_bounds_ms"], [first - impact[1], first - impact[0]], strict=True
        ):
            equal_numeric(actual, expected)
        assert c["relation"] == "impact_uncertainty" and impact[0] <= first <= impact[1]
        assert (
            c["inside_possible_fall_envelope"] is True
            and c["inside_guaranteed_fall_interval"] is False
        )
        independent_rows.append(
            {
                "sequence_id": sid,
                "onset_ms": list(map(float, onset)),
                "impact_ms": list(map(float, impact)),
                "first_alarm_ms": float(first),
                "first_alarm_relation": "impact_uncertainty",
                "active_endpoint_counts": len(expected_active),
                "active_after_latest_impact": after_impact,
            }
        )
    assert len(evaluated) == 233
    assert summary["first_alarm_possible_envelope_hits"] == 3
    assert summary["first_alarm_guaranteed_interval_hits"] == 0
    assert summary["first_alarm_inside_impact_uncertainty"] == 3
    assert summary["selected_after_latest_impact_active_endpoints"] == 5
    assert summary["possible_envelope_good_active_videos"] == 1
    return {
        "passed": True,
        "method": "Independent Decimal timestamp comparisons from original source metadata; original saved-window triples and CSV counts, without the new interval helper",
        "formal_sequence_rows": 23,
        "selected_saved_windows": len(evaluated),
        "selected_events": 3,
        "selected_active_endpoints": len(active),
        "old_hashes_preserved": len(preserved),
        "confusion_matrix": matrix,
        "cases": independent_rows,
        "test_or_model_or_feature_array_access": 0,
    }


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    path = OUT / "human-timing-comparison.png"
    if path.exists():
        raise FileExistsError("Preserve existing plot")
    comparisons, endpoints = load(OUT / "comparisons.json"), rows(OUT / "active-endpoints.csv")
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.3))
    for ax, c in zip(axes, comparisons, strict=True):
        onset, impact = c["onset_bounds_ms"], c["impact_bounds_ms"]
        ax.axvspan(
            onset[0] / 1000, onset[1] / 1000, ymin=0.1, ymax=0.85, color="#b8a4ce", alpha=0.55
        )
        ax.axvspan(
            onset[1] / 1000, impact[0] / 1000, ymin=0.1, ymax=0.85, color="#91c9c1", alpha=0.55
        )
        ax.axvspan(
            impact[0] / 1000, impact[1] / 1000, ymin=0.1, ymax=0.85, color="#eec279", alpha=0.55
        )
        for e in endpoints:
            if e["sequence_id"] == c["sequence_id"]:
                ax.plot(
                    float(e["available_at_ms"]) / 1000,
                    0.42,
                    "o" if e["current_same_track_confidence_good"] == "True" else "x",
                    color="#b83239",
                    markersize=8,
                )
        ax.axvline(c["first_alarm_ms"] / 1000, color="#b83239", linewidth=1.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlim(max(0, onset[0] / 1000 - 0.15), max(impact[1], c["event_end_ms"]) / 1000 + 0.15)
        ax.set_title(
            f"{c['sequence_id']}: video TP | first alarm {c['first_alarm_ms'] / 1000:.3f}s | current-good active endpoints {c['good_active_endpoint_count']}/{c['active_endpoint_count']}",
            fontsize=10,
            loc="left",
        )
        ax.set_xlabel(
            "Original video time (seconds); each marker is an active endpoint", fontsize=9
        )
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.12)
    handles = [
        Patch(color="#b8a4ce", alpha=0.55, label="Human onset uncertainty"),
        Patch(color="#91c9c1", alpha=0.55, label="Guaranteed interval from bounds"),
        Patch(color="#eec279", alpha=0.55, label="Human impact uncertainty"),
        Line2D(
            [0],
            [0],
            color="#b83239",
            marker="o",
            linestyle="",
            label="Active: current pose passes gate",
        ),
        Line2D(
            [0],
            [0],
            color="#b83239",
            marker="x",
            linestyle="",
            label="Active: current pose fails gate",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(
        "Existing wide30 Validation alerts vs self-reported human bounds\nNo exact impact time or new timing improvement claimed",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.89))
    with path.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    result = independent_audit()
    path = QA / "independent-verification.json"
    if path.exists():
        assert load(path) == result, "Independent audit differs from saved evidence"
    else:
        with path.open("x", encoding="utf8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    if args.plot:
        plot()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
