"""Append a lightweight review page, plot and independent audit to the new inventory."""

import base64
import bisect
import copy
import json
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import validation_alert_quality as m
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


def independent_audit():
    summary = m.csv_read(m.OUT / "validation-23.csv")
    original = m.csv_read(m.WIDE / "sequences.csv")
    enriched = m.csv_read(m.OUT / "windows-with-quality.csv")
    original_windows = m.csv_read(m.WIDE / "windows.csv")
    original_events = m.csv_read(m.WIDE / "events.csv")
    source_frames = m.csv_read(m.OUT / "source-frame-pose-quality.csv")
    samples = m.csv_read(m.OUT / "causal-sampled-pose-quality.csv")
    reported = m.read(m.OUT / "metrics-rechecked.json")["wide"]
    index = m.read(m.RUN / "dataset-index.json")
    threshold = m.read(m.RUN / "threshold.json")["threshold"]
    counts = Counter((int(r["label"]), r["prediction"]) for r in original)
    matrix = [
        [counts[(0, "0")], counts[(0, "1")], counts[(0, "")]],
        [counts[(1, "0")], counts[(1, "1")], counts[(1, "")]],
    ]
    tn, fp, _un = matrix[0]
    fn, tp, uf = matrix[1]
    assert matrix == reported["outcomes_with_unknown"]
    assert (tp + tn) / (tp + tn + fp + fn) == reported["sequence_accuracy"]
    assert (tp + tn) / len(original) == reported["operational_accuracy"]
    assert 2 * tp / (2 * tp + fp + fn) == reported["sequence_fall_f1"]
    assert tp / (tp + fp) == reported["sequence_fall_precision"]
    assert tp / (tp + fn + uf) == reported["end_to_end_fall_recall"]
    assert len(summary) == 23 and len(original_windows) == len(enriched)
    frame_checks = sample_checks = event_checks = 0
    reconstructed = []
    for source in m.val_sources(index):
        sid = source["sequence_id"]
        row = next(r for r in summary if r["sequence_id"] == sid)
        pose_path = m.local_path(index["sequences"][sid]["pose_path"])
        meta = m.read(pose_path.with_suffix(".json"))
        assert meta["split"] == "validation" and meta["sequence_id"] == sid
        with np.load(pose_path, allow_pickle=False) as arrays:
            points = arrays["keypoints"]
        direct_good, times, ids = {}, [], []
        fr_by_key = {
            (int(r["source_frame_id"]), r["track_id"]): r
            for r in source_frames
            if r["sequence_id"] == sid
        }
        for frame in meta["frames"]:
            times.append(frame["timestamp_ms"])
            ids.append(frame["source_frame_id"])
            good_tracks = []
            for i in frame["pose_indices"]:
                p = meta["poses"][i]
                good = int(np.count_nonzero(points[i, :, 2] >= 0.2)) >= 9
                assert (
                    str(good)
                    == fr_by_key[(frame["source_frame_id"], str(p["track_id"]))]["confidence_good"]
                )
                if good:
                    good_tracks.append(p["track_id"])
                frame_checks += 1
            if not frame["pose_indices"]:
                assert fr_by_key[(frame["source_frame_id"], "")]["has_pose"] == "False"
                frame_checks += 1
            direct_good[frame["source_frame_id"]] = good_tracks
        sample_rows = [r for r in samples if r["sequence_id"] == sid]
        direct_samples = {}
        for r in sample_rows:
            tick = float(r["timestamp_ms"])
            j = bisect.bisect_right(times, tick + 1e-7) - 1
            assert j >= 0 and times[j] <= tick + 1e-7
            missing = meta["frames"][j]["source_missing"] or tick - times[j] > 50 + 1e-7
            source_id = None if missing else ids[j]
            assert r["source_frame_id"] == ("" if source_id is None else str(source_id))
            expected_good = [] if missing else direct_good[source_id]
            assert json.loads(r["good_tracks"]) == expected_good
            direct_samples[int(r["sample_index"])] = (source_id, expected_good)
            sample_checks += 1
        ws = [r for r in original_windows if r["sequence_id"] == sid]
        ew = [r for r in enriched if r["sequence_id"] == sid]
        assert len(ws) == len(ew)
        endpoints = {int(w["sample_index"]) for w in ws}
        legal = {int(w["sample_index"]) for w in ws if w["raw_score"]}
        assert len(endpoints) == int(row["scheduled_unique_endpoints"])
        assert len(legal) == int(row["valid_unique_endpoints"])
        assert len(legal) / len(endpoints) == float(row["window_coverage"])
        for w, e in zip(ws, ew, strict=True):
            for key in w:
                assert w[key] == e[key]
            source_id, good_tracks = direct_samples[int(w["sample_index"])]
            good = (int(w["track_id"]) if w["track_id"] else None) in good_tracks
            assert e["current_same_track_confidence_good"] == str(good)
            assert e["alert_with_current_good_pose"] == str(good and w["state"] == "suspected_fall")
        # Calculate each triple directly, by track, rather than calling the aggregator replay.
        tracks = defaultdict(list)
        for w in ws:
            tracks[w["track_id"]].append(w)
        feature_meta = m.read(
            m.local_path(index["sequences"][sid]["feature_path"]).with_suffix(".json")
        )
        event_scores = []
        for track, track_windows in tracks.items():
            active = None
            for i, w in enumerate(track_windows):
                triple = track_windows[max(0, i - 2) : i + 1]
                ok = (
                    len(triple) == 3
                    and all(t["raw_score"] for t in triple)
                    and all(
                        int(triple[j + 1]["sample_index"]) == int(triple[j]["sample_index"]) + 1
                        for j in range(2)
                    )
                )
                score = min(float(t["raw_score"]) for t in triple) if ok else None
                assert score == m.number(w["event_score"])
                if score is not None:
                    event_scores.append(score)
                flagged = score is not None and score > threshold
                assert (w["state"] == "suspected_fall") == flagged
                if flagged:
                    if active is None:
                        active = {
                            "sequence_id": sid,
                            "track_id": track,
                            "confirmed_start_ms": w["available_at_ms"],
                            "last_active_ms": w["available_at_ms"],
                            "peak_score": float(w["raw_score"]),
                        }
                    else:
                        active["last_active_ms"] = w["available_at_ms"]
                        active["peak_score"] = max(active["peak_score"], float(w["raw_score"]))
                elif active is not None:
                    reason = (
                        w["reason"]
                        if not w["raw_score"]
                        else "time_gap"
                        if int(w["sample_index"]) != int(track_windows[i - 1]["sample_index"]) + 1
                        else "low_score"
                    )
                    reconstructed.append(
                        {**active, "end_ms": float(w["available_at_ms"]), "end_reason": reason}
                    )
                    active = None
                event_checks += 1
            if active is not None:
                reconstructed.append(
                    {**active, "end_ms": feature_meta["end_ms"], "end_reason": "sequence_end"}
                )
        assert (max(event_scores) if event_scores else None) == m.number(
            row["wide_sequence_K3_max"]
        )
    assert len(reconstructed) == len(original_events) == 14
    for computed, original_event in zip(reconstructed, original_events, strict=True):
        for key, value in computed.items():
            if key in {"confirmed_start_ms", "last_active_ms", "peak_score", "end_ms"}:
                assert float(value) == float(original_event[key])
            else:
                assert str(value) == original_event[key]
    return {
        "passed": True,
        "method": "Independent CSV counts, direct NPZ confidence checks, bisect past-only time mapping, direct per-track triples and event spans; no model/features loaded",
        "sequence_rows": 23,
        "window_records": event_checks,
        "event_records": len(reconstructed),
        "source_frame_pose_rows": frame_checks,
        "causal_sample_rows": sample_checks,
        "confusion_matrix_ADL_Fall_by_normal_fall_unknown": matrix,
        "exact_human_time_gt": 0,
        "test_artifacts_loaded": 0,
        "new_inference_training_calibration": 0,
    }


def lightweight_page():
    if (m.OUT / "review").exists():
        raise m.ContractError("Preserve existing lightweight review")
    bundle = m.read(m.OUT / "review-bundle.json")
    light = copy.deepcopy(bundle)
    target = m.OUT / "review"
    target.mkdir()
    (target / "frames").mkdir()
    n = 0
    for clip in light["clips"]:
        for f in clip["frames"]:
            content = base64.b64decode(f["image"].split(",", 1)[1], validate=True)
            assert __import__("hashlib").sha256(content).hexdigest() == f["source_sha256"]
            filename = f"frames/{clip['sequence_id']}-{f['source_frame_id']:03}.png"
            with (target / filename).open("xb") as stream:
                stream.write(content)
            f["image"] = filename
            n += 1
    light["transport"] = (
        "original_png_relative_urls; identity refers to original embedded review-bundle.json"
    )
    template = m.ROOT / "scripts/templates/validation-alert-review.html"
    payload = json.dumps(light, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    with (target / "index.html").open("x", encoding="utf8") as stream:
        stream.write(template.read_text("utf8").replace("__BUNDLE__", payload))
    return {
        "images": n,
        "page_bytes": (target / "index.html").stat().st_size,
        "all_images_original_png_hash_match": True,
        "human_answers_prefilled": False,
    }


def plot():
    destination = m.OUT / "validation-timeline.png"
    if destination.exists():
        raise m.ContractError("Preserve existing plot")
    rows = m.csv_read(m.OUT / "validation-23.csv")
    windows = m.csv_read(m.OUT / "windows-with-quality.csv")
    fig, (ax, coverage) = plt.subplots(
        1, 2, figsize=(15, 11), sharey=True, gridspec_kw={"width_ratios": [5, 2]}
    )
    colors = {
        "unknown": "#b7c1cb",
        "warming_up": "#ecc571",
        "normal": "#508aa6",
        "suspected_fall": "#cc4f51",
    }
    for i, r in enumerate(rows):
        group = defaultdict(list)
        for w in windows:
            if w["sequence_id"] == r["sequence_id"]:
                group[w["sample_index"]].append(w)
        for ws in group.values():
            best = (
                next((w for w in ws if w["state"] == "suspected_fall"), None)
                or next((w for w in ws if w["state"] == "normal"), None)
                or next((w for w in ws if w["state"] == "warming_up"), None)
                or ws[0]
            )
            time = float(best["available_at_ms"]) / 1000
            ax.add_patch(
                Rectangle((time, i - 0.22), 1 / 30, 0.44, color=colors[best["state"]], linewidth=0)
            )
            if (
                best["state"] == "suspected_fall"
                and best["current_same_track_confidence_good"] == "False"
            ):
                ax.plot(time, i, "x", color="#1c2230", markersize=6)
        phase = json.loads(r["AI_possible_phase_ms"]) if r["AI_possible_phase_ms"] else None
        if phase:
            ax.plot([t / 1000 for t in phase], [i - 0.32, i - 0.32], color="#998d71", linewidth=2)
        c = float(r["window_coverage"])
        coverage.barh(i, 1, color="#e1e6ea", height=0.5)
        coverage.barh(i, c, color="#508aa6", height=0.5)
        coverage.text(
            1.01,
            i,
            f"{r['valid_unique_endpoints']}/{r['scheduled_unique_endpoints']}",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(range(len(rows)), [f"{r['sequence_id']}  {r['wide_outcome']}" for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, 12)
    ax.set_xlabel("Available-at time (seconds); grid endpoints, not rendered video pixels")
    ax.set_title("Existing wide30 Validation: alerts and invalid endpoints")
    ax.grid(axis="x", alpha=0.15)
    coverage.set_xlim(0, 1.48)
    coverage.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    coverage.set_title("Valid unique endpoints / scheduled")
    for axis in (ax, coverage):
        axis.spines[["top", "right"]].set_visible(False)
    legend = [Patch(color=color, label=state) for state, color in colors.items()]
    legend += [
        Line2D([0], [0], color="#998d71", label="AI possible phase (NOT human GT)"),
        Line2D(
            [0],
            [0],
            color="#1c2230",
            marker="x",
            linestyle="",
            label="Alert tick: current pose fails confidence gate",
        ),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def report():
    path = m.OUT / "validation-records.md"
    if path.exists():
        raise m.ContractError("Preserve existing report")
    rows = m.csv_read(m.OUT / "validation-23.csv")
    events = m.csv_read(m.OUT / "events-with-quality.csv")

    def secs(x):
        return f"{float(x) / 1000:.3f}" if x else "—"

    def phase(x):
        return "–".join(f"{t / 1000:.3f}" for t in json.loads(x)) if x else "—"

    relation = {
        "not_applicable_ADL": "ADL不適用",
        "no_event": "無事件",
        "inside_AI_core": "AI core內",
        "inside_AI_uncertainty_margin": "AI不確定邊帶",
        "after_AI_possible": "AI possible後",
    }
    lines = [
        "# 全23支Validation：既有警示與品質紀錄",
        "",
        "主版本ur-isotropic-wide-20261012-001，30幀isotropic，固定門檻0.5341113805770874、strict >、K3。另列demo ur-sample-human-s1-001既有結果；沒有新模型、推論、訓練、校準或Test讀取。分數不是機率。",
        "",
        "影片：TP14／TN8／FP0／FN1／unknown0；可判定Accuracy22/23、含unknown整體Accuracy22/23＝95.65%；Precision14/14＝1；二元及整體Fall Recall14/15＝93.33%；二元F1與unknown Fall算漏報的F1均28/29＝0.965517。矩陣[ADL,Fall]×[normal,fall,unknown]＝[[8,0,0],[1,14,0]]。",
        "",
        "窗口：唯一合法端點2287/2710＝84.39%；所有track窗口紀錄2917筆，不能拿2917當唯一端點分母。14事件／14 TP，ADL零事件。完整窗口unknown理由與原幀品質見CSV。demo既有20/23／F1 .888889；本輪不换demo。",
        "",
        "good＝至少9/13關節confidence≥0.2，是工程品質門檻，並非人工確認骨架正確。active端點是K3狀態；當前good／同track因果原幀是獨立展示門檻診斷，不等同已驗證RGB實際可見警示。AI phase沒有完整人工時間真值。",
        "",
        "|影片|真值|wide影片結果|首次事件秒|合法唯一端點|有good骨架原幀|無骨架原幀|demo預測|首次事件對AI phase|",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            f"|{r['sequence_id']}|{'Fall' if r['label'] == '1' else 'ADL'}|{r['wide_outcome']}|{secs(r['first_alarm_ms'])}|{r['valid_unique_endpoints']}/{r['scheduled_unique_endpoints']}|{r['frames_any_confidence_good']}/{r['source_frames']}|{r['frames_no_pose']}|{'Fall' if r['demo_prediction'] == '1' else 'ADL' if r['demo_prediction'] == '0' else 'unknown'}|{relation[r['first_alarm_AI_relation']]}|"
        )
    lines += [
        "",
        "## 跌倒影片的警示時機（AI取樣診斷，人工精確時間全部尚未確認）",
        "",
        "13/15影片事件開始位於AI possible（14 TP中13/14）；6/15位於AI core（14 TP中6/14）；11/15在AI possible有當前good的active端點（14 TP中11/14）。7片首次事件落在寬鬆possible與core之間的不確定邊帶。這些比率不是教師成績，也不是人工時間命中率。fall-14首次1.533秒，超出AI possible末端1.468秒約65ms，仍需人工確認。",
        "",
        "|影片|AI possible秒|AI core秒|事件開始落possible／core|good active / 全active端點|possible內good active數|人工精確時間|",
        "|---|---|---|---|---:|---:|---|",
    ]
    for r in rows:
        if r["label"] == "1":
            lines.append(
                f"|{r['sequence_id']}|{phase(r['AI_possible_phase_ms'])}|{phase(r['AI_core_phase_ms'])}|{r['AI_possible_event_starts']}/{r['AI_core_event_starts']}|{r['current_good_active_endpoints']}/{r['active_window_endpoints']}|{r['AI_possible_good_active_endpoints']}|未標精確開始／落地|"
            )
    lines += [
        "",
        "## 14個事件的保存與顯示門檻",
        "",
        "last active不等於end：後者是下一個低分／低信心窗口關閉時刻，或影片終止時刻；不能把關閉時間當作持續警示直到落地。",
        "",
        "|影片|track|開始秒|最後active秒|關閉秒|關閉原因|good active / 全active|",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for e in events:
        lines.append(
            f"|{e['sequence_id']}|{e['track_id']}|{secs(e['confirmed_start_ms'])}|{secs(e['last_active_ms'])}|{secs(e['end_ms'])}|{e['end_reason']}|{e['current_good_active_endpoint_count']}/{e['active_endpoint_count']}|"
        )
    lines += [
        "",
        "## 原六片coarse guard：維持原定义",
        "",
        "4/6有phase內事件、2/6有phase內good active；零原定粗early事件。此與15片AI取樣phase不同，不能用13/15替代原驗收。",
        "",
        "|影片|原coarse phase秒|phase內事件數|phase內good active端點|",
        "|---|---|---:|---:|",
    ]
    for r in rows:
        if r["legacy6_phase_ms"]:
            lines.append(
                f"|{r['sequence_id']}|{phase(r['legacy6_phase_ms'])}|{r['legacy6_phase_event_starts']}|{r['legacy6_good_active_endpoints']}|"
            )
    lines += [
        "",
        "## 三段人工確認材料",
        "",
        "[開啟三段覆核頁](review/index.html)。每段只填開始最早最晚、落地最早最晚（不可確認／片段外均可），一個整段品質，最後本人確認並一次匯出JSON。不是逐幀218次標註；218張原PNG只用來播放／精確選幀。圖像hash與原來源一致，切換骨架只畫既有觀測，不修正任何點。頁面不顯示分數／AI答案。",
        "",
        "- fall-18：原幀1–65／0–2.135秒。TP首次1.433秒，唯一active端點當前good=false，窗口合法16/36、good原幀40/65。須辨認開始／落地及錯點、遮擋／出框，不能把合法窗口或影片TP當可見警示成功。",
        "- fall-09：原幀103–185／3.403–6.139秒。TP首次5.900秒，在原coarse 4.604–5.506之外，卻仍在較寬AI possible 3.904–6.139內；位於AI settling uncertainty 5.572–6.139。不能直接斷言落地後多久，需人工界限釐清。",
        "- fall-30：原幀1–70／0–2.302秒。TP只有1.866667秒一個active端點、當前good=false；距AI possible末端1.868僅1.333ms，來源整数ms與30Hz格點又不同。不能把這種邊界命中當準確時機。",
        "",
        "fall-19沿用已完成的本人phase：upright2/4合法、transition21/29、ground0/38；仍有跌倒過程連續合法窗口，但raw max .18390694與K3 max .13463649均未達.534111門檻，零事件／FN。人工作業的姿態phase不是精確開始／落地時刻，已完成的joint控制也沒解決漏報；本輪不重做100幀，不進行調參。",
        "",
        "## 本輪驗證與限制",
        "",
        "獨立核正式CSV矩陣／分母、2917窗K3與14事件、直接原Pose NPZ confidence及bisect因果原幀映射；synthetic18passed。既有真實輸出重核不冒充新real_feature_smoke或real_end_to_end，這兩項本輪未執行。原模型、快取、舊48個finalized artifact及修改前進度／驗收bytes保存。新測試首次lint只有import空行，format修復、原測試及初版source manifest封存；不是模型／結果修復。",
        "",
        "僅選三個診斷疑點，非盲測或全體代表樣本；其他影片的時間／骨架幾何正确仍缺人工GT。confidence、座標是否出框、AI sampled visibility都不是人體完整可見性真值。新報告不提高Accuracy、不改採用／demo／門檻／K3，不跑Test／正式Notebook／五分鐘影片，不公開上傳。",
        "",
        "完整檔案：validation-23.csv、source-frame-pose-quality.csv、causal-sampled-pose-quality.csv、windows-with-quality.csv、events-with-quality.csv、metrics-rechecked.json、fall19-existing-human-phase.json、review-selection.json、validation-timeline.png。大型review.html與review-bundle.json保留自含原PNG備份；平常用輕量review/index.html。",
    ]
    with path.open("x", encoding="utf8") as stream:
        stream.write("\n".join(lines) + "\n")


def main():
    if (m.QA / "independent-audit.json").exists():
        raise m.ContractError("Preserve the completed appendix audit")
    m.verify()
    audit = independent_audit()
    transport = lightweight_page()
    plot()
    report()
    m.save(m.QA / "independent-audit.json", {**audit, "review_transport": transport})
    print(json.dumps({"completed": True, "audit": audit, "review": transport}, ensure_ascii=False))


if __name__ == "__main__":
    main()
