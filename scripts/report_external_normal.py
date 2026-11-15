"""Render completed external-normal results and sampled AI annotations; no inference."""

import argparse
import csv
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fall_detect.io_utils import ContractError, read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "external-normal-20261012"
DEST = ROOT / "outputs" / f"{PREFIX}-report"
ANNOTATIONS = ROOT / "docs/annotations/ur-train-validation-20261012-v2"


def relative_url(path):
    return quote(os.path.relpath(ROOT / path, DEST).replace("\\", "/"), safe="/")


def build():
    if DEST.exists():
        raise ContractError("Preserve completed report; choose a new version for changes")
    verification_path = ROOT / f"docs/validation/{PREFIX}-verification.json"
    verification = read_json(verification_path)
    manifest = read_json(ANNOTATIONS / "manifest.json")
    phases = read_json(ANNOTATIONS / "fall-intervals.json")["sequences"]
    for name, digest in manifest["files"].items():
        if sha256(ANNOTATIONS / name) != digest:
            raise ContractError("Annotation version changed")
    with (ANNOTATIONS / "sampled-joint-regions.csv").open(encoding="utf-8-sig") as f:
        samples = list(csv.DictReader(f))
    assert len(samples) == manifest["sampled_frames"] == 340
    assert len(phases) == 47 and sum(p["label"] for p in phases) == 15
    assert verification["new_test_evaluations"] == 0
    sequences = []
    for phase in phases:
        sid = phase["sequence_id"]
        rows = [dict(r) for r in samples if r["sequence_id"] == sid]
        for row in rows:
            path = ROOT / row["path"]
            if sha256(path) != row["image_sha256"]:
                raise ContractError("Reviewed raw sample image changed")
            row["image"] = relative_url(row["path"])
        sheet = rows[0]["sheet"]
        if not (ROOT / sheet).is_file():
            raise ContractError("Missing raw contact sheet")
        sequences.append({**phase, "samples": rows, "sheet_url": relative_url(sheet)})

    DEST.mkdir(parents=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    axes[0].bar(["UR only", "UR + external"], [24, 45], color=["#56708f", "#009988"])
    axes[0].set_title("Normal training sources")
    axes[0].set_ylim(0, 54)
    for i, count in enumerate([24, 45]):
        axes[0].text(i, count + 1, str(count), ha="center")
    labels = ["Previous best", "External 128", "External 256"]
    accuracies = [22 / 23 * 100] + [
        c["metrics"]["sequence_accuracy"] * 100 for c in verification["checks"]
    ]
    axes[1].bar(labels, accuracies, color=["#56708f", "#eeaa33", "#eeaa33"])
    axes[1].set_ylim(0, 110)
    axes[1].set_title("Original UR Validation Accuracy (%)")
    axes[1].tick_params(axis="x", labelrotation=15)
    for i, acc in enumerate(accuracies):
        axes[1].text(i, acc + 2, f"{acc:.2f}%", ha="center")
    names = [
        "ur-isotropic-wide-20261012-001",
        "ur-external128-20261012-001",
        "ur-external256-20261012-001",
    ]
    losses = [verification["common_units_normal_val_mse"][n] for n in names]
    axes[2].bar(labels, losses, color=["#56708f", "#009988", "#009988"])
    axes[2].set_ylim(0, 0.14)
    axes[2].set_title("Normal Val MSE in common Train24 units")
    axes[2].tick_params(axis="x", labelrotation=15)
    for i, loss in enumerate(losses):
        axes[2].text(i, loss + 0.003, f"{loss:.5f}", ha="center")
    fig.suptitle(
        "More real normal data did not improve UR Validation Accuracy\n23 unchanged sequences; no new Test inference",
        fontsize=13,
    )
    plot = DEST / "results.png"
    fig.savefig(plot, dpi=150)
    plt.close(fig)

    payload = json.dumps(sequences, ensure_ascii=False).replace("<", "\\u003c")
    page = """<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UR Train／Validation：取樣標註檢視</title>
<style>body{font:16px system-ui,sans-serif;color:#e9edf4;background:#111923;max-width:1200px;margin:auto;padding:24px}h1{font-size:26px}p{line-height:1.65}.note{background:#263349;border-left:4px solid #edba5d;padding:16px}select{padding:10px;font-size:17px;background:#263349;color:white}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px}td,th{border-bottom:1px solid #394659;padding:8px;text-align:left}.table{overflow-x:auto}.V{color:#63d8ab}.P,.O{color:#edba5d}.C,.N{color:#ff9c9c}.U{color:#b0bdcf}.samples{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:18px}.sample{padding:12px;background:#1c2838}a{color:#83c6ff}.legend{line-height:1.8}</style>
<h1>UR Train／Validation：跌倒時段與可見區域</h1>
<p class="note">這是 Codex AI 檢視 340 張取樣影像的標註草稿，尚未由人員複核。不是完整逐幀、左右關節座標或官方 Ground Truth。低光、模糊、出框會保留不確定性；模型 confidence 另存診斷 CSV，沒有當成可見性答案。Train 僅正常活動；15 段跌倒均來自 Validation。Test 未標註、未評分。</p>
<p class="legend">V＝區域可見；P＝部分可見／自身遮擋／單側不明；O＝物件遮擋；C＝大部分出框；U＝模糊、暗或太小，無法可靠判定；N＝取樣圖中沒有目標人物。肩、肘、腕、髖、膝、踝為成對區域，不表示兩側關節皆準確。</p>
<label>序列 <select id="sequence"></select></label><p id="phase"></p>
<div class="table"><table><thead><tr><th>原始影格</th><th>秒</th><th>頭</th><th>肩</th><th>肘</th><th>腕</th><th>髖</th><th>膝</th><th>踝</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>原始影像取樣</h2><div class="samples" id="samples"></div>
<details><summary>整支原始影像接觸表</summary><img id="sheet" alt="原始取樣影格接觸表"></details>
<p>修改標註時請建立新版本，保留 v2 的 CSV／JSON／manifest 與 hash。這些取樣時段僅供診斷，影片判對不代表警示時間正確。</p>
<script>
const data=__DATA__; const regions=['head','shoulders','elbows','wrists','hips','knees','ankles'];
const selector=document.getElementById('sequence');for(const s of data){const o=document.createElement('option');o.value=s.sequence_id;o.textContent=`${s.sequence_id} · ${s.split} · ${s.label?'Fall':'ADL'}`;selector.append(o)}
const fmt=a=>a?a.map(v=>(v/1000).toFixed(3)).join(' – ')+' 秒':'無跌倒時段';
function render(){const s=data.find(s=>s.sequence_id===selector.value);document.getElementById('phase').textContent=s.label?`跌倒開始不確定區間：${fmt(s.onset_uncertainty_ms)}；完成下降／滾動不確定區間：${fmt(s.settled_uncertainty_ms)}。可確認下降核心：${fmt(s.core_descent_ms)}。可能下降區間：${fmt(s.possible_descent_ms)}。`:'來源 ADL 標籤，取樣影像為正常活動；沒有為正常 Train 捏造跌倒時段。';const tb=document.getElementById('rows');const gallery=document.getElementById('samples');tb.replaceChildren();gallery.replaceChildren();for(const r of s.samples){const tr=document.createElement('tr');for(const v of [r.source_frame_id,(Number(r.timestamp_ms)/1000).toFixed(3),...regions.map(k=>r[k])]){const td=document.createElement('td');td.textContent=v;td.className=v;tr.append(td)}tb.append(tr);const card=document.createElement('div');card.className='sample';const label=document.createElement('p');label.textContent=`原始影格 ${r.source_frame_id} · ${(Number(r.timestamp_ms)/1000).toFixed(3)} 秒`;const a=document.createElement('a');a.href=r.image;const img=document.createElement('img');img.src=r.image;img.loading='lazy';img.alt=label.textContent;a.append(img);card.append(label,a);gallery.append(card)}document.getElementById('sheet').src=s.sheet_url}
selector.addEventListener('change',render);render();
</script></html>""".replace("__DATA__", payload)
    viewer = DEST / "annotation-review.html"
    viewer.write_text(page, encoding="utf-8")
    assert "__DATA__" not in page and html.unescape("取樣標註") in page
    write_json(
        DEST / "manifest.json",
        {
            "source_sha256": sha256(__file__),
            "verification_sha256": sha256(verification_path),
            "annotation_manifest_sha256": sha256(ANNOTATIONS / "manifest.json"),
            "sampled_raw_images_hash_verified": len(samples),
            "annotation_sequences": len(sequences),
            "human_validated": False,
            "new_model_inferences": 0,
            "new_test_evaluations": 0,
            "files": {plot.name: sha256(plot), viewer.name: sha256(viewer)},
        },
    )
    print({"plot": str(plot), "review": str(viewer), "raw_images_hash_verified": len(samples)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("build",))
    build()
