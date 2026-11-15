"""Produce immutable Validation evidence and an executed, read-only results notebook."""

import argparse
import html
import os
import sys
from collections import Counter
from datetime import UTC, datetime

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

from fall_detect.config import project_path
from fall_detect.data.cache import load_pose_cache
from fall_detect.detection.events import score_sequence
from fall_detect.evaluation.course_report import load_course_results
from fall_detect.evaluation.metrics import sequence_metrics
from fall_detect.io_utils import ContractError, read_json, sha256, write_json
from fall_detect.training.experiment import Scorer, cache_for, verify_artifacts
from fall_detect.visualization.reports import evaluation_report


def evidence():
    freeze = read_json(project_path("docs/validation/course-before-test-freeze.json"))
    run = project_path("runs") / freeze["selected_run"]
    directory = project_path("outputs/course-ur-validation")
    if directory.exists():
        raise ContractError("Validation output exists; preserve prior evidence")
    scorer = Scorer(run)
    rows, windows, events, diagnosis = [], [], [], []
    # Detailed diagnosis reads Train/Validation. Test selection is already frozen.
    for source in scorer.index["split_manifest"]["sequences"]:
        if source["split"] == "test":
            continue
        sid = source["sequence_id"]
        cache = cache_for(scorer.index, sid)
        result, scored, alarms = score_sequence(
            cache, scorer.score(cache.features), scorer.threshold["threshold"]
        )
        pose_path = project_path(scorer.index["sequences"][sid]["pose_path"])
        observations, _ = load_pose_cache(
            pose_path, {"source_hash": cache.metadata["source_hash"], "split": source["split"]}
        )
        reason_counts = Counter(w["reason"] for w in scored if w["reason"])
        legal = [w for w in scored if w["event_score"] is not None]
        diagnosis.append(
            {
                **source,
                **result,
                "valid_features": len(cache.features),
                "raw_frames": len(observations),
                "resampled_frames": cache.metadata["n_samples"],
                "duration_ms": cache.metadata["end_ms"],
                "estimated_timestamp_frames": sum(f.timestamp_estimated for f, _ in observations),
                "frames_without_person": sum(not p for _, p in observations),
                "raw_track_ids": sorted({p.track_id for _, poses in observations for p in poses}),
                "window_track_reason_counts": dict(reason_counts),
                "legal_event_windows": len(legal),
                "first_legal_event_ms": legal[0]["available_at_ms"] if legal else None,
                "valid_window_times_ms": [r.end_ms for r in cache.records if r.quality == "valid"],
            }
        )
        if source["split"] == "validation":
            rows.append({**result, "label": source["label"]})
            windows.extend(scored)
            events.extend(alarms)
    metrics = {
        **sequence_metrics(rows),
        "split": "validation",
        "source_kind": "real",
        "artifact_hashes": verify_artifacts(run),
        "recipe_hash": scorer.index["recipe_hash"],
        "split_hash": scorer.index["split_manifest"]["split_hash"],
        "protocol_id": scorer.index["split_manifest"]["protocol_id"],
    }
    chosen = read_json(run / "calibration.json")["chosen"]["metrics"]
    assert all(chosen[k] == v for k, v in sequence_metrics(rows).items())
    directory.mkdir(parents=True)
    evaluation_report(directory, metrics, rows, windows, events)
    write_json(
        project_path("docs/validation/course-ur-window-diagnosis.json"),
        {
            "source_kind": "real",
            "scope": "train_and_validation",
            "selected_run": run.name,
            "formed_candidates": False,
            "created_after_freeze": True,
            "rows": diagnosis,
        },
    )
    paths = []
    for destination in (directory, project_path("outputs/course-ur-test")):
        paths.extend(destination.glob("*.csv"))
        paths.append(destination / "metrics.json")
    for relative in [
        "docs/validation/course-experiment-lock.json",
        "docs/validation/course-before-test-freeze.json",
        "docs/validation/course-ur-diagnosis.json",
        "docs/validation/course-ur-window-diagnosis.json",
        "outputs/refinement-test/metrics.json",
        "outputs/refinement-test/sequence-results.csv",
    ]:
        paths.append(project_path(relative))
    write_json(
        project_path("docs/validation/course-report-inputs.json"),
        {
            "created_at": datetime.now(UTC).isoformat(),
            "selected_run": run.name,
            "files": {
                str(p.relative_to(project_path("."))).replace("\\", "/"): sha256(p) for p in paths
            },
        },
    )
    result = load_course_results()
    print(
        "Verified formal Validation/Test rows, windows, events, unknown tables and hashes:",
        result["run"].name,
    )


def notebook():
    destination = project_path("notebooks/course-results.ipynb")
    if destination.exists():
        raise ContractError("Notebook exists; preserve existing executed result")
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [
        md("""# UR Fall 期末專案成果

實驗：真實 UR Fall camera 0 RGB；ST-GCN 特徵＋正常資料 AE。這本 Notebook **只讀正式輸出**，不訓練、不校準、不重新推論 Test。缺檔、hash、切分或分母不一致會報錯。

歷史 Test 已查看，以下為固定切分工程對照；不是未見過的獨立測試。教師尚未明定評估單位等細節，不換算課程分數。影片層級命中不代表起跌警示時間正確。"""),
        code("""from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from fall_detect.evaluation.course_report import load_course_results
from fall_detect.io_utils import read_json
from fall_detect.config import project_path

result = load_course_results()
m = result['test']['metrics']
print('Python:', sys.executable)
print('Run:', result['run'].name)
print('Recipe hash:', m['recipe_hash'])
print('Split hash:', m['split_hash'])
print('AE SHA256:', m['artifact_hashes']['autoencoder.pt'])
print('所有正式數字／事件／來源 hash 已核對')

def table(headers, rows):
    display(Markdown('| ' + ' | '.join(headers) + ' |\\n| ' + ' | '.join(['---'] * len(headers)) + ' |\\n' + '\\n'.join('| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows)))
def pct(value):
    return '未定義' if value is None else f'{value:.2%}'
"""),
        md("""## 1. 資料切分與方法

原始序列先切分，沒有重切。Scaler／AE只讀 Train ADL；正常Validation選checkpoint，完整Validation選門檻與候選。YOLO11m Pose是預訓練模型，映射OpenPose18後取凍結Kinetics ST-GCN分類頭前256D。AE重建誤差不是跌倒機率，不輸出七類動作。

30 fps／30幀窗口／stride5；連續3個有效窗口分數均大於門檻才確認事件。缺少合法連續分數為unknown，不能視為正常。未保證人物隔離。"""),
        code("""sources = result['split']['sequences']
table(['Split', 'ADL', 'Fall', '用途'], [[s, sum(r['split']==s and r['label']==0 for r in sources), sum(r['split']==s and r['label']==1 for r in sources), use] for s,use in [('train','Scaler / AE'),('validation','checkpoint / threshold / selection'),('test','frozen engineering comparison')]])
sampling = result['sources']['training_sampling']
counts = sampling['train_normal_feature_counts']
print('Train ADL:', len(counts), 'sequences /', sum(counts.values()), 'windows')
print('每序列有效窗範圍:', min(counts.values()), '-', max(counts.values()))
print('正常 Validation 特徵來源:', len(result['sources']['val_normal_feature_hashes']))
print('真實預訓練權重來源:')
table(['Role','Source','SHA256'], [[role,w['source_page'],w['sha256']] for role,w in result['sources']['weights'].items()])
"""),
        md("""## 2. Loss 與事前鎖定候選

三候選最多100 epochs、patience10。窗口等權32維latent／序列均衡32維／序列均衡16維。門檻選Validation balanced sequence accuracy；候選依operational accuracy、F1、全體跌倒召回、較少FP、較早候選排序。Test後停止調參。

Loss是MSE，不是Accuracy。CUDA曾有CuBLAS決定性警告，保留實際hash；不承諾重新訓練逐位元一致。"""),
        code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, history) in zip(axes, result['history'].items()):
    epochs = [int(r['epoch']) for r in history]
    train = [float(r['train_loss']) for r in history]
    val = [float(r['val_normal_loss']) for r in history]
    best = int(np.argmin(val))
    ax.plot(epochs, train, label='Train ADL')
    ax.plot(epochs, val, label='Validation ADL')
    ax.axvline(epochs[best], color='gray', linestyle=':')
    ax.set(title=name.replace('course-ur-', ''), xlabel='Epoch', ylabel='MSE')
    ax.legend()
plt.tight_layout()
plt.show()
rows=[]
for c in result['freeze']['candidates']:
    v=c['validation_metrics']
    h=result['history'][c['run']]
    rows.append([c['run'], len(h), min(h,key=lambda r:float(r['val_normal_loss']))['epoch'], f\"{c['threshold']:.6f}\", pct(v['operational_accuracy']), pct(v['sequence_fall_f1']), f\"{v['TP']}/{v['TN']}/{v['FP']}/{v['FN']}\"])
table(['Run','Epochs','Best epoch','Threshold','Val all accuracy','Val F1','TP/TN/FP/FN'],rows)
print('凍結後選定:', result['freeze']['selected_run'])
"""),
        md("""## 3. 正式 UR Test 成績與分母

二元指標只計可判定影片。另列unknown、coverage、全體正確率及全體跌倒召回，不能只挑較高數字。一次確認事件就將整支影片判陽性，可能來自跌倒前誤报；未量測精確警示時機。"""),
        code("""tp,tn,fp,fn=[m[k] for k in ('TP','TN','FP','FN')]
table(['指標','值','分母／公式'],[
 ['Accuracy（可判定）',pct(m['sequence_accuracy']),f'{tp+tn}/{m[\"n_evaluable\"]}'],
 ['Precision（可判定）',pct(m['sequence_fall_precision']),f'{tp}/{tp+fp}'],
 ['Recall（可判定Fall）',pct(m['sequence_fall_recall']),f'{tp}/{tp+fn}'],
 ['F1（可判定）',pct(m['sequence_fall_f1']),f'{2*tp}/({2*tp}+{fp}+{fn})'],
 ['全體 operational accuracy',pct(m['operational_accuracy']),f'{tp+tn}/{m[\"n_total\"]}'],
 ['全體 Fall recall',pct(m['end_to_end_fall_recall']),f'{tp}/{m[\"by_class\"][\"Fall\"][\"n_total\"]}'],
 ['Sequence coverage',pct(m['sequence_coverage']),f'{m[\"n_evaluable\"]}/{m[\"n_total\"]}'],
 ['Unknown',m['n_unknown'],f'n_total={m[\"n_total\"]}; n_evaluable={m[\"n_evaluable\"]}'],
 ['Window coverage',pct(m['window_coverage']),'有效端點／全部排程端點']])
fig, axes=plt.subplots(1,2,figsize=(10,4))
outcomes=np.array(m['outcomes_with_unknown'])
for ax,values,labels,title in [(axes[0],outcomes[:,:2],['Normal','Fall'],f'Evaluable n={m[\"n_evaluable\"]}'),(axes[1],outcomes,['Normal','Fall','Unknown'],f'All n={m[\"n_total\"]}')]:
    ax.imshow(values,cmap='Blues',vmin=0)
    for (r,c), value in np.ndenumerate(values):
        ax.text(c,r,str(value),ha='center',va='center',fontsize=22,color='white' if value>values.max()/2 else 'black')
    ax.set(xticks=range(len(labels)),xticklabels=labels,yticks=[0,1],yticklabels=['ADL','Fall'],xlabel='Prediction',ylabel='Truth',title=title)
plt.tight_layout(); plt.show()
"""),
        md("""## 4. 與既有混合模型分開比較

相同23支UR Test，舊模型Train混合UR／Le2i／CAUCA，新模型Train只用UR ADL。舊門檻與run原封保留。樣本小且Test曾查看，這是工程對照；改善不能解讀成未知場景可靠度。"""),
        code("""table(['Model','UR TP/TN/FP/FN/U','Accuracy evaluable','F1 evaluable','Accuracy all','Fall recall all'], [[name, '/'.join(str(v[k]) for k in ('TP','TN','FP','FN','n_unknown')),pct(v['sequence_accuracy']),pct(v['sequence_fall_f1']),pct(v['operational_accuracy']),pct(v['end_to_end_fall_recall'])] for name,v in [('Mixed historical',result['mixed_ur']),('UR-only selected',m)]])
print('整體正確支數差:', (m['TP']+m['TN'])-(result['mixed_ur']['TP']+result['mixed_ur']['TN']))
print('誤報支數差:', m['FP']-result['mixed_ur']['FP'])
"""),
        md("""## 5. 錯誤案例與警示時間

以下完整列出23支Test，不用剪輯或截圖反推成績。first_alarm是影片中的確認時間，並非已驗證的起跌延遲。unknown影片保留其原始標籤及原因。展示影片另選Validation正常、跌倒、誤報與漏報，僅用來解釋行為。"""),
        code("""table(['Sequence','Truth','Prediction','Sequence score','First alarm (s)','Window coverage','Reason'],[[r['sequence_id'],r['label'],'unknown' if r['prediction'] is None else r['prediction'],'—' if r['sequence_score'] is None else f\"{r['sequence_score']:.4f}\",'—' if r['first_alarm_ms'] is None else f\"{r['first_alarm_ms']/1000:.2f}\",pct(r['window_coverage']),r['unknown_reason'] or ''] for r in result['test']['rows']])
print('錯誤／unknown案例:', [r['sequence_id'] for r in result['test']['rows'] if r['prediction'] != r['label']])
diagnosis=read_json(project_path('docs/validation/course-ur-window-diagnosis.json'))
table(['Validation case','Valid features','Legal event windows','No-person raw frames','Unique tracks','Prediction'],[[r['sequence_id'],r['valid_features'],r['legal_event_windows'],r['frames_without_person'],len(r['raw_track_ids']),r['prediction']] for r in diagnosis['rows'] if r['split']=='validation' and r['sequence_id'] in ['adl-04','adl-20','fall-02','fall-19']])
"""),
        md("""## 6. 限制與重現

- 現有結果未達課程UR參考Accuracy90%／F1 0.87，且官方切分與unknown規則尚未確認，不能直接換算分數。
- 程式功能通過、Notebook數字吻合、畫面清楚，都不代表辨識品質足夠。骨架追蹤／時間窗失效及正常與跌倒分數重疊仍存在。
- 歷史Test已查看；不以重新切分或新run名稱宣稱恢復獨立性。
- 未驗證起跌onset與事件延遲。預訓練模型來源與額外資料比較皆明示。
- 本Notebook是real_end_to_end成果；既有synthetic只是管線測試，real_feature_smoke只是資源／特徵測試，兩者不當課程成績。

重啟kernel後Run All只核對檔案；正式推論入口為`python -m fall_detect`。環境使用專案`.venv`，工具依賴在`requirements/course-results.txt`。"""),
        code("""# 最後再次核對；本Notebook沒有訓練／校準／Test推論呼叫。
verified=load_course_results()
assert verified['test']['metrics']==m
print('PASS: 正式輸出、指標分母、事件、所有輸入hash與frozen模型一致。')
"""),
    ]
    nb = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "name": "course",
                "display_name": "Fall detection (.venv)",
                "language": "python",
            }
        },
    )
    kernels = project_path(".artifacts/course-kernels")
    write_json(
        kernels / "course/kernel.json",
        {
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Fall detection (.venv)",
            "language": "python",
        },
    )
    # Connection files stay in the project; the notebook kernel is always .venv.
    runtime = project_path(".artifacts/course-jupyter-runtime")
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    manager = KernelManager(
        kernel_name="course", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)])
    )
    client = NotebookClient(
        nb, km=manager, timeout=180, resources={"metadata": {"path": str(project_path("."))}}
    )
    client.execute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, destination)
    render_html(nb, destination.with_suffix(".html"))
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
    write_json(
        project_path("docs/validation/course-notebook-execution.json"),
        {
            "created_at": datetime.now(UTC).isoformat(),
            "kernel_python": sys.executable,
            "source_kind": "real_end_to_end",
            "code_cells": len(code_cells),
            "all_executed": True,
            "errors": 0,
            "notebook_sha256": sha256(destination),
            "input_hashes_verified": True,
            "html_sha256": sha256(destination.with_suffix(".html")),
        },
    )
    print(destination)


def render_html(nb, path):
    sections = []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            sections.append('<div class="markdown">' + html.escape(cell.source) + "</div>")
            continue
        sections.append(
            "<details><summary>程式</summary><pre>" + html.escape(cell.source) + "</pre></details>"
        )
        for output in cell.outputs:
            data = output.get("data", {})
            if "image/png" in data:
                sections.append('<img src="data:image/png;base64,' + data["image/png"] + '">')
            elif "text/markdown" in data:
                # Tables generated here have no arbitrary raw HTML.
                lines = data["text/markdown"].splitlines()
                if lines and lines[0].startswith("|"):
                    sections.append(
                        "<table>"
                        + "".join(
                            "<tr>"
                            + "".join(
                                "<td>" + html.escape(v.strip()) + "</td>"
                                for v in line.strip("|").split("|")
                            )
                            + "</tr>"
                            for i, line in enumerate(lines)
                            if i != 1
                        )
                        + "</table>"
                    )
                else:
                    sections.append("<pre>" + html.escape(data["text/markdown"]) + "</pre>")
            elif output.output_type == "stream":
                sections.append("<pre>" + html.escape(output.text) + "</pre>")
    path.write_text(
        '<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>UR Fall 成果 Notebook</title><style>body{font:17px/1.65 "Microsoft JhengHei",sans-serif;background:#f5f7fb;color:#17243a;max-width:1200px;margin:45px auto;padding:25px}.markdown{white-space:pre-wrap;margin:38px 0 20px}table{border-collapse:collapse;font-size:14px;width:100%;margin:22px 0}td{border:1px solid #ccd3df;padding:9px;word-break:break-word}tr:first-child{background:#dce8f6;font-weight:bold}img{width:100%;height:auto;background:white}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:14px}details{font-size:13px}</style><body>'
        + "".join(sections)
        + "</body></html>",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["evidence", "notebook"])
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    globals()[args.stage]()
