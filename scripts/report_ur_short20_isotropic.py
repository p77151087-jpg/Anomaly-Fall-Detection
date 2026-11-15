"""Read verified outputs and publish this completed experiment's report/figures."""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ur_short20_isotropic as workflow

from fall_detect.io_utils import ContractError, read_json, sha256, write_csv, write_json


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def report():
    lock = workflow.verify_lock()
    results = read_json(workflow.evidence("validation"))
    verification = read_json(workflow.evidence("verification"))
    if not verification["passed"]:
        raise ContractError("Independent verification required")
    root = workflow.ROOT
    target = root / "docs" / f"{workflow.PREFIX}.md"
    if target.exists():
        raise ContractError("Preserve completed report")
    output = root / "outputs" / workflow.PREFIX
    output.mkdir(exist_ok=True)
    reference, candidate = results["reference"], results["candidate"]
    before, after = reference["metrics"], candidate["metrics"]
    ref = {r["sequence_id"]: r for r in reference["predictions"]}
    comparison = [
        {
            "sequence_id": r["sequence_id"],
            "label": r["label"],
            "reference_prediction": ref[r["sequence_id"]]["prediction"],
            "candidate_prediction": r["prediction"],
            "reference_first_alarm_ms": ref[r["sequence_id"]]["first_alarm_ms"],
            "candidate_first_alarm_ms": r["first_alarm_ms"],
            "reference_coverage": ref[r["sequence_id"]]["window_coverage"],
            "candidate_coverage": r["window_coverage"],
        }
        for r in candidate["predictions"]
    ]
    write_csv(output / "sequence-comparison.csv", comparison)
    indexes = [read_json(workflow.SOURCE_INDEX), read_json(workflow.INDEX)]
    valid_sets = []
    for index in indexes:
        meta = read_json(
            root / Path(index["sequences"]["fall-19"]["feature_path"]).with_suffix(".json")
        )
        valid_sets.append({r["sample_index"] for r in meta["records"] if r["quality"] == "valid"})
    added = sorted(valid_sets[1] - valid_sets[0])
    removed = sorted(valid_sets[0] - valid_sets[1])
    difference = {
        "shared_valid_endpoints": len(valid_sets[0] & valid_sets[1]),
        "added_valid_sample_indexes": added,
        "removed_valid_sample_indexes": removed,
        "added_available_at_ms": [i * 1000 / 30 for i in added],
        "normal_until_ms": lock["coarse_intervals"]["fall-19"]["normal_until_ms"],
        "added_before_coarse_normal_end": all(
            i * 1000 / 30 < lock["coarse_intervals"]["fall-19"]["normal_until_ms"] for i in added
        ),
        "coarse_annotations_are_not_ground_truth": True,
    }
    write_json(output / "fall19-window-difference.json", difference)
    source_csv = root / "outputs/isotropic-refinement-20261012" / workflow.BASE / "windows.csv"
    new_csv = output / workflow.RUN / "window-results.csv"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].bar(
        [0, 1],
        [before["operational_accuracy"] * 100, after["operational_accuracy"] * 100],
        color=["#226699", "#c46236"],
    )
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["30-frame reference", "20-frame candidate"],
        ylim=(0, 105),
        ylabel="Overall sequence accuracy (%)",
        title="Same 23 Validation videos",
    )
    for i, metric in enumerate((before, after)):
        axes[0].text(
            i,
            metric["operational_accuracy"] * 100 + 2,
            f"{metric['TP'] + metric['TN']}/23",
            ha="center",
        )
    for ax, path, result, title in zip(
        axes[1:],
        (source_csv, new_csv),
        (reference, candidate),
        ("fall-19: 30 frames", "fall-19: 20 frames"),
    ):
        rows = [r for r in csv_rows(path) if r["sequence_id"] == "fall-19"]
        x = [float(r["available_at_ms"]) / 1000 for r in rows]
        raw = [float(r["raw_score"]) if r["raw_score"] else np.nan for r in rows]
        k3 = [float(r["event_score"]) if r["event_score"] else np.nan for r in rows]
        ax.plot(x, raw, label="Raw AE error", color="#226699")
        ax.plot(x, k3, label="K3 minimum", color="#c46236")
        ax.axhline(result["threshold"], color="black", ls="--", label="Frozen threshold")
        lo, hi = lock["coarse_intervals"]["fall-19"]["fall_phase_ms"]
        ax.axvspan(lo / 1000, hi / 1000, alpha=0.12, color="#777777", label="Coarse AI phase")
        ax.set(
            xlabel="Source time (s)",
            ylabel="AE error (own Train scaler)",
            title=title,
            ylim=(0, 0.6),
        )
        ax.legend(fontsize=8)
    fig.savefig(output / "validation-comparison.png", dpi=160)
    plt.close(fig)
    table = "\n".join(
        f"| {r['sequence_id']} | {r['reference_first_alarm_ms'] / 1000:.4f} | {r['candidate_first_alarm_ms'] / 1000:.4f} |"
        if r["reference_first_alarm_ms"] is not None and r["candidate_first_alarm_ms"] is not None
        else f"| {r['sequence_id']} | {str(round(r['reference_first_alarm_ms'] / 1000, 4)) if r['reference_first_alarm_ms'] is not None else '無'} | {str(round(r['candidate_first_alarm_ms'] / 1000, 4)) if r['candidate_first_alarm_ms'] is not None else '無'} |"
        for r in comparison
        if r["label"] == 1
    )
    totals = [
        (
            sum(r["n_valid_endpoints"] for r in result["predictions"]),
            sum(r["n_scheduled_endpoints"] for r in result["predictions"]),
        )
        for result in (reference, candidate)
    ]
    old19 = read_json(workflow.evidence("diagnosis"))["fall19"]
    new19 = candidate["fall19_diagnosis"]
    text = f"""# UR-only：20 幀＋保留身體比例實驗（2026-10-12）

**實際完成單一候選訓練與完整 Validation；沒有提升，不採用。** 新候選 `{workflow.RUN}` 共39 epochs，最佳Epoch{verification["best_epoch"]}，Validation 20/23＝86.96%、Fall F1 24/27＝.8889，低於30幀等比例最高研究候選22/23＝95.65%／28/29＝.9655，Accuracy下降8.70百分點。原demo預設 `ur-sample-human-s1-001` 保留；其既有86.96%與最高候選95.65%不混稱為已部署成績。本輪沒有恢復／下載外部資料、沒有Test、沒有額外候選。

## 方法、鎖定與實作

[訓練前計畫]({workflow.PREFIX}-plan.md)及[不可覆寫lock](validation/{workflow.PREFIX}-lock.json)固定20幀、stride1、30FPS、short_20、per_frame_isotropic；原AlphaPose／Human TSSTG真實權重、13→14點、分類頭前雙流512D方法不變。AE256/64、seed42、lr.001、wd.001、noise0、batch32、sequence sampling、max100、patience10、min_delta1e-6均沿用wide。只有原24支UR Train ADL，無shift／鏡像／外部資料。品質.2／9真實點／.8、因果重採樣與缺失、strict >／K3不變。舊20幀per_frame結果不是本輪per_frame_isotropic結果。

核心 `prepare/train --train-validation-only` 新增明確opt-in。新index保留原70支split及hash，47支Train／Validation全部prepared，Test23支均not_prepared且無pose／feature路徑；不偽造prepared、不讀Test特徵、不混30幀Test快取。訓練檢查全部Train／Validation含Fall均準備完成，缺片／smoke／舊Test路徑都拒絕。預設完整訓練仍需完整prepared；scoped run禁止Test evaluate。核心API／CLI及風險測試一起更新。

固定10支pilot通過全部10檢查。真實ST-GCN strict load及20幀完整forward／重接head parity通過；特徵512D有限、時間戳因果、品質及完整端點排程通過，30等比例／20per_frame／20等比例recipe hash三者不同。正常平均coverage比值1.002633、六粗時段合法K3總數比值.969072，均過事前gate。固定樣本及逐片證據見[pilot](validation/{workflow.PREFIX}-pilot.json)；gate不測Accuracy。

完整47支features全部由相同觀測Pose及真實凍結ST-GCN以20幀重新計算，沒有沿用30幀特徵。Pose array hash逐片與來源相同、原來源身份及權重hash重驗，Pose本輪沒有新RGB forward；ST-GCN實際CUDA forward有執行。正式Train24ADL 3,888窗、正常Validation8支1,515窗、Validation23支2,623窗，合計6,511窗。原split未重切，Train僅ADL。

正式AE CUDA訓練39回合，正常Val最佳Epoch29／MSE .11289029724330399，依該checkpoint校準原23支Validation，balanced_sequence_accuracy目標、FP平手及较高門檻规则沿用wide。凍結門檻 **{candidate["threshold"]}**；沒有以Test挑epoch／門檻。兩個模型Scaler不同，較低正常Loss不當成辨識改善。

## 正式Validation比較與分母

單位為原始影片sequence。兩者n_total=23、n_evaluable=23、unknown=0，所以可判定二元Accuracy與包含unknown的整體Accuracy各自分母均23。Fall為positive；Precision分母為预测Fall，Recall分母為可判定真Fall；全體Fall召回另以全部15支Fall計。

| 指標 | 30幀等比例wide基準 | 20幀等比例候選 |
|---|---:|---:|
| 可判定二元Accuracy | 22/23＝95.65% | 20/23＝86.96% |
| 包含unknown整體Accuracy | 22/23＝95.65% | 20/23＝86.96% |
| Fall F1（可判定） | 28/(28+0+1)＝.9655 | 24/(24+0+3)＝.8889 |
| Precision（可判定） | 14/14＝100% | 12/12＝100% |
| Recall（可判定） | 14/15＝93.33% | 12/15＝80% |
| 全體Fall召回 | 14/15＝93.33% | 12/15＝80% |
| TP／TN／FP／FN／U | 14／8／0／1／0 | 12／8／0／3／0 |
| ADL誤報（全部ADL分母） | 0/8 | 0/8 |
| 漏報（全部Fall分母） | 1/15：fall-19 | 3/15：fall-19／18／30 |
| unknown（全部影片分母） | 0/23 | 0/23 |
| 影片coverage | 23/23＝100% | 23/23＝100% |
| 窗口coverage | {totals[0][0]}/{totals[0][1]}＝{before["window_coverage"] * 100:.4f}% | {totals[1][0]}/{totals[1][1]}＝{after["window_coverage"] * 100:.4f}% |

2×2（列ADL／Fall、欄normal／suspected_fall）：基準 `[[8,0],[1,14]]`，候選 `[[8,0],[3,12]]`。2×3加unknown欄：基準 `[[8,0,0],[1,14,0]]`，候選 `[[8,0,0],[3,12,0]]`。窗口分母是各自窗長後全部預定端點；短窗暖機少10格，因此窗口總數不同，增加有效窗不等同每幀可判定或增加獨立影片。

## fall-19：品質、时间窗與分數

原粗略跌倒phase1468–2369ms。30幀實際span966.67ms、20幀633.33ms（30FPS下T-1個間隔）。相同Pose hash，整片有效端點23/71→33/81，track_gap37→27、low_confidence11→21；品質規則未弱化。新增{len(added)}個合法端點為sample index {added}，時間{min(difference["added_available_at_ms"]):.2f}–{max(difference["added_available_at_ms"]):.2f}ms，全部在粗略正常前段；共享{difference["shared_valid_endpoints"]}個、移除{len(removed)}個。精確差異見[端點診斷](../outputs/{workflow.PREFIX}/fall19-window-difference.json)。

**可能跌倒phase內仍是27個預定端點、9個有效端點、9個合法K3，沒有增加。** 基準phase最高raw .18390694、K3 {old19["phase_max_event_score"]:.9f}，門檻{reference["threshold"]:.9f}；候選raw {new19["phase_max_raw_score"]:.9f}、K3 {new19["phase_max_event_score"]:.9f}，門檻{candidate["threshold"]:.9f}，仍未跨門檻，無事件／無時段內良好骨架警示。分數屬各自Scaler及AE，不以raw数值較大當成改善。

因此不能將漏報直接歸因於没有骨架。短窗增加了前段可用窗口，但沒有增加此粗phase的合法K3，也没有救回fall-19；「減少正常歷史稀釋／裁切無效窗可改善辨識」在這個鎖定候選沒有得到Accuracy改善的支持。並未量化所有漏報原因的因果贡献；AI粗时段、confidence與可見性診斷不是人工真值。

## 警示時間與影片判對分開

完整[逐影片比較CSV](../outputs/{workflow.PREFIX}/sequence-comparison.csv)包含23支預測／coverage／首次警示。以下秒數是影片來源時間的實際K3確認時間，**不是人工onset延遲**。12支共同TP中10支首次事件較早、2支相同；fall-18／30則由有事件退步為沒有事件。沒有以刪減警示或剪輯挑成功片段作成績。

| Validation Fall | 基準首次事件（秒） | 候選首次事件（秒） |
|---|---:|---:|
{table}

六個舊粗時段事件起點命中由4降至2，仍0早報；失去fall-18／30命中。fall-09事件5.9000→5.6333秒，仍超出舊phase4.604–5.506秒，影片TP不能稱警示时间合格。候選只fall-29／25有粗phase事件；當前良好骨架警示端點分別3／2。所有六片[逐時段結果](validation/{workflow.PREFIX}-validation.json)保留。新的14個事件不等同14支TP，12支影片計TP。

事前採用guard未通過：Accuracy/F1下降、兩支基準判對片退步、粗時段命中减少；無FP／unknown增加及coverage增加不能抵銷。**不採用、原預設保留，單一候選實驗結束，不追加調參。**

## 驗證、保全及限制

- synthetic：33相關風險測試全過；全套首次271 passed／1历史預設Human權重路徑skip／2本機HTTP沙箱權限失敗，兩項以允許loopback環境重測2 passed。總274項中273通過、1 skip，保留原失敗XML與log，不偽稱首次全綠。新使用的真實sample Human／20幀ST-GCN strict與forward實際通過，skip不代表本輪權重缺失。[風險XML](validation/pytest-ur-short20-isotropic-risk.xml)／[全套XML](validation/pytest-ur-short20-isotropic-full.xml)／[loopback重測](validation/pytest-ur-short20-isotropic-loopback.xml)。
- real_feature_smoke：固定10片genuine20／512D gate、完整47片真實ST-GCN重新抽取、39epoch AE、23Val凍結評估；獨立sklearn重算Accuracy／Precision／Recall／F1／兩矩陣與分母、每窗K3／状态及14完整事件開始／結束／峰值／原因全部相同。Train-only Scaler逐值相同，Train24與正常Val8 hash、原23Val門檻來源及最佳正常Val checkpoint核對通過。[獨立驗收](validation/{workflow.PREFIX}-verification.json)。
- real_end_to_end：本輪未跑。没有新RGB Pose擷取／demo／Test；已有Pose沿用hash保全，不能把新ST-GCN抽取与功能測試称完整端到端性能驗收。Test全部23支not_prepared，沒有新Test特徵／評分；歷史Test已查看，不稱全新盲測。
- {verification["preserved_files"]}個舊run／設定／TrainVal相關完整cache／split／原权重／既有結果檔hash不變。原raw身份於prepare重新hash驗證，Pose原array相同；舊30幀cache／外部封存資料未覆寫。拒絕4個完成後stage mutation；新run有自己的recipe **{lock["recipe_hash"]}**、config、scaler、checkpoint、threshold、Loss及來源hash。
- 初版流程於訓練前修正CLI prepare派送；首次獨立驗收讀CSV漏處理BOM，修正為utf-8-sig後全通過。原source／lock／失敗log及新hash鏈分開保留，沒有重訓、重校準或改選擇／停止條件。[派送修復](validation/{workflow.PREFIX}-dispatch-repair.json)／[驗收器修復](validation/{workflow.PREFIX}-verifier-repair.json)。
- Graph服務恢復，Tier2先查search／trace／snippet及coverage；初代新脚本239–242语法parse_partial直接读取並修正。後續src及實驗／驗收／測試路徑coverage metadata_match／no_recorded_issue；ignored cache與upstream以精确source／hash／實測補核對。coverage為best effort，不是完整正確性證明。
- Validation只有23支且反覆使用、自訂split未保證人物隔離；預訓練Pose／Human ST-GCN不是UR-only從零學習，Human动作權重既有Le2i来源如實保留。完整人工骨架／onset GT、教師評估協定、跨場景泛化／完整CPU實驗仍未確認。CuBLAS deterministic warn_only限制保留，不承諾跨硬體重訓逐位元相同。
- 正式成果Notebook與五分鐘影片继续暫緩，沒有公開上傳、不自行換算課程分數。

![Validation比較與fall-19分數](../outputs/{workflow.PREFIX}/validation-comparison.png)

## 成果與可執行命令

模型：`runs/{workflow.RUN}`。正式47片cache：`data/cache/{workflow.RUN}`；pilot另置real_feature_smoke。正式[metrics](../outputs/{workflow.PREFIX}/{workflow.RUN}/metrics.json)、sequence-results／window-results／events／2×3／混淆矩陣及Loss曲線均保留。鎖定／診斷／pilot／prepared／validation／verification與实际log位於`docs/validation/{workflow.PREFIX}-*`。

```powershell
Set-Location 'C:\\成大專案\\fall-detection-anomaly'
.\\.venv\\Scripts\\python.exe scripts/ur_short20_isotropic.py status
.\\.venv\\Scripts\\python.exe scripts/verify_ur_short20_isotropic.py
.\\.venv\\Scripts\\python.exe -m fall_detect loss --run runs/{workflow.RUN}
```

前兩條只讀重驗，不跑Test；Loss命令顯示39回合真實紀錄。流程依序lock／pilot／prepare／training已完成，重跑皆拒絕覆寫／追加候選。若只重驗風險測試：

```powershell
.\\.venv\\Scripts\\python.exe -m pytest tests/test_train_validation_scope.py tests/test_human_context.py tests/test_human_isotropic.py tests/test_training_cache.py tests/test_events_metrics.py --basetemp .pytest_cache/short20-review -q
```

核心新模式的可用介面為 `python -m fall_detect prepare --config <新設定> --train-validation-only --reuse-poses-index <來源index>`，再 `python -m fall_detect train --config <新設定> --train-validation-only --index <該index> --run-id <新ID>`；本輪已结束，不以此另加候選。
"""
    target.write_text(text, encoding="utf8")
    evidence = {
        "report_sha256": sha256(target),
        "report_source_sha256": sha256(Path(__file__)),
        "figure_sha256": sha256(output / "validation-comparison.png"),
        "comparison_csv_sha256": sha256(output / "sequence-comparison.csv"),
        "fall19_window_difference": difference,
        "preserved_files": len(lock["preserved_files"]),
    }
    workflow.save("report", evidence)
    workflow.verify_lock()
    print({"report": str(target), "fall19_difference": difference, "window_counts": totals})


if __name__ == "__main__":
    report()
