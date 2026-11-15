"""Generate the completed comparison report from immutable formal outputs."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ur_normal_methods import BASE, ROOT, E, O, save, verify_lock

from fall_detect.io_utils import ContractError, atomic_bytes, read_json, sha256, write_csv
from fall_detect.models.normal_alternatives import candidate_rank


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def report():
    target = ROOT / "docs/ur-normal-methods-20261012.md"
    if target.exists() or (E / "selection.json").exists():
        raise ContractError("Completed report exists; preserve previous generation")
    lock = verify_lock()
    verification = read_json(E / "verification.json")
    assert verification["passed"]
    geometry = read_json(E / "geometry-diagnosis.json")
    results = (
        read_json(E / "svm-results.json")["candidates"]
        + read_json(E / "ae-results.json")["candidates"]
    )
    selected = max(results, key=lambda r: candidate_rank(r["metrics"], r["spec"]["order"]))
    better = [
        r["spec"]["id"]
        for r in results
        if r["recognition_better_than_wide"]
        and r["metrics"]["n_unknown"] <= lock["reference_metrics"]["n_unknown"]
    ]
    decision = {
        "selected_new_candidate": selected["spec"]["id"],
        "recognition_adopted": bool(better),
        "eligible_recognition_candidates": better,
        "research_reference_retained": BASE,
        "demo_default_retained": "ur-sample-human-s1-001",
        "selected_new_rank": list(candidate_rank(selected["metrics"], selected["spec"]["order"])),
        "test": "unknown_for_all_new_methods",
        "display_separate": True,
        "stop": "six SVM and two AE completed; no grid extension",
    }
    save(E / "selection.json", decision)
    save(
        E / "stage3-decision.json",
        {
            "self_supervised_training": "not_run",
            "diagnostic_support": geometry["self_supervised_supported"],
            "geometry_rule_sha256": sha256(E / "geometry-rule.json"),
            "geometry_diagnosis_sha256": sha256(E / "geometry-diagnosis.json"),
            "reason": "Raw observed XY as well as pooled512 fall19 legal windows overlap normal Train neighborhoods; zero consecutive supporting endpoints under the prior rule. No controlled support that ST-GCN alone compresses an otherwise abnormal observed pose. Stop rather than blind normal-only fine tuning.",
            "not_a_claim_of_correct_representation": True,
            "test_used": False,
        },
    )
    rows = []
    for r in results:
        m = r["metrics"]
        spec = r["spec"]
        training = read_json(ROOT / "runs" / spec["id"] / "training.json")
        rows.append(
            {
                "run": spec["id"],
                "kind": spec["kind"],
                "nu": spec.get("nu"),
                "gamma": spec.get("gamma"),
                "epochs": training["epochs"],
                "best_epoch": training.get("best_epoch"),
                "threshold": r["threshold"],
                "TP": m["TP"],
                "TN": m["TN"],
                "FP": m["FP"],
                "FN": m["FN"],
                "unknown": m["n_unknown"],
                "n_total": m["n_total"],
                "n_evaluable": m["n_evaluable"],
                "overall_accuracy": m["operational_accuracy"],
                "binary_accuracy": m["sequence_accuracy"],
                "fall_f1": m["sequence_fall_f1"],
                "precision": m["sequence_fall_precision"],
                "recall": m["sequence_fall_recall"],
                "all_fall_recall": m["end_to_end_fall_recall"],
                "window_coverage": m["window_coverage"],
                "phase_hits_of_six": sum(bool(p["phase_event_starts"]) for p in r["phases"]),
                "phase_visible_hits_of_six": sum(
                    bool(p["visible_phase_alert_ms"]) for p in r["phases"]
                ),
                "early_events": sum(len(p["early_event_starts"]) for p in r["phases"]),
                "false_positives": ",".join(r["false_positives"]),
                "false_negatives": ",".join(r["false_negatives"]),
                "delta_accuracy_vs_wide": m["operational_accuracy"]
                - lock["reference_metrics"]["operational_accuracy"],
            }
        )
    write_csv(O / "candidate-comparison.csv", rows)
    comparisons = []
    for name in [BASE] + [r["spec"]["id"] for r in results]:
        for row in read_rows(O / name / "sequence-results.csv"):
            comparisons.append({"run": name, **row})
    write_csv(O / "all-sequence-comparison.csv", comparisons)
    phases = [
        {"run": BASE, **p} for p in read_json(E / "diagnosis-stage1.json")["reference_phases"]
    ]
    phases += [{"run": r["spec"]["id"], **p} for r in results for p in r["phases"]]
    save(
        O / "phase-comparison.json",
        {
            "interval_source": "preexisting AI coarse sample intervals, not human temporal GT",
            "phases": phases,
        },
    )
    labels = ["Wide AE"] + [f"SVM {i + 1}" for i in range(6)] + ["Denoise AE", "GRU AE"]
    accuracy = [lock["reference_metrics"]["operational_accuracy"]] + [
        r["metrics"]["operational_accuracy"] for r in results
    ]
    f1 = [lock["reference_metrics"]["sequence_fall_f1"]] + [
        r["metrics"]["sequence_fall_f1"] for r in results
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(x - 0.18, accuracy, 0.36, label="Overall Accuracy (n=23)")
    ax.bar(x + 0.18, f1, 0.36, label="Fall F1 (evaluable n=23)")
    for i, a in enumerate(accuracy):
        ax.text(i - 0.18, a + 0.012, f"{a * 100:.2f}%", ha="center", fontsize=8)
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylim=(0, 1.12),
        ylabel="Metric",
        title="UR-only Validation: six fixed SVM and two AE; Test unknown",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(O / "comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, name in zip(axes, [BASE, selected["spec"]["id"]]):
        ws = [
            w for w in read_rows(O / name / "window-results.csv") if w["sequence_id"] == "fall-19"
        ]
        t = [float(w["available_at_ms"]) / 1000 for w in ws]
        raw = [float(w["raw_score"]) if w["raw_score"] else np.nan for w in ws]
        k3 = [float(w["event_score"]) if w["event_score"] else np.nan for w in ws]
        threshold = lock["reference_threshold"] if name == BASE else selected["threshold"]
        ax.plot(t, raw, label="Raw anomaly score")
        ax.plot(t, k3, label="Legal K3 minimum")
        ax.axhline(threshold, color="red", linestyle="--", label="Frozen Validation threshold")
        ax.axvspan(1.468, 2.369, color="orange", alpha=0.15, label="AI coarse phase, not GT")
        ax.set(title=name, ylabel="Model-specific score")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Source video seconds")
    fig.tight_layout()
    fig.savefig(O / "fall19-scores.png", dpi=150)
    plt.close(fig)
    table = [
        "| 新候選 | Epochs／最佳 | Accuracy | Fall F1 | Precision | Recall | TP/TN/FP/FN/U | 窗口coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    matrix = [
        "| 新候選 | 2×2（ADL/Fall × normal/fall） | 2×3（加unknown） | 門檻 | 粗phase事件／可見命中（各/6） |",
        "| --- | --- | --- | ---: | --- |",
    ]
    names = []
    for i, (r, row) in enumerate(zip(results, rows)):
        spec = r["spec"]
        m = r["metrics"]
        name = (
            f"SVM {i + 1} nu={spec['nu']}, gamma={spec['gamma']}"
            if spec["kind"] == "svm"
            else "去噪wide AE"
            if spec["kind"] == "denoise"
            else "時序GRU AE"
        )
        names.append(name)
        table.append(
            f"| {name} | {row['epochs']}/{row['best_epoch'] or '—'} | {m['TP'] + m['TN']}/23={m['operational_accuracy'] * 100:.2f}% | {m['sequence_fall_f1']:.4f} | {m['sequence_fall_precision']:.4f} | {m['sequence_fall_recall']:.4f} | {m['TP']}/{m['TN']}/{m['FP']}/{m['FN']}/{m['n_unknown']} | {m['window_coverage'] * 100:.2f}% |"
        )
        matrix.append(
            f"| {name} | {np.array(m['outcomes_with_unknown'])[:, :2].tolist()} | {m['outcomes_with_unknown']} | {r['threshold']:.12g} | {row['phase_hits_of_six']}/{row['phase_visible_hits_of_six']} |"
        )
    sensitivity = []
    for r in results:
        balanced = read_json(ROOT / "runs" / r["spec"]["id"] / "calibration.json")[
            "diagnostic_historical_balanced_calibration"
        ]
        sensitivity.append(
            f"- {r['spec']['id']}：歷史balanced校準Accuracy={balanced['metrics']['operational_accuracy'] * 100:.2f}%、F1={balanced['metrics']['sequence_fall_f1']:.4f}；本輪主要成績仍用事前新規則。"
        )
    timing = [
        "| 粗phase影片 | wide首次事件秒 | 最佳新SVM首次事件秒 | wide phase可見端點 | 最佳SVM phase可見端點 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    baseline_rows = {r["sequence_id"]: r for r in read_rows(O / BASE / "sequence-results.csv")}
    chosen_rows = {
        r["sequence_id"]: r for r in read_rows(O / selected["spec"]["id"] / "sequence-results.csv")
    }
    baseline_phases = {
        r["sequence_id"]: r for r in read_json(E / "diagnosis-stage1.json")["reference_phases"]
    }
    chosen_phases = {r["sequence_id"]: r for r in selected["phases"]}
    for sid in lock["coarse_intervals"]:
        old = baseline_rows[sid]["first_alarm_ms"]
        new = chosen_rows[sid]["first_alarm_ms"]
        timing.append(f"| {sid} | {float(old) / 1000:.4f}" if old else f"| {sid} | 無")
        timing[-1] += f" | {float(new) / 1000:.4f}" if new else " | 無"
        timing[-1] += (
            f" | {len(baseline_phases[sid]['visible_phase_alert_ms'])} | {len(chosen_phases[sid]['visible_phase_alert_ms'])} |"
        )
    text = (
        """# UR-only正常資料異常偵測方法比較（2026-10-12）

**已實際完成六組One-Class SVM及兩個AE、共62個新AE epochs，沒有超過最高Validation wide；不採用新模型。** 新候選最高21/23＝91.30%、Fall F1 .9333，比基準22/23＝95.65%／.9655少1支判對（Accuracy下降4.35百分點）。SVM06與GRU持平，按事前固定順序選SVM06作新方法比較代表；這不是部署採用。原demo仍`ur-sample-human-s1-001`，最高研究參照仍`ur-isotropic-wide-20261012-001`。全部新方法Test未知，未評估。

## 教師方向、方法契約與歷史版本

重新讀取54頁教師簡報及10頁Sample PDF並保存原檔hash。簡報37–40頁明示跌倒作為異常、One-class／Autoencoder及特徵抽取示例，39頁圖與Sample3頁图為ST-GCN→Normalizer→AE。48頁UR90%／F1 .87是參考基準；不自行換算課程分數。Sample3–4頁主張AE、第6頁報70支Accuracy .843／F1 .845，但附件缺完整AE訓練／校準／評估，不能斷言其成績全部來自七類分類器。

根規格、AGENTS及課程計畫已區分教師方向、使用者要求、歷史工程預設與新選擇。固定ST-GCN、不微調、30幀、MLP AE、K3不再是永久研究禁令；本輪第一二階段刻意保留30iso表示、真實權重及品質/K3以隔離方法。歷史lock／設定／資料／成果不改寫，修改前五份契約逐位元封存。

| 歷史版本（均為已完成Validation） | Accuracy／Fall F1 | 方法與身份 |
| --- | --- | --- |
| demo預設 ur-sample-human-s1-001 | 20/23＝86.96%／.8889 | 原30幀per_frame，128/32 AE；目前預設 |
| 最高參照 ur-isotropic-wide-20261012-001 | 22/23＝95.65%／.9655 | 30幀per_frame_isotropic，256/64 AE；研究參照，未混稱已部署 |
| ur-short20-isotropic-20261012-001 | 20/23＝86.96%／.8889 | 20幀isotropic，256/64，39epochs；不採用 |
| ur-human-context20-001 | 20/23＝86.96%／.8889 | 舊20幀per_frame；不同於20iso |
| ur-human-isotropic-001 | 21/23＝91.30%／.9286 | 30iso原較小AE |
| ur-joint-shift-20261012-001 | 21/23＝91.30%／.9286 | Train限定關節位移，不採用 |
| 舊isotropic denoise128/32 | 21/23＝91.30%／.9286 | 本輪denoise改為wide256/64，不重訓此舊run |
| Train鏡像候選 | 21/23＝91.30%／.9286 | 已完成，不重複 |
| wide Train殘差對照 | 22/23＝95.65%／.9655 | Accuracy持平，粗時段另有改善，未改demo |

wide歷史Test已查看且已保存結果，僅見`ur-isotropic-wide-test-20261012.md`作歷史脈絡；本輪未讀Test特徵、未評估、不據其失敗片或成績調參。反覆Validation及非人物隔離限制不因新run消失。

## 資料、鎖定與真正執行

事前計畫與svm-lock固定六候選、分數方向、門檻／排序／採用／停止規則。原70支split不變：Train24 ADL、Validation8ADL15Fall、Test8ADL15Fall。既有30iso47片快取逐bytes複製到新目錄，保留原feature recipe `713ee59a0853f20641cb7f664df2920359baa0734807acfa5cf86fb87a4e2f5b`；八異常模型另有不同method recipe，時序history另存專用NPZ與provenance。新index Test23片not_prepared、無Pose／feature路徑，scope讀取明確拒絕Test。沒有外部恢復／下載。

三固定原窗口（Train adl05／06、Val fall19）嚴格真實ST-GCN forward及重接原頭parity通過，512D與原快取最大差0。Pose本輪沿用原觀測與hash；沒有新RGB Pose抽取。Train原3638 features、正常Val1427、全Val2363，共6001；這不是獨立影片6001支。Scaler只fit原24 Train，八模型逐值與wide一致。

SVM用全部原Train標準化窗擬合RBF，nu=.01/.05/.10 × gamma=scale/.001、tol=.001、cache512MB、shrinking=true、max_iter=-1；分數=-decision_function，越大越異常，非機率。六組已完成即停。

SVM及第一階段診斷後另寫ae-lock。去噪512-256-64-256-512（296000參數）、ReLU／linear output，Train-only標準化輸入Gaussian std .1、clean target MSE，40epochs、最佳30、正常Val MSE .1154182501。時序512→128 ReLU→GRU64 encoder；context64+固定position→GRU128 decoder→Linear512（243840參數），8連續同影片同track合法窗口，缺失／gap清空、無padding／未來。Loss為全部8步重建MSE，異常分數為最後一步512D MSE；22epochs、最佳12、正常Val MSE .2602489313。兩者seed42、Adam lr.001／wd.001、batch32、sequence-balanced Train、max100／patience10／min_delta1e-6；只有8正常Val選checkpoint。推論固定B=1，沒有依Fall選epoch。

時序Train3449 histories、正常Val1364、全Val2174；原影片24／23不變，history從既有觀測衍生，不增加獨立影片。clean原快取與時序張量／noise來源分開保存；noise按專用seed42 generator在Train即時產生，不改原觀測、Val不增強。

新門檻與候選依包含unknown整体Accuracy→Fall F1→全體Fall Recall→較少FP，候選最後按固定順序。strict >、K3，門檻平手取較高值，各自校準，沒有共用數值。歷史wide保持原balanced校準門檻.5341113805770874，沒有回溯重校準。

## 正式Validation與分母

全部新候選n_total=23、n_evaluable=23、n_unknown=0，所以可判定二元Accuracy和含unknown整体Accuracy各分母23；Fall F1依可判定TP/FP/FN，全體Fall Recall分母15，正常FP率分母8、unknown率分母23。不是事件數或窗口數當Accuracy分母。

"""
        + "\n".join(table)
        + """

SVM1–5與去噪：Precision14/16、Recall14/15、F1=28/(28+2+1)；SVM6與GRU：Precision14/15、Recall14/15、F1=28/(28+1+1)。基準wide Precision14/14、Recall14/15、F1=28/(28+0+1)。全部FN fall-19；前五SVM／去噪FP adl-37與adl-35，SVM6／GRU只FP adl-35。沒有救回基準漏報，沒有新增其他Fall漏報；新增正常誤報使Accuracy退步。

"""
        + "\n".join(matrix)
        + """

窗口coverage原排程分母2710，不以合法輸入縮分母：wide／六SVM／去噪2287/2710＝84.3911%；GRU2112/2710＝77.9336%。實際window CSV各2917行含多track，同sample_index去重後2710端點；feature行數也可能含多track，不能混用。影片coverage全部23/23，不代表逐幀可判定。

同一新模型改用歷史balanced校準的敏感度（不新增模型，不作選擇）：

"""
        + "\n".join(sensitivity)
        + """

這八組的主要與敏感度Accuracy／F1相同，不能以校準排序不同解釋本輪退步；仍不證明所有模型／協定都等價。

## 已由實驗支持的現象與尚未證實的原因

- fall-19粗phase1468–2369ms有9合法窗口、9合法K3、7個當前良好Pose端點；合法窗口在1500–1766.67ms。wide raw最高.18390694、K3 .13463649，均低於.53411138，raw過門檻0。這個漏報不是「完全沒有骨架」，也不是K3單獨壓掉已過門檻的峰值。後段品質／track gap確實減少覆蓋，但不能因此把整個漏報全部歸因缺骨架。
- wide 的fall-19影片K3分數低於全部8個正常Val。用同一分數與strict門檻救回它，至少8正常片誤報；已驗證單純降低門檻無法同時保住原誤報率。六SVM及两替代AE仍漏同片，這是這八個有限候選的結果，不是所有方法不可行的證明。
- 原30 per_frame七類頭與isotropic異常模型分開。fall-19粗phase原頭FALLING argmax0，不能說分類器早已辨識它。fall-09有13個匹配端點原頭FALLING而wide未警示；其phase有2個raw跨門檻，但最高K3 .50871474仍低於.53411138。因此分類對照與AE表示／重建計分／連續規則存在不同反應；不把七類當AE成績。
- fall-18／30在wide有正式事件，這些影片TP保留；當前良好Pose可見警示0只列展示限制，不回頭否認TP。

表示／骨架品質／正常資料覆蓋不足仍是可能假設，沒有完整人工关節或onset GT，不能宣稱已量化因果。

## 自監督方向的有界診斷與停止

在看結果前固定geometry-rule：若同一fall-19粗phase有至少3連續當前良好Pose端點，觀測30×14×2等比例XY的Train鄰近距離超過正常Val p95、而512D距離在正常Val p95內，才支持「原觀測異常變化可能被表示壓縮」並研究一個微調候選。兩種Scaler及最近鄰都只fit原Train；與Val比較只是診斷，不把Val搬進Train。

原XY normalVal p50=.41046／p95=1.84468；512D p50=.47615／p95=.82296。fall-19九合法窗口原XY距離.34822–.87161、512D .42610–.74694，兩者都在各自正常p95內。支持端點0、最長連續0；**未取得足以支持本輪微調假設的受控對照證據，因此自監督訓練未執行並停止。** 這不是表示正確的證明；條件、資料及結果保留，可在有新診斷證據時另立實驗，不能以完成清單盲目微調。

## 事件時間與展示另列

最佳SVM的六粗phase事件起點命中由wide4/6到5/6、粗正常前段早報仍0；phase有當前良好Pose警示的影片數均2/6。fall-09首次事件5.9→5.0333秒，進入舊AI粗phase，屬時序診斷改善，Accuracy卻由22/23降21/23，不稱準確率提高。以下是來源時間，不是人工onset延遲；完整23片與所有175新事件均保存。

"""
        + "\n".join(timing)
        + """

其他候選也有較早／更多可見phase端點，但帶來誤報，不能只挑成功畫面。舊可見警示guard作展示品質檢查，與正式指標分開，不當教師成績公式。

## 驗證、保全、限制與成果

獨立驗證八模型47,630分數逐值相同、184影片CSV之sklearn指標／2×2／2×3／分母、每窗口K3／状态與175完整事件（起迄／峰值／原因）、Train-only Scaler、來源hash／正常Val最佳checkpoint／Validation門檻全一致；時序provenance與原觀測重新核對無未來、跨影片、跨track或缺失補值。23,925舊run／raw／cache／設定／權重／輸出hash全部不變；原契約hash在封存副本吻合。完成stage及證據拒絕覆寫，不追加候選。

synthetic：47相關風險測試全過，完整回歸狀態見final-qa。real_feature_smoke：三窗真ST-GCN／原七類頭／六SVM與兩AE真實擬合及23Val評估。real_end_to_end：未跑，沒有新RGB Pose或demo／Test；不能把功能測試當完整性能驗收。Test未知、歷史Test已查看，不能稱全新盲測。Validation僅23片且反覆選模、split非人物隔離、預訓練Human含既有Le2i動作／跌倒來源，專案UR-only指新增正常學習資料。AIphase與confidence不當人工GT；CuBLAS deterministic warn_only與跨硬體重現限制保留。Notebook及五分鐘影片繼續暫緩，舊成品保留、不公開上傳。

前置freeze第一次因Pose cache API缺expected參數退出，尚未開始模型擬合；原source與不完整cache／output另存prelock-failed，修正後三窗parity差0才锁定及訓練。沒有覆寫舊run或額外候選。文件apply_patch首次找不到驗收標題時原文件未修改，後續正確標題更新；不影響模型。

成果：新八run位於`runs/ur-ocsvm-01…06-20261012-001`、`runs/ur-denoise-wide-methods-20261012-001`、`runs/ur-temporal-gru-methods-20261012-001`。設定各自JSON、原30iso私有cache與時序NPZ在`data/cache/ur-normal-methods-20261012-001`；鎖定／來源／診斷／驗證／hash／測試在`docs/validation/ur-normal-methods-20261012-001`。正式每候選metrics、sequence/window/event CSV與混淆矩陣在`outputs/ur-normal-methods-20261012-001/<run>`；全部比較CSV、phase JSON及圖位於上層。

![方法比較](../outputs/ur-normal-methods-20261012-001/comparison.png)

![fall19模型分數](../outputs/ur-normal-methods-20261012-001/fall19-scores.png)

```powershell
Set-Location 'C:\\成大專案\\fall-detection-anomaly'
.\\.venv\\Scripts\\python.exe -X utf8 scripts/ur_normal_methods.py status
.\\.venv\\Scripts\\python.exe -X utf8 scripts/verify_normal_methods.py
Import-Csv outputs/ur-normal-methods-20261012-001/candidate-comparison.csv | Format-Table run, overall_accuracy, fall_f1, FP, FN, unknown
```

status／verify為只讀，不訓練／不跑Test；freeze／svm／ae-lock／ae與report在完成後拒絕重跑覆寫。新方法不自動接舊AE demo loader，需另立通用推論／展示驗收才可部署。舊demo預設未改。
"""
    )
    atomic_bytes(target, text.encode("utf-8"))
    print(decision, flush=True)


if __name__ == "__main__":
    report()
