# 大容量候選 UR-FALL Test 評估（2026-10-12）

依使用者明確要求，只評估已凍結的 `ur-isotropic-wide-20261012-001` 一次，**Test Accuracy＝18/23＝78.26%，Fall F1＝20/25＝.8000**。沒有重訓、重切、改門檻或依Test選模。本次授權補充前輪完成後「不跑Test」的範圍，不修改其歷史lock或採用guard。

## 凍結條件與執行

- 真實 Human Pose／ST-GCN 等比例512D特徵＋正常Train AE；hidden256／latent64，已完成34epochs。Scaler／AE只用原24支Train ADL／3638窗；checkpoint只用正常Validation Loss。
- 門檻固定為 **.5341113805770874**，原Validation balanced sequence Accuracy校準；strict `>`、連續3個合法窗，原30幀／stride1與品質規則不變。
- 原Test為8支ADL＋15支Fall，共23支；協定 `urfall-cam0-rgb-sequence-v1`。split hash `0b9a0f0cf3609c478afcbbe028130c7d519e26b573135fbd739c074c68bc93c5`。
- AE hash `afdb6f1a37bb1ada3e13cffe10bc0c23466f0fcf908e567840c46f3bf5ad300b`，Scaler hash `a6dd9dbb68f85fa3bd6f358998d341d4b7d40317597e35db592f71ee4007eb19`；完整模型／設定／split／門檻hash見正式metrics及評估前快照。
- 使用專案 `.venv`，核心CLI實際執行、exit code0；CUDA對真實快取作新AE推論，沒有重新抽取Pose／ST-GCN。

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe -m fall_detect evaluate --run runs/ur-isotropic-wide-20261012-001 --split test --output outputs/ur-isotropic-wide-test-20261012-001
```

輸出已存在，同一路徑會拒絕覆寫；本次不再次執行評估。可直接讀取保存結果：

```powershell
Get-Content outputs/ur-isotropic-wide-test-20261012-001/evaluation-summary.md
Import-Csv outputs/ur-isotropic-wide-test-20261012-001/sequence-results.csv | Format-Table
```

## 正式結果與分母

| 項目 | 結果／定義 |
|---|---|
| TP／TN／FP／FN | 10／8／0／5 |
| 影片unknown | 0；可判定23／總23 |
| 可判定二元Accuracy | (10+8)/23＝78.2609% |
| 含unknown全體Accuracy | (10+8)/23＝78.2609% |
| 可判定Fall F1 | 2×10/(2×10+0+5)＝.8000 |
| Fall precision | 10/(10+0)＝100% |
| 可判定／全體Fall recall | 10/15＝66.6667%；Fall unknown0，所以二者相同 |
| 序列coverage／有效窗口coverage | 100%／81.0718% |

8支正常活動全部判對，15支跌倒辨識10支、漏5支。影片unknown0不表示整片每個影格都有合格骨架。

| 漏報序列 | 最大合法K3分數 | 有效端點／排程端點 |
|---|---:|---:|
| fall-27 | .173377573 | 50／63 |
| fall-23 | .284247309 | 30／46 |
| fall-21 | .307647467 | 16／26 |
| fall-20 | .213687986 | 79／81 |
| fall-03 | .469225764 | 186／188 |

五支均有合法K3分數，但未跨凍結門檻；這是現象描述，不回用它們調門檻、模型或品質規則。

## 與已保存成績對照

| 已評估版本 | Validation Accuracy／F1 | Test Accuracy／F1 | Test漏報／誤報 |
|---|---|---|---|
| 舊 isotropic | 91.30%／.9286 | 82.61%／.8462 | 4／0 |
| 大容量 wide | 95.65%／.9655 | **78.26%／.8000** | 5／0 |

大容量的Validation增加1支判對，但此Test少1支判對，新增漏報fall-03。表格只對照既有保存結果，沒有新增其他候選Test或據此改選模型。**Validation提升沒有延續到這組Test，不能宣稱泛化改善。** 正常重建Loss、單一候選最高Validation或課程數字達標都不能替代實際Test表現。

預設仍 `ur-sample-human-s1-001`。大容量先前因新TP無當前良好骨架警示而未通過採用guard，此結論本次保持不變，沒有改寫為由Test決定採用／淘汰。

## 驗收、證據及限制

13項相關風險測試通過，0 skipped；使用synthetic cache驗證凍結評估、Scaler／訓練隔離、快取、K3與unknown，是程式證據。正式數字為 `real_feature_smoke`：真實影片已保存的Pose／ST-GCN特徵上，實際凍結AE推論。這輪沒有新 `real_end_to_end` 原影片重抽或新demo，不將synthetic通過當辨識效果。

sklearn從23條逐影片CSV重算Accuracy、F1、precision、recall及混淆矩陣全部一致；獨立從2814條窗口記錄／2133個有效raw scores重建每支影片最高合法K3及嚴格門檻，23支完全一致，事件有無相符。11筆事件不是Accuracy分母。混淆矩陣PNG已目視確認為ADL行8／0、Fall行5／10。

評估前後 **1103個既有檔案hash全部不變**，含44個舊凍結run、原設定／split、相關快取、4組真實權重、核心Python、前次Test輸出、前輪Validation lock／結果／驗收。指定模型、Scaler與Validation門檻完全相同；核心程式未修改，沒有新增訓練或校準。

歷史Test已多次查看、Validation反覆選模，這是原Test切分的一次指定候選評估，**不是新的獨立盲測**。原切分未保證人物隔離，只有23支Test；教師逐影格／逐影片及官方split仍未確認，不能直接換算課程得分。此工程Test低於課程資料中的參考Accuracy90%／F1 .87。影片級TP只代表全片有事件，不證明警示時間正確；AE重建分數不是跌倒機率。

指定評估與驗收已完成，無外部執行阻塞。沒有以此Test開始下一輪優化；正式Notebook／五分鐘影片仍依前述辨識品質優先指示暫停，沒有公開上傳。

## 成果

- [正式metrics](../outputs/ur-isotropic-wide-test-20261012-001/metrics.json)、[23支逐影片結果](../outputs/ur-isotropic-wide-test-20261012-001/sequence-results.csv)。
- [窗口](../outputs/ur-isotropic-wide-test-20261012-001/window-results.csv)、[事件](../outputs/ur-isotropic-wide-test-20261012-001/events.csv)、[含unknown矩陣](../outputs/ur-isotropic-wide-test-20261012-001/outcomes-with-unknown.csv)、[混淆矩陣圖](../outputs/ur-isotropic-wide-test-20261012-001/confusion-matrix.png)、[核心CLI摘要](../outputs/ur-isotropic-wide-test-20261012-001/evaluation-summary.md)。
- [評估前hash／授權快照](validation/ur-isotropic-wide-test-20261012-before.json)、[獨立重算及hash驗證](validation/ur-isotropic-wide-test-20261012-verification.json)、[13項pytest XML](validation/pytest-wide-test-20261012.xml)。
