# UR-only AE 正常活動泛化實驗

2026-09-18。延續「繼續優化」，固定原UR split及原640骨架／ST-GCN快取，另建run。Notebook與正式影片暫不更新；歷史Test與重複使用的Validation均非獨立泛化證據。

## 先驗診斷

[特徵診斷](validation/feature-diagnosis.json)僅讀Train／Validation。Train有659×256特徵，256維皆非零變異，標準差約0.098–4.310，中位0.864。各Train ADL平均重建誤差約0.028–0.097，各Validation ADL約0.100–0.305。Validation ADL最大的10維誤差貢獻約15–31%，沒有證據說一兩個接近零變異特徵主導整體誤報。分數及事件使用共用B=1 Scorer；逐維NumPy分析僅診斷，不替代正式計分。

這支持繼續研究正常活動泛化差異，但不能證明AE過擬合是唯一原因。fall-19／fall-29仍缺少部分倒地骨架，正則化不會憑空補回觀測。

## 訓練前固定的三個候選

| run | Adam weight decay | 標準化特徵輸入 Gaussian noise std |
| --- | ---: | ---: |
| ur-ae-decay-001 | 0.001 | 0 |
| ur-ae-denoise-001 | 0 | 0.1 |
| ur-ae-both-001 | 0.001 | 0.1 |

皆為原128/32 AE、sequence均衡抽樣、seed42、learning rate0.001、batch32、最多100epochs／patience10，Train ADL擬合原scaler。擾動只施於Train輸入，目標始終是乾淨正常特徵；Validation、門檻校準、推論不加噪音。噪音使用獨立seeded generator，不改抽樣RNG。既有設定新增欄位預設0，維持原推論及特徵recipe。

去噪AE概念參考[Vincent等，ICML 2008原論文](https://icml.cc/2008/papers/592.pdf)：從受擾動輸入重建原輸入。本次Gaussian標準差與組合是工程候選，不是論文對UR Fall的成功保證。

checkpoint仍只按正常Validation loss；門檻仍按全Validation balanced sequence accuracy（unknown保留各類分母）。三個新候選與既有course-ur-balanced-001比較，依Validation全體accuracy、Fall F1、全體Fall recall、較少FP、較早候選排序；baseline排第一，平手不任意換模型。

完成這三個候選即停止，不追加相近參數追分。只有選定結果同時達可判定Accuracy≥90%、含unknown全體Accuracy≥90%、F1≥0.87，才保存新的Test前凍結紀錄並評分固定Test一次。Test歷史已查看，不宣稱獨立重現。

## 操作

```powershell
Set-Location -LiteralPath 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe scripts/ae_generalization.py status
```

事前設定、來源／程式hash與23個舊frozen run見`docs/validation/ae-generalization-lock.json`。`lock`及已完成的`training`拒絕覆蓋，`test`受Validation gate控制；原始資料、舊run與既有影片保留。

## 已完成的實測

三個新run皆完成100epochs，共300epochs。以下皆為固定Validation：8 ADL、15 Fall，23/23可判定，unknown=0；故本表可判定Accuracy與含unknown全體Accuracy相同。

| run | TP / TN / FP / FN | Accuracy（全體與可判定） | Fall F1 | 正常Val最佳Loss |
| --- | --- | --- | --- | --- |
| 舊course-ur-balanced-001 | 11 / 5 / 3 / 4 | 16/23 = 69.57% | 0.7586 | 0.206712 |
| ur-ae-decay-001 | 12 / 5 / 3 / 3 | 17/23 = 73.91% | 0.8000 | 0.220158 |
| ur-ae-denoise-001 | 7 / 7 / 1 / 8 | 14/23 = 60.87% | 0.6087 | 0.202584 |
| ur-ae-both-001 | 12 / 5 / 3 / 3 | 17/23 = 73.91% | 0.8000 | 0.218929 |

依事前平手規則選定`ur-ae-decay-001`，門檻0.3430730998516083。這只是Validation排名，未推廣為可靠偵測器；90%／0.87目標仍未達。去噪降低正常Val Loss卻降低F1，不能把Loss下降當成辨識改善。訓練曲線中的Train Loss是重建項，去噪候選的輸入受擾動；Adam weight decay不包含在該Loss數值內。

[原始四組結果](validation/ae-generalization-validation.json)、[正式訓練log](validation/ae-generalization-training.log)、[錯誤案例](validation/ae-generalization-errors.json)可查核。Scaler與基準逐位元相同；四組Validation重新以共用Scorer計算皆與保存指標相符；23個舊run及3個新frozen run均未變。

## 影片核對揭露：數字提高不等於警示時機改善

唯一改變影片級判斷的是fall-09。新模型3.63秒警示，但目視人物仍站立；5.30／6.00秒人物倒地時為UNKNOWN。此case在「整段任何合法K3警示」的影片級算法下由FN變TP，**不能宣稱成功在跌倒時警示**。沒有可靠起跌／落地標註，尚不計算事件時機Accuracy、延遲或frame-level F1。

![fall-09目視時間軸](../outputs/ae-generalization-validation/fall-09-timeline.png)

兩支Validation另做真實影像→Pose→ST-GCN→AE→影片推論：fall-09 185幀、adl-04 149幀，共334幀。49個可比特徵窗分數差0，未被即時品質遮罩的顯示狀態與cache一致，事件完全一致。H264播放副本完整解碼、抽幀目視確認。這證明流程一致，並未證明警示準確。[整合證據](validation/ae-generalization-integration.json)。

- [fall-09實驗短片](../outputs/ae-generalization-validation/fall-09-preview.mp4)
- [adl-04正常活動](../outputs/ae-generalization-validation/adl-04-preview.mp4)

此輪沒有新Test評分；命令`test`已實測在gate未過時拒絕，且沒有建立Test前凍結紀錄。歷史舊模型Test仍是全體11/23=47.83%，可判定11/22=50%、F1=0.56、unknown1；不是新候選的Test成績。

## 驗證與尚未完成

完整pytest **148 passed／1 skipped**，skip為原Human四個外部權重缺少，現用YOLO／Kinetics路徑的真實權重測試已執行。12則警告涉及CuBLAS非完全deterministic與上游相容性，不能宣稱跨GPU逐位元可重現。Ruff與70個檔案格式檢查通過。第一次完整測試中的去噪單元測試因float32加減消去誤差容限過嚴而失敗，修正測試相對容限為1e-5後，13項相關測試及最終完整回歸通過；失敗XML保留，不隱藏。

尚未達90%、尚未建立可靠警示時機指標、倒地骨架覆蓋仍不足。正式Notebook與五分鐘影片繼續暫停；僅保留本輪診斷短片。下一個需要解決的工作是Train／Validation起跌與落地時間標註、倒地期間關節／追蹤覆蓋，再另立有限候選；不靠再次微調同組AE或Test門檻追分。教師評估單位／切分規則仍待確認，但不是本輪不能訓練的原因。

只讀查看結果用上方`status`。重播實验模型可另存新影片（避免覆蓋現有成品）：

```powershell
$previewPath = 'outputs/demo/ur-ae-decay-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.mp4'
.\.venv\Scripts\python.exe -m fall_detect demo --run runs/ur-ae-decay-001 --input data/raw/urfall/fall-09 --output $previewPath --overlay simple --no-display
```
