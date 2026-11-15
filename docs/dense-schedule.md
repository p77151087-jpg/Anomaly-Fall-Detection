# UR Fall 密集計分實驗

2026-09-18。使用者要求繼續優化準確率。此輪保留ST-GCN特徵＋正常AE、固定原split、Train ADL scaler／AE及Validation校準。歷史Test已查看，重複Validation亦非獨立泛化證據。

## 診斷與固定候選

舊stride5每166.7ms計算一次30幀窗口，K3需三次連續超標；第一個與第三個窗口端點相隔333.3ms。fall-25在2.133、2.300秒的raw AE score已是0.498、0.976，但K3分數仍僅0.125、0.211，接著骨架中斷。這證明確認規則會抑制該片尾端的短暫高分，不證明所有漏報都只差一個窗口。

固定兩個候選：stride2、stride1；窗口仍30幀、30fps，品質仍0.2信心／12個真實關節／80%好幀，K3不變。第一至第三端點間隔分別133.3ms、66.7ms，是不同事件持續時間的工程取捨，可能增加短暫誤報。demo新鮮度依既有兩個stride規則，亦隨計分頻率縮短。舊設定與舊run皆保留。

新cache從原已驗證Pose重建ST-GCN特徵，不改人物ID、不填補缺失、品質不放寬。允許的Pose cache重用只新增stride變動；品質、追蹤、權重、split或窗口長度更改仍拒絕。先用Train adl-05/24/38及Validation adl-04/20、fall-02/09/19/29/25做10支真實pilot。gate為：共同窗口特徵絕對差≤1e-6、Pose array hash相同、原可判定序列不退化unknown。這是實作一致性gate，不是Accuracy。

通過後每個候選各建完整新cache及1個新AE run，固定上一輪ur-ae-decay-001的128/32 AE、sequence抽樣、weight decay0.001、無去噪、seed42、最多100epochs／patience10。只用Train ADL擬合，正常Validation選checkpoint、全Validation balanced accuracy校準。密集窗口高度重疊，不等於增加獨立影片數；Train仍24支。

## 時間檢查與選擇規則

在看新候選分數前，依四支Validation原始影格建立粗略檢查區間：

| 序列 | 可見站立截止（秒） | 明顯跌倒／落地過程區間（秒） |
| --- | ---: | --- |
| fall-09 | 4.304 | 4.604–5.506 |
| fall-19 | 1.167 | 1.468–2.369 |
| fall-29 | 1.735 | 2.035–2.936 |
| fall-25 | 1.568 | 1.868–2.802 |

來源為本機`outputs/short-window-diagnosis/`的實際時間戳抽幀。這是單一AI目視的粗略工程檢查，區間間隙不判定，未經人工複核，並非官方起跌／落地ground truth，不計算時間Accuracy或延遲。已出現的提早警示不能因延續到跌倒區間而算正確：僅計算在區間內新開始的事件。

候選須比舊模型多至少一支在上述跌倒区間新開始警示，站立截止前事件總數不增加，且Validation全體Accuracy與F1不低於舊模型17/23、0.80，才進入候選排名。通過者連同舊模型依全體Accuracy、F1、全體Fall recall、較少FP、較早候選（舊模型先）選擇。

固定兩個候選完成即停止，不繼續網格追分。只有通過時間檢查且同時達Validation可判定與全體Accuracy90%、F1 0.87，才凍結新模型後評分Test一次。Test只可在先前完整cache製作時提取固定特徵，不參與擬合與選擇。正式Notebook與五分鐘影片仍暫停。

## 命令

```powershell
Set-Location -LiteralPath 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe scripts/dense_schedule.py status
```

lock、pilot、training與test皆有保留既有結果的檢查。原始證據檔名前綴`docs/validation/dense-schedule-`；pilot標為`real_feature_smoke`，正式訓練與評估才標`real_end_to_end`，synthetic測試不列辨識效果。

## 實際結果：未提高整體Accuracy

兩個候選各完成10支pilot，共同有效窗口92／179個特徵差皆0，Pose array hash相同，原可判定序列未退化unknown。接著各完成70支固定特徵準備及70epochs AE訓練，共140epochs；正常Train特徵659→1630／3261，來源仍24支ADL，Scaler分別只擬合各候選的Train ADL。Test僅固定特徵準備，未拿來選模型／門檻、未評分。

| 模型 | TP / TN / FP / FN | Validation Accuracy | Fall F1 | 粗略跌倒區間內有新事件的案例數 | 四案例站立截止前事件數 |
| --- | --- | --- | --- | ---: | ---: |
| 保留ur-ae-decay-001（stride5） | 12 / 5 / 3 / 3 | 17/23 = 73.91% | 0.80 | 0 | 1 |
| ur-stride2-decay-001 | 9 / 7 / 1 / 6 | 16/23 = 69.57% | 0.72 | 1 | 0 |
| ur-stride1-decay-001 | 9 / 7 / 1 / 6 | 16/23 = 69.57% | 0.72 | 1 | 0 |

三組Validation皆8 ADL／15 Fall，23支可判定、影片級unknown0，所以可判定與含unknown全體Accuracy相同；這不代表每一幀皆可判定。兩個新候選雖少2支正常誤報，卻多3支跌倒漏判，未通過包含Accuracy／F1要求的採用條件；不取代舊候選，未達90%／0.87。

選定門檻分別0.7115532159805298、0.804909348487854（不同run的AE score不能當成機率或直接跨run比較）。以已保存的Validation門檻候選作只讀診斷，兩個新模型最高全體Accuracy也只有16/23；沒有藉這項診斷改動已凍結門檻。

完整數字與逐序列案例：[評估結果](validation/dense-schedule-validation.json)、[重新計算的Validation明細](validation/dense-schedule-errors.json)、[原始訓練log](validation/dense-schedule-training.log)。四個時間區間只是事前固定工程檢查，不是整體事件Accuracy。

## 真實影片核對

使用stride1模型重新從影像執行三支Validation：fall-25／fall-09／adl-04，共419幀。284個未被即時品質遮罩的可比特徵窗分數差0、狀態相符、事件相同；H264播放副本完整解碼，抽幀目視通過。

fall-25在2.267–2.333秒連續3個顯示影格出現FALL ALERT，畫面確實處於向前跌落過程，隨後因骨架品質／追蹤不足回到UNKNOWN。fall-09原本3.63秒站立時的提早警示消失，但本輪整支仍漏判，不能把「沒有提早亂報」當成「成功偵測跌倒」。部分人物倒地後超出影像邊界，無法只靠調整計分頻率補回画面外的關節。

- [fall-25：短暫正確時段警示](../outputs/dense-schedule-validation/fall-25-preview.mp4)
- [fall-09：提早警示消失，但仍漏判](../outputs/dense-schedule-validation/fall-09-preview.mp4)
- [adl-04：正常活動](../outputs/dense-schedule-validation/adl-04-preview.mp4)
- [整合核對原始證據](validation/dense-schedule-integration.json)

## 驗收與後續界線

36項相關風險測試及完整**153 passed／1 skipped**通過，完整測試37.08秒；skip只為原Human四個權重缺少，現用YOLO／Kinetics路徑已驗證。12則警告涉及CuBLAS非完全deterministic及上游相容性；不宣稱跨環境逐位元可重現。Ruff與70檔格式檢查通過。26個既有frozen run、2個新run及來源index hash核對保留。

`test`命令已實測拒絕未達Validation門檻的執行，沒有建立新Test凍結紀錄。[驗證總表](validation/dense-schedule-verification.json)、[完整pytest](validation/pytest-dense-schedule-final.xml)、[Test gate](validation/dense-schedule-test-gate.json)。Notebook及正式五分鐘影片仍暫停；無訓練留在背景。

目前最高影片級Validation仍73.91%／F1 0.80，且有前輪已揭露的警示時間限制；歷史Test全體11/23=47.83%、可判定11/22=50%、F1=0.56、unknown1，並非本輪新模型成績。90%目標、可靠跌倒事件時機、官方評估協定確認仍未完成。

此輪排除了「只加密計分就能提升整體Accuracy」這個有限假設，未證明所有時間／特徵方案無效。下一個研究方向應是正常活動與跌倒的特徵分離：例如事前限定的多時間尺度或保留軀幹傾斜／下降速度的表示，仍接正常資料AE；尚未實作或證明有效。可靠的起跌／落地人工標註可協助比較警示時機，教師未知事項不應被當成本輪無法訓練的理由。
