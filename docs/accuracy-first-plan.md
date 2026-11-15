# 先改善辨識效果，再製作正式交件

2026-09-18，依使用者最新指示重新排序工作。上一輪Notebook與影片保留為歷史工程紀錄，暫停更新、重製及公開上傳；不把它們當成已達課程要求的正式交件。

## 達標條件與資料界線

目標為UR Fall Accuracy至少90%、Fall F1至少0.87。教師的官方切分／逐影片或逐影格規則仍未知，使用者亦未收到補充；先沿用固定原始序列協定，同時要求包含unknown的operational accuracy至少90%，避免藉排除unknown達標。官方協定確認後才能宣稱與課程基準直接可比，不自行換算分數。

本輪先在Train／Validation診斷及選擇，保留全部Test历史已查看的限制。不重切、不把Test拿來調門檻；本輪候選未通過Validation目標時，不追加Test評分，也不製作新版Notebook／期末影片。Validation已多次使用，亦不能稱為獨立泛化成績。

## 第一個有證據的改善方向

已觀察到部分Validation跌倒的有效骨架窗口只覆蓋前段。例如fall-19僅5窗，後半段沒有合法特徵。既有YOLO只有原角度影像推論；程式已有將真實影像旋轉、把座標轉回原畫面的路徑，但設定驗證目前只允許Lightweight OpenPose使用。

先驗證YOLO在原角度骨架品質不足時，補做±90度真實影像推論是否恢復躺倒期間骨架。只採用實際模型觀測，不補假關節、不旋轉輸出影片、不修改品質／K3規則。衍生neck不計入品質門檻。這是待驗證方法，不先宣稱有效。

第一階段pilot限定Train adl-05／adl-24與Validation adl-04／adl-20／fall-02／fall-18／fall-19／fall-29。候選僅一個：YOLO原角度品質不足時做[0,90,-90]，其餘參數沿用。比較真實有效窗口、合法連續窗、追蹤中斷與原始影格觀測，不用Test評分。

pilot gate：四支Fall有效窗口總數至少增加30%，四支ADL總數不低於原先90%，且沒有原可判定序列退化為unknown。通過才建立70支新recipe快取（Test僅固定特徵擷取、不評分），再比較事先固定的3個AE候選：window/latent32、sequence/latent32、sequence/latent16；最多100epochs／patience10。僅Train ADL擬合，正常Validation選checkpoint，Validation balanced accuracy校準，候選依全體accuracy、F1、全體Fall recall、較少FP、較早候選選擇。全部另開run及保存hash。

若此方向未通過，不追加同一參數網格追分；以失敗證據檢查特徵表示及預訓練域差異，另立下一個有限假設。加epochs與改善展示不是優先方法。

## 可能需要外部協助

1. 向老師確認評估單位、資料切分及unknown規則；目前不知道不阻塞工程改善。
2. 同學附件已查核，沒有可用的ST-GCN模型權重；若之後取得含來源／訓練資料說明的權重，可作另立候選的研究資源。現有Kinetics ST-GCN已能執行，不能把另一套Human權重缺少說成本轮無法改善的必然原因。
3. 若要驗證警示時間，需可靠的起跌／落地標註；可以先整理Train／Validation待標註片段及格式，之後請使用者協助核對。不要求為湊分而修改Test標籤。

不保證在現有資料與方法下必然達90%；每一步以實際Validation證據決定，若外部資料／權重或標註成为主要瓶頸，明確提出需要的項目。

## 旋轉pilot實測結果

事前鎖定見[lock](validation/accuracy-first-rotation-lock.json)，原始結果見[pilot](validation/accuracy-first-rotation-pilot.json)。真實YOLO與ST-GCN已對2支Train及6支Validation抽取新特徵，屬於`real_feature_smoke`，沒有計算本候選Accuracy。

| 序列 | 原有效特徵窗 | 旋轉候選有效窗 |
| --- | ---: | ---: |
| Train adl-05 | 30 | 30 |
| Train adl-24 | 9 | 9 |
| Validation adl-04 | 24 | 24 |
| Validation adl-20 | 30 | 41 |
| Validation fall-02 | 9 | 11 |
| Validation fall-18 | 3 | 4 |
| Validation fall-19 | 5 | 5 |
| Validation fall-29 | 10 | 10 |

ADL合計93→104，Fall合計27→30；Fall僅增加11.1%，低於事前30%門檻。沒有原可判定序列退化為unknown，但兩支困難Fall仍未恢復更多有效窗。這證明本次有限旋轉方式效益不足，尚不能認定所有旋轉或所有Pose方法無效，也未證明特徵表示是唯一根因。

依停止條件，沒有建立三個候選的AE run、沒有新Test評分、沒有更新Notebook或影片。既有23個frozen run及來源index hash均保留。已加上重複執行保護，已存在pilot／Test產出時拒絕覆蓋。`training`會在pilot未通過時拒絕執行。

實際操作與只讀狀態檢查（已完成的lock／pilot不可重跑覆蓋）：

```powershell
Set-Location -LiteralPath 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe scripts/accuracy_first.py status
```

## 參考論文補查與下一步

第45張引用Galvão等人的[A Framework for Anomaly Identification Applied on Fall Detection](https://www.researchgate.net/publication/351830327_A_Framework_for_Anomaly_Identification_Applied_on_Fall_Detection)（DOI:10.1109/ACCESS.2021.3083064）。II-A用Kinetics400 ST-GCN；II-C寫10次平均；III-B的單資料集採ADL 90%訓練、5%驗證、餘下ADL與全部Fall測試；III-C還寫依Test AUC選Hyperopt候選。本專案禁止Test選模，不照搬這一點。這些是論文設定，未確認老師要求逐項採用。

工程判斷：現有24/8/8 ADL與0/15/15 Fall固定切分不同於論文，分母與訓練量會影響可比性；維持既有資料界線，不重切來宣稱獨立Test。下一輪應先查核時間窗／特徵池化、特徵正規化及AE容量，從Train／Validation證據選一個有限假設。不能因看見論文成績就保證換正規化或更深AE會達標；本輪尚未實作這些候選。

補查也確認：使用者指出的UR_Fall_dataset.zip是結果圖片包，原始資料已在本機`data/raw/urfall`，不能把70張截圖當成補充動作訓練序列。[附件稽核](validation/course-sample-dataset-audit.json)。
