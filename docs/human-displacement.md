# 保留窗口內人體位移的實驗（2026-09-18）

使用者持續優化授權下，短窗20／15幀兩候選已依事前規則完成並停止：分別86.96%／82.61%，未採用、沒有新增Test。接著另立一個不同的前處理假設，仍只使用Train／Validation，不延伸短窗大小或AE網格。

## 新確認的資訊流失與待驗證假設

目前Human原始前處理將每一影格的骨架x／y各自縮放到[-1,1]，因此移除跨影格的人體整體平移與尺寸改變。以固定骨架加入87像素向下位移的數值反例，ST-GCN points最大差僅3.58e-7、motion最大差7.15e-7。這是`synthetic`數學診斷，不是跌倒辨識成績；既有演算法與上游一致，不能稱為程式錯誤。

假設：同一因果30幀窗口共用x／y範圍，可保留下墜軌跡，讓凍結512D特徵＋正常AE辨識早期下降。它會改變預訓練模型的輸入分布，低信心位置離群值也可能影響整窗；不能預先宣稱有效。

## 明確界線與候選

只新增 `ur-human-displacement-001`。`skeleton.coordinate_normalization: per_window` 明確啟用；預設per_frame保持原行為及全部舊recipe hash。窗口仍30幀，stride1，13→14點與confidence不變，沒有填補缺失、改善／偽造骨架或改顯示狀態；所有端點、有效性及K3必須與原30幀完全一致。新recipe hash、cache、run另存。

Scaler／AE只用原24支Train ADL；128／32 AE、sequence sampling、wd=.001、noise0、seed42、至多100epochs／patience10；正常Validation Loss選checkpoint，Validation balanced sequence accuracy選門檻。候選僅此一組，沒有新增seed／AE或校準目標。

Pilot沿用10支Train／Validation與六段事前粗略時間區間，要求真實pose array hash完全一致、有效／無效窗口記錄逐項一致、512D特徵有限且與原前處理有可量測差異；不讀Test分數。通過才準備70支並訓練。模型選擇沿用Accuracy、F1、全體召回、少FP、同分保留原模型；時間檢查要求命中數不降、早報不增，原三支漏報若改成TP還必須有事件開始於指定區間。

Validation全體／可判定Accuracy≥.90、F1≥.87與時間檢查同時通過，才凍結選定模型做一次Test工程對照；之後不依Test調參。若本候選仍失敗，即結束本次優化工作並报告限制。歷史Test已看過，Validation反覆使用，不能稱為獨立泛化成績。正式Notebook／期末影片仍等待辨識品質。

## 實作與階段紀錄

已保存36個舊run、所有舊設定／manifest、三組來源index及pose／feature共843個檔案hash，另保存5個修改檔案的上一版本。舊實驗程式的原始碼鎖保持原樣，會對新程式版本拒絕重跑；使用本輪status檢查檔案與當前結果。

前處理、位移反例、causal window、缺失／退化姿態、原recipe相容、pose reuse範圍、真實Human權重與既有訓練風險測試46 passed，17.62秒。功能通過不是辨識改善。以下為事前鎖定且已執行的順序；已完成步驟會拒絕重跑覆蓋。

```powershell
.\.venv\Scripts\python.exe scripts/human_displacement.py lock
.\.venv\Scripts\python.exe scripts/human_displacement.py pilot
.\.venv\Scripts\python.exe scripts/human_displacement.py training
.\.venv\Scripts\python.exe scripts/human_displacement.py status
```

## 實際結果：三個新候選共147epochs，未提高Accuracy

10支Train／Validation pilot通過：Pose array hash與有效／無效窗口完全相同，真實512D特徵有限且確實改變；ADL平均coverage比、六段跌倒區間合法窗比均1.0。依原定流程完成70支固定前處理，Test僅產生固定特徵，未計分或用於選擇；Train24支正常影片3638窗，AE完成30epochs，最佳正常Validation Loss為0.101714510（epoch20），門檻0.303333074只由Validation選定。

| 模型／本輪候選 | 新訓練epochs | TP / TN / FP / FN / unknown | 可判定Accuracy | 含unknown整體Accuracy | F1 |
| --- | ---: | --- | --- | --- | ---: |
| 原30幀 `ur-sample-human-s1-001`，保留 | 0（既有38） | 12 / 8 / 0 / 3 / 0 | 20/23 = 86.96% | 20/23 = 86.96% | .8889 |
| 20幀 `ur-human-context20-001` | 34 | 12 / 8 / 0 / 3 / 0 | 20/23 = 86.96% | 20/23 = 86.96% | .8889 |
| 15幀 `ur-human-context15-001` | 83 | 12 / 7 / 1 / 3 / 0 | 19/23 = 82.61% | 19/23 = 82.61% | .8571 |
| 30幀共用座標尺度 `ur-human-displacement-001` | 30 | 11 / 8 / 0 / 4 / 0 | 19/23 = 82.61% | 19/23 = 82.61% | .8462 |

所有數字都是同一固定Validation的8支ADL＋15支Fall，並非新Test。所有23支均有可判定窗口，因此本表两種Accuracy分母均23；這不表示每個影格都可判定。原模型全體跌倒召回12/15=80%，窗口coverage84.39%，仍漏fall-19／18／30。原模型balanced accuracy恰為90%，但它不是課程要求的普通Accuracy，不能拿來宣稱已達90%。

本輪共用尺度補到fall-30，但新增fall-29、fall-11漏報，仍漏fall-19、fall-18，跌倒召回11/15=73.33%。正常Loss較小不能抵銷辨識退步。按事前選擇規則保留原模型；20幀持平亦不替换。三個候選均已停止，沒有追加seed、AE、門檻或依Test挑選，沒有新Test。

### 個案事件與可見畫面必須分開

共用尺度在六段粗略人工區間的命中由2變3，早報仍0；整體保護條件失敗的主因是Accuracy／F1下降。原保存JSON中舊`timing()`工具的scope文字寫「four」，實際傳入與保存的是六個案例；本報告按六列解讀，不改寫歷史凍結檔案。這些區間不是正式逐影格GT，不能推導時間Accuracy。

| 共用尺度案例 | 窗口事件紀錄 | 新鮮影格畫面核對 |
| --- | --- | --- |
| fall-30 | 首次事件1.800秒，落在事前粗略區間內 | 警示期間目前Pose為low_confidence；全片0幀顯示FALL ALERT，不能稱實際警示改善 |
| fall-09 | 首次事件5.900→4.833秒，另有5.100秒事件 | 提前事件均被目前Pose品質遮罩，畫面仍到5.900秒才警示，共8幀；5.5秒UNKNOWN |
| adl-04 | 無事件、判正常 | 暖機後NORMAL，沒有可見誤報 |

`UNKNOWN`保留無法判定語意，沒有為了展示效果延長或偽造FALL ALERT。原模型fall-29重跑仍只有2個可見警示影格。個案倒地時身體出框、Pose品質下降與合法窗口分數不足是已觀察事實；尚不能斷言哪一個因素單獨造成所有漏報。

## 最終驗證與保留

完整回歸174 passed／1 skipped（81.06秒），新Human真實權重測試有執行；唯一skip是歷史預設路徑仍缺四項權重，現在使用的sample-human資源可用。12則既有CuBLAS／上游相容性警告保留，未承諾GPU跨次訓練逐位元一致。Ruff lint通過；format check有一個純換行建議（`windows.py`），其餘76檔符合格式。該行屬訓練前鎖定原始碼，保留精確bytes，沒有為消除格式提示改寫原始碼hash或歷史鎖。

最新四支真實影像推論共503幀、313個可比窗口，分數最大差0；事件與可見狀態吻合，UNKNOWN不沿用分數，H264完整解碼。已目視三支新候選與原模型接觸表，另抽查fall-09的4.83／4.93／5.10／5.90秒；這是流程一致性證據，並非辨識改善。此前短窗三支234幀／105窗的獨立紀錄保留。

重新由每組Train ADL特徵擬合Scaler，mean／scale完全相同；訓練來源hash、Validation門檻來源、23支Validation指標與history均重算吻合。36個既有run、843個舊cache檔案、設定／manifest／真實權重hash皆保留；70支Pose array hash相同，47支Train／Validation窗口全部記錄相同（Test在這個檢查僅核對Pose hash，不分析品質或分數）。三個重跑／Test gate拒絕均實測，沒有新Test檔。新增驗證程式第一次嚴格比較遇到保存指標多一個balanced accuracy欄位，補上相同公式後全部通過，兩份log保留；未改任何訓練、門檻或推論結果。

完整机器證據：[驗證報告](validation/human-displacement-verification.json)、[Validation](validation/human-displacement-validation.json)、[最終測試XML](validation/pytest-human-displacement-full.xml)、[新候選真實影片核對](validation/human-displacement-integration-ur-human-displacement-001.json)。`synthetic`數值反例、`real_feature_smoke`的10支pilot、`real_end_to_end`的訓練與原影像重跑各自保存，不混合宣稱辨識能力。

## 尚未達成與後續方向

目前仍未達普通Accuracy90%，沒有足夠證據保證現有資源一定能達標。歷史Test已看過，Validation反覆用於改善亦有選擇偏差，不能當成獨立泛化成績。Human凍結表示曾用含跌倒的Le2i七類動作預訓練；本專案僅Scaler／AE為UR正常Train-only，沒有改成七類分類，但也不能稱整條管線從未接觸跌倒標籤。老師的逐影片／逐影格計分及切分方式仍未明定，不自行換算課程分數。

本次無外部權重或執行環境阻塞。正式Notebook與五分鐘期末影片依使用者要求繼續暫停；本次短片僅為工程診斷，未公開上傳。

下一輪應先完成Train／Validation跌倒起迄與可見關節標註，把「窗口事件、目前Pose品質、可見警示」並列評估，再決定是否有足夠有效下降觀測支持特徵改善；本輪不能靠縮窗或改正規化補回鏡頭外的關節。可協助提供的資料是新場景、全身持續入鏡的正常活動與安全既有跌倒素材，以及教師評分單位的答覆；任何新資料需在查看結果前固定用途／分組，不拿重切現有Test當新獨立測試。這些是後續建議，尚未實作或證明能提高Accuracy。

## 現在可直接操作

以下只核對已完成結果，不會重訓或覆蓋：

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe scripts/human_displacement.py status
.\.venv\Scripts\python.exe scripts/verify_human_optimization.py
.\.venv\Scripts\python.exe -m fall_detect loss --run runs/ur-human-displacement-001
Start-Process 'outputs/human-displacement-validation/ur-human-displacement-001/fall-30-preview.mp4'
```

新候選影片：[正常活動](../outputs/human-displacement-validation/ur-human-displacement-001/adl-04-preview.mp4)、[fall-09](../outputs/human-displacement-validation/ur-human-displacement-001/fall-09-preview.mp4)、[fall-30未顯示警示](../outputs/human-displacement-validation/ur-human-displacement-001/fall-30-preview.mp4)。保留的原最佳模型：[fall-29重跑](../outputs/human-displacement-validation/ur-sample-human-s1-001/fall-29-preview.mp4)。
