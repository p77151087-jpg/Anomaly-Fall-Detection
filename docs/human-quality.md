# 骨架翻轉推論與正常活動抽樣（2026-09-18）

依使用者「好，幫我處理」完成本輪準確度優化，**沒有提高Accuracy**。翻轉骨架pilot未通過，停止該候選；正常型態抽樣AE實際完成33回合，Validation仍20/23＝86.96%、F1=.8889。依事前平手規則保留 `ur-sample-human-s1-001`，沒有追加Test。

## 已完成的診斷

已讀取39個既有frozen run與1165個來源cache的hash，保留設定、權重manifest與舊原始碼；新實驗只寫新目錄。抽查Train／Validation正常活動，`adl-35`為伏地挺身一類地面活動、下半身部分出框，Train `adl-31`有相似地面活動；`adl-11`為沙發坐下／側躺，`adl-40`為地面運動。因此不能聲稱Train完全缺少這些動作。

Train3638個有效窗口的七類頭预测為Standing831、Walking1395、Sitting733、Lying167、Stand up334、Sit down106、Fall Down72。這是模型預測，**不是人工動作標註，也不作為本輪抽樣或AE的訓練標籤**。正常影片也可能被七類頭判成Fall Down。抽查只涵蓋最高AE誤差附近的四張影像，不是完整逐影格標註。

已知三支漏報fall-19／18／30在預先鎖定粗略跌倒區间，當前Pose合格端點分別7／14／10，總31；有效特徵窗口9／16／16，總41。人工粗略時段不是正式事件GT。此輪先測「目前可見關節是否能更穩定」，不靠降低骨架品質門檻或複製遺失關節。

證據：[正常窗口統計](validation/human-flip-normal-diagnosis.json)、[正常影像抽查](validation/human-quality-normal-review.json)、[跌倒時段診斷](validation/human-quality-phase-diagnosis.json)。

## 事前固定的兩個獨立候選

1. `ur-human-flip-001`：選配AlphaPose水平翻轉熱圖平均。只用同一目前影像的原始／翻轉crop，還原COCO17左右對應後平均，再切13點、decode、NMS。保留原單次推論預設與舊recipe hash；啟用後有新recipe/cache。這對照上游`InferenNet`的平均算式；sample實際ResNet50快速路徑沒有此平均，不能稱為修正sample重現錯誤。
2. `ur-human-patterns-001`：保留原骨架與512D特徵。僅用Train ADL經Train Scaler標準化的特徵做固定KMeans8，seed42、n_init10、max_iter300；每支Train影片總抽樣機會相等、同影片內各有樣本的群組機會相等。群組是特徵型態，不是七類動作。AE輸入仍是ST-GCN特徵，重建目標仍是正常資料；K3、品質與MSE判斷不變。

翻轉先跑12支Train／Validation pilot：六支ADL平均有效窗coverage至少基準95%，原三個漏報粗略跌倒區間當前合格Pose端點至少35（基準31、+10%且至少+3），有效窗合計至少41、全部保留合法K3，才訓練其AE。此階段僅`real_feature_smoke`，不計Accuracy。

各允許一個AE，最多100回合／patience10，架構128/32、wd=.001、noise0、seed42；Train正常資料擬合Scaler／AE，正常Validation選checkpoint，全部Validation按balanced_sequence_accuracy校準門檻。按整體Accuracy、F1、全體跌倒召回、較少FP排序，平手保留基準；不可有Accuracy／F1退步、unknown或粗略早報增加，新補漏報須在粗略跌倒區間有可見AE警示。未達90%普通Accuracy／F1 .87及時間檢查，不追加Test。

不組合兩候選，不追加cluster數、翻轉shift、confidence門檻或seed搜尋。歷史Test已查看、反覆使用Validation的偏差持續揭露。[事前規則](validation/human-quality-lock.json)、[實作與設定hash](validation/human-quality-implementation.json)。

## 實際結果與驗收

翻轉pilot已完成12支真正影像推論：三支原漏報的跌倒區間當前合格Pose端點仍31，未達35；有效特徵窗合計仍41，fall-18由16→18、fall-30由16→14互相抵銷。六支ADL平均coverage比值0.96584，雖高於95%保留下限，仍略降，未達整體gate。沒有該候選AE、Calibration或Test成績，不能把pilot稱為辨識改善。[原始pilot](validation/human-quality-pilot.json)。

正常型態抽樣候選完成33回合，最佳正常Validation Loss為.14262269（epoch23），高於基準.13753477；不能將Train Loss下降當作更會辨識跌倒。Train-only Scaler與舊版逐位元相同，分群中心／assignment／抽樣權重全部重算一致。

| 正式Validation版本 | TP／TN／FP／FN／unknown | 二元Accuracy | 全體Accuracy | 二元F1 | 全體跌倒召回 |
| --- | --- | --- | --- | --- | --- |
| 原基準，保留 | 12／8／0／3／0 | 20/23＝86.96% | 20/23＝86.96% | .8889 | 12/15＝80% |
| 正常型態抽樣，未採用 | 12／8／0／3／0 | 20/23＝86.96% | 20/23＝86.96% | .8889 | 12/15＝80% |

固定Validation為8ADL＋15Fall；兩者unknown0，因此可判定及全體Accuracy分母皆23，F1為可判定二元結果，全體跌倒召回分母15。有效窗口coverage84.39%，影片可判定不等於每幀可判定。原漏報`fall-19`、`fall-18`、`fall-30`不變，沒有誤報。

`adl-35`仍是形成校準門檻的正常困難個案，最高合法K3從.68801838變.76480126。三支漏報在新模型的最高合法K3／門檻比值約.172／.375／.540，皆未超過1，沒有證據支持這個固定抽樣方法改善分離。分群不是語意標籤，也不代表已補足所有正常姿態。

26項相關測試及**完整210 passed／1 skipped／12 warnings**通過；skip僅歷史預設Human權重路徑，新鏡像真實權重有執行。[完整XML](validation/pytest-human-quality-full.xml)。

四支真實串流診斷共723幀、526個可比分數差0，窗口狀態與事件吻合，H264全片解碼並抽幀目視。原模型fall-09的AE CSV／事件與修改前逐位元相同；有既有對照的ACTION CSV也相同。新候選fall-09可見警示仍5.9秒、fall-30仍無警示，adl-35無AE誤報；七類頭在fall-30下降時仍會顯示STANDING UP，不把七類輸出當AE成績。[整合證據](validation/human-quality-integration.json)。

39舊run、1165原cache、設定／manifest／模型hash保留，新1個完成run與翻轉pilot另存。新權重manifest獨立保存，原manifest未被prepare改寫。synthetic風險測試、`real_feature_smoke`翻轉pilot、`real_end_to_end`正式訓練／評估／串流分開。

成果檔案：

- [逐影片Validation、時間檢查及選模結果](validation/human-quality-validation.json)，另匯出`outputs/human-quality-validation/<run>/`的metrics、sequences、windows、events。
- `runs/ur-human-patterns-001/`：33回合Loss、AE、Scaler、凍結設定；`sources.json`含Train-only分群與抽樣完整provenance。
- [正常資料人工抽查清單](../outputs/human-quality-diagnosis/normal-activity-review.csv)及同資料夾五份原始影像接觸表；每支只有四张抽查圖，**不能當正式動作／關節GT**。`adl-37`也已目視為正常側躺於床。
- `outputs/human-quality-streaming/`：四份診斷影片、推論／事件／動作CSV、H264副本與接觸表；不是正式五分鐘影片。
- [最終保存與拒絕重跑／Test gate檢查](validation/human-quality-verification.json)。

90%普通Accuracy仍未達，不能以balanced accuracy90%代替。歷史Test已查看、反覆Validation限制不變，教師切分／計分單位仍未確認，不自行換算課程分數；沒有新外部資源阻塞。已確認影像出框與骨架觀測不足，但未證明它們能完整解釋全部漏報。新外部正常影片、完整可見關節及逐幀動作標註尚未取得；不能假裝已完成這些資料補充。正式Notebook／五分鐘影片仍暫停。

## 操作

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe scripts/human_quality.py status
.\.venv\Scripts\python.exe scripts/verify_human_quality.py
.\.venv\Scripts\python.exe -m fall_detect demo --run runs/ur-sample-human-s1-001 --input data/raw/urfall/fall-09 --output outputs/quality-review-best.mp4 --overlay auto
```

前兩條為唯讀核對；最後一條另存保留模型的demo，輸出存在時拒絕覆寫。已完成的`freeze`、`pilot`、`training`不可重跑覆蓋；`test`因未達目標必須拒絕。舊實驗的source lock不改写；目前來源核對請用本輪命令。無新公開上傳。
