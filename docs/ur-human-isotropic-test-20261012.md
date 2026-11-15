# 最高 Validation 候選的 UR-FALL Test 結果（2026-10-12）

依使用者明確要求，已用 `ur-human-isotropic-001` 跑原始固定 Test。**影片級 Accuracy 19/23＝82.61%，Fall F1＝0.8462**；未達老師文件列出的參考 Accuracy90%／F1 .87。這是現有工程協定的結果，教師官方切分與評估單位仍未確認，不能直接換算課程得分。

## 凍結與執行

此版本在查看本輪Test結果前，依既有Validation最高普通／全體Accuracy21/23選定；Scaler與AE早已完成訓練及凍結，只讀Train24支正常活動的3638個有效窗。這次沒有重訓、重切、校準或改門檻，沒有比較其他候選Test後換模型。

- 模型：`runs/ur-human-isotropic-001`，凍結Human ST-GCN512D特徵＋正常資料AE，`per_frame_isotropic`，30幀／stride1。
- 門檻：**0.4529072046279907**，Validation校準，嚴格大於；每個track連續3個合法窗口形成K3分數。
- 原Test：**8支ADL＋15支Fall，共23支**；協定`urfall-cam0-rgb-sequence-v1`。序列切分沒有更動。
- 使用已準備、具來源hash的真實Pose／ST-GCN特徵快取，實際重新載入凍結Scaler／AE在CUDA計分。此次不重新逐影片擷取Pose／ST-GCN，也不以synthetic或七類分類頭代替Test辨識。
- 執行前保存候選、門檻、23支Test清單、92個相關Pose／feature快取檔、真實權重與41個既有run的hash：[評估前快照](validation/ur-human-isotropic-test-20261012-before.json)。

原實驗未通過「目前合格Pose可見警示」的採用條件，這次使用者授權只評估最高Validation候選；原guard、原保留模型及舊`test_scored:false`歷史報告原樣保存，沒有追溯改寫實驗結論。

## 正式成績

| 影片級指標 | 已有 Validation | 此次 Test |
| --- | ---: | ---: |
| 可判定Accuracy | 21/23＝91.30% | **19/23＝82.61%** |
| 包含unknown全體Accuracy | 21/23＝91.30% | **19/23＝82.61%** |
| 可判定Fall F1 | .928571 | **.846154** |
| Fall Precision | 100%（13/13） | **100%（11/11）** |
| 可判定／全體Fall Recall | 86.67%（13/15） | **73.33%（11/15）** |
| TP／TN／FP／FN／unknown | 13／8／0／2／0 | **11／8／0／4／0** |
| 有效窗口端點coverage | 84.39% | **81.07%** |

Test可判定及全體Accuracy分母均為23，沒有因排除unknown而提高Accuracy；F1＝2×11/(2×11＋0＋4)＝22/26。整支影片unknown0不等於每一影格或時間窗都可判定，亦不等於七類動作Accuracy。

| 真實類別＼預測 | 正常 | 跌倒 | unknown |
| --- | ---: | ---: | ---: |
| ADL，8支 | 8 | 0 | 0 |
| Fall，15支 | 4 | 11 | 0 |

四支漏報都有合法K3分數，分類為正常；沒有無法載入快取而造成的unknown。

| 漏報序列 | 全片最高合法K3分數 | 有效窗口端點coverage |
| --- | ---: | ---: |
| fall-27 | .219817 | 79.37% |
| fall-23 | .306093 | 65.22% |
| fall-21 | .343025 | 61.54% |
| fall-20 | .228193 | 97.53% |

上述分數都是AE重建誤差所衍生的K3分數，**不是跌倒機率**。記錄漏報供審閱，沒有用Test分數回調門檻或品質條件。Validation与Test的Accuracy相差8.70個百分點；不能以先前91.30%的Validation數字冒稱Test成績。

## 驗證與成果

既有評估／unknown分母／校準與Test隔離風險測試 **17 passed，0 skipped**。另以sklearn從原始Test逐序列CSV独立重算Accuracy、Fall F1與混淆矩陣，全部吻合；從窗口CSV最高合法K3重建嚴格門檻決策，23支結果全部吻合，警示有無與事件CSV一致。

正式輸出含2814條窗口記錄、2133筆有效特徵分數及15筆事件；同一影片可能有多track記錄／事件，這些數量不是Accuracy分母。Accuracy始終依23條原始影片結果計算。混淆矩陣PNG已實際查看，與8／0／4／11吻合。

41舊run、92個相關快取檔、真實上游權重、全部核心Python原始碼，以及原Validation結果hash皆在評估後核對不變；凍結模型、Scaler、門檻及split一致。沒有修改核心程式，因此沒有追加與本次風險無關的完整回歸或新增模型測試。[最終重算與hash證據](validation/ur-human-isotropic-test-20261012-verification.json)｜[17項pytest](validation/pytest-isotropic-test-20261012.xml)。

檔案位於 `outputs/ur-human-isotropic-test-20261012-001/`：

- [metrics.json](../outputs/ur-human-isotropic-test-20261012-001/metrics.json)：正式指標、分母、split／recipe與凍結artifact hash。
- [sequence-results.csv](../outputs/ur-human-isotropic-test-20261012-001/sequence-results.csv)：全部23支結果及漏報分數。
- [window-results.csv](../outputs/ur-human-isotropic-test-20261012-001/window-results.csv)、[events.csv](../outputs/ur-human-isotropic-test-20261012-001/events.csv)：窗口與事件時間。
- [confusion-matrix.png](../outputs/ur-human-isotropic-test-20261012-001/confusion-matrix.png)、[outcomes-with-unknown.csv](../outputs/ur-human-isotropic-test-20261012-001/outcomes-with-unknown.csv)：二元與含unknown混淆矩陣。
- [evaluation-summary.md](../outputs/ur-human-isotropic-test-20261012-001/evaluation-summary.md)：核心CLI產出的摘要。

## 限制與操作

這次是既有Test切分的首次新候選評分，**不是全新獨立盲測**：歷史Test已被其他實驗查看，Validation也反覆用於選擇；未保證人物隔離，只有23支Test，不能把此數字直接推廣至其他場景。影片TP僅代表全片有事件，不證明警示時間正確。此輪沒有新逐影格GT、警示延遲驗收或重新擷取影片的端到端展示。synthetic測試只作程式驗證；正式數字來自真實資料快取的凍結AE評估，證據分開。

使用者要求的Test評估已完成，無外部阻塞。新訓練、調參、七類逐影格評估、正式Notebook／五分鐘影片仍未執行，公開上傳未授權。

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
# 核心評估命令；若只是閱讀成績，直接開上述metrics即可。
# 此資料夾已存在，原命令會拒絕覆寫；需要重算時必須另選新輸出名稱。
.\.venv\Scripts\python.exe -m fall_detect evaluate --run runs/ur-human-isotropic-001 --split test --output outputs/ur-human-isotropic-test-20261012-002
```
