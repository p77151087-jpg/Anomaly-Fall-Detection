# Sample 流程實作與比較

2026-09-18。承接使用者「按照他們的流程實作」：Human 人物裁切／AlphaPose → 13點正規化加肩中點 → 位置／位移雙流 ST-GCN → 分類頭前512D → UR Train ADL Scaler／AE → Validation門檻。沒有把七類分類輸出當成AE，也沒有重訓姿態或動作模型。

## 資源與實作

- 與附件同源的[Human專案](https://github.com/GajuuzZ/Human-Falling-Detect-Tracks)，固定commit `7ed2faa4d6147dfd576f58869b6c25545208af35`。原上游Drive連結重試仍失敗；改用[上游Issue117](https://github.com/GajuuzZ/Human-Falling-Detect-Tracks/issues/117)所列公開社群鏡像，另存`weights/pretrained/sample-human/`及`weights/sample-human-manifest.json`。不是作者公布checksum認證；不能證明與同學影片實際模型逐位元相同。
- 三個網路已嚴格載入真實tensor-only checkpoint，未使用隨機特徵。TinyYOLO 34,720,299 bytes；AlphaPose ResNet50 162,603,012 bytes；TSSTG 24,708,522 bytes；完整hash另存manifest。
- 沿用已有Human介接及數值相容前處理；新增預設關閉的`pose.use_track_proposals`。啟用時，依sample的main.py將Kalman預測人體框送入**當下影像**重新做AlphaPose，沒有把預測框或旧骨架當成新觀測。框裁至原圖、排除非有限／過期框，仍要求實際關節品質，沒有跨缺失影格補骨架。
- 舊設定的recipe fingerprint保持不變；新選項為true會產生不同cache識別。修改前程式留在`docs/validation/sample-human-legacy-source/`；舊run、資料、設定、權重manifest保留。
- 取得的是包含Le2i跌倒類別的預訓練表示。Scaler／AE只用固定UR Train ADL，不能稱全流程未接觸跌倒或完全無監督。

## 有限實驗

僅兩候選：`ur-sample-human-s5-001`及`ur-sample-human-s1-001`。共用30幀窗、K3、9/13實測關節confidence≥.2、80%好幀、128/32 AE、sequence sampling、weight decay .001、seed42、max100 epochs／patience10。只差stride5與stride1；鎖定後不追加候選。保留原始序列切分與既有unknown定義，Test不参与選擇。

先在10支Train／Validation影片驗證真實特徵；再固定處理70支影片（Test僅固定特徵處理），訓練／校準兩個AE。選擇同時要求Validation Accuracy／F1不下降、四段既有粗略跌倒時段至少多一段警示、站立前段誤報事件不增加。這四段為工程目視診斷，不能稱官方時間準確率。

課程參考目標90%／.87與時間檢查未達時，停止本輪，不追加Test。歷史Test已查看、多輪Validation亦非獨立評估。正式Notebook／五分鐘影片仍暫停。

## 操作

在專案根目錄使用`.venv`：

```powershell
.\.venv\Scripts\python.exe scripts/fetch_sample_weights.py
.\.venv\Scripts\python.exe scripts/sample_human.py lock
.\.venv\Scripts\python.exe scripts/sample_human.py pilot
.\.venv\Scripts\python.exe scripts/sample_human.py training
.\.venv\Scripts\python.exe scripts/sample_human.py status
```

已完成階段會拒絕覆蓋；`status`只讀核對。實際執行狀態與成績在後續驗收段落記錄，以上描述本身不代表辨識改善。

### 前測實測完成

10支Train／Validation、兩種stride均通過真實特徵前測。兩者共同193個有效窗特徵差0；sample的兩個ST-GCN stream類別與固定上游版本AST一致，附件TwoStream.forward多回傳中間特徵，本專案直接取分類頭前512D並核對分類頭重建一致。舊YOLO recipe hash保持不變。這是real_feature_smoke／介接證據，尚不是Accuracy改善。詳細見`validation/sample-human-pilot.json`及`sample-human-pilot-parity.json`。

### 第一候選完成

`ur-sample-human-s5-001`訓練72 epochs；正常Validation最佳Loss 0.146722727。Validation TP7／TN8／FP0／FN8／unknown0，Accuracy15/23=65.22%、F1=0.63636、全體跌倒召回7/15=46.67%。所有23支均可判定，所以可判定與包含unknown的Accuracy分母相同。固定門檻0.7126387358；只讀檢查原門檻候選，最高operational accuracy仍15/23，未改門檻。四段既有粗略時段檢查0段跌倒期新警示、0次過早警示，未通過採用條件。這是新流程的實測結果，不能稱辨識改善。

## 本輪實測結論

兩候選共110 epochs（72／38），依事前規則選定`ur-sample-human-s1-001`。全部正常Train仍為24支；stride5共728窗，stride1共3,638窗，密集重疊窗不代表新增獨立影片。兩種recipe各完成70支固定特徵处理；Scaler／AE只使用Train ADL。

| Validation 23支影片 | TP | TN | FP | FN | unknown | Accuracy（可判定／全體皆23） | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 舊最佳ur-ae-decay-001 | 12 | 5 | 3 | 3 | 0 | 17/23=73.91% | 0.8000 |
| Human stride5 | 7 | 8 | 0 | 8 | 0 | 15/23=65.22% | 0.6364 |
| **Human stride1（本輪選定）** | **12** | **8** | **0** | **3** | **0** | **20/23=86.96%** | **0.8889** |

改善是固定Validation上少3支正常誤報，跌倒召回仍12/15=80%，漏報`fall-19`、`fall-18`、`fall-30`。新門檻0.6880183816；沒有把AE誤差當機率。只讀檢查已保存的門檻候選，stride1最高全體Accuracy仍20/23，沒有事後換門檻。**balanced accuracy=90%不等於課程所稱Accuracy=90%；本輪整體仍只有86.96%。**

時間檢查：四段既有Validation粗略時段中，新警示開始落在跌倒時段的影片0→2（fall-29為2.633秒、fall-25為2.467秒），過早警示事件1→0。此為有限工程檢查，沒有可靠的逐幀onset ground truth，不報正式警示延遲或時間Accuracy。fall-09直到5.9秒才警示，晚於粗略跌倒時段；fall-25雖內部事件在時段內，畫面只有2.50秒一個影格顯示FALL ALERT，其餘因品質遮罩呈UNKNOWN。骨架仍有高可信度錯點與出框問題，不能稱展示已穩定。

**未達Accuracy90%，不追加Test；F1數值已高於參考.87，但不能因此宣稱課程要求全部達成。** Test gate已實跑並拒絕，沒有before-test freeze／新Test分數。既有歷史Test已看過，多輪Validation亦非獨立評估；本輪改善不能推廣成未知場景準確率。正式Notebook／五分鐘影片未重製，未公開上傳。這輪有限候選已完成並停止。

## 驗證與成果

- 完整回歸160 passed／1 skipped，32.38秒，12則上游相容性／CuBLAS可重現性警告。skip針對保持原狀的舊Human權重路徑；新鏡像真實權重CPU測試已執行。不能宣稱跨次GPU逐位元可重現。相關風險測試23 passed／1 skipped。
- 三支初始Validation診斷片共419影格，285個可比較窗口分數差0，事件相符，所有未遮罩狀態相符；unknown沒有沿用舊分數。H264副本完整解碼。這是real_end_to_end一致性證據，不是額外Accuracy評估。fall-29另作已知命中個案展示，選片原因如實揭露。
- synthetic結構／邊界測試、real_feature_smoke前測、real_end_to_end訓練與真實影片三類證據分開。
- 主要證據：`validation/sample-human-lock.json`、`sample-human-pilot.json`、`sample-human-validation.json`、`sample-human-errors-s1.json`、`sample-human-integration-ur-sample-human-s1-001.json`、`pytest-sample-human-final.xml`。
- 新run：`runs/ur-sample-human-s1-001/`與`runs/ur-sample-human-s5-001/`，包含AE、scaler、門檻、Loss曲線、原始訓練記錄及凍結hash。舊run／設定／cache／資料保留。
- 示範：`outputs/sample-human-validation/ur-sample-human-s1-001/`中的`*-preview.mp4`；包含正常、晚警示、短暫警示等診斷案例，不是五分鐘成品。

查看結果：

```powershell
.\.venv\Scripts\python.exe scripts/sample_human.py status
.\.venv\Scripts\python.exe -m fall_detect demo --run runs/ur-sample-human-s1-001 --input data/raw/urfall/fall-29 --output outputs/demo/sample-human-fall29.mp4 --overlay simple --no-display
```

不必重新下載／重訓即可使用選定run；若重跑正式實驗須另建新run及事前紀錄，不能覆蓋本輪。

最終四片驗證合計518幀、333個可比窗口差0、事件及可見狀態一致。fall-29可見FALL ALERT只有2影格（2.70／2.733秒），同樣仍有品質遮罩造成的閃爍。目視抽查已完成，詳細見`validation/sample-human-verification.json`。28個舊凍結run及原manifest／index hash再次確認未變；2個新run也與凍結hash一致。
