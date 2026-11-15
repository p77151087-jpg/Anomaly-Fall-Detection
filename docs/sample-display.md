# Sample 式七類動作顯示（2026-09-18）

使用者要求移除主畫面的 UNKNOWN。此次將動作顯示改為獨立的原 Human 七類分類分支：左側大字呈現 STANDING、WALKING、SITTING、LYING、STANDING UP、SITTING DOWN、FALLING，未累積30次實際觀測時顯示 PENDING。AE 特徵、品質、分數、門檻、K3、事件和評估分母不變；不是把 unknown 計成正確，也不是把七類分類換算成 AE 成績。

## 顯示契約

- `demo --overlay auto`：真實 Human Pose／ST-GCN 使用新 `sample` 模式，其他 recipe 仍用 `simple`。可顯式指定 `sample`；不相容 recipe 拒絕。`actions` 保留先前受 AE 窗口品質限制的版本，`simple`／`debug`／`presentation` 保留。
- 七類分類只用相同 track 最近30次實際確認的 Pose 觀測，原始 `original30/per_frame` 前處理，再執行已嚴格載入的凍結 Human 七類模型完整 forward。不使用重採樣複製影格，不補骨架，不使用未來影格，不以 AE 分数或姿勢規則猜動作。
- AE 使用 isotropic／短窗口／per_window 時，動作分支仍獨立使用原30幀逐幀前處理，不把不同特徵空間直接套入分類頭。新增 forward 不更新模型參數或 BatchNorm buffers。
- 與 sample 相同，湊滿歷史後取七類分數 argmax，不以当前 Pose 的9/13品質條件抹掉動作標籤。品質不足時顯示小字 `Pose uncertain` 和觀測更新時間；這不代表七類判斷正確或分數是校準機率。
- 原 sample 從 tracker 內部歷史累積，本專案從 Pose backend 實際輸出的 confirmed track 開始，暖機可能較晚；不宣稱完整上游影片逐位元相同。短暫缺幀保留實際歷史，但無當前觀測的畫面不顯示舊動作。超過 `track_max_age` 個來源影格，或對應名義時間上限（現設定1秒），清除該 track 歷史。解析度變化也重新暖機。
- 人物消失時大字留空，小字 `Tracking lost`；幾何退化時 `Pose unavailable`。缺人不硬填 STANDING／LYING。低信心與出框骨架線不畫，原始座標、品質及推論輸入不修改。
- 右欄 `AE EVENT HISTORY` 保存已由原 K3 規則觸發的事件，含 `ALERT RECORDED`、track ID 和首次確認時間。小字 `Current` 另列目前 AE 狀態；目前無法判定時仍明說 unavailable。歷史事件留下來，不等於目前姿勢已確認跌倒。
- 原 `.csv`、`-events.csv` 保留 AE unknown 及原因。`-actions.csv` 增加實際觀測影格清單、起迄時間及 `pose_good`；這裡 `state=valid` 表示分類有合法數值，**不表示 Pose 合格或動作標籤正確**。新增 `-display.csv` 記錄每幀歷史事件數及時間，可對照事件 CSV。

## 實際驗收

核心與介接已完成，39項針對性測試通過，包含真實權重 full forward 對照、AE 特徵／模型 state 不變、低信心分類、實際30觀測、不同 track、缺幀與長間隔、因果新鮮度、K3事件歷史及影片整合。完整回歸 **232 passed／1 skipped／12 warnings**；skip是既有測試的舊預設權重路徑，本輪真實Human checkpoint有實際執行。Ruff check通過，七個變更Python檔format通過。首次執行的7項 setup error 是 Windows 既有 pytest 暫存目錄存取限制，保留錯誤 XML；相同測試在取得目錄存取後全部通過，沒有把錯誤視為通過。[完整pytest](validation/pytest-sample-display-full.xml)。

改版前已保存41個凍結run、8764個cache檔案、44份設定、5份權重manifest、37個舊媒體檔hash，以及三個修改前核心檔案：[快照](validation/sample-display-before.json)。本輪不重訓、不新增Test評分、不改選模規則。

真實驗證已完成六支既有 Validation 影片，共888幀。六組新舊 AE 推論／事件 CSV **逐位元相同**，含559筆可見有效AE分數、167筆AE unknown；其餘為暖機。674筆七類分類逐筆從凍結Pose快取取出其30次實際觀測，獨立重播原模型full forward，七分數最大差0、argmax一致，無未來資料。包含81笔當前Pose品質不足但仍有合法七類數值的分類，品質小字有保留。

| 模型／片段 | 輸出幀 | 七類分類幀 | 當前低品質仍分類 | 無當前動作 | AE事件历史首次時間 |
| --- | ---: | ---: | ---: | ---: | --- |
| 原基準 fall-09 | 185 | 153 | 7 | 2 | 5.9000秒 |
| isotropic adl-35 | 283 | 251 | 0 | 2 | 無 |
| isotropic fall-09 | 185 | 153 | 7 | 2 | 4.9667秒 |
| isotropic fall-19 | 100 | 65 | 45 | 5 | 無 |
| isotropic fall-18 | 65 | 15 | 4 | 20 | 無 |
| isotropic fall-30 | 70 | 37 | 18 | 3 | 1.8667秒 |

每支其餘30幀為PENDING（含來源暖機觀測在網格的持有），不是30次分類失敗；「無當前動作」大字留空並顯示Tracking lost。以上是顯示計數，沒有逐影格動作GT，不能叫七類Accuracy或辨識coverage改善。六支H264完整解碼、尺寸640×596／30FPS／幀數確認，全部接觸表已目視；低品質不畫骨架線但動作仍可顯示。事件歷史每幀核對原事件CSV，時間沒有提前或偽造；fall-09的4.9667秒是既有窗口事件，不是把原5.0秒合格Pose警示變快。

41舊run、8764 cache、44設定、5 manifest、37舊媒體以及三個封存原始碼的hash已逐一核對不變。既有影片覆寫拒絕已實測（exit1），舊實驗source lock不改寫。[真實整合與全部hash](validation/sample-display-integration.json)｜[核心差異](validation/sample-display-core.diff)｜[相關測試](validation/pytest-sample-display-focused.xml)。

動作分支新增一次完整ST-GCN forward，本輪含Pose和渲染的處理速度約5.70–8.68 FPS，輸出影片時間軸仍30FPS；此為離線成品，沒有宣稱能以30FPS即時推論。不同時段執行的舊影片meta較快，未作同條件效能benchmark，不能把整個差異都歸因於單一環節。

## 成績與限制

本次僅改善展示，AE 成績不变：保留預設 `ur-sample-human-s1-001` Validation TP12／TN8／FP0／FN3／U0，Accuracy20/23=86.96%、F1=.8889；最高影片級候選 `ur-human-isotropic-001` TP13／TN8／FP0／FN2／U0，Accuracy21/23=91.30%、F1=.9286，剩餘漏報fall-19／18。兩者可判定與包含unknown全體Accuracy分母皆23，不等於每影格可判定。

isotropic 的 fall-30 在1.8667秒已有真實K3事件；改版留下此事件紀錄，但當前 Pose 的有效性未改善，不能據此回頭宣稱通過原「目前合格Pose可見警示」採用條件。因此最高候選仍另存，沒有取代原保留模型。七類動作尚無完整逐影格GT／Accuracy驗收：實際fall-30下降時仍出現STANDING UP，adl-35正常活動在3.30秒有一幀FALLING（原頭分數約.1768，七類argmax）；AE沒有因此新增事件。移除大字UNKNOWN會更完整暴露原分類器的判斷與錯誤，不能宣稱更準。歷史Test已查看、Validation反覆使用、教師評估單位／切分未明的限制仍在。

synthetic 測試只驗證程式契約；真實checkpoint固定输入測試是 real_feature_smoke；原影片重新Pose→ST-GCN→AE／動作→渲染的六片才是 real_end_to_end。沒有把三者混為辨識效果。此輪顯示工作已完成，無外部阻塞；仍待改善七類錯判、AE漏報、獨立測試與正式GT。正式成果Notebook、五分鐘影片與公開上傳仍依先前指示暫停，沒有以介接通過宣稱課程全部達標。

## 操作

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
# 核對已產出的真實影片、分數、事件與所有舊成果hash（完整hash較耗時）
.\.venv\Scripts\python.exe scripts/verify_sample_display.py
# 對新的影片另存demo；auto會選sample，輸出已存在就拒絕覆寫
.\.venv\Scripts\python.exe -m fall_detect demo --run runs/ur-human-isotropic-001 --input data/raw/urfall/fall-09 --output outputs/my-sample-demo.mp4 --overlay auto
# 亦可使用原保留基準
.\.venv\Scripts\python.exe -m fall_detect demo --run runs/ur-sample-human-s1-001 --input data/raw/urfall/fall-09 --output outputs/my-baseline-sample-demo.mp4 --overlay sample
.\.venv\Scripts\python.exe -m pytest tests/test_sample_display.py tests/test_action_display.py tests/test_streaming_video.py -q
```

短片放在 `outputs/sample-display/<run>/<sequence>-preview.mp4`，同目錄保留原渲染檔與全部sidecar。若要重做影片，使用新的輸出名稱，不刪除舊成果。`verify_sample_display.py --render --write`只供尚無整合報告的首次產出，完成後會拒絕重寫。
