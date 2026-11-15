# fall-19可見左手肘座標控制診斷

日期2026-10-13；ID `ur-pose-coordinate-diagnostic-20261013-001`。結果：**AI暫定座標使分數小幅上升，fall-19仍漏報；人工確認座標尚待覆核。** 不採用、不改正式Accuracy、demo或舊成果。

## 實際操作與控制

依使用者授權及[事前契約](ur-pose-coordinate-diagnostic-20261013-plan.md)，選定原Validation fall-19第44–51幀／track1／human13 joint3（人物左手肘、畫面右側）。原人工分類標註把這8幀列為骨架不符合人物，但沒有正確關節座標。AI依原640×480 RGB一次指定可見手肘暫定位置，不依分數搜尋、未經人工確認、不是joint真值。介入前鎖定程式、計畫、8點、模型／Scaler／校準hash。

| 原source幀 | 原XY | AI暫定XY |
|---|---|---|
|44|377.03,315.84|376,354|
|45|375.39,324.18|377,369|
|46|377.36,346.61|377,385|
|47|377.49,357.97|376,401|
|48|378.92,382.42|378,417|
|49|377.49,407.51|378,432|
|50|378.58,417.83|379,448|
|51|379.31,427.91|378,449|

只覆寫深拷貝Pose的8組XY；confidence、bbox、detection confidence、track、timestamps、missing、layout及quality不改。不補其他遮擋或出框關節，不插值、不新增偵測，不調品質。原100觀測與衍生Pose／features分開保存。

原wide30幀per_frame_isotropic、stride1、因果resampler、嚴格真實ST-GCN 512D、Train-only Scaler、AE256/64、MSE、threshold **0.5341113805770874**、strict >及K3全部凍結。無新fit、訓練、checkpoint選擇或門檻校準；只對已存在Validation觀測做介入推論。Train adl-05及正常Validation adl-04各前3個合法窗口是原樣固定控制。

原recipe `713ee59a0853f20641cb7f664df2920359baa0734807acfa5cf86fb87a4e2f5b`；衍生AI sensitivity recipe `3e064d37d0859093e05e36cf04662549eb5c6e6ed33e4a5010a2a65a001b2a2d`。這是刻意以固定舊model測輸入敏感度，不宣稱新recipe能直接部署或與舊model訓練契約相容。

## 實際分數及事件

|指標|原快取／sham|AI暫定位置介入|差異|
|---|---:|---:|---:|
|max raw MSE|0.1839069426|0.2003316581|+0.0164247155|
|max K3 sequence score|0.1346364915|0.1510627866|+0.0164262950（12.20%）|
|raw過門檻窗口|0|0|0|
|K3過門檻窗口|0|0|0|
|合法端點／排定端點|23/71|23/71|0|
|覆蓋率|32.39%|32.39%|0|
|事件／首次警示時間|0／null|0／null|無改善|
|影片判斷|normal（FN）|normal（FN）|仍漏報|
|unknown影片|0/1|0/1|0|

AI K3距離固定門檻仍0.3830485940。10個合法窗口包含介入來源幀，9個raw分數上升、1個下降；下降窗口是available_at=1466.67ms、delta=-0.01079993。其他13個合法窗口特徵及分數完全一致。最後合法窗口仍1766.67ms；兩組正常控制共6窗特徵與分數也完全一致。

![固定AE及K3分數](../outputs/ur-pose-coordinate-diagnostic-20261013-001/score-comparison.png)

### 人工姿態阶段與分母

阶段依「端點時間以前最近的原source frame」之人工整段姿態標籤歸類，非完整30幀全屬該阶段，也不是精確事件GT。全部71排定端點被歸類，23合法端點分母一致。

|姿態阶段（原source幀）|原合法／排定|介入合法／排定|警示|
|---|---:|---:|---:|
|upright1–32|2/4|2/4|0|
|transition33–61|21/29|21/29|0|
|ground62–100|0/38|0/38|0|

transition原本就有合法連續窗口，不能將漏報全歸沒有骨架。ground仍無合法窗口，分數是null，不能填0或宣稱已改善落地後警示。原人工「33–61／62–100」是整段姿態阶段，精確開始與落地時刻仍未知。

`phase-comparison-by-source-frame.json`是報告的完整阶段分母；初步`phase-comparison.json`直接採毫秒區間，兩個量化端點落在區間間隙，因此只涵蓋69點，原樣保留並由reading-note註明，不用於報告分母。圖上色帶只作原source姿態時間示意。

## 支持的結論與未知

- 此特定XY介入會改變同track、含介入幀之ST-GCN表示及AE分數，影響方向並非每窗都增加。
- 就這組AI暫定點位，raw本身也低於固定門檻，故本次不只是K3連續規則未滿足；K3放寬仍不能由這些raw分數跨越原門檻。未因此降低門檻。
- 局部輸入更動不足以修復本片漏報；分數上升不等於偵測改善，不能稱Accuracy提升。
- 尚未有人類正確joint座標，不能稱錯點已被客觀修正、不能歸因唯一原因，也不能排除其他關節、特徵表示、重建誤差或門檻的影響。
- 只研究單joint8幀，未對其他影片／完整Validation做模型比較；手動Validation介入不是可部署Pose演算法。即使日後本片修正後判對，也不能替換正式23影片Accuracy或採用新模型。

原最高Validation `ur-isotropic-wide-20261012-001`仍TP14TN8FP0FN1U0、整體與可判定Accuracy22/23=95.65%、Fall F1=.9655（28/29）、Precision14/14、Recall14/15；正式CSV未修改。本次單片診斷FN1、TP0、unknown0，沒有正常片全片評估，不能報整體Precision/F1改善。demo仍`ur-sample-human-s1-001`；20幀isotropic20/23／.8889及512解析度16/23／.7500是不同歷史實驗，均未採用。

## 驗證與保存

- **synthetic：54passed**，包含點位identity／NaN／越界／假human／hidden不得重建、深拷貝、原confidence不變、Test路徑先拒絕、因果缺失、原快取／時序／K3規則；ruff與format通過。
- **real_feature_smoke：完成**，嚴格真實ST-GCN SHA `1cac9a6da99d5d67da8ab58ee7633c5e923d85fba668673f028792819750a149`、23×512有限及原forward校核、sham特徵最大delta0、兩個Train/Val正常控制最大delta0。Pose来自既有嚴格真實權重cache，本輪不重新跑Pose。
- **real_end_to_end：有限控制推論完成**，原Pose觀測→ST-GCN→固定Scaler／AE→K3／事件的fall-19敏感度；無新RGB Pose全流程、完整Validation重新評分、Test或demo全流程驗收。
- 獨立NumPy scaling、獨立AE forward及手算K3／事件，核對cached、sham、intervention共69分數、213窗口、零事件；逐Pose所有非介入字段及causal source membership／23與71分母一致。
- 12,710舊run、cache、weights、configs、outputs及此片原RGB檔案SHA未變；完整性hash只讀檔案，不將Testcache作numpy載入或Test模型評估。五舊契約原bytes封存，舊299解析度成果manifest含可修改契約以封存副本核對。未宣稱全原始資料庫重新逐檔hash。
- prepare與diagnose防覆寫；AI不能作human匯入、無Test訪問；實際拒絕另存QA。新原始／衍生cache隔離；原split未重分。
- Graph Tier2查詢／trace／coverage後讀源，相關文件無記錄gap，訊號best effort非完整性保證；raw/cache/run設計排除以原檔核對。一次猜測resampling.py路徑不存在，改由graph定位skeleton/windows.py，未宣稱該不存在路徑被驗證。

## 需要的人工介入與命令

人工覆核頁已提供第44–51幀，請確認／移動可見左手肘；看不清選「看不清／不可確認」會記xy=null，不補造。**只需8個點位決策，完成後匯出一個JSON，不重做100幀。** 尚未替使用者作human確認或匯出；human stage=pending。頁面有草稿保存與重新載入，未滿8幀／未填覆核者及勾自述時禁止匯出。原Download與100幀標註保留。

在專案根目錄PowerShell執行：

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
# 只讀結果（不重跑）
.\.venv\Scripts\python.exe -X utf8 scripts/pose_coordinate_diagnostic.py status
# 獨立核對（不訓練、不校準、不讀Test特徵）
.\.venv\Scripts\python.exe -X utf8 scripts/verify_pose_coordinate_diagnostic.py
# 覆核服務；若8810已運作，直接開網址即可
.\.venv\Scripts\python.exe -X utf8 scripts/pose_coordinate_diagnostic.py serve --port 8810
# 開 http://127.0.0.1:8810/review/index.html；匯出後才執行：
.\.venv\Scripts\python.exe -X utf8 scripts/pose_coordinate_diagnostic.py diagnose --human-coordinates 'C:\Users\Huayi\Downloads\fall-19-left-elbow-human-coordinates.json'
```

最後一行只供收到真實人工JSON後使用；目前未執行。每份人工內容產生自己的recipe與human-hash子目錄，已存在會拒絕，不覆盖AI或原成果。`prepare`及無human的`diagnose`已完成，同名再次呼叫會拒絕，請用status／verify，不刪資料重跑。

成果：同名`outputs`（summary、三組scores/windows/events、逐窗delta、8點JSON、圖片、review/index.html）、`data/cache`（獨立sham／衍生Pose／feature）、`runs`（僅固定參照manifest，training=not_run）、`docs/validation`（lock、五舊契約副本、保全manifest、synthetic／independent／UI／拒絕證據）。

不採用；Test未知、未載入特徵／解碼影像／評估，不用歷史Test選設定，不宣稱新盲測、不自行換算課程分數。正式Notebook、五分鐘影片及公開上傳繼續暫緩。
