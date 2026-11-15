# fall-19人工左手肘座標控制結果

2026-10-13，接續`ur-pose-coordinate-diagnostic-20261013-001`；人工stage **human-b01d622686b5**。已實際匯入使用者覆核JSON並重算，**分數幾乎不變，fall-19仍漏報；不採用。** 先前[AI敏感度結果](ur-pose-coordinate-diagnostic-20261013.md)原樣保留，pending為當時狀態，本次人工步驟已完成。

## 輸入與固定控制

原檔`C:/Users/Huayi/Downloads/fall-19-left-elbow-human-coordinates.json`，SHA256 `ec742381a71a775c33621c224348bea6d8bca9bfc5884ae334b3bb5e6b38a193`；覆核者唐懷毅，自述時間2026-10-13T06:53:11.301Z、human_self_reported／human_confirmed=true。JSON作標註資料讀取，不當成程式或指令；原bytes封存，未改寫上傳檔。

8幀review決策均驗證來源、split、pose／proposal hash、原timestamp、track1、human13 joint3、原XY／confidence及原影像SHA。第44–50幀7點visible，只改XY；第51幀not_visible_or_uncertain、xy=null，保留原座標，不補造。不可確認是人工可見性判斷，不覆寫本次刻意凍結的原信心值／窗口品質，也不能稱51已確認正確。

|原幀|原XY|人工XY|移動距離px|
|---|---|---|---:|
|44|377.03,315.84|376.82,315.68|0.258|
|45|375.39,324.18|374.83,323.65|0.776|
|46|377.36,346.61|376.82,345.56|1.179|
|47|377.49,357.97|376.82,357.51|0.810|
|48|378.92,382.42|379.81,381.41|1.346|
|49|377.49,407.51|375.83,408.30|1.845|
|50|378.58,417.83|377.82,417.26|0.945|
|51|379.31,427.91|不可確認，保留原點|未介入|

人工選點接近原座標，沒有確認之前AI暫定的下移位置。原整體骨架「不符合人物」標籤不指定哪個joint錯誤；不能由此直接把手肘歸為主要錯點。人類自述7點不是全骨架真值或多名標註者一致性證據。

完全沿用原[事前鎖定契約](ur-pose-coordinate-diagnostic-20261013-plan.md)及source lock：原wide30幀per_frame_isotropic／stride1、嚴格真實ST-GCN512D、原Train-only Scaler、AE256/64、MSE、threshold0.5341113805770874、strict >、K3、原品質／track／時間凍結。不fit、不重訓、不選checkpoint、不校準，不調點位找分數。Train adl-05與正常Validation adl-04各3合法窗為不變控制。本輪不跑完整Validation候選比較或Test。

人工recipe `b01d622686b5d81b6ed4866f6399e9a960862f036880339a5df7fe08c4e3627d`；舊base recipe `713ee59a0853f20641cb7f664df2920359baa0734807acfa5cf86fb87a4e2f5b`。衍生Pose與feature另存；固定舊model刻意測输入敏感度，不宣稱新recipe可直接部署。

## 分數與事件比較

|指標|原／sham|AI暫定點位（未確認）|本次人工選點|
|---|---:|---:|---:|
|max raw MSE|0.1839069426|0.2003316581|0.1837749630|
|max K3|0.1346364915|0.1510627866|0.1351487935|
|raw過門檻窗|0|0|0|
|K3過門檻窗|0|0|0|
|合法／排定窗口|23/71|23/71|23/71|
|合法覆蓋|32.39%|32.39%|32.39%|
|事件／首次警示時間|0／null|0／null|0／null|
|fall-19影片判斷|normal（FN）|normal（FN）|normal（FN）|
|unknown影片|0/1|0/1|0/1|

人工raw最大值微降0.0001319796；K3最大值微升0.0005123019（约0.38%），距固定門檻仍0.3989625871。10合法窗口含介入幀，5個raw增加>1e-6、4個減少>1e-6、1個變化≤1e-6；其他13合法窗口完全不變。原Pose以外的信心、bbox、偵測、track、時間、品質及排程保持一致。兩正常控制的特徵／分數delta=0，sham23×512亦與原cache完全一致。

![原、AI與人工分數對照](../outputs/ur-pose-coordinate-diagnostic-20261013-001/human-b01d622686b5/score-comparison.png)

以端點以前最近原source幀的人標整段姿態分類，原／AI／人工均upright2/4、transition21/29、ground0/38，合計23/71。最後合法窗口1766.67ms不變；落地後無合法分數是null，不填0。無任何警示，不能宣稱警示時間改善。姿態33–61與62–100非精確開始／落地時間真值。

## 結論、驗證與界線

這份人工選點在固定流程下未改善漏報；已完成這次八點覆核的匯入與實際驗證，不需要重做100幀。不能把之前AI分數上升當成「人類確認修正錯點後改善」的證據。就本次點位raw與K3皆低於固定門檻，並非只有K3未滿足；但小幅單joint介入無法排除其他關節、表示、重建誤差、門檻或整體骨架品質的問題，也沒有支持盲目追加另一輪Pose模型／微調。

- synthetic：原54風險測試及source lock保持不變，未改核心程式而未重跑相同網格；新增實際檔案匯入、7visible＋1uncertain处理、同內容重复拒絕及bytes保全檢查。
- real_feature_smoke：嚴格真實ST-GCN SHA `1cac9a6da99d5d67da8ab58ee7633c5e923d85fba668673f028792819750a149`、forward核對／23×512有限／sham与6正常控制一致；無新Pose推論。
- real_end_to_end：有限既有Pose觀測→ST-GCN→固定Scaler／AE→K3事件控制已實跑；不是完整RGB Pose、全Validation／Test或demo驗收。
- 獨立手動K3／事件、NumPy scaling及AE forward核對由`independent-verification-2.json`記錄，包含AI及人工兩stage共138分數、426窗口、零事件，全部通過；本片來源／非XY字段／因果性／分母及12,710舊成果hash一併核對不變。
- 原AI final-qa的55檔含可修改契約，更新前先逐hash確認並封存契約bytes，後續以原檔或副本對照；AI舊report、score、cache、summary、lock均不覆寫。原下載JSON原樣保存。

正式wide最高Validation仍22/23＝95.65%、Fall F1 .9655，正式CSV／混淆矩陣不改；手動單片介入不重算正式Accuracy。demo仍ur-sample-human-s1-001，20幀及512解析度歷史版本保持獨立。不採用，Test未知、未讀特徵／解碼影像／評估；未參照歷史Test挑設定、不宣稱新盲測、不換算課程分數。Notebook／五分鐘影片／公開上傳仍暫緩。

## 檔案與只讀命令

`outputs/ur-pose-coordinate-diagnostic-20261013-001/human-b01d622686b5/`：summary、comparison、coordinates-used、scores、windows CSV、事件JSON、window-deltas及圖。

`data/cache/ur-pose-coordinate-diagnostic-20261013-001/human-b01d622686b5/`：獨立derived-pose、sham-features、intervention-features及metadata；`docs/validation/ur-pose-coordinate-diagnostic-20261013-001/human-b01d622686b5/`：原匯入JSON、來源檢查／圖譜／舊契約副本／核對／防覆寫／final-QA證據。

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
Get-Content 'outputs\ur-pose-coordinate-diagnostic-20261013-001\human-b01d622686b5\summary.json' -Raw
.\.venv\Scripts\python.exe -X utf8 scripts/verify_pose_coordinate_diagnostic.py
```

人工診斷命令已實際執行完成，以下保留作來源紀錄；同內容再執行會拒絕覆寫，請用上述只讀命令。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/pose_coordinate_diagnostic.py diagnose --human-coordinates 'docs/validation/ur-pose-coordinate-diagnostic-20261013-001/human-b01d622686b5/imported-human-coordinates-original.json'
```
