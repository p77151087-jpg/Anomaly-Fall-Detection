# fall-19人工整副骨架覆核：六點控制已實跑，仍漏報

2026-10-13，`ur-fall19-whole-coordinate-20261013-001`。已驗證使用者提供的三幀39項覆核、保留原JSON，並對六個可見錯位點實際重算嚴格真實ST-GCN→凍結Scaler／AE→固定門檻／K3。**K3最大分數上升9.36%，但沒有事件、fall-19仍FN，沒有Accuracy或警示時間改善，不採用。**

## 事前規則與人工證據

依[本輪事前規則](ur-fall19-whole-coordinate-20261013-plan.md)，只接受這份JSON與原定位bundle／RGB／Pose／manifest／時間／track1／human13關節identity。原檔SHA `9b4f3216924a04896578187249f85d7dab4d855068e961620816d516a38f7429`，bytes另存human-review-original.json；標註內容只作資料，不作執行指令。

使用者唐懷毅明示human_self_reported／human_confirmed=true、reviewed_at有時區；三幀44、47、49，各13點，共39項：26 matches_original、6 mislocated_visible、7 not_visible_or_uncertain。是本人自述覆核，沒有多人獨立關節真值驗證。

|影格|可見修正點（human13）|與原點距離px|
|---|---|---:|
|47|5 左腕|10.127|
|47|6 右腕|8.127|
|47|9 左膝|22.888|
|47|10 右膝|5.080|
|47|11 左踝|34.769|
|49|1 左肩|9.135|

44全部13點被本次標為符合原點，沒有替換。這與早先100幀整體pose_check=wrong_or_drift不一致；兩份不同粒度的人工證據均原樣保留，沒有改寫舊分類或宣稱已消除標註歧義。47左髖／右踝與49鼻／左髖／左膝／左踝／右踝共7項不可確認，xy=null，保留原觀測與信心／品質；尤其49右踝的大跳動疑點並未得到可靠修正座標。未要求追加100幀紀錄或補猜遮擋位置。

## 實際控制與成績

從原30幀per_frame_isotropic快取開始，**沒有合併舊手肘介入**。保持原confidence、bbox、track、時間、缺失／品質、stride1／30幀／512D以及原wide AE256/64、Train-only Scaler、嚴格大於門檻 .5341113805770874、K3。沒有fit、訓練、選checkpoint、校準或增加候選。

新recipe `565c5532169725504bcbc09c5bb179f03e46300095509f190ff3f1544d3224f2` 與原 `713ee59a0853f20641cb7f664df2920359baa0734807acfa5cf86fb87a4e2f5b` 分開。新cache另存原不變sham、derived-pose與intervention-features，沒有建立假的AE訓練run或改寫原array hash。

|fall-19限定控制|原wide／sham|本次人工六點|差異|
|---|---:|---:|---|
|max raw MSE|.1839069426|.1838027239|−.0001042187|
|max K3|.1346364915|.1472322047|+.0125957131（+9.355%）|
|raw／K3超門檻窗口|0／0|0／0|不變|
|合法／排程窗口|23/71＝32.39%|23/71＝32.39%|不變|
|unknown窗口records|48/71|48/71|不變，分數空值不是0|
|影片unknown|0/1|0/1|可判定但判normal|
|跌倒影片結果|FN|FN|未救回|
|警示事件／首次警示|0／null|0／null|没有警示時間改善|
|最後合法窗可用時間|1766.67ms|1766.67ms|不變|

7個合法窗含47或49原source而受影響，raw 3升／4降；其餘16合法窗特徵與分數完全不變。K3上升後仍離門檻 .3868791759，不能把局部分數變化當辨識改善。分數為異常重建誤差，不是跌倒機率。

本片二元矩陣（列真實ADL／Fall、欄預測normal／fall）原與介入均`[[0,0],[1,0]]`：TP0、TN0、FP0、FN1、U0。可判定Accuracy **0/1**、含unknown整體Accuracy **0/1**，Fall Recall **0/1**、F1 **0.0**；沒有陽性預測，Precision分母0，記null。這是一片控制，沒有完整正常影片對照；不能由FP0推論整體誤報改善。

## 正式Validation與版本分開

原wide正式CSV原樣保留並從23列重新核對：TP14／TN8／FP0／FN1／U0，二元矩陣`[[8,0],[1,14]]`；可判定Accuracy22/23與含unknown整體Accuracy22/23均95.65%，Fall Precision14/14＝1、Recall14/15＝.9333、F1＝.965517。原CSV SHA `ab85b313a35616edab748bc6ba7237bcb8b8b42a806c89c5f286b6453c841285` 與先前鎖定inventory一致。**本次沒有產生新的完整23片正式候選成績，局部人工介入不替換任何正式列。**

- 最高正式Validation：`ur-isotropic-wide-20261012-001`，仍22/23／.9655。
- demo預設：`ur-sample-human-s1-001`，歷史Validation20/23／.8889；本輪不換模型。
- 已完成20幀isotropic：`ur-short20-isotropic-20261012-001`，歷史20/23／.8889，不與本輪30幀修正混稱。
- 舊人工左手肘：K3 .1351487935、仍FN；本輪六點K3 .1472322047、仍FN。來源／joint／recipe各自保存，不合併後宣稱更佳候選。
- 舊AI手肘移點：K3 .1510627866，但點位未由人工確認；其較高分數不能當修正成功真值。

## 支持的結論與仍未解的原因

1. 本次本人覆核提供六個可見錯位點，修正會改變對應連續窗口中的真實512D特徵及部分AE分數；其他16窗、固定兩正常控制的六窗不變。支持局部輸入敏感度，未證明錯點是漏報的充分或唯一原因。
2. 人工整段transition33–61仍有21/29合法窗，原與修正raw均未過固定門檻，故不能把漏報全歸「沒有骨架」，也不能只歸K3阻擋了已超門檻分數。
3. upright階段2/4合法、ground62–100階段0/38合法；本次僅改可見XY，沒有新增倒地觀測／合法窗，也未解決裁切／遮擋。後段不可確認關節仍是限制，不能說全副骨架已修好或已排除Pose表示问题。
4. AE表示、正常重建方式／門檻與這種局部姿態的異常可分性仍是原因假設；此次没有調門檻、微調模型或證明新異常方法改善。縮短窗、解析度、SVM／替代AE等已有歷史失敗結果，不重複網格。

姿態phase是整段人工階段、以窗端點最近過去source對照，不是精確跌倒開始／落地時間真值。影片判對與警示落在跌倒時間仍須分開；本片兩者皆未改善。

![原與人工修正K3、固定門檻及整段姿態階段](../outputs/ur-fall19-whole-coordinate-20261013-001/k3-comparison.png)

## 驗證與保存

- **synthetic**：本輪新validator／防覆寫／split隔離與舊座標／定位風險79passed；另事件／cache／Train-only擬合隔離13passed，共92。後組使用合成Train／Val／Test，沒有讀真Test。首次79項中的4項因預設沙箱temp無法建立而setup error、75passed，改新workspace basetemp後79passed；原因另記，未以放寬assert通過。5新source lint通過。
- **real_feature_smoke**：嚴格載入真實ST-GCN SHA `1cac9a6da99d5d67da8ab58ee7633c5e923d85fba668673f028792819750a149`、forward verified；sham與原23×512全部相同、有限。原Train adl-05與正常Val adl-04各固定前三合法窗特徵delta0、分數相同；通過後才六點介入。
- **限定real_end_to_end**：已實際從原Pose快取重抽六點介入ST-GCN、固定Scaler／AE、K3與本片事件。不是重新從RGB估Pose、完整23片Validation／Test或demo驗收。
- **獨立核對**：NPZ逐Pose只改6個joint XY、其餘字段與信心全不變；NumPy獨立線性層／ReLU／MSE重算69分數（容差1e-6）、手工因果hold-source／213窗口／K3／零事件與分母全過；另6正常控制分數一致、phase合法2/4／21/29／0/38相同。原正式23列CSV Accuracy／F1／矩陣與分母另核對。
- **保全與停止**：209份有界舊來源／wide根artifact／原三片cache／舊座標控制／定位成果與原source hash核對，四mutable契約更新前bytes存prior-contracts；不是全專案全檔保全聲明。原定位final-qa的40artifact先逐一核對再封存，原final-qa／plan／report保持歷史pending內容，接續結果另立本報告。實際再次prepare／diagnose兩次皆拒絕，當時22份新output／cache檔不變。原正式CSV另對歷史inventory SHA。
- **Graph Tier2**：使用codebase-memory先search／trace／snippet及coverage；索引ready，相關source未記錄缺口。trace仍有歷史副本callee／heuristic錯配，以當前import與exact source核實；outputs／cache／raw／run為設計排除，直接讀原檔。best effort，沒有全graph完成驗證聲明。

沒有fit、訓練／校準，沒有讀取Test特徵／解碼Test影像／評估Test；此介入的Test效果未知，歷史已查看Test不稱新盲測、不作調參依據。正式wide、demo、舊run／資料／權重／hash不改；Notebook、五分鐘影片及公開上傳仍暫緩。這份三幀人工覆核已完成本階段需求，不需再匯出一百個檔案。

## 檔案與可直接執行命令

- `outputs/ur-fall19-whole-coordinate-20261013-001/`：原人工JSON、39項audit、6點corrections.csv、三組分數／窗口／事件、window-deltas、causal-window-sources、summary與比較圖。
- `data/cache/ur-fall19-whole-coordinate-20261013-001/`：隔離derived-pose、sham／intervention features及recipe metadata。
- `docs/validation/ur-fall19-whole-coordinate-20261013-001/`：source／protocol lock、209舊hash、四原契約bytes、synthetic XML、独立核對／正式CSV核對／防覆寫、graph與final manifest。

只讀狀態與重核已完成成果（不重跑介入／訓練）：

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe -X utf8 scripts/whole_pose_coordinate_diagnostic.py status
.\.venv\Scripts\python.exe -X utf8 scripts/whole_pose_coordinate_diagnostic.py verify
```

`prepare --human-review ...`、`diagnose`與`audit_whole_pose_outputs.py`本輪已實際執行；再次建立同目錄會拒絕。獨立核對可直接只讀執行 `scripts/verify_whole_pose_coordinate_diagnostic.py`，不要以原`--output`覆写已有核對。
