# 23支Validation：影片判定、警示時機與骨架品質盤點

本輪ID：`ur-validation-alert-quality-20261013-001`。已完成既有23支Validation紀錄整理、獨立核對與三段人工覆核材料。只讀既有成果，沒有訓練、特徵／模型推論、校準、Test影像／快取／成績訪問；沒有換模型或提升Accuracy。

## 版本与指標

|版本|角色|整體／二元Accuracy|Fall Precision|Fall Recall整體／二元|Fall F1|TP/TN/FP/FN/U|
|---|---|---|---|---|---|---|
|ur-isotropic-wide-20261012-001|最高Validation參照；30幀isotropic、256/64 AE|22/23＝95.65%／22/23|14/14＝1|14/15＝93.33%／14/15|28/29＝.965517|14/8/0/1/0|
|ur-sample-human-s1-001|目前demo預設；30幀per_frame、128/32 AE|20/23＝86.96%／20/23|12/12＝1|12/15＝80%／12/15|24/27＝.888889|12/8/0/3/0|

unknown影片均0，兩Accuracy分母都是23。F1分母＝2TP+FP+FN；另列把Fall unknown算漏報的整體F1，本次unknown0故相同。wide矩陣[ADL,Fall]×[normal,fall,unknown]＝[[8,0,0],[1,14,0]]。唯一漏報fall-19；demo另漏fall-18／30，不能把wide警示紀錄說成demo實際警示。

wide合法唯一窗口端點2287/2710＝84.39%；所有track窗口紀錄2917筆，不拿2917作unique endpoint分母。共14事件、14 TP，8 ADL零事件，序列unknown0與窗口invalid分開。模型數值門檻固定.5341113805770874、strict >、K3，不是跌倒機率。

完整逐片表、15片時序表及14事件明細：[23支紀錄](../outputs/ur-validation-alert-quality-20261013-001/validation-records.md)；[可篩選CSV](../outputs/ur-validation-alert-quality-20261013-001/validation-23.csv)。來源原幀品質、因果採樣品質、每窗raw／K3／unknown原因、每事件最後active與關閉原因各自另存CSV。

## 影片判對不等於時機或展示通過

原六片coarse guard維持4/6在phase內開始事件、2/6在phase內有當前good的警示、粗early0；後來15片AI取樣phase較寬，13/15在possible開始事件（14 TP中13/14）、6/15在core開始（14 TP中6/14）、11/15在possible有good active（14 TP中11/14）。7片首次位於AI不確定邊帶；fall-14首次1.533秒比AI possible末端1.468晚約65ms。這些都是AI診斷比率，不是人工時間準確率或教師計分公式，也不替代舊guard。

good是9/13關節confidence≥.2的原工程條件，不保證點位正確／人物可見；同track因果原幀品質僅作active窗口端點展示診斷，未重跑RGB renderer逐幀可見警示验收。事件關閉end_ms也不等於last_active_ms，不能把整個事件關閉區間視為持續可見警示。

fall-19沿用已完成本人整段姿態phase：upright合法2/4，transition21/29，ground0/38。raw max .18390694與K3 max .13463649均低固定門檻，零事件／FN；跌倒過程確有連續合法窗口，不是全片沒有骨架。本輪不重做100幀或已完成的joint覆核；既有phase仍沒有精確開始／落地界限，其他14片也未有人工作此時間界限。

![既有23支影片窗口、警示端點與覆蓋](../outputs/ur-validation-alert-quality-20261013-001/validation-timeline.png)

## 三段人工確認：只需三筆、一份JSON

覆核頁：`http://127.0.0.1:8830/review/index.html`，目前已在in-app browser開啟。也可直接用本機瀏覽器開[本機HTML](../outputs/ur-validation-alert-quality-20261013-001/review/index.html)，維持同層frames資料夾即可；草稿以bundle hash隔離。

|片段|原幀／原時間|為何確認|
|---|---|---|
|fall-18|1–65／0–2.135秒|影片TP、首次1.433秒，但只有一個active端點且當前good=false；合法16/36、good原幀40/65。確認跌倒／落地時刻與主要錯點或遮擋。|
|fall-09|103–185／3.403–6.139秒|首次5.900秒超出舊coarse 4.604–5.506，仍在較寬AI possible 3.904–6.139內；需分辨跌倒過程、首次落地與後續躺好，不能直接宣稱已落地後多少秒。|
|fall-30|1–70／0–2.302秒|首次1.866667秒，唯一active端點當前good=false；距AI possible末端1.868僅1.333ms。確認時刻，避免把格點與整數來源時間的邊界命中當準確警示。|

每段看原片，填「開始時刻最早／最晚」與「落地時刻最早／最晚」，或選看不清／片段外；再選整段骨架品質（可加錯點幀段備註），勾本人查看。兩個數字是單一時刻可能範圍，**不是整段姿態phase**。先不顯示骨架看時間，再開骨架看對齊。頁面不預填AI答案／分數、不填不可見座標、不代選本人確認。

三段完成後填覆核者、勾本人確認，按一次「匯出一份正式JSON」，將`validation-three-clips-human-review.json`交回對話。只有三段各一筆，218張原PNG是播放器逐幀素材，不是218次人工紀錄。這是依已知結果挑的診斷樣本，非盲測、非全體代表樣本；人工仍pending，目前0/3，精確時間未知。

## 驗證、保全與證據分級

- synthetic：18個Python風險測試通過，含Validation-only metadata/array路徑、Test拒絕、strict >／K3／track及缺失重置、unknown分母、因果原幀、人工時間界限、來源路徑逃逸及防覆寫；離線假DOM22断言通過，涵蓋欄位禁用、時間順序、逐片草稿、3/3與本人確認匯出保護、正式／草稿差異。假DOM填答沒有進實際瀏覽器，也沒有產生真人標註。
- existing_real_result_audit：獨立手算23序列指標／分母、2917原窗raw至K3及14事件，直接原Pose NPZ核3576原幀／track紀錄，bisect past-only核3377採樣映射；與正式輸出一致。218張覆核PNG bytes／SHA與來源一致。
- 實際UI：本機頁及首張PNG HTTP200；真瀏覽器三片導覽、source frame／time、末幀disabled、原骨架切換、0/3及正式匯出disabled確認；沒有勾human／填座標／匯出真人JSON。截圖及HTTP紀錄保存。初沙箱socket拒絕／服務timeout留錄；停止該服務、經自動批准改用僅127.0.0.1服務後成功，無外傳。
- real_feature_smoke／real_end_to_end：本輪均未執行，既有分數重核不當新模型或RGB至AE全流程證據。
- 329個有界來源hash與上輪48個finalized artifact核對；修改前進度／驗收bytes另存prior-contracts，AGENTS／根規格／舊run／快取／hash／報告均保留。新測試最初一處import空行lint修正，原測試／初版manifest保存；首次Windows路徑pin更新失敗未改manifest，修正後一致。無結果、模型或原契約修復。
- graph先查與coverage：原／新增結構查詢、exact snippet與當前原碼；335 evidence paths分三批（工具128上限），13 no_recorded_issue、322設計排除；排除原始檔／cache/run/output直接讀檔及核SHA。LSP部分指到歷史副本、最初suffix歧義／trace失敗及猜測renderer不存在都保留限制；不用graph聲稱完整或已做RGB顯示驗收。

沒有新的辨識或警示時間改善結論。維持wide参照、原demo／門檻／K3，Test未知、本輪未訪問或評估；新版正式Notebook／五分鐘影片／公開上傳仍暫緩。下一步等這三段本人覆核，才把警示與人工時間範圍對照；不自動把標註拿去訓練或重新校準。

## 可直接執行命令

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
# 只讀核對既有來源、前輪保全及本輪檔案
.\.venv\Scripts\python.exe -X utf8 scripts/validation_alert_quality.py verify
# 本機覆核；若8830服務已在執行，直接開上述URL即可
.\.venv\Scripts\python.exe -X utf8 scripts/validation_alert_quality.py serve --port 8830
# 收到本人JSON後只驗格式、來源bundle與時間範圍，不訓練／評分
.\.venv\Scripts\python.exe -X utf8 scripts/validation_alert_quality.py validate-review --annotation 'C:\Users\Huayi\Downloads\validation-three-clips-human-review.json'
```

`build`與`finish_validation_alert_quality.py`都是本輪新目錄一次性建立命令，已執行完成；再次執行必須拒絕，不用它們覆蓋成果。QA位於`docs/validation/ur-validation-alert-quality-20261013-001/`，完整報表、六CSV／JSON／圖與覆核材料位於`outputs/ur-validation-alert-quality-20261013-001/`。
