# 三段Validation實際畫面：demo與wide快取重播驗證

ID：`ur-validation-display-20261105-001`。已實際完成fall-18／fall-09／fall-30的六支模型單片、三支左右並排短片及本機播放器，並核對原CSV事件與編碼後畫面。所有來源都是既有Validation RGB、原Pose、保存的AE／七類分數及本人標註；沒有重訓、模型新推論、校準、Test影像／快取／成績訪問或預設模型替換。正式Notebook／五分鐘影片／公開上傳仍暫緩。

## 先看結果

**事件歷史顯示與CSV起點一致；當前AE提示會繼續套用品質門檻。** 因此wide的fall-18／30可以同時顯示「ALERT RECORDED」與「Current: unavailable」。歷史紅字不是當前可判定警示，七類ACTION的FALLING也不是AE事件。

|影片|模型|影片判定|CSV首次事件秒|當前AE alert幀|歷史ALERT RECORDED幀|獨立ACTION FALLING幀|
|---|---|---|---:|---:|---:|---:|
|fall-18|demo預設|FN|無|0/65|0/65|6/65|
|fall-18|wide參照|TP|1.433333|0/65|22/65|6/65|
|fall-09|demo預設|TP|5.900000|8/82|8/82|44/82|
|fall-09|wide參照|TP|5.900000|8/82|8/82|44/82|
|fall-30|demo預設|FN|無|0/70|0/70|13/70|
|fall-30|wide參照|TP|1.866667|0/70|14/70|13/70|

分母是本輪短片的媒體影格，非獨立影片或新增訓練資料數。每支TP只有一個事件；22／14幀是原歷史面板保留的顯示，不是22／14次偵測。fall-09兩模型各8個active端點、5個後續端點超過人工最晚落地5.973秒，仍不是5支影片誤報。正式CSV的事件閉合時間／peak亦由原EventState重播逐欄核對，沒有改事件規則。

### fall-18

原預設模型`ur-sample-human-s1-001`沒有AE事件。wide在1.433333秒有一個合法K3事件，但當前同track骨架未通過原confidence條件，當前AE顯示不可判定；事件歷史自該影格起記錄並保留22幀。兩模型的獨立動作頭當時都顯示FALLING，不能把該標籤當作demo AE已判對。

![fall-18編碼後實際事件影格](../outputs/ur-validation-display-20261105-001/frames/fall-18-decoded-first-event.png)

### fall-09

兩模型首次AE事件皆5.900秒，當前Pose通過原門檻，畫面Current: alert與ALERT RECORDED同時出現，持續8個格點到6.133333秒；原事件於來源結尾6.139秒關閉。截取短片前仍重播完整歷史，沒有在3.433333秒片段起點重新warmup或重置事件。

![fall-09編碼後實際事件影格](../outputs/ur-validation-display-20261105-001/frames/fall-09-decoded-first-event.png)

### fall-30

demo無AE事件；wide於1.866667秒觸發一次，但當前品質遮蔽，Current不可判定，歷史紅字保留14幀。下一個1.900秒窗口因low_confidence關閉事件，不能把後續歷史紅字都算成active。1.866667距人工最晚落地界限1.868僅約1.333ms，仍無法據此保證落地前預警。

![fall-30編碼後實際事件影格](../outputs/ur-validation-display-20261105-001/frames/fall-30-decoded-first-event.png)

## 來源、原renderer與時間契約

- demo為`ur-sample-human-s1-001`，原30/per_frame／128-32 AE、門檻.6880183816；wide為`ur-isotropic-wide-20261012-001`，原30/per_frame_isotropic／256-64 AE、門檻.5341113806。各自讀凍結threshold／config／正式Validation scores，strict >、K3、stride1、9/13 confidence≥.2、窗口good ratio .8、score hold倍數2不變。
- 三片兩模型Pose NPZ SHA、逐點／bbox／track／原時間及RGB source hash完全一致。七類快取取自既有`sample-display/ur-human-isotropic-001`：原分類方法是原30/per_frame的獨立forward，與AE recipe無關。已核對每列的最後30個實際觀測、quality、時間因果、七分數有限／argmax與標籤；fall-09與demo本身的既有action CSV逐欄完全相同。只複用分類顯示資料，沒有複用該run的AE門檻或分數。
- 原`draw_sample`及`EventHistory`直接使用，原renderer檔案未修改。原demo會先觀察未遮蔽的AE事件，再套當前Pose／score age顯示gate；快取重播保留相同順序。沒有呼叫`Scorer`、backend、ST-GCN、分類頭或AE；實際render程序的模型模組載入列表為空。權重／Scaler檔僅核SHA，不反序列化。
- native畫面之外加模型角色／影片判定、原來源與格點時間、CSV狀態與本人時間範圍；native標題、骨架、色彩、目前不可判定及歷史行為不變。CSV原分數僅作有標示的紀錄對照，不冒充unknown時的即時可用分數；所有分數都不是跌倒機率。

|片段|本人覆核原時間範圍ms|實際輸出格點ms|影格數／影片長度|原圖末幀|
|---|---|---|---|---|
|fall-18|0–2135|0–2133.333333|65／2.166667秒|第65幀未落在最後格點，未補造|
|fall-09|3403–6139|3433.333333–6133.333333|82／2.733333秒|第185幀未落在最後格點，未補造|
|fall-30|0–2302|0–2300|70／2.333333秒|第70幀未落在最後格點，未補造|

每個媒體影格另列原sample_index／grid timestamp／source id／source timestamp及media offset。30Hz格點可能重複或略過部分原影格，不把sample index當source id。native時鐘只顯示小數兩位，底部與CSV保留完整格點時間；不拿顯示值1.87秒對人工界限作精確判斷。30fps是播放格點，沒有宣稱模型新推論即時效能。

本人三段開始／落地界限原樣使用，首次事件仍全在落地時刻不確定範圍；包絡命中不當作已確認時機正確或落地前預警。good只表示原confidence門檻，仍可能出框或錯點，沒有藉這次播放宣稱骨架幾何正確。

## 實際驗證

- synthetic：35個cached replay／原sample display／骨架顯示風險測試通過，涵蓋K3、strict >、缺失重置、track／stale、歷史先於mask、分類頭獨立、壞骨架不畫、原資料不改、防覆寫及真H.264 codec合成片解碼。另16個HTTP Range解析／真檔案分段與416邊界測試通過，合計51。
- cached_real_render_replay：六單片＋三並排均實際H.264/yuv420p編碼，個別640×820、並排1280×820、30fps。全數逐幀解碼651個唯一成品影格；獨立由原JSON／NPZ／CSV重算640個完整模型格點及466窗口的K3／因果映射／quality／history，對434個個別native畫面及並排兩側核原header字形／顏色／model title。並排panel與個別編碼平均絕對差最大約1.292/255；像素核對全部通過。
- 原分數與CSV事件逐欄對照保留；畫面Current與raw CSV active的兩個差異均由原quality gate解釋，而非事件漏寫或降低門檻。13張事件前／當下／後lossless配對圖及三張實際解碼事件圖另存。
- 真瀏覽器：本機H.264載入、三片切換、單片版本、播放到結尾、半速、逐幀及跳到事件均檢查，原格點1.433333／5.900000／1.866667對應正確；頁面與影片只供127.0.0.1。
- 最初獨立審核器將尚未存在的track誤設persistent track_gap，暖機核對失敗；初版程式與訊息另存，只修審核器，原重播／CSV／九片不變。初版靜態HTTP服務不支援Range，瀏覽器seekable為0；新增單一byte-range服務及16項測試後可正常跳轉，保留初版source／inputs／失敗紀錄。AST驗證render build／畫布／資料載入函式未變，所有九片SHA於服務修正前後一致。
- 308份有界舊成果／來源hash及255份目前輸入核對（原253份render輸入另加2份Range工具／測試），兩份修改前進度／驗收bytes封存；原run、模型、資料、Pose、分數、人工JSON／bundle、先前報告與hash不覆寫。實際重跑build與gallery均拒絕，30個既有新成果檔案前後SHA一致；服務修正前後九片SHA亦一致。範圍不是全專案所有檔案的窮盡盤點。
- Graph先查並核coverage，現有renderer exact source與原demo流程已讀；NumPy.full等heuristic連結不作依據。初次索引曾記錄兩個無關automatic-human13 CSV parse timeout，最終索引已無skipped；另有一個無關synthetic非JSON partial未用於本輪。353個有界來源／成果路徑最終coverage為71個no_recorded_issue、282個設計排除，來源以原檔、像素與SHA另核，不宣稱graph完整驗證。

本輪沒有real_feature_smoke或新的RGB→Pose→ST-GCN→AE real_end_to_end；證據是**真實既有資料至實際編碼畫面的重播驗證**。沒有新的Accuracy／Recall／F1或警示時間改善，wide正式仍22/23、Fall F1 .965517；demo仍20/23、.888889，沒有改預設或弱化原採用／品質／時序條件。fall-19仍未解決，本輪未納入它或Test。

## 交付與命令

[本機互動播放器](http://127.0.0.1:8841/index.html)：可選三片、左右並排／demo／wide、半速、逐幀、跳到wide首次事件。全部九支影片在`outputs/ur-validation-display-20261105-001/videos/`；[逐影格CSV](../outputs/ur-validation-display-20261105-001/display-frames.csv)、[摘要CSV](../outputs/ur-validation-display-20261105-001/summary.csv)、[來源與媒體時間JSON](../outputs/ur-validation-display-20261105-001/summary.json)。QA在同名`docs/validation/`。

|影片|並排|demo單片|wide單片|
|---|---|---|---|
|fall-18|[並排](../outputs/ur-validation-display-20261105-001/videos/fall-18-compare.mp4)|[demo](../outputs/ur-validation-display-20261105-001/videos/fall-18-demo.mp4)|[wide](../outputs/ur-validation-display-20261105-001/videos/fall-18-wide.mp4)|
|fall-09|[並排](../outputs/ur-validation-display-20261105-001/videos/fall-09-compare.mp4)|[demo](../outputs/ur-validation-display-20261105-001/videos/fall-09-demo.mp4)|[wide](../outputs/ur-validation-display-20261105-001/videos/fall-09-wide.mp4)|
|fall-30|[並排](../outputs/ur-validation-display-20261105-001/videos/fall-30-compare.mp4)|[demo](../outputs/ur-validation-display-20261105-001/videos/fall-30-demo.mp4)|[wide](../outputs/ur-validation-display-20261105-001/videos/fall-30-wide.mp4)|

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe -X utf8 scripts/validation_display_replay.py verify
.\.venv\Scripts\python.exe -X utf8 scripts/verify_validation_display.py
# 只在服務尚未執行時啟動；原8840不支援seek的服務已停
.\.venv\Scripts\python.exe -X utf8 scripts/validation_display_replay.py serve --port 8841
```

`build`和gallery建立已完成，再跑應拒絕，不用它們覆寫成果。新播放器只讀、沒有本人標註匯出或修改既有標註功能。正式Notebook／五分鐘影片仍暫緩，不公開上傳。
