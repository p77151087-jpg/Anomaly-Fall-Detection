# 三段本人標註驗證與既有Validation警示時間對照

ID：`ur-validation-human-timing-20261105-001`。使用者已完成fall-18／fall-09／fall-30覆核。原JSON bytes已封存，SHA256為`cd36880401f9c7daee8b57124d8b2d65017abdf238f6012df47843fb9d7ea0c2`。

已實際完成格式、來源、時間及本人自述確認驗證，並對照既有警示。三片在wide參照模型都是影片TP；三個首次警示均落在「落地時刻不確定範圍」，不能據此確認在落地前或落地後，也不能把3/3可能包絡命中當成3/3已確認跌倒過程警示。本輪沒有辨識或時間改善，沒有訓練、模型／特徵推論、校準或Test訪問。

## 標註驗證

- schema1、正確原bundle `a4a8f0ceac863b96c75781a0f9a1dfea65d48fc1a0cce6020231367ad4810d73`；重新計算完整bundle內容hash一致。
- 恰好3支唯一Validation，reviewer非空、`human_self_reported`、`human_attested=true`、逐片`reviewed=true`，非草稿；三片的開始／落地均有排序正確的原影格整數界限，來源、時間戳與尺寸吻合。
- 218張bundle PNG重新解碼SHA、原Pose metadata的RGB source identity、輕量覆核頁PNG三者一致；三份metadata都是原Validation／原wide recipe／原split。沒有載入Pose或特徵NPZ、權重或AE／Scaler，沒有解碼或讀取Test。
- 兩端點依原頁定義解讀為**單一開始／落地時刻的不確定範圍**，不是整段姿態phase。這是本人自述，不宣稱已由第二位獨立覆核者確認內容，也不以取中點製造精確時間真值。備註是資料，不當成指令。

## 如何判斷「是否落在跌倒時段」

開始時刻為`[Olo,Ohi]`、落地時刻為`[Ilo,Ihi]`：

- `[Olo,Ihi]`是**可能跌倒包絡**；落在其中只能說可能在跌倒過程。
- 若`Ohi≤Ilo`，`[Ohi,Ilo]`是所有允許界限都涵蓋的**保證區間**；等於最早落地邊界仍不當成嚴格落地前。
- 警示`t<Ilo`才確定早於落地，`t>Ihi`才確定晚於落地；在`[Ilo,Ihi]`內就是落地時刻不確定範圍。
- 警示使用既有CSV的`available_at_ms`，人工界限使用原影格`timestamp_ms`。30Hz sample_index不當source_frame_id，不先四捨五入再比邊界。

這是診斷區間運算，不更改原教師方向、影片級指標、舊品質／時序guard或模型採用規則。

## 逐片結果（主版本：ur-isotropic-wide-20261012-001）

|影片|開始界限：原幀／秒|落地界限：原幀／秒|首次警示秒|可能包絡|保證區間|結論|
|---|---|---|---:|---|---|---|
|fall-18|22–34／0.700–1.101|40–47／1.301–1.534|1.433333|命中|未命中：1.101–1.301|落地時刻不確定範圍內|
|fall-09|130–138／4.304–4.571|142–180／4.705–5.973|5.900000|命中|未命中：4.571–4.705|落地時刻不確定範圍內|
|fall-30|36–47／1.167–1.534|50–57／1.635–1.868|1.866667|命中|未命中：1.534–1.635|落地時刻不確定範圍內|

已覆核且具兩個有效時間界限的分母為3：可能包絡3/3、保證區間0/3、落地不確定範圍3/3、確定落地前0/3、確定落地後的首次警示0/3。**0/3確定落地前不等於證明3/3都落地後**。這三片是先前依已知疑點挑的子集，不能外推為15片Fall或23片Validation的人工時序準確率；另12片Fall仍未有本輪開始／落地覆核，fall-19已有姿態phase亦非這種精確時刻界限。

首次警示相對未知實際時刻的延遲保留整個範圍：

|影片|相對開始延遲ms|相對落地延遲ms|解讀|
|---|---:|---:|---|
|fall-18|332.333–733.333|−100.667至+132.333|可能落地前約101ms，亦可能落地後約132ms|
|fall-09|1329–1596|−73至+1195|不能確定是否落地後1.195秒；範圍跨過零|
|fall-30|332.667–699.667|−1.333至+231.667|只比最晚落地界限早1.333ms，不能據此宣稱準時或提前預警|

![本人時間界限與既有警示端點](../outputs/ur-validation-human-timing-20261105-001/human-timing-comparison.png)

## 首次、後續active與展示品質分開

|影片|active端點數|其中當前同track confidence-good|可能包絡內active|保證區間內active|超過最晚落地的active|最後active／事件關閉秒|
|---|---:|---:|---:|---:|---:|---|
|fall-18|1|0|1|0|0|1.433333／1.466667（low_score）|
|fall-09|8|8|3|0|5|6.133333／6.139000（sequence_end）|
|fall-30|1|0|1|0|0|1.866667／1.900000（low_confidence）|

fall-09在6.000、6.033333、6.066667、6.100、6.133333秒的5個後續active端點，確定超過最晚落地5.973秒；這表示警示延續至落地後，**不是首次警示已證明太晚**，也不把它們算成5支影片誤報。事件關閉時間不是最後active時間，兩者不混算。

三片整段品質都標為`mostly_matches`；原備註保留，提到落地／倒地後部分身體出框，fall-18後段無骨架、fall-30倒地後骨架顯示異常。這支持「落地附近／之後有觀測或展示限制」的診斷，但不是逐joint座標真值、逐幀品質標籤，也未證明骨架是唯一問題。整段mostly_matches與單一警示端點confidence不足可以同時成立。

在可能包絡內有當前good active的影片1/3（僅fall-09）；fall-18／30各唯一active端點的當前good仍為false。此為原工程confidence門檻對照，沒有重跑RGB renderer，不稱實際畫面可見警示率或人體骨架正確率。原六片coarse／15片AI診斷資料原樣保留，不用這次寬鬆包絡取代其驗收。

## 正式影片級成績與demo

|版本|原角色|23支Validation結果|這三片|
|---|---|---|---|
|ur-isotropic-wide-20261012-001|最高Validation參照，30幀isotropic／256-64 AE|TP14 TN8 FP0 FN1 U0；兩Accuracy22/23＝95.65%；P14/14、R14/15、F1 28/29＝.965517|全部TP；三個首次時間如上|
|ur-sample-human-s1-001|目前demo預設，30幀per_frame／128-32 AE|既有20/23、F1 .888889，本輪不改|fall-18／30仍FN無警示；fall-09 TP且首次5.900秒|

wide矩陣[ADL,Fall]×[normal,fall,unknown]仍[[8,0,0],[1,14,0]]；unknown0使可判定與含unknown整體Accuracy分母同為23，Recall14/15，兩種F1同為28/29。fall-19仍為唯一wide漏報，本輪沒有對它新增任何模型／座標介入。正式分數與門檻.5341113805770874／strict >／K3都不變，不稱新評估模型或成績提升。

## 保存與驗證

- 新目錄保存原JSON、標註audit、3列比較JSON／CSV、10個active端點、3事件、233選定窗口、218原影格時間來源表及23列human補充表。原23列內容逐欄一致，新表只增本輪human欄位；舊bundle／AI phase／pending歷史報告不覆寫。
- synthetic：最初21個新時間邊界／缺標／交疊／不四捨五入／因果性與防覆寫測試，加18個既有split／Test拒絕／K3／human來源風險，39 passed；保全路徑修復後新增兩種分隔符／封存副本遭改仍須拒絕測試，最終23新＋18既有＝41 passed。lint／format通過。
- existing-real-result-audit：獨立程式不使用新時間分類helper，直接從原Pose JSON時間戳以Decimal比較，核23影片指標／分母、233原窗口K3、3事件、10active、3組延遲界限、5個最晚落地後端點及原同track因果來源。與正式CSV和本輪輸出一致。
- 263份有界舊hash保全（包含231個舊outputs檔、前輪29個final QA artifact及來源契約），兩份修改前進度／驗收bytes另存。實際重複build拒絕及新outputs hash不變另記；不宣稱全專案所有檔都做了窮盡盤點。
- 更新進度文件後的首次保全檢查發現Windows反斜線與正斜線比對錯誤，誤查已更新文件而未查封存原bytes。原始程式／tests／inputs manifest與失敗訊息先另存，再只修復路徑正規化、重核SHA；沒有放寬hash，也沒有重算／改動時間、警示或標註結果。
- real_feature_smoke、real_end_to_end本輪均未執行；沒有模型／特徵陣列載入、訓練、推論、校準、Test影像／快取／成績訪問或新Test評估。新方法Test未知，不使用歷史Test改選。
- Graph先查index／search／exact snippet／coverage；舊generation、heuristic trace把dict.get及非相關caller連到別處的限制保留，實際以當前原碼和只讀檔案核對。成果／cache設計排除不冒充graph驗證，coverage是best effort。

本人覆核輸入階段已完成3/3；仍有的是**落地真實時刻的不確定性**，不是漏交或格式驗證失敗。本輪不需要再做大量逐幀紀錄。若將來確實要判定落地前後，才另行縮小關鍵落地界限；目前不猜時刻、不重標、不調參，原demo與Notebook／五分鐘影片暫緩維持，沒有公開上傳。

## 檔案與可直接執行命令

成果：`outputs/ur-validation-human-timing-20261105-001/`；QA：`docs/validation/ur-validation-human-timing-20261105-001/`。[逐片CSV](../outputs/ur-validation-human-timing-20261105-001/comparisons.csv)、[完整JSON](../outputs/ur-validation-human-timing-20261105-001/comparisons.json)、[23列補充表](../outputs/ur-validation-human-timing-20261105-001/validation-23-human-supplement.csv)。

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
# 只讀驗本輪來源、原標註副本及新舊hash
.\.venv\Scripts\python.exe -X utf8 scripts/validation_human_timing.py verify
# 獨立重核時間／正式CSV／233窗口／3事件，不重畫、不重寫既有證據
.\.venv\Scripts\python.exe -X utf8 scripts/verify_validation_human_timing.py
```

`build --annotation ...`已完成，再執行應拒絕。驗證程式只有首次`--plot`建立圖；圖已存在時不再加該參數，不覆寫成果。
