# fall-19第44–51幀全骨架錯點定位

2026-10-13，`ur-fall19-joint-localization-20261013-001`。已實際核對8幀原RGB／原Pose，產生104筆關節與96筆骨段量測、逐幀並排圖，選出**44、47、49三幀**供人工確認。主要待確認範圍為髖／膝／踝，**沒有已確認新錯點或辨識改善**；人工覆核仍pending。

## 來源與界線

依[事前範圍](ur-fall19-joint-localization-20261013-plan.md)，只讀原Validation fall-19。原30幀isotropic Pose recipe `713ee59a0853f20641cb7f664df2920359baa0734807acfa5cf86fb87a4e2f5b`，同track1、human13原13點。原圖640×480；放大僅放大像素，不稱影像解析度提升。joint0鼻、1/2左右肩、3/4左右肘、5/6左右腕、7/8左右髖、9/10左右膝、11/12左右踝，已對照核心HUMAN13_NAMES；第14個肩中點是ST-GCN衍生點，沒有原觀测關節真值，不要求人工另標。

核對原Pose NPZ／metadata SHA、原manifest及人工補註manifest SHA、8影像SHA與原cache來源身份；既有review-data的每個關節與原cache逐值一致、原時間完全一致。8幀原reported quality=valid、current_good_pose=true；原人類整體pose_check均為wrong，但整體wrong沒有指定每個joint錯在哪裡。

本輪沒有使用Train去擬合；也沒有讀Test影像／特徵或評估。沒有ST-GCN／Pose重新推論、AE訓練／校準、分數比較、點位替換或事件修改。舊人工左手肘七點接近原模型／51不可確認的結果保留，不重做它們，不把AI之前移點的分數增加當成人工修正成功。

## 全8幀的疑點

AI逐幀目視原圖及原點；下表是**待確認觀察**，human_confirmed=false。跳動、投影骨段長度、低信心與身體彎腰／遮擋都不能直接當錯點真值。

|幀|原時間ms|觀察與量測|本輪人工頁|
|---|---:|---|---|
|44|1434|起始對照；下肢點與上身／手臂／足部投影重疊，足部輪廓較後段清楚。沒有範圍外前幀位移，delta為空。|選取|
|45|1468|左腕5相較44幀位移44.48px，其中y向上42.46px，疑似靠近前臂；46幀又移50.82px。額外上半身疑點，未確認。|只存圖／報告|
|46|1501|右膝10位移55.16px、右踝12的投影y高於膝；彎腰與遮擋可影響位置／上下順序。|只存圖／報告|
|47|1534|左右踝11/12疑似落在前臂附近；膝9與肘3靠近，髖7/8近上身。需要辨識真正下肢可見性。|選取|
|48|1568|左右踝11/12相較47幀位移88.83／101.16px，疑似移到手腕附近；不能只由位移直接判錯。|只存圖／報告|
|49|1601|右踝12相較48幀位移67.77px、y向上67.72px；減去13點中位位移後殘差83.01px，疑似回到膝附近；遮擋明顯。|選取|
|50|1635|右踝12又移71.96px，y477.92接近下緣；精確下肢位置難從原圖確認。|只存圖／報告|
|51|1668|原骨架仍存在；既有人工左肘不可確認。下肢遮擋／模糊，不增添必填圖，也不猜點。|只存圖／報告|

位移只比較本範圍相鄰44–51原source幀及同track；不跨缺失、不看未來補點，44幀沒有43幀資料參與。殘餘位移=`joint位移−13點位移的逐軸中位數`的長度，用來區分共同移動與單點跳動；未擬合新閾值，也不是模型分數。骨段長度只是2D投影量，沒有以固定身體比例判定錯點。

![第47幀原圖與原骨架對照](../outputs/ur-fall19-joint-localization-20261013-001/review/frame-047.png)

## 人工工作只有三張與單一匯出檔

[本機覆核頁](http://127.0.0.1:8821/index.html)已實際開啟並保留；另有自含原圖的離線[index.html](../outputs/ur-fall19-joint-localization-20261013-001/review/index.html)。每張先檢查7–12六個髖／膝／踝，共18項；全13點都有顯示，上半身可選補註，不自動標成正確。

1. 可辨原點符合：選「原點符合此關節」。
2. 明確錯位而且正確位置可見：先點原圖的新位置，再按確認。
3. 疑似錯位但找不到可靠正確位置：選「疑似錯位，但正確位置無法確認」，xy=null。
4. 看不清／遮擋／不能判定是否錯：選「看不清／遮擋／無法判定」，xy=null。

左右指人物解剖左右，不依畫面左／右簡化。必要18項、姓名、本人自述確認皆完成後，匯出一個`fall-19-whole-skeleton-human-review.json`。來源影格／時間／track／原XY及confidence／來源hash保留；草稿key綁定新bundle SHA，與舊手肘頁隔離。頁面沒有分數或AI修正點，不依分數選位置；本輪沒有替使用者填寫、勾本人確認或匯出人類標註。

收到人工結果後才判斷是否存在足以做控制修正的可見錯點。若全部下肢都不可確認，不補造整副骨架，也不進行以猜測座標救漏報的實驗。只確認分類而沒有正確點位，不能當成已完成修正推論。

## 已完成的驗證

- synthetic：10個Python風險測試通過（同track連續性、時間、非有限值、平移與單joint殘差、confidence不等同正確率、無修改原觀測、Test metadata先拒絕、原目錄防覆寫、joint映射）；離線Node VM使用假DOM／假點位驗證20項UI斷言通過，包括放大點選換算、不可確認null、草稿重載、末幀導覽及匯出gate。沒有在真人頁自述確認或產生人工檔。
- 初次Python測試6個失敗源於新合成fixture將tuple乘整數當數值位移；改成NumPy向量後，同10項全過。首次XML原樣保存，未改診斷輸出或核心程式；新source ruff check通過。
- 真實觀測核對：8原圖／原Pose／原人類整體分類及時間／SHA核對，104關節CSV與96骨段CSV獨立從原NPZ以math.hypot／statistics.median重算通過；3內嵌圖bytes完全相同，沒有新feature推論。
- 保存：340份既有重要成果／原來源／契約，加上3份本輪事前source／template／plan lock，共343份hash一致；進度與驗收文件更新前bytes另存。保全範圍列在old-assets-before.json，並非宣稱全專案所有檔案均完整盤點。independent-verification的343是這個合計，不能全部解讀為舊檔。
- 實際重複inspect及package均拒絕，當時26個新輸出／QA檔before/after hash不變。新成果與舊手肘／Pose解析度／wide等run分開，不建立假訓練run。
- 實際瀏覽器：3幀頁載入、原图／骨架切換、末幀next禁用、未點新位置的修正拒絕、0/18匯出disabled、reload後仍未填寫均確認；畫面保存review-page.png。local HTTP200、1253129 bytes。UI行為的草稿／匯出完整組合由synthetic VM驗證，未冒充人類做真頁完整匯出。
- 限制原樣記錄：最初新tab attach timeout、沙箱loopback Win10013／8820無法連線，改以經review的只綁127.0.0.1服務8821通過HTTP及真瀏覽器。Full-page screenshot API失敗，改用已支援viewport screenshot；沒有關閉網路／瀏覽器保護。CIM盤點權限不足，Get-Process preflight未見舊Python；新隔離目錄本來不存在。
- Graph Tier2：索引ready、結構先search／trace／snippet並check coverage；原source無記錄缺口，raw／cache／outputs設計排除後直接核原檔。部分trace將函數連至歷史副本，已以現在import／exact source辨識，不宣稱完整graph或全專案驗證。證據另存graph-evidence.json。
- real_feature_smoke與real_end_to_end本輪都**未執行**。本輪實際證據是既有真實影像／Pose診斷，沒有新辨識成績。

正式最高Validation仍wide22/23＝95.65%、Fall F1 .9655，fall-19既有漏報狀態不改；demo仍ur-sample-human-s1-001。沒有採用新模型／點位方案，沒有Accuracy或警示時間改善證據。Test未讀取／未評估，舊成果與舊已查看Test限制保留。Notebook／五分鐘影片／公開上傳仍暫緩。

## 成果与可直接執行命令

`outputs/ur-fall19-joint-localization-20261013-001/`含original-frames.json、joint/bone-measurements.csv、8幅all-frames並排圖、all-eight-contact.png、ai-assessment.json、summary.json與review三幀頁。`docs/validation/ur-fall19-joint-localization-20261013-001/`含事前lock、舊hash／契約bytes、獨立核對、synthetic XML／UI記錄、graph及真實頁截圖。

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe -X utf8 scripts/localize_fall19_joints.py verify
```

若本機服務已停止，可啟動（已在執行時不要重複啟動）：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/localize_fall19_joints.py serve --port 8821
```

瀏覽器開`http://127.0.0.1:8821/index.html`。build命令inspect／package已執行，重跑會拒絕覆寫；獨立核對輸出也採exclusive create，後續讀取已有核對JSON或執行上面的只讀verify即可。
