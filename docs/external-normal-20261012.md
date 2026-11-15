# 新增真實正常資料與 Train／Validation 取樣標註

2026-10-12。本輪已實際取得32支新正常RGB影片、嚴格重抽真實Pose／ST-GCN、完成取樣標註及兩個事前固定的混合正常資料AE。**新候選Validation Accuracy均21/23＝91.30%、Fall F1 .9333，低於既有最高22/23＝95.65%／.9655；未採用，沒有辨識效果改善。** 沒有新Test推論、校準或選模；原預設保留。

## 實際取得的正常活動

| 官方來源 | 新下載影片 | 可用Train影片 | 新正常特徵窗 | 說明 |
|---|---:|---:|---:|---|
| [UMAFall v8](https://figshare.com/articles/dataset/UMA_ADL_FALL_Dataset_zip/4214283) | 8 | 8 | 3,662 | 上下樓、走路、慢跑、床上躺下起身、彎腰、跳躍、椅子坐下起身示範 |
| [Ivan Ursul video dataset v2](https://figshare.com/articles/dataset/Sensor-Based_Fall_Detection_Dataset_with_2017_Activities_from_29_Subjects/28596332) | 24 | 13 | 169 | 8個來源受試者、5場地的ACT13／18／24；可用影片來自其中7人 |
| 合計 | 32 | 21 | 3,831 | 11支無符合原品質／時間窗規則的特徵，保存影片／快取但不訓練 |

兩來源官方授權CC BY4.0；保存官方JSON、URL、版本、影片SHA256、ZIP成員CRC32與實際尺寸。使用新增真實攝影，沒有從原UR Test或既有Le2i／CAUCA留出集搬影片。與既有外部清單比較整支影片SHA，0個相同檔案，不能證明人物或語意完全獨立。

UMAFall主要是穿戴感測資料，這次取用另附的8支動作示範，不能宣稱新增了19人RGB資料。示範影片含片頭、感測圖表，原始影像取樣檢視後只留完整位於攝影正常活動區間的窗口；原影片及完整快取保留。8支共11,914幀。

Ursul的VideoDataset.zip約2.53GB，讀中央目錄後只Range下載24個ADL成員，未下載Fall成員。選擇於模型實驗前固定為受試者01／02／03／04／06／07／08／09、ACT13／18／24、方向代碼F、trial1、第一個符合的場地；05在目錄缺ACT18／24，鎖定前排除，非依辨識結果排除。F是來源代碼，不能當正面攝影證據；實際取樣多為背向走路、彎腰拾物、跨台階。片段僅56–57幀／30FPS、約1.9秒，共1,356幀。

UMA整包ZIP已核對官方MD5 `2fa62988eacd2158a77928ce52b0b37f`，本機SHA256 `ddbc9144b41ac57a9673dbd8d0f453e60a91166d209a530cf38151e45cf27a79`。**Ursul完整大ZIP未下載，沒有驗證整包官方MD5**；個別成員均驗證HTTP206範圍、ZIP名稱／格式／長度／CRC32，另記本機SHA256，TLS驗證開啟。

原Train24支UR ADL／3,638窗，本輪 **24UR＋21外部＝45支正常影片、3,638＋3,831＝7,469窗**。外部均明示external_train，沒有加入原UR Validation／Test分母。窗口高度重疊，新窗3,662/3,831來自UMA示範，不能稱7,469個獨立樣本或證明人物隔離。

- [資料鎖定](validation/external-normal-20261012-data-lock.json)、[32支來源清單](validation/external-normal-20261012-data.json)。
- [真實特徵證據](validation/external-normal-20261012-prepared.json)、[裁切區間／矩陣hash](validation/external-normal-20261012-training-inputs.json)。
- raw在`data/raw/external-normal-20261012/`，快取在`data/cache/external-normal-human-isotropic-20261012-001/`，不加入Git。
- [來源原始影像接觸表](../outputs/external-normal-review-20261012/)。

## 跌倒時段與可見性

正式完成版本為[v2 manifest](annotations/ur-train-validation-20261012-v2/manifest.json)，涵蓋47支原Train／Validation：24Train ADL、8Val ADL、15Val Fall。Train按契約沒有跌倒，不捏造時段；Test未標註。

實際檢視340張原始取樣影像，每Fall12張、每ADL5張。15支Fall記錄最後尚未下降／第一張下降／最後仍在下降或滾動／第一張穩定的來源影格及觀測時間，另存開始與穩定不確定區間、內側下降核心、外側可能下降區間。這是**AI取樣影像診斷，非人工複核、官方完整逐幀GT或精確警示延遲驗收**；取樣間隔與事前已知道模型錯誤的偏差保留。

頭、肩、肘、腕、髖、膝、踝7區域逐圖記錄V可見、P部分可見／自身遮擋／單側不確定、O物件遮擋、C大部分出框、U模糊／暗／太小無法可靠判定、N未見目標。成對區域不分左右，非每個Human13關節座標GT；低光影片保留U，未以confidence替代可見性。

另存106,808行Human13自動診斷，含原Pose confidence、影像邊界、既有品質結果，明示不是可見性GT。340原圖SHA已核對。取樣時段僅用診斷，沒有跌倒標註加入Scaler／AE，也不改舊採用guard／品質／事件規則。

- [直接檢視原始影像／時段／標註HTML](../outputs/external-normal-20261012-report/annotation-review.html)。
- [取樣可見區域CSV](annotations/ur-train-validation-20261012-v2/sampled-joint-regions.csv)。
- [47支序列／15跌倒時段JSON](annotations/ur-train-validation-20261012-v2/fall-intervals.json)。
- [Human13自動診斷CSV](annotations/ur-train-validation-20261012-v2/automatic-human13-diagnostics.csv)。

初版v1在寫manifest時因NumPy整數不能JSON序列化而失敗；不完整v1保留，修正計數後改建v2，未覆寫／冒稱v1完成。

## 固定實驗與結果

沿用真實AlphaPose Human13、雙流ST-GCN分類頭前串接512D特徵＋正常AE，沒有改七類分類作為異常成績。新增32支原RGB **13,270幀**實際嚴格Pose／ST-GCN forward。recipe為原等比例縮放`713ee59a0853f20641cb7f664df2920359baa0734807acfa5cf86fb87a4e2f5b`；30幀窗口／stride1／品質／K3不變。

模型訓練前固定資料、128/32及256/64容量、seed42、max100／patience10、選擇與停止條件，見[模型鎖定](validation/external-normal-20261012-model-lock.json)。Scaler／AE僅45支Train ADL；正常Val8支只選checkpoint，原Val23支只校準門檻／評估。須嚴格超過舊22/23、F1不降低、誤報／unknown不增加並滿足舊guard才採用。兩容量完成即停止，未追加容量／seed／品質／門檻掃描。

| 模型 | 新AE epochs | TP | TN | FP | FN | 影片unknown | Accuracy | Fall F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原最高ur-isotropic-wide-20261012-001 | 0 | 14 | 8 | 0 | 1 | 0 | 22/23＝95.65% | 28/29＝.9655 |
| ur-external128-20261012-001 | 68 | 14 | 7 | 1 | 1 | 0 | 21/23＝91.30% | 28/30＝.9333 |
| ur-external256-20261012-001 | 27 | 14 | 7 | 1 | 1 | 0 | 21/23＝91.30% | 28/30＝.9333 |

合計95新AE epochs，兩者FP為adl-35、FN為fall-19。二元矩陣列ADL／Fall、欄normal／suspected_fall為`[[7,1],[1,14]]`；含unknown矩陣`[[7,1,0],[1,14,0]]`。可判定二元Accuracy分母23，全體含unknown Accuracy分母23，此次影片unknown0所以相同；Precision／Recall皆14/15，Fall F1 28/30，全體Fall召回14/15。影片coverage1.0不代表各窗口皆可判定，窗口coverage仍 .843911439。

新門檻`.49245548248291016`／`.4758157730102539`由Val校準後凍結；AE分數為重建誤差，非跌倒機率。兩者未過Accuracy／F1／誤報採用條件，**原預設ur-sample-human-s1-001不變**；原預設Val20/23＝86.96%／.8889和最高候選95.65%是不同模型，不能混稱部署成績。

不同Scaler下Loss不能直接比較；獨立核對將重建殘差表達在共同原Train24尺度：

| 模型 | 共同尺度正常Val MSE |
|---|---:|
| 原isotropic128 | .1151348558 |
| 原isotropic256 | .1156163662 |
| 外部正常128 | .1125923567 |
| 外部正常256 | .1153724109 |

小模型正常重建誤差略降，但跌倒／正常分離沒有改善且新增誤報；不能稱Loss下降即Accuracy提高，也不能由本次結果推論增加任何真實資料都沒用。

兩模型在新取樣診斷均14/15可能下降區間有事件、12/15有當前良好Pose警示；確認下降核心內事件分別7／8段。這些受取樣不確定範圍影響，不是正式定位指標。**fall-19可能下降區間有17合法特徵窗，仍無警示／事件**，不能說該區間完全無窗口。原影像出框、視角、窗口混入跌倒前正常姿態為證據與可能因素，尚未完整消融隔離唯一原因。舊6段粗時段命中4→5、早報0，不覆蓋舊失敗紀錄，也不等於準時警示。

本輪新Test評估0。保存的wide歷史Test仍18/23＝78.26%／.8000，舊isotropic128 Test19/23＝82.61%／.8462；新混合正常模型Test未知。原70支split hash `0b9a0f0cf3609c478afcbbe028130c7d519e26b573135fbd739c074c68bc93c5`不變。歷史Test已看、小Val反覆使用、人物隔離未證、老師逐影片／逐影格與切分協定未知，不宣稱新盲測／泛化改善／課程全部達標。

## 驗證與保全

- synthetic：36相關風險測試全過／0 skipped，涵蓋ZIP路徑／HTTP Range、外部Train-only label／recipe、完整窗口區間、觀測時間、矩陣subset／hash、Val保留／Test拒絕、Scaler及K3指標。
- real_end_to_end範圍限定：新增32正常RGB／13,270幀真實Pose／ST-GCN抽取，不是新原UR demo／Test成績。
- real_feature_smoke：真實特徵上95epochs訓練／校準、45來源／7,469列Train-only Scaler逐值吻合、訓練hash／最低正常ValLoss checkpoint吻合；46支次CSV指標／嚴格K3／事件重算及新15段時段重算通過。
- 1,160實驗前舊檔hash保留，含run／核心source／設定／split／權重／歷史Test。Test檔僅做保全hash，沒有讀特徵陣列或依Test選擇。10次重跑既有stage的覆寫／追加訓練全被拒絕，合計1,430受檢檔前後不變。
- 首次36測試35過／1失敗：Windows `Path.is_absolute()`未拒絕`/root/file`，補leading slash防護。取得／準備原v1 source完整封存，原lock／資料／特徵／模型／門檻不改；[修復記錄](validation/external-normal-20261012-source-repair.json)綁定精確新舊hash，status允許這次修復。未重訓／改模型結果；初次失敗XML保留，全過另存新XML。
- graph index／search／coverage回覆Transport closed；依AGENTS先嘗試，後以資料reader／split／cache／訓練／計分相關源碼及測試補核對，未取得新generation／coverage，不聲稱完整graph覆蓋。核心未修改。

[獨立驗證](validation/external-normal-20261012-verification.json)｜[最終QA](validation/external-normal-20261012-final-qa.json)｜[正式測試XML](validation/pytest-external-normal-20261012-final.xml)｜[結果圖](../outputs/external-normal-20261012-report/results.png)。

## 未完成事項與後續資料

兩個固定候選已完成，沒有Accuracy改善。仍缺完整逐幀／左右Human13關節人工複核及座標GT、精確跌倒開始／結束、未看過的獨立辨識及警示時間驗收。取樣標註可檢視，但不能替代上述工作。

新增窗集中於UMA示範、短片難提供30幀穩定姿態。後續宜先人工複核本輪fall-19／adl-35，再收集更長、完整身體入鏡、彎腰／坐躺起身等正常攝影，或對固定Train／Val另立骨架品質實驗。這是未來方向，沒有算成本輪改善。

已找到[SAFER-Activities](https://huggingface.co/datasets/SAFER-Activities/SAFER-Activities)，官方有長錄影、逐動作開始／結束CSV、人物／視角split。但檔案需登入且同意分享聯絡資料，**未取得、未訓練**；日後使用仍應保留其官方Test／非實驗室Test，不挪到Train。這是額外來源的外部存取限制，不阻塞本輪32支已完成工作，沒有代發信／提交條款。

正式Notebook／五分鐘影片仍依Accuracy優先指示暫停，未公開上傳。HTML是取樣標註檢視工具，不是正式成果Notebook／影片。

## 可直接執行的只讀操作

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe scripts/external_normal_data.py status
.\.venv\Scripts\python.exe scripts/prepare_external_normal.py status
.\.venv\Scripts\python.exe scripts/ur_annotation_review.py status
.\.venv\Scripts\python.exe scripts/external_normal_train.py status
.\.venv\Scripts\python.exe scripts/verify_external_normal.py
Start-Process -FilePath '.\outputs\external-normal-20261012-report\annotation-review.html'
```

verify會重核Train／Val快取與模型，不執行Test，不帶--write則保留原驗證檔。既有freeze／acquire／prepare／curate／training／build拒絕重跑覆寫，新增實驗須新版本／run。

正式新模型在runs/ur-external128-20261012-001及runs/ur-external256-20261012-001；每模型在[outputs/external-normal-20261012](../outputs/external-normal-20261012/)均有metrics.json、sequences.csv、windows.csv、events.csv。raw／cache本機保留不加入Git，來源／hash清單保存。
