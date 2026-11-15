# 期末專案調整計畫

2026-10-12 接續全域動態階段已完成：只新增一個原像素骨架90D中心位移／log尺度的正常5近鄰側分支，融合凍結wide；原24Train ADL擬合、23Val校準，沒有新AE／SSL epochs。Val22/23／F1 .9655與wide持平，fall19仍FN、FP0、U0；2917窗口狀態／14事件時間／23首次警示均未改變，不採用、不宣稱Accuracy或時序提高。60風險通過（初次暫存setup errors保留）、18003分數／3638影片排除Train定標／完整CSV事件獨立核對與24620舊assets保全完成。Test23仍未準備／未知，無demo部署、Notebook／影片／上傳。[完整成果](ur-motion-novelty-20261012.md)｜[事前單一候選契約](ur-motion-novelty-20261012-plan.md)。

2026-10-12 方法比較已完成：六SVM＋兩AE實際62epochs，最高新候選21/23／F1 .9333低於wide22/23／.9655，未採用、原demo保留。SVM fall09粗時段事件5.9→5.0333秒與phase命中4→5另列，不稱Accuracy提高。raw骨架／表示受控診斷未支持本輪SSC，選配停止；47風險、完整去重296passed／1歷史skip、8模型47,630分數／175事件獨立核對、23,925舊assets保全完成。新Test未知，Notebook／影片仍暫緩。[完整方法比較](ur-normal-methods-20261012.md)。

2026-10-12 新授權：擴大原正常UR Train ADL上的One-class與AE方法比較。教師簡報37–40頁將跌倒視為異常，列One-class／AE；固定ST-GCN、30幀、MLP及K3屬歷史工程預設。使用者已授權時序AE與有診斷依據的正常自監督微調。本輪六組固定SVM，再依診斷鎖最多兩個AE；原24Train、原23Val、無新Test，外部保持封存。比較按含unknown Accuracy／Fall F1／全體Fall Recall／較少FP／固定順序；展示guard與辨識成績分開，基準90%／.87不換算課程分數。歷史契約與結果原樣保存，詳見根規格最新研究契約；新版Notebook／影片仍暫緩。

最新單一20幀＋per_frame_isotropic實驗（2026-10-12）：原24UR Train ADL無增強、47Train／Val真實ST-GCN重新抽取，39epochs／最佳29；Validation20/23／F1 .8889低於最高wide22/23／.9655，fall-19仍漏、fall-18／30退步，不採用。明確Train／Validation-only模式保持Test23支未準備／無路徑，Test不讀／不評估；33相關測試、總273回歸通過／1歷史skip及完整CSV／K3／14事件獨立驗證、1,031舊檔保全完成。短窗新增fall-19前段10窗，粗phase仍9合法K3，不能當已證實原因或辨識改善。Notebook／五分鐘影片仍暫緩。[成果與限制](ur-short20-isotropic-20261012.md)。

最新使用者範圍修正（2026-10-12）：外部正常raw／cache已退出原路徑並封存，混合run只留歷史比較。課程後續AE僅原24UR Train ADL及其明示增強；不挪Val／Test、不重切、不以增強列數冒稱更多原影片。固定1%關節位置擾動＋單一AE已29epochs完成，Val21/23／F1 .9286，未超最高22/23／.9655、不採用；36相關測試、Train-only Scaler／23支K3及1,435舊檔保全通過。原split／教師未定事項／歷史Test限制保留，Notebook／影片暫停。[本輪成果與驗收](ur-only-shift-20261012.md)。本條優先於以下歷史外部訓練範圍。

最新外部正常資料工作（2026-10-12）：32真實RGB新增下載／21可用，原Train24增至45支／7,469窗；47Train／Val有340圖AI取樣及15段Val跌倒時段，非完整人工GT。固定2AE95epochs均Val21/23／F1 .9333，未超過原最高22/23／.9655，不採用；36風險測試、46支次K3及1,160舊檔hash通過，無新Test。資料／標註／實驗／驗收階段實際完成範圍與不足另記，Notebook／影片依Accuracy優先指示暫停。[本輪實作與限制](external-normal-20261012.md)。

最新Train／Validation診斷與有限實驗（2026-10-12）：24原Train影片／3638重疊窗，4個子集學習曲線228epochs、共同尺度Val正常誤差下降；2鏡像AE84epochs皆Val21/23，後續2Train正常殘差profile新增epochs0、皆22/23。未提高最高Accuracy、不依Test降低基準或換模型；37風險測試及1110舊檔hash保全通過，所有新候選未跑Test，Notebook／影片暫停。[資料量影響、成績、限制與命令](normal-data-support-20261012.md)。

最新指定Test更新（2026-10-12）：使用者在三候選完成後授權只評估大容量wide一次，不依Test調參。原門檻／split下Test TP10／TN8／FP0／FN5／U0，Accuracy18/23＝78.26%、Fall F1 .8000，Validation95.65%未延續；未改模型、校準、預設或舊guard。13相關測試、獨立CSV／K3及1103舊檔hash核對通過。歷史Test已查看、教師官方協定仍未知；正式Notebook／五分鐘影片暫停。[本次結果與驗收](ur-isotropic-wide-test-20261012.md)。

最新Train／Validation改善（2026-10-12）：依使用者禁止Test調參指示，診斷後固定三個等比例特徵AE，實際訓練121epochs，最高wide的Validation22/23＝95.65%、F1 .9655，無誤報／unknown；新增TP當前良好骨架警示guard未過，原預設保留。26相關測試、獨立CSV／K3驗收及1039舊檔hash保全通過。所有新候選未跑Test；舊Test82.61%／.8462仍是已查看結果，Notebook／五分鐘影片繼續暫停，官方協定可比性未確認。[本輪完整成績與限制](isotropic-refinement-20261012.md)。

最新Test更新（2026-10-12）：依使用者明確指示評估最高Validation候選`ur-human-isotropic-001`，固定原門檻與23支Test，Accuracy19/23＝82.61%、F1 .8462，TP11／TN8／FP0／FN4／U0；沒有回頭改模型／切分／校準或舊guard。Validation91.30%不是Test成績，目前Test未達課程參考值，歷史Test已查看限制保留，正式Notebook／五分鐘影片仍暫停。[正式結果與驗證](ur-human-isotropic-test-20261012.md)。以下保留各階段當時狀態。

最新已完成（2026-09-18）：AI目視關鍵影格診斷後，單一保留身體比例候選49epochs，影片級Validation提高至21/23=91.30%、F1=.9286；新補fall-30無可見警示，未通過事前guard，最高影片級候選另存，原預設保留。這是反覆使用的Validation，未新增Test，不能宣稱課程已全部達標；正式Notebook／五分鐘影片繼續暫停。215 tests passed／1歷史路徑skip、888幀真實核對、40舊run及8450cache保留。[方法、數字、短片及限制](human-isotropic.md)。以下保留歷史紀錄。

最新骨架／正常活動工作（2026-09-18）：翻轉骨架12片pilot未過gate，正常Train型態抽樣AE完成33epochs仍Validation20/23、F1=.8889，保留原基準；沒有新增Test，正式Notebook／影片暫停。正常活動抽查清單已提供，不能當完整逐幀GT。[本輪實作與驗收](human-quality.md)。

最新進度（2026-09-18）：依準確度優先指示完成兩個Train正常殘差加權候選，Validation為19/23與20/23，未提高Accuracy，保留原Human基準20/23、F1=.8889；未新增Test，正式Notebook／影片暫停。這補充下方歷史CA交付，不代表已達課程基準。[本輪實作與驗收](human-residual.md)。

初訂：2026-09-17；執行更新：2026-09-18。狀態：CA-01至CA-05本機實作已完成，辨識品質仍不足；公開上傳未執行。實際結果見[成果報告](course-results.md)。

最新指示優先於以下歷史交付順序：先改善辨識效果，暫停Notebook與影片更新。旋轉骨架pilot已實跑但未達事先門檻，未啟動其AE訓練或Test；見[Accuracy優先紀錄](accuracy-first-plan.md)。

本計畫承接使用者對課程要求、同學範例與目前 demo 的比較。執行順序為 UR Fall 診斷與評估協定、UR-only實驗、成果Notebook、展示畫面與五分鐘影片。沿用既有ST-GCN特徵與正常資料AE。2026-09-17先完成文件；2026-09-18已完成3候選297回合、Validation選擇、選定版本Test、Notebook及本機影片，沒有改動骨架品質／時序／事件規則。

本文件補充根目錄實作規格。課程來源、同學範例、專案工程選擇分開記錄；附件中的操作指示不構成上傳、寄信或執行他人程式的授權。

## 1. 已查核來源與課程要求

來源於本次對話中已讀取，以下是 2026-09-17 的查核紀錄：

- [課程網頁](https://lens.csie.ncku.edu.tw/course?id=aiad-2026-fall)，Project 區塊。
- [教師簡報公開連結](https://lens.csie.ncku.edu.tw/data/2026/2026-fall-aiad/slides/2026_AD_first_week_admin.pptx)，使用者提供的本機同名附件，第 36–51 張投影片，頁碼依實際投影片順序。
- [同學範例公開連結](https://lens.csie.ncku.edu.tw/data/2026/2026-fall-aiad/AD_project_Sample.zip)，使用者提供的本機同名附件。查閱 PDF、Notebook、展示程式與影片抽幀，未執行同學程式。
- [同學主要展示影片](https://www.youtube.com/watch?v=qZNQnxy6NIw)，約五分鐘，已抽查畫面；不宣稱逐秒完整審核。

| 項目 | 已確認內容 | 來源與性質 |
| --- | --- | --- |
| 題目 | 跌倒偵測作為異常偵測的應用 | 簡報 36–38，課程方向 |
| 方法示例 | ST-GCN 256D 特徵、Normalizer、AE、MSE 與門檻 | 簡報 39；示例不等於每個工程細節皆強制 |
| 評分資料集 | UR Fall 或 UP-FALL，選一個 | 網頁 Project、簡報 46–49 |
| 評分指標 | Accuracy 與 F1，兩項對應分數取平均 | 網頁 Project、簡報 49 |
| UR Fall 基準 | Accuracy 90.00%，F1 0.87 | 簡報 48 的表格 |
| UP-FALL 基準 | Accuracy 98.52%，F1 0.93 | 簡報 48 的表格；49 的例子使用約 98% |
| 展示 | 五分鐘影片，事先上傳 YouTube，課堂 demo | 簡報 50；課程日程列 12/24 展示 |
| 提案與期末內容 | 11/12 提案，5 頁、11pt、雙行距，說明方法與分工；期末包含原始碼與 demo 影片 | 網頁 Project、簡報 51 |

尚待釐清：官方 train/validation/test 切分、逐幀或逐影片等評估單位、unknown 如何計分、可否用額外資料集訓練，以及評分公式正文的 `×10` 與例子的 `×100` 不一致。未確認前不自動換算課程分數、不宣稱自訂協定直接重現基準。這些未定事項不阻塞 Train/Validation 診斷、成果整理與畫面開發。

2026-09-18補查：第45張有引用論文，不能概括說「提供資料都沒有切分資訊」。論文的實驗設定與「教師是否要求學生完全遵循」應分開；已將來源、方法差異及不可沿用的Test選模記於[論文補查](accuracy-first-plan.md#參考論文補查與下一步)。

本機附件實際 SHA256（不是上游公布的雜湊）：

| 附件 | 使用者提供位置 | SHA256 |
| --- | --- | --- |
| 教師簡報 | `C:/Users/Huayi/Downloads/2026_AD_first_week_admin.pptx` | `8e71b5649186683c2019b3ed10e984ed182324d15baf1bdfbeeb6e4a10221167` |
| 同學範例 | `C:/Users/Huayi/Downloads/AD_project_Sample.zip` | `02ea5eeae3964f4fc7560ac3e462a66f0b83da1efe8bfc929f929510e0cbc200` |

## 2. 同學範例可借鏡的部分與限制

Notebook `Accuracy and F1 score calculation.ipynb` 讀取 70 支 UR 影片的代表圖片，依檔名中的判定結果統計 TN29、FP11、TP30、FN0，得到 Accuracy 59/70 = 84.3%、F1 60/71 = 84.5%。2026-09-18已逐一檢查內層`Term_project/UR_Fall_dataset.zip`：只有40張ADL及30張Fall JPG，沒有原始連續影格、影片或模型權重。Notebook只展示圖片與統計檔名，沒有訓練或執行影片推論；這些截圖不足以重驗整段影片的事件及警示時間。附件沒有充分交代 UR 的訓練／驗證／測試角色，不能據此判定有資料洩漏，也不能直接與本專案 23 支 UR Test 比高下。[檔案清單、Notebook原始儲存格與hash](validation/course-sample-dataset-audit.json)。

範例 `main.py`、`main_avi.py` 的展示判斷取七類動作分類頭的 argmax，顯示 Standing、Walking、Fall Down 等類別與分數。報告提出 AE 流程，但附件展示程式沒有提供可驗證的 AE 重建誤差判斷；不能推論該組在附件之外完全沒做 AE。

借鏡其「正常活動、跌倒案例、績效整理」編排及容易辨識的標示。專案的 AE 分數不得乘以 100 後當跌倒機率，也不直接顯示本流程沒有產生的七類動作名稱。多人熱影像是範例的補充展示，未在已讀課程要求中列為必要項目。

## 3. 目前基準與保留範圍

目前已完成的混合資料基準為 `caucafall-yolo11-balanced-001`：YOLO11m Pose、COCO17 到 OpenPose18 的既有映射、凍結 Kinetics ST-GCN 分類頭前的 256D 特徵、正常 Train 擬合 Scaler／AE，再以 Validation 校準門檻。該 run 依其已鎖定設定，以 balanced sequence accuracy 選門檻，以 operational accuracy、跌倒召回、較少誤報的順序選模型；不能描述成已按原規格預設的 F1 校準，也不因課程重視 F1 就回頭更動既有門檻。

原 Human 雙流 512D 是歷史路徑，不能把其 Le2i 七類權重來源套用到目前模型。各預訓練模型的來源仍須如實陳述；不宣稱全流程從未接觸跌倒或完全無監督。

以下只摘錄既有 [AE 強化結果](refinement-training.md)，不是本輪新增實驗：

| 既有 Test 範圍 | 正確／全部 | operational accuracy | TP／TN／FP／FN／unknown |
| --- | --- | --- | --- |
| UR Fall | 10/23 | 43.5% | 3／7／1／11／1 |
| Le2i | 27/38 | 71.1% | 15／12／4／7／0 |
| CAUCAFall | 20/20 | 100.0% | 10／10／0／0／0 |
| 合計 | 57/81 | 70.4% | 28／29／5／18／1 |

UR 的 43.5% 是 unknown 留在分母的整體正確率。依既有規格，可判定部分的二元 Accuracy 為 10/22 = 45.5%，跌倒 F1 為 6/(6+1+11) = 33.3%；全體跌倒召回為 3/15 = 20.0%。不同定義必須分開標示，不拿排除 unknown 的指標單獨當作全體表現。

CAUCAFall 20/20 不能取代課程指定資料集成果，也不能推廣成未知場景可靠度。保留所有原始資料、既有 run、hash、設定及失敗案例。歷史 Test 已被查看；後續即使另建 run 或重新切分，也不能抹去這項限制或重新稱為未見過的獨立測試。

## 4. 方法與評估契約

1. 主課程實驗選 UR Fall。先建立使用現有可用特徵 recipe 的 UR-only 基準：Scaler／AE 僅讀 UR Train ADL、正常 UR Validation 選 checkpoint、UR Validation 校準。現有混合模型另列擴充比較，不能把它改名成 UR-only。
2. 先從 Train／Validation 做診斷：正常與跌倒分數分布、正常活動誤報、跌倒漏報、骨架品質、track 中斷、時間窗與 unknown。把原因分為已證實與待驗證假設，不先認定再加 epochs 就會改善。
3. 每輪在訓練前記錄有限候選、選擇指標、平手規則與停止條件；不在看過 Test 後追加候選追逐分數。固定模型、門檻、split、recipe 後才做該輪 Test 工程對照。
4. 延用原始序列先切分的隔離；保持 Test 影片不進 Train，Scaler／AE 不以 Test 擬合。新 UR-only run 與混合 run 分開命名與記錄訓練資料；重用特徵前核對來源、切分角色、recipe、權重及內容 hash。
5. 教師協定尚未釐清前，使用原規格的自訂 sequence 評估：任何確認事件為陽性、有合法 event score 而無事件為陰性、完全無合法 event score 為 unknown。這是工程選擇，不是已確認的課程計分規則。
6. 同時呈現可判定序列的 Accuracy／Precision／Recall／F1、2×2 混淆矩陣、含 unknown 的 2×3 表、n_total／n_evaluable／n_unknown／coverage，以及全體 operational accuracy 與全體跌倒召回。保留原公式，不臨時把 unknown 填成 normal 或偷偷移除分母；沒有可用樣本時明示未定義。
7. 影片曾警示不代表警示時機正確。保留首次警示時間，展示正常前段誤報等失敗案例；只有核對可靠 onset 標註後，才以獨立事件協定報延遲或提前預警指標。
8. 預訓練權重嚴格載入、取分類頭前特徵、正常 AE 與 Test 隔離等既有契約繼續適用。synthetic、real_feature_smoke、real_end_to_end 分開呈現。

## 5. 開發順序與可驗收成果

以下為實際交付及驗收狀態；不能把功能通過解讀為高準確率。

| 階段 | 工作 | 完成證據 | 目前狀態 |
| --- | --- | --- | --- |
| CA-01 | 課程UR協定與Train/Validation診斷 | `docs/urfall-course-protocol.md`、`docs/urfall-diagnosis.md`、course-ur-diagnosis.json | 本機完成；教師細節待確認 |
| CA-02 | UR-only基準及有限候選 | 3個新run、訓練前lock、Test前freeze；24支Train ADL／659窗；正式Test23支 | 已完成；成績不足、停止追加 |
| CA-03 | 成果Notebook | `notebooks/course-results.ipynb`與HTML，7 cells成功、2圖內嵌、hash／分母核對 | 已完成並重啟kernel再驗證 |
| CA-04 | 中文demo畫面 | 四支Validation／四種真實狀態；debug／presentation log與events逐位元相同 | 已完成；67個未遮罩窗分數差0 |
| CA-05 | 五分鐘期末展示 | course-demo-script.md；有聲／無聲本機成品300秒；13段抽幀與完整解碼 | 本機完成／未上傳 |

先完成 CA-01，再依診斷鎖定 CA-02。CA-03 和 CA-04 可用既有凍結結果先開發，最終成果須換成明確指定的同一版本。CA-05 整合完成後，更新驗收文件；規劃完成不等於辨識品質通過。

### 成果 Notebook

- 讀取正式評估輸出的 metrics、sequence-results、events、unknown 表，以及對應 run 的訓練紀錄。記錄 run ID、模型／split／recipe hash、資料來源、評估單位與所有分母。
- 表格、混淆矩陣、Loss 和文字摘要由這些證據產生；核對逐序列統計與 metrics 一致，不手填成績或由剪輯畫面反推準確率。
- 展示成功、誤報、漏報及無法判定案例，包含抽樣／選片原則。展示用 Validation 案例與 Test 成績分開標示。
- 預設只讀既有產出，缺少檔案或 hash／分母不符就明確報錯；執行全部儲存格不應默默啟動訓練、校準或重新測試。
- 清楚標示條件式二元指標與全體指標；教師協定未確認前，基準表只作來源參考，不自動換算課程分數。

### Demo 畫面

| 核心狀態 | 展示文字 | 視覺提示 |
| --- | --- | --- |
| warming_up | 分析中：收集動作片段 | 中性色，不能當正常 |
| normal | 正常活動 | 綠色加文字 |
| suspected_fall | 疑似跌倒！ | 紅色加文字 |
| unknown | 暫時無法判定 | 黃色加簡短原因，例如骨架不足 |

- 人物附近顯示編號與大字狀態；底部資訊區顯示原影片時間、異常分數與門檻，避免小字擠在骨架上。文字不能只靠顏色區分。
- AE 重建誤差不是機率，不顯示虛構的「跌倒 XX%」，也不把 normal 細分為未由本流程辨識的 Walking／Sitting。
- 展示模式可將精確 `score_age_ms`、內部原因碼移至除錯模式；保留「分數更新於影片 t 秒」等可辨識的新鮮度提示，unknown 時不留舊分數造成誤導。CSV／JSON 保留原始 age、原因與狀態。
- 這是對原規格「畫面標示 score age」的呈現調整，不修改 `2*stride/target_fps` 的分數保留上限、品質失效或警示觸發規則。
- 選用可用且授權可記錄的中文字型，實際檢查缺字、遮擋與可讀性。技術退路仍可輸出英文，但正式中文展示驗收須有可讀中文。
- UI 改版前後用相同輸入及固定 run 對照分數、狀態、事件與時間戳；完整解碼影片，確認播放速度和來源時間一致。

### 五分鐘期末影片

建議編排：方法與正常訓練資料、正常活動、跌倒案例、Accuracy／F1 與混淆矩陣、失敗案例與限制。UR Fall 為主要證據，額外資料集明標為擴充。片段、分數與模型版本一致；顯示原影片時間，不靠剪掉誤報或改播放速度營造即時效果。五分鐘是課程要求，不把單支十秒推論輸出當完整交件。

## 6. 驗收與本次完成範圍

- 文件：課程要求附來源、工程預設及待釐清事項分開；根規格、README、進度、驗收頁均能找到本計畫。
- 實作：CA-01 至 CA-05 完成時才逐項勾選，保留真正執行的命令、結果及限制。資料／模型／時序／事件規則有變動才跑相關風險測試並更新證據。
- 成果：UI 可讀與 Notebook 可執行不等於性能改善；性能改善需在既定協定與分母下證明。95% 是先前工程目標，不是本次查到的統一期末通過門檻。
- 2026-09-17僅完成文件；2026-09-18已完成上述本機成果。136 tests passed／1舊Human資源skip，20個舊run不變。新UR-only Test全體11/23（47.8%）、可判定Accuracy50%／F1 56%、unknown1；相比混合模型多抓4支跌倒也多3次誤報。沒有公開上傳或自動換算課程分數。
