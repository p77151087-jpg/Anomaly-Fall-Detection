# 大容量落差、正常資料量及 Train 限定改善（2026-10-12）

使用者要求找出大容量Test更差的原因、只用Train／Validation改善，並詢問資料太少是否有關。本輪封存Test，只保全既有輸出／hash，不新增Test推論，不以Test漏報分數／身分選擇模型、門檻、品質或資料。保留ST-GCN512D＋正常AE，不改七類分類、不重切split、不重寫歷史結果；正式Notebook／五分鐘影片仍暫停。

## 已證實與尚未識別的原因

保存的大容量Test78.26%／F1 .8000比舊isotropic82.61%／.8462多漏一支，兩者FP0／影片unknown0。原split、Pose／ST-GCN快取、品質規則及有效窗口覆蓋率相同，只改AE及各自在Validation校準的門檻。因此差異定位於AE分數與門檻分離，不能歸因不同骨架輸入或漏算分母。沒有以Test做反事實門檻／模型掃描，不能單憑一支差異斷言「大模型一定嚴重過擬合」。

新Train／Validation診斷：原AE140064參數、大容量296000（2.11倍）；乾淨Train平均MSE .0568165→.0562264，正常Val .1151349→.1156164，沒有改善，Val／Train比2.03→2.06。大容量多判對一支Val，卻沒有較好的平均正常重建泛化，95.65%不是普遍更好的證據。

僅對保存的Validation作逐一省略正常影片診斷，只有省略最高正常分數影片會改變校準：原adl-37省略時門檻 .4529072→.4172828；大容量adl-35省略時 .5341114→.4313048。這顯示僅8支正常Validation中，少數困難正常活動對門檻影響大。**未採用省略後的門檻，未移除困難案例，原split不變**；這是敏感度診斷，不是新驗證成績。

## 資料太少是否有關：學習曲線

原Train只有24支ADL，3638有效stride1窗口；按30幀不重疊選取有效端點只剩135個。這些仍來自24支影片，不等於135個獨立人物／場景。Validation只有23支，每多錯一支影響4.35百分點；歷史Test已看、Validation反覆選模，偏差貢獻尚未量化。

結果產生前固定兩個Train影片排列seed42／43，各取8／16支巢狀子集。模型seed42、AE128／32及原選模流程固定，每組最多100epochs／patience10。共用核心train／Scaler／AE，只替換Train矩陣；checkpoint仍由原正常Validation選擇。70支原split不變；子集是診斷，不當新Train／Test切分，不校準分類門檻、不選為正式候選、不跑Test。

四組實際68／78／39／43＝**228epochs**。每個子集Scaler只用其Train正常資料，所以原生Loss的尺度不同；另把重建誤差換到**共同Train24尺度**，該Scaler僅供診斷比較，不回用子集訓練。

| 原Train影片數 | 原生正常Val Loss（兩個排列） | 共同Train24尺度的正常Val MSE |
|---|---|---|
| 8 | .226409／.228942 | .223885／.197463 |
| 16 | .123328／.148789 | .125955／.138108 |
| 24（既有模型） | .115135 | .115135 |

兩個排列在共同單位下都呈現增加原Train影片後誤差降低，**支持正常資料覆蓋／多樣性不足可能有關**。但影片組成、每epoch的batch數及早停回合同時不同，並非完全隔離訓練步數的因果實驗；24支既有模型也沿用舊CUDA環境設定。它不能量化Test下降多少由資料少造成，也不能保證增加影片就一定達標。骨架出畫面、短跌倒上下文與正常／跌倒特徵分離問題仍在；影像沒有拍到的身體，增加AE訓練不能直接恢復。135不重疊端點亦不是統計上的等效獨立樣本數。

## 固定 Train 鏡像實驗：已完成，沒有Accuracy提升

診斷後、特徵增強／辨識結果前固定 `ur-mirror128-20261012-001`（128／32）及 `ur-mirror256-20261012-001`（256／64）。只對24原Train ADL已觀測30幀骨架作固定水平鏡像與左右關節互換，經真實凍結ST-GCN。原＋鏡像1:1，共7276訓練列，**不是新增24支真實影片**；不補造缺失關節、不改品質／時序／K3或Validation輸入。

共用核心train／calibrate／Scorer／event規則，只有明示Train矩陣供應者使用已驗證的衍生矩陣。Scaler／AE只用此Train正常矩陣，modelseed42、sequence sampling、wd.001、noise0，正常Val選checkpoint、完整Val balanced sequence Accuracy選門檻，最多各100epochs／patience10。兩組實際完成42／42＝**84epochs**。

| 候選 | 最佳正常Val Loss（各自Scaler單位） | TP／TN／FP／FN／U | 二元／全體Accuracy | 可判定Fall F1 |
|---|---:|---|---:|---:|
| mirror128 | .103395 | 13／8／0／2／0 | 21/23＝91.30% | 26/28＝.9286 |
| mirror256 | .103950 | 13／8／0／2／0 | 21/23＝91.30% | 26/28＝.9286 |

原生正常重建Loss較低，Scaler單位不同不能直接比較幅度；另以同一Train24單位重算mirror128／256為 **.105392／.105900**，相對原.115135確有較低正常重建誤差。即使如此，Accuracy仍未超過最高Val參照wide22/23＝95.65%，不能當跌倒召回改善。鏡像兩組漏報均fall-19／18，粗略時段命中均3，比參照4少，沒有救回更多跌倒。兩組均未通過採用guard。比較基準沒有因wide的Test較差而降回21/23；所有鏡像候選未跑Test，原預設保留。

事前guard：Accuracy嚴格高於22/23、F1不下降、無新增FP／unknown、舊六時段至少4命中／0早報、新救回漏報有當前良好骨架警示；平手保留參照。固定兩個鏡像AE已跑完並停止，沒有在此實驗追加其他容量／變換／seed／Test。

## 配置註記修復與保全

首次訓練入口因strict config拒絕`experiment_training_inputs`註記，未建立任何候選run或讀取候選辨識結果。保留原source v1、原兩個配置、診斷／候選lock與hash；從原設定派生唯一移除註記的`*-core.yaml`，模型參數、seed、資料及規則逐值不變。註記改放新run的sources，綁定augmentation profile／實際矩陣／工作流程／修復hash，並在校準前凍結新run來源。沒有改舊run或核心程式。

原dataset-index用於追溯與推論；profile另列原、鏡像及合併矩陣hash。sources的Train hash是實際合併NPZ，非冒稱原快取。配置修復完整紀錄為`normal-data-support-20261012-implementation-repair.json`；新source hash與原source封存核對，未追溯修改原lock。新CUDA實驗設定CUBLAS_WORKSPACE_CONFIG=:4096:8，不重跑原實驗，不保證跨硬體逐位元相同。

## 後續獨立的殘差尺度實驗

鏡像有限實驗完成後，使用者原有持續優化授權仍適用；另於結果前固定兩個既有Train殘差計分模式，以最高Val wide為來源：`ur-isotropic-wide-channel-20261012-001`（train_channel_mse）及`ur-isotropic-wide-stream-20261012-001`（train_stream_mse）。只用原24Train ADL估計重建殘差尺度，固定逆平均誤差／floor公式，沿用凍結AE／Scaler，新增AE epochs＝0。完整Val重新校準／選擇，保持原窗口品質、K3及相同採用guard；兩模式跑完即停止，不新增Test。此為獨立score實驗，不追加到鏡像候選；舊per_frame殘差實驗不冒稱已驗證本isotropic特徵。

兩模式均已完成，AE／Scaler與來源wide逐位元相同，新增訓練epochs＝0，沒有冒稱重新訓練。結果如下；兩組皆無新增判對影片，平手依規則保留參照，不切換預設。

| 殘差模式 | 凍結Val門檻 | TP／TN／FP／FN／U | 二元／全體Accuracy | 可判定Fall F1 | 粗略時段命中／早報 |
|---|---:|---|---:|---:|---|
| channel | .318937838 | 14／8／0／1／0 | 22/23＝95.65% | 28/29＝.9655 | 5／0 |
| stream | .412708282 | 14／8／0／1／0 | 22/23＝95.65% | 28/29＝.9655 | 5／0 |

兩者仍漏fall-19；fall-09首次事件5900→5100ms，增加一個舊粗略時段命中。沒有正式時間GT，也沒有新demo逐影格檢視，不能冒稱警示延遲成績達標；新增Accuracy＝0。fall-18的1433ms事件時當前骨架限制仍在。這是有限案例的時間觀察，沒有依Test調整任何規則。

## 驗證與成果

27項相關synthetic風險測試通過；鏡像可逆、信心／觀測輸入不變、Train正常標籤隔離、拒絕Test、矩陣hash與既有AE／K3均驗證。真實ST-GCN權重strict load，24支原首／尾共48特徵parity逐值完全相同（最大差0）、原窗口record／品質／時間完整相同。

`real_feature_smoke`：四個真實特徵學習曲線、Train鏡像特徵重新經凍結ST-GCN、兩組AE實際訓練／校準／23支Val；sklearn二元指標／全體分母、46支次K3／事件及原規則重算一致，兩Scaler與7276個Train衍生列重新fit逐值一致。`real_end_to_end`：本輪沒有重新從原影片抽取Pose或新demo；synthetic通過不當辨識效果。

1110既有檔案hash不變，含原Test輸出、44舊凍結run、核心source、設定、原快取／權重與先前Validation證據。後續殘差派生前亦另保存兩鏡像run檔案hash，結果後均相同。新來源／特徵／模型另列新路徑。初期graph先查且coverage無記錄缺口；後續MCP連續回報Transport closed，殘差模組完整直接讀取，coverage無法刷新，這個限制明示，不作完整graph結論；仍可用原始碼及測試完成工作。

殘差另有10項相關synthetic測試通過；本次合計 **27＋10＝37 passed／0 skipped**（兩份XML）。獨立重算兩殘差mode的Train來源24／3638、逆殘差權重、AE／Scaler相同hash、46支次Val CSV／K3／事件與採用guard一致。全部四候選Val的影片unknown0，所以二元／含unknown全體Accuracy分母皆23；鏡像／殘差全體Fall召回分別13/15＝86.67%及14/15＝93.33%。時間案例不替代這些分母。

本次**228個學習曲線＋84個鏡像＝312個實際新AE epochs**，另2個殘差score新增epochs0。4個未校準學習曲線僅供資料量診斷，2鏡像及2殘差是兩個各自事前鎖定、已停止的有限改善實驗。所有新候選Test未知；不以舊wide78.26%／.8000或原isotropic82.61%／.8462改選基準，不宣稱提高Test、課程得分或新的盲測。

證據：`docs/validation/normal-data-support-20261012-{diagnostic-lock,diagnosis,common-scale,candidate-lock,mirror,validation,verification}.json`及配置修復紀錄。學習曲線模型在`outputs/normal-data-support-20261012/learning/`，Train衍生資料在`data/cache/ur-train-mirror-20261012-001/`；原始影像／權重／快取不加入Git。兩鏡像正式Val輸出在`outputs/normal-data-support-20261012/<run>/`，比較圖同目錄；殘差獨立紀錄另用`isotropic-residual-support-20261012`前綴。

## 操作、限制及未完成項目

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe scripts/normal_data_support.py status
.\.venv\Scripts\python.exe scripts/verify_normal_data_support.py
.\.venv\Scripts\python.exe scripts/isotropic_residual_support.py status
```

完成後freeze／diagnose／lock／mirror／training與殘差freeze／derive都拒絕覆寫，已實際驗證。配置修復後原選擇不變，Ruff通過，比較圖目視。資料量curve、鏡像與殘差各自有完整來源／模型／Loss／設定／門檻／窗口／事件／hash；對應兩份驗收及最終QA保存。

已完成本次可執行的診斷與有限實驗，**沒有提高目前最高Val Accuracy**。資料多樣性確實影響正常重建，但這不能恢復相機外的關節、保證正常與跌倒可分，或保證Test達標。後續最有用的協助是補更多真實正常活動、不同人物／拍攝位置與地面活動，並標註Train／Validation跌倒起迄與關節可見性；新增真實資料另立來源及協定，不挪用現有Test，不重切來宣稱新盲測。只加鏡像／重複窗口不等於增加獨立場景。

仍未完成：fall-19辨識及穩定可見警示、教師官方評估單位／split確認、所有新候選Test（依指示不跑）、暫停中的正式Notebook／五分鐘影片。graph服務有連線限制，但已用直接原始碼與測試完成受影響驗證，沒有阻塞本次實驗；未公開上傳。
