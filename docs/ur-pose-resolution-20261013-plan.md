# 骨架品質對照：單一高解析度人物crop候選（事前契約）

2026-10-13。使用者授權繼續骨架品質對照，實際確認fall-19漏報能否改善；僅Train／Validation，不跑Test，不覆蓋舊成果。本契約在任何新候選Pose推論、特徵重建、AE訓練或校準前固定。

## 假設與範圍

本人fall-19補註確認transition f33–61、ground f62–100是整段姿態階段，精確事件時點未知；f44–51的原Pose被人工判wrong_or_drift，但confidence規則仍good。原三路transition21/29合法且Raw／K3低門檻，ground0/38合法。前段低分與後段觀測不足並存，未證實唯一原因。

可驗證假設：原AlphaPose的256×192 crop／64×48 heatmap可能使人物局部定位受限；同一ResNet50真實權重採512×384 crop及相應較密熱圖，可能改變錯点或穩定性，進而改善異常分離。這是輸入解析度對照，不是已確認原因；放大crop不創造原圖資訊、不恢復出框身體，也不能把較高信心當成正確點位。

歷史Human flip TTA、YOLO960／低偵測門檻／旋轉候選已停止，本輪不重跑其網格。**唯一候選 `ur-pose-resolution-20261013-001`**；全47片同樣處理，不只修fall-19，不以其姿態標籤選取輸入。教師方向仍是正常資料學習與跌倒異常；512×384是本實驗選擇。

## 固定設定與隔離

- 比較基準 `ur-isotropic-wide-20261012-001`：原Validation TP14TN8FP0FN1U0，Accuracy22/23＝95.65%、Fall F1 .9655。demo仍 `ur-sample-human-s1-001`，與基準分開。
- 僅改Pose的input_height 256→512、input_width192→384。人物偵測器仍384／confidence .45、相同權重與NMS／track proposals／track規則；不換Pose、不旋轉、不flip、不補或插值關節、不放寬品質。
- 30幀／stride1／per_frame_isotropic、真實凍結ST-GCN分類頭前512D、quality .2／9實測關節／80%好幀、bounded_jitter及K3／strict >均沿用wide。
- AE256/64、seed42、最多100epochs／patience10／min_delta1e-6、lr.001／wd.001、batch32、sequence sampling、noise0，其餘同wide。原24Train ADL的全部合法窗fit Scaler／AE；正常Validation8影片選最低實際MSE checkpoint。
- 原70序列split不改。完整47Train／Val prepared；23Test維持not_prepared且無feature／pose路徑。不解碼Test特徵／影像、不評估Test。舊Test只可做bytes保全hash，不按歷史結果選設定。外部正常資料繼續封存。
- 新設定、新run、新recipe hash、cache、權重manifest、資料manifest及輸出均為獨立目錄；舊run／資料／設定／權重／cache／hash與契約先保存。CIM不可用時用psutil盤點程序，記錄限制；不得和其他程序共用可寫目錄。

## 執行與停止

先跑固定四支完整原序列pilot：Train adl-05／adl-24；Validation adl-04／fall-19。只有技術gate：三套預訓練網路strict真實權重、512×384 Pose forward有限、512D分類頭parity、來源ID／時間／尺寸一致、品質不變、合法窗口與快取隔離可核對，正常Train／Val均有合法特徵。pilot不以confidence或fall-19分數升降決定新增設定；不計Accuracy。

gate通過才完整prepare47片，四支pilot同recipe完整cache可經來源／hash嚴格驗證後複用，避免重抽。完整prepare後只訓練一次AE、只校準一次、只輸出一次完整Validation；技術失敗保存原因及輸出，不隨機特徵fallback，不調尺寸、容量、seed或品質。全部有限工作完成即停止。

門檻採與wide相同的既有 `balanced_sequence_accuracy` 校準（unknown保留分母；較少FP、較高threshold解平手），僅完整Validation23支；不同recipe不沿用舊AE／Scaler／數值門檻。候選與基準比較依含unknown整体Accuracy、Fall F1、全體Fall Recall、較少FP、基準優先解平手。單一變因對照不以新Loss降低或更多合法窗冒稱Accuracy提高。

## 驗收與採用

正式列出可判定二元分母23-U與含unknown整體分母23，2×3混淆表、Precision／F1、含unknown全體Fall Recall分母15、FP／FN／unknown清單、端點coverage及事件時間。Precision／可判定Fall Recall與全體Recall分開；window／frame分母不混稱影片數。

比較每支原23Validation，確認fall-19影片判定及transition／ground的實際事件、其他原TP是否退步、正常誤報和unknown是否增加。既有六片AI粗phase僅診斷，另列事件命中／早報及當前good警示展示guard；本人fall-19phase也是姿態區間，不能計精確onset延遲。原人工pose_check僅評原Pose，新Pose只作並排可視診斷，不偷用舊標籤稱新Pose正確率。

採用要求全部通過：含unknown Accuracy嚴格高於22/23、F1≥.9655、全體Fall Recall≥14/15、FP≤0、影片unknown≤0，無原TP退步；fall-19須在本人transition或ground有新事件且當時當前Pose達原品質，六片粗時序／可見guard不退步、無新增粗早報。若僅影片判對但時間／可見guard不足，列為辨識結果，預設仍保留；guard屬工程展示檢查，非教師計分公式。平手／Loss或coverage增加不採用，任何未通過即保留demo，不跑Test。

## 必要驗證與證據

synthetic測試：唯一配置差異、recipe隔離、pilot來源、Test／Val訓練隔離、缺失不造Pose、原品質與事件規則、來源／track／因果時間及比較不修改原觀測。相關既有cache／train scope／Scaler／校準／K3風險測試一併執行。

real_feature_smoke：實際Pose／ST-GCN strict真實forward、完整47片新cache、新正常AE訓練及完整23Val評估；獨立重算Train-only Scaler、最佳正常Val checkpoint、保存重載分數、正式CSV指標／混淆矩陣／分母、逐窗口K3與完整事件、Validation門檻排序／hash防覆寫。原RGB新抽取如實記錄；完整Test／demo的real_end_to_end仍未執行，不混稱完成。

每階段更新進度／驗收，全部成功與失敗保存；最終新增報告、只讀verify命令。新版Notebook／五分鐘影片及公開上傳仍暫緩，Test未知，不換算課程分數。
