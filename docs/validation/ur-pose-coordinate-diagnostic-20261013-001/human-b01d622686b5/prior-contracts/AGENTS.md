# 維護規則

## 2026-10-13 可見關節座標控制診斷

- 使用者另授權 `ur-pose-coordinate-diagnostic-20261013-001`：凍結原wide30 isotropic ST-GCN／Scaler／AE／門檻／K3，只對Validation fall-19第44–51幀track1的可見左手肘XY做小規模控制；不是追加512解析度模型候選。事前契約及來源hash另存。
- AI暫定座標已實跑：raw max .18391→.20033、K3 max .13464→.15106，仍低固定 .53411，23/71合法窗不變、仍FN／零事件；不採用、不改正式Accuracy或demo。不能称人工修正已成功。
- 人工關節座標仍pending；舊人工pose_check是分類標籤，不是joint真值。8點覆核頁只匯出一份JSON，human確認後使用不同recipe／cache另存。看不清／出框點不補造；原confidence、bbox、track、時間與品質不改。禁止依分數搜尋座標。
- 不fit、不重訓、不校準、不載入Test特徵／解碼Test影像／評估Test。完整性hash不作Test模型選擇。只許Train adl-05與正常Val adl-04各3合法窗做固定不變控制；原資料／成果／舊契約副本保留。詳見 `docs/ur-pose-coordinate-diagnostic-20261013.md`。

## 2026-10-13 骨架品質對照授權

- 本候選已結案：AE32epochs／正常Val最佳22，Val整體16/23／F1 .7500，fall-19仍FN，低於wide；不採用、原demo保留、Test未知且未評估。不得在同一run追加候選或覆寫；成果與只讀命令見 `docs/ur-pose-resolution-20261013.md`。
- 使用者已授權 `ur-pose-resolution-20261013-001` 單一骨架品質候選：相同真實AlphaPose權重，人物crop由256×192改512×384；人物偵測、追蹤、原品質、30幀isotropic 512D、wide AE設定與K3保持一致。
- 原24Train ADL／23Validation；新recipe、cache、權重manifest與run另存，Test23保持not_prepared且無路徑、禁止Test特徵／影像解碼及評估。既有候選／契約／hash保存，不重跑舊網格。
- 依 `docs/ur-pose-resolution-20261013-plan.md` 先technical pilot，再完整47片新Pose／特徵及一個正常AE。原人工pose_check只標原Pose，不能沿用來宣稱新Pose正確；時間phase非精確事件GT。舊motion階段已完成，下面「只做motion」為當時範圍。

- 遵守上層 codebase-memory 指示：結構探索先查 graph，檢查 coverage，再讀必要原始碼。
- 根目錄 `Fall_Detection_Codex_Implementation_Spec.md` 是方法及驗收契約。
- 使用 `.venv`，不修改系統 Python。核心入口為 `python -m fall_detect`。
- 先原始序列切分，Train 僅 ADL。Scaler/AE/門檻禁止使用 test 選擇。
- Pose/ST-GCN 必須嚴格載入真實權重；不可降級為随机特徵或分類頭分數。
- 修改資料、模型、品質、時序或事件規則後跑相關風險測試；更新 `docs/current-progress.md` 和驗收文件。
- synthetic、real_feature_smoke、real_end_to_end 證據必須分開。
- 原始資料、上游 checkout、權重、快取與環境不可加入 Git；來源及實際 hash 必須保存。

## 2026-10-12 正常資料方法比較授權

- 教師方向是以正常資料學習、將跌倒視為異常；簡報列出 One-class classification 與 Autoencoder，ST-GCN→Normalizer→AE 是示例。
- 使用者已授權另建 One-class 模型、替代 AE／時序表示，以及有診斷依據的 Train ADL 自監督微調。固定 ST-GCN、不微調、30 幀、小型 MLP AE、K3 是歷史工程預設，不是永久研究禁令。
- 本輪專案學習資料仍限原 24 支 UR Train ADL；學習式前處理僅 Train 擬合，正常 Validation 選 checkpoint、完整 Validation 校準與比較。不得新增 Test 評估、搬動 Fall 或 Val／Test 增強副本進 Train。
- 每階段先鎖定有限候選、校準／選模／採用／停止規則。舊契約與 lock 原樣保留；任何新方法均保存自己的設定、權重、來源、recipe 與快取。
- 辨識指標與展示品質分開。舊可見警示 guard 保留為展示檢查，不改寫為教師指定計分公式；分數不是機率。
- 新版 Notebook／五分鐘影片暫緩，不公開上傳。詳見根規格最新授權節與方法比較報告。

## 2026-10-12 接續階段：全域動態正常近鄰

- 前輪六 SVM／兩 AE 網格已停止，接續只做 `ur-motion-novelty-20261012-001`：凍結 wide AE 加原骨架因果 90D 軌跡的正常近鄰側分支；事前計畫 `docs/ur-motion-novelty-20261012-plan.md`。
- 90D 衍生表示與原 512D ST-GCN 快取分開，來源影片／track／時間／品質保持一致。近鄰與定標只 fit 原 Train ADL，Test 保持未準備、不評估。此為新實驗選擇，非教師指定或永久工程限制。
