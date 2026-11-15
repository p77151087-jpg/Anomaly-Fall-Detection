# 維護規則

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
