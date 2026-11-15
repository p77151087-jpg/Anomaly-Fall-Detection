# 維護規則

- 遵守上層 codebase-memory 指示：結構探索先查 graph，檢查 coverage，再讀必要原始碼。
- 根目錄 `Fall_Detection_Codex_Implementation_Spec.md` 是方法及驗收契約。
- 使用 `.venv`，不修改系統 Python。核心入口為 `python -m fall_detect`。
- 先原始序列切分，Train 僅 ADL。Scaler/AE/門檻禁止使用 test 選擇。
- Pose/ST-GCN 必須嚴格載入真實權重；不可降級為随机特徵或分類頭分數。
- 修改資料、模型、品質、時序或事件規則後跑相關風險測試；更新 `docs/current-progress.md` 和驗收文件。
- synthetic、real_feature_smoke、real_end_to_end 證據必須分開。
- 原始資料、上游 checkout、權重、快取與環境不可加入 Git；來源及實際 hash 必須保存。
