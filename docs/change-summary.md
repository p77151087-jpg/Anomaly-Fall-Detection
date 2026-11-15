# 本次變更

由只有實作規格文件的 Git 工作區，建立 `src/fall_detect` Python 套件、唯一 CLI、設定、型別、來源管理、資料／模型／訓練／事件／評估／影片流程與風險測試。原始規格文件保留不改。

- 新增 `.gitignore`，隔離上游 checkout、Python 環境、原始資料、模型權重、cache、run 與影片。
- 固定 Human commit；嚴格 tensor-only 權重讀取；CPU/CUDA 明確傳入；分類頭前抽取 512D，凍結網路。
- 固定完整 70 序列 split；下載小型 Train／Validation 真實資料；不先切窗口再混合分組。
- Reader 以原檔 frame ID／timestamp 讀取，數值排序、中文路徑、重複／缺失／壞檔／時間檢查；缺圖明確保留 missing。
- 新增因果重採樣、全序列固定預定端點、品質策略與分開的骨架／特徵 cache，驗證版本、識別、shape、有限值和對應關係。
- 新增 Train-only scaler、MLP AE、正常 validation early stopping、原子保存、reload；測試資料更動不影響模型選擇。
- 新增共用 K=3 事件流程與 float64 nextafter 校準、已校準 run 的 artifact hash 核對。
- 新增 sequence metrics、unknown、coverage、2×3 表、獨立評估目錄、文字與圖表、headless 影片及時間對應 CSV。
- synthetic 和 real_feature_smoke 路徑、狀態、報告分離；正式 demo 拒絕 synthetic run 或未校準模型。
- 反覆驗證時修正 tensor advanced-index clamp、detector confidence 欄位、缺少時間軸的 coverage 分母、缺圖時間缺口、最佳 checkpoint 與 min_delta early-stop 分工，以及畫面無效後不復用舊正常狀態。

上述為2026-09-11原Human方案紀錄，當時缺四個模型資源。沒有做Git commit；原始權重／資料／環境維持Git忽略。

## 2026-09-17 第二條路

- 依使用者選擇新增獨立OpenPose18／256D recipe、共用backend factory、固定來源commit及兩個checkpoint的strict load；prepare與demo共用介接。原Human設定與快取不混用。
- 完整載入Lightweight模型三個refinement stages，推論取原作者demo使用的第一stage；原始ST-GCN抽分類頭前256D。CPU局部停用過慢的oneDNN路徑，仍使用相同真實權重。
- 下載全70段資料並處理11,936張RGB；新增每序列獨立session、最多4段並行下載。adl-37最後20張缺時間戳，僅顯式允許尾端30fps推估並保存estimated旗標；中間缺口仍拒絕。
- 實際產生777個有效特徵窗，完成正常Train AE的100 epochs、Validation校準、固定Test及三支真實demo。
- Scorer固定B=1，保證校準／demo的嚴格門檻比較不因批次浮點差異而變；來源記錄分開保存擷取與訓練時程式hash。此修正在Test前完成。
- demo EOF使用實際來源結尾時間，新增尾端非整格時間測試；最終54 passed、1原Human資源skip。正式run凍結hash未改。
- 更新使用、來源、資料、環境、驗收及結果文件。Test效果不足如實保留：8 unknown、7/15跌倒警示、7/8正常誤報；沒有拿synthetic數字充當成果，也沒有用Test重調此run。

## 2026-09-17 程式內 Loss 畫面

- 新增核心 loss 命令與「查看Loss曲線.cmd」，提供繁體中文本機介面、run 切換、曲線、最佳回合與逐回合表格。
- train 每回合原子保存 training-progress.json；--show-loss 開啟自動刷新的曲線。原訓練選模與凍結實驗不變。
- 已完成真實紀錄與瀏覽器驗證、synthetic 實際 AE 即時更新、76 passed／1 skipped 全套測試；最後缺省數量顯示修正後7項再通過。
- 操作與證據見 docs/loss-viewer.md、docs/validation/loss-viewer-checks.json。


## 2026-09-17 接續訓練

新增可選正常影片均衡抽樣、可追溯抽樣來源與Loss提示；完成三組AE實驗577回合但整體準確率未提升。修正Windows原子替換短暫鎖定的有界重試、訓練失敗狀態與畫面。詳見[further-training.md](further-training.md)。
