# 驗收結果

日期：2026-09-11。**pytest：40 passed、1 skipped；skipped 為真實 checkpoint 因四個外部資源缺失而受阻。** `ruff check src tests` 通過。原始測試證據為 `docs/validation/pytest.xml`；標準 CLI 的命令、實際退出碼與日誌見 `docs/validation/cli-checks.json` 和同目錄 `.log`。

本文件用 passed／blocked／not_run 明確区分。synthetic 通過只證明程式，不等於真實權重、資料或辨識效能通過。

| 規格第 15 節項目 | 狀態 | 實際證據與範圍 |
| --- | --- | --- |
| 輸入檔案與時間 | passed | test_data_contracts：1/2/10 numeric sort、frame-ID 時間對齊、缺圖保留、重複／倒退拒絕。另對 290 張真實 RGB 解碼並對齊 CSV |
| Split 隔離 | passed | 完整 70 ID、ADL 24/8/8、Fall 0/15/15、train 僅 ADL、來源 ID 不重複，manifest hash 驗證。實際完整資料下載為 not_run |
| 骨架排列 | passed (synthetic) | COCO17 保留索引、13→14 肩平均含 confidence、非正方形 width/height |
| 前處理 parity | passed (synthetic) | test_skeleton_timing 對固定上游 normalize_points_with_size／scale_pose 數值一致、不改原輸入、zero span／NaN／Inf 拒絕 |
| 張量與抽取點 | passed (synthetic); blocked (real) | CPU/CUDA 上原雙流 points/motion、512D、重接 fcn＋sigmoid 重現原 forward；無真實預訓練權重 |
| Checkpoint | passed (synthetic rejection); blocked (real) | 受限 strict load、統一 module. prefix、shape／key／HTML 拒絕；真實權重測試明確 skipped |
| 凍結模型 | passed (synthetic) | AE backward/optimizer 前後 ST-GCN state_dict（含 BN buffers）完全不變，無 grad |
| Scaler／Test 隔離 | passed (synthetic) | 改 test 特徵重跑後 scaler、AE tensor、threshold 相同；evaluate 前後 artifact hash 不變 |
| 因果性 | passed (synthetic) | 改未來資料不改較早窗口、重採樣不用未來觀測、換 track／sequence／缺口處理 |
| 缺失與 coverage | passed (synthetic) | no_person／track_gap／低品質、固定端點分母、短片 null；時間軸缺失令整體 coverage=null，不能縮分母 |
| 序列與事件 | passed (synthetic) | 三高才警示、三低可判 normal、unknown 重設、跨人物不能湊 K；保存實際確認時間 |
| 校準一致性 | passed (synthetic) | max rolling-min sequence score 與逐窗 EventState 在多組分數和門檻相符 |
| 門檻邊界 | passed (synthetic) | score==threshold 不觸發，向下 float64 nextafter 觸發，JSON 重載一致 |
| 指標 | passed (synthetic) | 手算 TP=2/FP=1/FN=1/TN=1、F1、2×3、operational accuracy 與 end-to-end recall；unknown 不當 TN |
| 保存與重載 | passed (synthetic) | Scaler NPZ／AE strict state dict／threshold 重載分數一致；修改 frozen threshold 拒絕 |
| Batch／streaming | passed (synthetic) | 相同骨架存 pose cache，再批次與 streaming，窗口分數／狀態／events 一致；含一個中途 missing |
| Device | passed (core/synthetic); blocked (full real) | CPU AE全流程、CPU/CUDA TSSTG與FastPose結構forward、明確 cuda 不可用會報錯；完整真實 CPU/CUDA 推論未驗證 |
| 影片輸出 | passed (synthetic); blocked (real) | mp4v/MJPG 實際編解碼；85 frames MP4 640×480/30fps、狀態CSV一致、無人立即unknown且不復用舊正常、無GUI；圖及影片抽幀經視覺檢查 |

## CLI 預期退出狀態

| 情境 | 預期 | 意義 |
| --- | ---: | --- |
| 所有八個子命令 `--help` | 0 | 參數文件可執行 |
| core doctor CPU／CUDA | 0 | 核心環境與codec可用 |
| full doctor | 2 | 權重未驗證、只下載2/70；正確報阻塞 |
| prepare --inventory-only | 2 | 紀錄70段，全體資料不完整 |
| prepare | 2 | 缺真實權重，不以random features繼續 |
| real smoke | 2 | 保留290張真實輸入驗證；blocked_external_asset |
| synthetic smoke | 0 | train→calibrate→evaluate→video可執行，正式metrics=null |
| train／calibrate／evaluate／demo 缺正式資源或run | 2 | 明確失敗，不生成假正式成果 |

## 仍受阻或未執行

- `blocked_external_asset`：detector.pth、yolov3-tiny-onecls.cfg、fast_res50_256x192.pth、tsstg-model.pth 四個上游 Drive 資源下載失敗。精確路徑與接續命令見 README。
- `blocked`：真實圖像→pose→512D；strict 真實 checkpoint＋真實forward；真實CPU/CUDA整合。
- `not_run`：剩餘68段camera0 RGB完整下載及全資料prepare。按規格先等小型真實模型整合，避免在模型受阻時處理完整資料。
- `blocked`：UR Fall正常AE訓練、真實validation校準、frozen Test評估、真實demo；沒有任何可報告的正式F1／accuracy。
- `not_run`：官方逐幀posture CSV的RGB對齊、人工event onset標註與延遲評估；第一版主要指標未使用它們。

索引亦檢查此次 src/tests 的30個主要證據路徑，generation `2026-09-11T10:40:19Z` 均 metadata_match、沒有記錄parse gap。忽略範圍包括 __pycache__、上游、環境、資料與輸出；上游與必要輸出已直接讀取或實測。此為best-effort索引訊號，不當作完整程式正確性的證明。
