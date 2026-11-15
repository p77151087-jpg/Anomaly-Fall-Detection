# 95% 目標：骨架與時間處理改善

本輪目標為整段影片 `operational_accuracy >= 95%`，所有無法判定仍留在分母。沿用固定人物／原始序列切分，Train 僅 ADL，ST-GCN 凍結。舊 Test 已被歷次實驗看過，最後成績屬既有測試集上的工程對照，不能宣稱全新獨立泛化驗證。

## 固定比較與選擇方式

本輪比較既有 CAUCAFall 基準、1 ms 時間容差版、YOLO11 版，以及 YOLO11＋有界時間抖動版。只有 Train ADL 更新 Scaler／AE；正常 Validation Loss 選 checkpoint；Validation 的 balanced sequence accuracy 選門檻，unknown 保留在每個類別分母。候選依 Validation operational accuracy、跌倒召回、較少誤報依序選擇。選擇完成、hash 凍結後才執行 Test，不追加候選追逐 Test 分數。

鎖定紀錄：`validation/target95-lock.json`、`target95-yolo-lock.json`、`target95-jitter-lock.json`。這是探索性工程實驗，Validation 已多次使用，沒有把 Validation 宣稱為未知資料成績。

## 已查證問題與新方法

- UR Fall 整數毫秒時間戳會讓 33.67 ms 的最近觀測超過原 33.33 ms 上限。顯式 `timestamp_tolerance_ms: 1` 只延長失效判定，不移動時間格點、不讀取未來骨架；預設仍為 0，舊 fingerprint 保留。
- 部分 UR Fall Train／Validation 序列平均 33.3 ms，單次拍攝時間卻抖動至 47 ms。另設 `resampling_hold_policy: bounded_jitter`：最多保留最近真實骨架 `max(1.5×target interval, declared source interval)`；30 Hz 為 50 ms，與 tracker 名義重設界線一致。此為明訂策略對照，並非宣稱 47 ms 內永遠沒有掉幀。明確 source_missing、沒有骨架及 track 缺口照常失效，未對缺點做插值。
- 新骨架 recipe 為 `yolo11m_coco17_openpose18_stgcn_kinetics_v1`。凍結官方 COCO 預訓練 YOLO11m Pose（640px），經原尺寸座標輸出轉為 OpenPose18；使用既有未修改 tracker，不開骨架平滑；仍用原始 Kinetics ST-GCN 分類頭前 pooled 256D，重新訓練 AE，不沿用舊 Scaler／門檻。
- 17 個 COCO 測量點均一對一映射，neck 只由兩個可觀測肩點取中點，信心取兩肩最小值；任一肩缺失則 neck 缺失。衍生 neck 不計入「至少 12 個真實關節」品質分數，維持一窗 80%、30 幀／stride5／K3。

## 權重與環境

官方來源：[YOLO11 文件](https://docs.ultralytics.com/models/yolo11/)、[官方 release](https://github.com/ultralytics/assets/releases/tag/v8.3.0)。原始 `yolo11m-pose.pt` 為 42,459,307 bytes，實際 SHA256 `29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2`。官方未提供此 release asset 的 digest，本專案保存實際下載 hash。

固定 `ultralytics==8.3.228`，精確 hash 的官方物件 checkpoint 一次轉為 float32 tensor-only state dict，691 個鍵值嚴格載入 `yolo11m-pose`／nc=1／17×3 結構；正式推論只以受限 `weights_only=True` 載入轉換檔。轉換檔 SHA256 `9ed4d6879c271b554149ea93ec71c9531da17863a64e0e80945d9c11c668ad3d`。並非隨機初始化推論。YOLO 原始碼與權重為 AGPL-3.0；既有 ST-GCN、資料授權沿用各來源。

可重建資源：

```powershell
.\.tools\Scripts\uv.exe pip install --python .venv/Scripts/python.exe --no-deps -r requirements/yolo11-tested.txt
.\.venv\Scripts\python.exe -m fall_detect fetch-assets --config configs/yolo11-caucafall.yaml --models
```

本機新依賴只加入 `.venv`，原 Torch／OpenCV 保留。原始影片、權重、upstream、快取、run 未納入 Git。YOLO CPU 與原 OpenPose CPU／CUDA 已實測；完整 YOLO CPU 資料流程尚未驗證。

## 完成結果

時間容差版實跑 94 epochs，Validation 可判定 50/58→55/58，但平衡門檻下 TP16／TN20／FP0／FN19／unknown3，整體 36/58=62.1%，低於基準 38/58=65.5%，不直接替換。

YOLO 8 段真實 feature smoke 通過預定 coverage gate，可判定 5→6 段。此處只驗證骨架／特徵，不是 AE 準確率。之後完成 231/231 全新骨架及 ST-GCN 擷取；時間抖動版的 231 份骨架陣列 hash 與 YOLO 原版完全相同，只重新計算時間窗與真實 ST-GCN 特徵。

本輪實際新增 **293 回合** AE 訓練，三個獨立 run 全部保存：

| Run | 實跑／最佳 Epoch | 正常 Val 最低 MSE | Val 正確／全部 | Val unknown |
| --- | --- | --- | --- | --- |
| 原 `caucafall-extension-001` | 97／87（既有） | 0.198277 | 38/58 = 65.5% | 8 |
| `caucafall-timestamp-001` | 94／84 | 0.201846 | 36/58 = 62.1% | 3 |
| `caucafall-yolo11-001` | 99／89 | 0.155140 | 42/58 = 72.4% | 2 |
| **`caucafall-yolo11-jitter-001`** | **100／95** | **0.149559** | **43/58 = 74.1%** | **0** |

不同 run 的 Scaler／特徵分布不同，不把跨 run 的 Loss 高低等同辨識率。選定 jitter 版，門檻 `0.3282490372657776`；Validation TP25／TN18／FP4／FN11。它比第一組 YOLO 多偵測 2 支跌倒，也多 3 支正常誤報，依事前整體正確率規則選定。未隱藏取捨。

選定版本正常 Train **92/92 段、6,096 窗**；正常 Val **22/22 段、1,026 窗**。舊基準為 87/92 段、5,276 窗，正常 Val 21/22 段、637 窗。

只有選定且凍結的 jitter 版執行本輪 Test：

| 相同 Test 集 | 舊版 | 新版 | TP／TN／FP／FN／unknown |
| --- | --- | --- | --- |
| **全部 81 支** | **48/81 = 59.3%** | **57/81 = 70.4%** | 30／27／7／16／1 |
| CAUCAFall 20 支 | 15/20 = 75.0% | **18/20 = 90.0%** | 10／8／2／0／0 |
| Le2i 38 支 | 25/38 = 65.8% | 29/38 = 76.3% | 16／13／3／6／0 |
| UR Fall 23 支 | 8/23 = 34.8% | 10/23 = 43.5% | 4／6／2／10／1 |

**95% 目標尚未達成**，CAUCAFall 也仍是 90%，不是 95%。全資料整段 unknown 10→1，正確判定增加 9 支（+11.1 個百分點），總跌倒召回 22/47=46.8%→30/47=63.8%。仍有 16 支跌倒未警示、7 支正常誤報。這輪改善主要來自恢復原本無法判定的影片；單純再降低 Loss 不保證能消除剩餘分數重疊。

CAUCAFall 的 10 支跌倒影片皆有觸發，但此處是**整段影片**標準；沒有驗證每個警示是否發生在正確的起跌時間。資料為既有研究資料中的安排動作，不是實際事故部署成績。下一階段應以新的獨立驗證資料檢查正常／跌倒表示能否分離，並先核對事件標註；不能持續依已看的 Test 調門檻來湊 95%。若改用跌倒標籤訓練分類器，會改變目前「正常資料 AE」的方法，必須另立協定。

## 真實展示、Loss 與驗收

- 原始 Validation `Subject.5/WalkS5.avi`、`FallForwardS5.avi` 全新推論，輸出 `outputs/target95-demo-walk.mp4`（338 幀）、`outputs/target95-demo-fall.mp4`（299 幀），均完整解碼。62＋45 個有效端點 raw score 差為 0，狀態與事件一致，沒有遮罩例外。行走無警示；跌倒展示在 4.4667 秒首次警示，因無已核對 onset 標註，不宣稱為偵測延遲。
- 已查看正常／躺倒畫面，骨架及綠色 normal／紅色 suspected_fall 顯示合理。原始 YOLO checkpoint 與轉換檔在真實 Train adl-05 影像的輸出差為 0。
- **程式內 Loss 已切換新 run**，100 回合、最佳 Epoch95／0.149559、兩條實際曲線及資料數量均核對。雙擊「查看Loss曲線.cmd」亦可從選單選擇此 run。
- 全套 **132 passed／1 skipped／12 warnings**（29.99 秒）。skip 僅原 Human 四個資源缺失；YOLO 真實 CPU、OpenPose／ST-GCN 真實 CPU／CUDA、缺點映射、品質計數、因果性／真缺失、cache 隔離、原 fingerprint 及 batch／streaming 風險測試通過。ruff、format、compileall 通過。
- 選定前後 frozen hashes 不變，原模型保留；原始 100 支 CAUCAFall hash 再次核對。新預訓練模型不是重新訓練，只有 AE 學習正常資料。

機讀證據：`validation/target95-results.json`、`target95-before-test-freeze.json`、`target95-integration.json`、`target95-final-checks.json`、`pytest-target95-final.xml`。正式 Test 完整 CSV 與圖表在 `outputs/target95-test/`。

## 使用新版

在專案根目錄 PowerShell 執行：

```powershell
# 已訓練模型的 Loss
.\.venv\Scripts\python.exe -m fall_detect loss --run runs/caucafall-yolo11-jitter-001 --port 0

# 對自己的新影片推論；輸出檔名必須尚未存在
.\.venv\Scripts\python.exe -m fall_detect demo --run runs/caucafall-yolo11-jitter-001 --input "你的影片.mp4" --output outputs/my-yolo-result.mp4 --no-display
```

本輪重跑記錄由 `scripts/target95_compare.py` 呼叫共用核心流程，拒絕覆寫已存在的 run。一般使用者直接用上述已完成模型，不必再次訓練。
