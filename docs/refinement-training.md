# YOLO11 骨架特徵：AE 訓練強化

日期：2026-09-17。延續 `caucafall-yolo11-jitter-001`，目標仍為提高整段影片的正確判定率；不把事前預測跌倒或警示延遲混入本輪指標。

## 診斷與固定實驗

原版 AE 訓練在 100 回合上限結束，最低正常 Validation Loss 出現在第 95 回合。因此先比較增加訓練上限是否有益。Train ADL 共 92 支、6096 個真實 ST-GCN 特徵窗，其中 UR Fall 24 支／659 窗、Le2i 38 支／3983 窗、CAUCAFall 30 支／1454 窗；單支有效窗數介於 9 與 302。逐窗抽樣讓較長影片出現更多次，因此另比較每支序列總抽樣權重相同的策略。

壓縮層另從 32 縮為 16 維，檢查限制 AE 容量是否能改善正常與異常分數的分離。這只是待驗證假設，不預設較小 Loss 或較小模型一定能提高跌倒辨識率。

| 候選 Run | 抽樣 | 壓縮層 | 回合上限／耐心值 |
| --- | --- | --- | --- |
| `caucafall-yolo11-long-001` | 每個窗等權 | 32 | 200／20 |
| `caucafall-yolo11-compact-001` | 每個窗等權 | 16 | 200／20 |
| `caucafall-yolo11-balanced-001` | 每支序列等總權重 | 32 | 200／20 |
| `caucafall-yolo11-balancedcompact-001` | 每支序列等總權重 | 16 | 200／20 |

其餘設定固定：seed42、隱藏層128、Adam 0.001、batch32、30幀／stride5／K3、YOLO11m→OpenPose18→凍結 Kinetics ST-GCN 256D。原始資料切分、骨架品質規則、時間規則及真實特徵快取均不變，Train 僅 ADL。Scaler 仍以所有 Train ADL 有效窗擬合；序列抽樣只改 AE 訓練時抽到各窗的機會，不增加資料數量，也不宣稱 Scaler 或正常 Validation Loss 已改成等序列權重。

## 選擇與驗收規則

實驗開始前寫入 [refinement-lock.json](validation/refinement-lock.json)，鎖定四組設定、快取 SHA256 與 16 個既有 frozen run。核心 API 仍為 `fall_detect.training.experiment.train/calibrate`，薄包裝腳本為 `scripts/refine_yolo_training.py`。

1. checkpoint 只依正常 Validation Loss；門檻只依完整 Validation 的 balanced sequence accuracy。
2. 四組全部完成後，與原基準依 Validation operational accuracy、跌倒召回率、較少誤報依序比較；完全相同保留較早候選，基準列在最前。
3. 候選與門檻凍結後，只對單一選定新模型執行 Test。若原版勝出，不重跑相同 Test。
4. 不在看到 Test 後新增候選、調門檻或改選另一個版本。Test 早已在歷次工作中被查看，本輪只作既有資料集的工程對照，不宣稱全新獨立泛化驗證。
5. 跑抽樣、正常訓練與 Test 隔離、模型保存重載、門檻及 Loss viewer 相關測試；選定新模型再跑真實原影片的端到端示範。

Loss 曲線與每回合紀錄仍存入各 run，原有程式內 Loss 頁面可選取查看。這次不修改核心訓練或推論程式，只增加固定實驗設定與可重跑的比較腳本。

## 執行結果

四組實際完成 **521 回合**，依事前 Validation 規則選定 **`caucafall-yolo11-balanced-001`**。只有此新模型執行本輪 Test，沒有依 Test 改選模型或調門檻。

| Run 簡稱 | 實跑／最佳 Epoch | 正常 Val 最低 MSE | Val 正確／58 | TP／TN／FP／FN |
| --- | --- | --- | --- | --- |
| 原 jitter | 100／95（既有） | 0.149559 | 43（74.1%） | 25／18／4／11 |
| long | 165／145 | 0.147498 | 43（74.1%） | 25／18／4／11 |
| compact | 131／111 | 0.235954 | 41（70.7%） | 20／21／1／16 |
| **balanced** | **144／124** | **0.143568** | **47（81.0%）** | **26／21／1／10** |
| balancedcompact | 81／61 | 0.221947 | 46（79.3%） | 25／21／1／11 |

全部 Validation 58 支均可判定。長訓練版前 100 回合 Train／Validation Loss 與原版逐值完全一致，增加回合使 Loss 下降但沒有提高分類正確數。16 維壓縮層減少誤報卻增加漏報；不能把低誤報單獨包裝成全面改善。

選定 balanced 版的門檻為 `0.36089396476745605`，嚴格大於、連續 K3；正常訓練仍是 92 支／6096 窗，正常驗證為 22 支／1026 窗。Scaler 與原版 hash 相同，資料與特徵未變，模型依 seed42 從頭重訓。

### 固定 Test 結果

| Test 範圍 | 原 jitter | 新 balanced | 新 TP／TN／FP／FN／unknown |
| --- | --- | --- | --- |
| **全部 81 支** | **57/81 = 70.4%** | **57/81 = 70.4%** | 28／29／5／18／1 |
| CAUCAFall 20 支 | 18/20 = 90.0% | **20/20 = 100.0%** | 10／10／0／0／0 |
| Le2i 38 支 | 29/38 = 76.3% | 27/38 = 71.1% | 15／12／4／7／0 |
| UR Fall 23 支 | 10/23 = 43.5% | 10/23 = 43.5% | 3／7／1／11／1 |

**CAUCAFall 的這 20 支 Test 超過 95%，但整體 95% 目標仍未達到。** 混合資料集整體正確率沒有上升，少 2 次誤報的同時，多漏掉 2 支跌倒。整體跌倒召回由 30/47=63.8% 降至 28/47=59.6%。新版不是全面優於舊版，兩個版本都保留，必須連同取捨一起報告。

CAUCAFall 僅 20 支 Test、且既有 Test 曾被前幾輪工作查看；20/20 不能推廣為未知場景一定有 100%，也不是新的獨立評估。此次原始切分、Test 分母與 unknown 規則保持不變。所有成績仍為整段序列辨識，未驗證起跌時間或提前預警能力。

### 真實影片與程式驗證

- 真實 Validation 原影片重新推論：S05 Walk（338 幀）、FallForward（299 幀）、SitDown（389 幀）、S10 Kneel（391 幀），共 1417 幀，全部完整解碼。
- Walk／SitDown／Kneel 沒有警示，FallForward 於影片 4.4667 秒確認警示；不是 4.4667 秒偵測延遲。舊版曾誤報的 Validation 坐下與跪下案例，在本輪同一門檻下沒有再誤報。
- 與快取推論比較 **252 個有效視窗**，raw score 最大差 **0.0**；狀態及事件完全一致，0 遮罩例外。已查看新版坐下綠色 normal 與跌倒紅色 suspected_fall 的畫面。
- 相關風險測試 **25 passed**（11.20 秒）：抽樣、資料隔離、scaler／模型重載、門檻及 Loss viewer。ruff／compileall 通過。無核心推論程式變更，因此沒有無理由重跑所有第三方模型測試。
- 16 個原 frozen run、4 個新模型、Test 前凍結 hash 與共用特徵 index 均核對未變。原始資料、權重、快取、runs 及 outputs 仍未加入 Git。
- 程式內 Loss 已選定 balanced 版，144 回合、最佳124／0.143568。新示範保存在 `outputs/refinement-demo-*.mp4`；另提供完整解碼通過的 H.264 `*-browser.mp4` 觀看副本。

機讀證據：`validation/refinement-lock.json`、`refinement-validation-diagnosis.json`、`refinement-before-test-freeze.json`、`refinement-results.json`、`refinement-integration.json`、`pytest-refinement.xml`。完整 Test 表格及圖在 `outputs/refinement-test/`。各候選 Loss／checkpoint 均完整保留，沒有仍在背景執行的訓練。

## 使用與後續界線

```powershell
# 本輪選定模型的程式內 Loss
.\.venv\Scripts\python.exe -m fall_detect loss --run runs/caucafall-yolo11-balanced-001 --port 0

# 新影片的真實推論（輸出檔名須尚未存在）
.\.venv\Scripts\python.exe -m fall_detect demo --run runs/caucafall-yolo11-balanced-001 --input "你的影片.mp4" --output outputs/my-balanced-result.mp4 --no-display
```

這輪證明平均序列抽樣可改善部分場景，但 AE 容量與回合數的調整尚不能解決跨資料集分數重疊。後續若要追求整體 95%，應取得新的獨立人物／場景驗證資料，核對跌倒時間與骨架品質，再設計新的實驗。單純繼續依這份 Test 反覆調整不構成可信的提升。
