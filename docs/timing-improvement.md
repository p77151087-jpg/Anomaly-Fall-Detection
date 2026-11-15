# 低 FPS 修正、重新訓練與門檻比較（2026-09-17）

本輪已完成程式修正、161 段資料的特徵重建、真實 AE 訓練 88 回合及兩段真實影片整合驗證。資料可用率提高；降低誤報的替代門檻會增加漏報。所有下列辨識數字來自 **Validation 校準集**，本輪沒有重跑 Test，不能視為獨立泛化成績。

## 修正的原因

Le2i 四段正常 Validation 的影片是 24.000384 FPS，原影格間隔約 41.666 ms。舊程式重採樣到 30 FPS 時，只容許已到達的觀測保留 33.333 ms，導致約每五個取樣點產生一次假缺幀，連穩定骨架也無法形成有效窗口。

新 `configs/openpose18-le2i-timing.yaml` 使用 `source_interval`：允許保留最近已到達的觀測至來源的正常影格間隔，以影片標頭 FPS 為依據，不使用未來影格、未來時間間隔或虛構關節。真正缺幀、超時、追蹤中斷與低品質仍保留 unknown。舊配置仍使用 `target_interval`，原指紋與凍結 run 不變。

四段 Le2i 正常 Validation 有效特徵窗從 0/0/0/0 恢復至 38/49/34/37。沒有放寬 12 關節、80% 品質要求，也沒有填補標註空白。

## 實際訓練

原始混合 split 不變：161 段來源；正常 Train 62 段、Validation 38 段（12 正常、26 跌倒），其餘為 Test。Train 仍只使用正常資料。此次只修正時間政策，透過新 `prepare --reuse-poses-index` 嚴格核對原始來源、split、权重、設定與 pose array hash 後，沿用完全相同的真實關節與 track ID，所有 ST-GCN 特徵重新計算並另存；不是再次執行全資料 Pose forward。

| 正常資料 | 修正前 | 修正後 |
|---|---:|---:|
| Train 可用來源 | 45/62 | 57/62 |
| Train 特徵窗 | 3,766 | 4,180 |
| Validation 可用來源 | 7/12 | 11/12 |
| Validation 特徵窗 | 128 | 286 |

`le2i-timing-001` 完成 88 epochs，依正常 Validation 選到 Epoch 78，MSE **0.2042546279311597**，該回合 Train MSE 0.09839485963565882。最新 Epoch 88 的 Validation MSE 0.208518。Scaler 和可用驗證資料與舊版本不同，不能單憑較低 Loss 說辨識比較準。

## 保留規格基準與替代門檻

規格預設仍是最大化序列跌倒 F1。修正後它將所有可判定的正常影片都判成異常，因此另做 `le2i-timing-balanced-001` 作探索性比較。它沿用完全相同的 AE、Scaler 與 Loss CSV，**沒有再訓練一次**；Loss 畫面會顯示沿用提示。

替代目標 `balanced_sequence_accuracy` 為 `0.5 × (TP / 全部跌倒段數 + TN / 全部正常段數)`，unknown 保留於分母。只用 Validation 選門檻，同分仍取較少 FP、較高門檻。這是觀察過本輪 Validation 問題後新增的替代協定，不是預先登錄實驗，也不取代規格基準。

| 同一批 38 段 Validation | 舊 Le2i | 修正後／規格 F1 | 修正後／平衡門檻 |
|---|---:|---:|---:|
| run | le2i-extension-001 | le2i-timing-001 | le2i-timing-balanced-001 |
| 門檻（strict >） | 0.150868 | 0.140551 | 0.492939 |
| 正常影片誤報／全部正常 | 6/12 | 10/12 | 1/12 |
| 正常影片正確判定 | 0 | 0 | 9 |
| 正常影片 unknown | 6 | 2 | 2 |
| 跌倒影片警示／全部跌倒 | 14/26 | 20/26 | 12/26 |
| 可判定但未警示的跌倒 | 0 | 0 | 8 |
| 跌倒影片 unknown | 12 | 6 | 6 |
| 全部 unknown | 18/38 | 8/38 | 8/38 |

所以時間修正確實提高可用率，但 F1 基準誤報並沒有降低。平衡門檻相對修正後 F1 少了 9 段誤報，也少了 8 段跌倒警示。其全部跌倒召回僅 46.15%，整體正確判定為 21/38＝55.26%；不能把排除 unknown 的數值當成全體準確率。兩版本應一併呈現，尚不能稱為可靠跌倒警報系統。

## 使用方式

查看這次真正重新訓練的 Loss：雙擊根目錄「查看Loss曲線.cmd」，選 `le2i-timing-001`；也可以在專案根目錄執行：

```powershell
.\.venv\Scripts\python.exe -m fall_detect loss --run runs/le2i-timing-001 --port 0
```

使用平衡門檻測試自己的影片（將路徑換成實際檔案；目前這是研究展示）：

```powershell
.\.venv\Scripts\python.exe -m fall_detect demo --run runs/le2i-timing-balanced-001 --input "C:\影片\測試.mp4" --output outputs/demo/my-balanced-demo.mp4 --no-display
```

如需比較規格預設，只將 `--run` 換為 `runs/le2i-timing-001`，並使用不同輸出檔名。MP4 旁有逐取樣 CSV、事件 CSV 與實際參數 JSON。

重現時用新的 run ID（已存在 run 不可覆寫）：

```powershell
.\.venv\Scripts\python.exe -m fall_detect prepare --config configs/openpose18-le2i-timing.yaml --reuse-poses-index data/cache/le2i-extension-selective/dataset-index.json
.\.venv\Scripts\python.exe -m fall_detect train --config configs/openpose18-le2i-timing.yaml --run-id le2i-timing-next --show-loss
# 關閉訓練完成後保留的 Loss 服務，再執行校準。
.\.venv\Scripts\python.exe -m fall_detect calibrate --run runs/le2i-timing-next
.\.venv\Scripts\python.exe -m fall_detect calibrate --run runs/le2i-timing-next --objective balanced_sequence_accuracy --output-run runs/le2i-timing-next-balanced
```

不需要重新訓練即可使用本輪已完成的兩個 run。替代校準必須同時指定 objective 和新的 output-run，禁止原地改掉舊門檻。

## 實測與限制

- **92 passed／1 skipped／11 warnings，17.76 秒**。跳過的是原 Human 方案缺四資源；替代方案真實 checkpoint CPU/CUDA forward 已測。CUDA deterministic warn-only、舊套件棄用及 CPU backend 警告仍有記錄。未驗證完整 CPU 訓練與即時效能。
- 新測試涵蓋 24 FPS 假缺幀、30 FPS 舊行為、因果性、真正缺幀、hash 隔離、不可覆寫重用、替代門檻保留 unknown、來源 run 不變及校準不讀 Test。測試中的 synthetic 僅為功能證據。
- 兩段 Validation 影片重新使用真實 Pose、ST-GCN、AE 執行。Home_02/video_47 正常影片的 48 個可比端點、UR fall-02 的 7 個端點 raw score 最大差均為 0。
- 正常影片另 1 個已計分端點因當下骨架低品質，畫面正確隱藏分數。48 個可比點中 44 個狀態相同，另 4 點因離線 cache 有標註區間／無標註 demo 沒有此限制而不同；两邊均無警示事件。不宣稱所有狀態完全相同。UR 七點狀態與完整事件皆相同。
- 已查看實際影片抽幀：正常影片有正常骨架與分數；fall-02 有疑似跌倒警示。僅屬兩段整合檢查，沒有驗證事件起始準確性、延遲或新場景泛化。
- 已在瀏覽器核對兩個新 run 的 88 回合、Epoch 78 最佳值、曲線、正常資料數量及「沒有重複訓練」提示。舊三個 run 與新 F1 基準的凍結 hash 均不變。

證據：[根因](validation/improvement-normal-validation-diagnosis.json)、[修正與 pilot 鎖定](validation/improvement-timing-lock.json)、[完整特徵重建](validation/improvement-timing-preparation.json)、[校準結果](validation/improvement-timing-results.json)、[真實整合](validation/improvement-timing-integration.json)、[全套測試](validation/pytest-improvement-full.xml)、[最後檢查](validation/improvement-final-checks.json)。
