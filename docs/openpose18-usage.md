最新擴充模型 `le2i-extension-001` 已訓練完成，直接使用見[Le2i操作指南](le2i-extension.md)。本頁保留原基準的使用方式。

# 第二條路：使用方法

2026-09-17 已完成規格第3.2節的備援路徑、完整資料、真實訓練及Test。使用 `configs/openpose18.yaml`；不用補原 Human 的四個檔案。目前已有可直接使用的 `runs/openpose18-baseline-001`，不用重新訓練。

**這是效果仍不足的基準模型：** Test 15段跌倒中7段警示，8段正常活動中7段誤報，23段中8段unknown。完整數據見 [結果摘要](results-summary.md)。

## 先看已完成的成果

- [跌倒示範影片](../outputs/real_end_to_end/openpose18/fall-02-final.mp4)：Validation fall-02，包含警示及躺倒後的unknown。
- [正常活動示範影片](../outputs/real_end_to_end/openpose18/adl-05-final.mp4)：Train adl-05；不能用此訓練片表現代表Test。
- [正式Test報告](../outputs/real_end_to_end/openpose18/test-evaluation/evaluation-summary.md)：固定23段Test的結果。

分析自己的影片可直接使用下方「分析自己的影片」命令。需要檢查模型安裝才執行smoke；不必每次分析影片都重跑訓練、校準與Test。

## 開始

在 PowerShell 中執行。此電腦環境與替代權重已安裝好，不用啟用 venv。

```powershell
cd 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe -m fall_detect doctor --config configs/openpose18.yaml --mode core
.\.venv\Scripts\python.exe -m fall_detect smoke-test --config configs/openpose18.yaml --mode real --device cuda --output outputs/real_feature_smoke/my-openpose18-check
```

real smoke 使用固定的 Train adl-05、Validation fall-02；強制重新執行模型，不用舊特徵冒充本次推論。輸出 `smoke-report.json`，`passed: true` 代表兩段都有真實、有效256D特徵。此步驟尚未分類正常／跌倒。

`--device cpu` 可用於 CPU；本機 CPU 姿態估計比 GPU 慢很多，不承諾即時。

## 重新取得資源及完整實驗

需要在其他電腦重建時，先照根目錄 README 建立 Python 環境，再執行：

```powershell
.\.venv\Scripts\python.exe -m fall_detect fetch-assets --config configs/openpose18.yaml --models
.\.venv\Scripts\python.exe -m fall_detect fetch-assets --config configs/openpose18.yaml --dataset urfall-cam0-rgb
.\.venv\Scripts\python.exe -m fall_detect prepare --config configs/openpose18.yaml --inventory-only
.\.venv\Scripts\python.exe -m fall_detect prepare --config configs/openpose18.yaml --device cuda
.\.venv\Scripts\python.exe -m fall_detect train --config configs/openpose18.yaml --run-id openpose18-baseline-001
.\.venv\Scripts\python.exe -m fall_detect calibrate --run runs/openpose18-baseline-001
.\.venv\Scripts\python.exe -m fall_detect evaluate --run runs/openpose18-baseline-001 --split test
```

逐步執行，任何一步報錯就先處理；不要把錯誤退出當作成功。資料與prepare可續跑。已存在的 run 不覆蓋，重做實驗要改 run-id，例如 `openpose18-baseline-002`。train要求70個序列都已處理，但參數更新只使用24個正常Train序列；驗證正常資料選checkpoint、Validation選門檻，Test只評估。

## 分析自己的影片

本機真實 train→calibrate 已完成。把 `C:\影片\我的影片.mp4` 改成實際影片位置，在專案根目錄執行；demo 讀取run保存的設定，不另帶 `--config`。

```powershell
.\.venv\Scripts\python.exe -m fall_detect demo --run runs/openpose18-baseline-001 --input 'C:\影片\我的影片.mp4' --output outputs/demo/my-result.mp4 --device cuda --no-display
```

輸出：`my-result.mp4`、逐影格CSV、事件CSV、設定與速度JSON。輸出檔已存在時需換檔名。

- `warming_up`：尚未收集足夠時間窗。
- `normal`：目前異常分數未達警示條件。
- `suspected_fall`：同一人物連續三個有效窗超過門檻，表示疑似跌倒／異常，不保證必然跌倒。
- `unknown`：沒有骨架、追蹤中斷或品質不足，不能當作正常。

預訓練姿態估計器在躺倒或遮擋時可能漏點、分裂人物；須同時看coverage、unknown與漏報，不能只看可判定片段的accuracy。固定30fps/30幀短窗與Lightweight OpenPose均是本專案選擇，未宣稱重現原论文成績或已獲老師確認。

## 檔案位置

| 內容 | 路徑 |
| --- | --- |
| 替代設定 | `configs/openpose18.yaml` |
| 模型來源／hash | `weights/openpose18-manifest.json` |
| 兩個模型檔 | `weights/pretrained/openpose18/` |
| 共用原始資料及固定切分 | `data/raw/urfall/`、`data/manifests/split-v1.json` |
| 新骨架與256D特徵快取 | `data/cache/openpose18/` |
| 訓練及門檻 | `runs/openpose18-baseline-001/` |
| 本次正式測試指標 | `outputs/real_end_to_end/openpose18/test-evaluation/` |
| 已完成的示範影片 | `outputs/real_end_to_end/openpose18/*-final.mp4` |

模型、資料、快取與環境維持Git忽略；来源、實際hash、設定、驗收摘要會保留在專案文件。
