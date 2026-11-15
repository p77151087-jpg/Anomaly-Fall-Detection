# 在程式中查看 Loss

在檔案總管開啟 `C:\成大專案\fall-detection-anomaly`，雙擊 **查看Loss曲線.cmd**。程式會啟動本機檢視頁並自動開啟瀏覽器，不需要重新訓練，也不需要安裝網頁套件。保留啟動時的終端視窗；按 Ctrl+C 可以關閉服務。

畫面上方的「訓練紀錄」可以選擇 `openpose18-rotation-001`、`le2i-extension-001` 或 `openpose18-baseline-001`。新一輪旋轉對照實跑100回合，最佳Epoch95、Loss0.499633，[結果與限制](rotation-training.md)另列。畫面包含 Train／Validation 曲線、最新 Loss、最佳回合、逐回合數值及 SVG／CSV 下載。滑鼠移到曲線上可查看該回合數值。圖表會配合視窗寬度顯示。

## 命令列開啟

在專案根目錄 PowerShell 執行：

```powershell
.\.venv\Scripts\python.exe -m fall_detect loss --run runs/le2i-extension-001
```

預設網址為 `http://127.0.0.1:8765`。若該連接埠已使用，加入 `--port 0`，程式會選可用的連接埠並印出網址。雙擊啟動檔已採此方式。`--no-browser` 可只啟動服務，再自行開啟印出的網址。

## 訓練時同步看曲線

需要建立新實驗時，在原本的 train 命令加上 `--show-loss`：

```powershell
.\.venv\Scripts\python.exe -m fall_detect train --config configs/openpose18-le2i.yaml --run-id le2i-next-001 --show-loss
```

此命令會**真的重新訓練**，run ID 必須尚未存在。它自動開啟該 run 的曲線，每個完整 Epoch 結束後更新，頁面每 2 秒讀取一次。剛開始尚未完成第一回合時顯示等待，不會預填數值。訓練完成後曲線保留，終端按 Ctrl+C 結束檢視服務。若訓練途中按 Ctrl+C，訓練本身也會中止。

也可以先開 `loss`，在另一個終端執行原本的 `train`；新 run 會出現在選單中。`--runs-dir` 可指定其他包含 run 子目錄的資料夾，例如明確分開保存的 synthetic 測試。即時功能適用於此更新後建立的新 run；舊 run 已保存的完成曲線可直接讀取。

## 畫面數值與狀態

- Loss 是 AE 正常特徵的重建 MSE。Train 是回合內更新模型時、按樣本加權的 minibatch 平均；Validation 是回合結束後使用固定模型計算的正常驗證平均。
- 「最佳 Validation Loss」使用已完成 run 的最低正常驗證 Loss；訓練中標為「目前最低」。沒有 Test 曲線，也不將 Loss 換算成準確率。
- Le2i 顯示 79 回合，最佳 Epoch 69、Loss 0.264816。原基準為 100 回合，最佳 Epoch 90、Loss 0.496766。各自正規化不同，不宜直接比較 Loss 大小。
- 畫面保留 Le2i 四段正常 Validation 無有效特徵的限制。未提供特徵數量的舊／測試紀錄顯示「未提供」，不當成零。
- synthetic 會明確標記為程式測試資料。若超過 30 秒沒有新回合，提示檢查訓練終端；沒有新資料不等於模型仍在執行。
- 連線或紀錄驗證失敗時清除曲線並顯示錯誤。完成紀錄會檢查 epoch／最佳 Loss；若已有 frozen 或 training hash 清單，也會核對原 CSV 的 hash。

檢視頁只讀取紀錄，不載入模型、不重新校準或評估 Test。新訓練增加原子寫入的 `training-progress.json`，不改變最佳 checkpoint、early stopping 與既有 CSV 規則。

已實際驗證：兩個真實 run 切換、79 個曲線資料點與表格、滑鼠讀值、HTTP CSV 下載、錯誤／空資料／新 run、實際 synthetic AE 的逐回合更新與完成狀態。全套 76 passed／1 skipped；最後缺省數量顯示修正後，7 項相關測試再次通過。證據見 [loss-viewer-checks.json](validation/loss-viewer-checks.json)。報告圖片仍可使用 [Loss 圖與圖說](loss-figures.md)。


## 本輪時間修正後的紀錄

選擇 `le2i-timing-001` 可查看新完成的88回合，最佳Epoch78、Loss0.204255；正常Train57/62段4180窗、正常Validation11/12段286窗。`le2i-timing-balanced-001`沿用相同訓練，只校準另一個門檻，畫面會顯示提示。兩者Loss相同是正確結果，辨識門檻取捨見[timing-improvement.md](timing-improvement.md)。


## 接續訓練三組比較

最新 `le2i-seqbalanced-001` 為172回合，最佳Epoch142／0.197624；`le2i-long-002` 為250回合，`le2i-seqcompact-001` 為155回合。均衡抽樣會顯示Train Loss計算方式提示。`le2i-long-001` 是已保留的165回合失敗紀錄，畫面明示中斷，不是可用模型。這輪Loss下降但整體正確判定率未超過55.3%，詳見[完整比較](further-training.md)。
