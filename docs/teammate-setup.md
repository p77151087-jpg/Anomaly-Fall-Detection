# 組員從 Git 直接使用

適用於 Windows x64。電腦只需先安裝 Git，Python 和套件會由啟動程式放在專案內，不修改系統 Python。

## 第一次使用

在 PowerShell 執行：

```powershell
git clone https://github.com/p77151087-jpg/Anomaly-Fall-Detection.git
cd Anomaly-Fall-Detection
.\start.cmd
```

啟動程式會自動建立隔離環境，下載既有 AE 模型、預訓練資源及兩段範例，再產生並開啟示範影片。第一次需連網下載，GPU 套件較大；完成後不需重新訓練。下次直接執行 `start.cmd`，既有環境和資源會重用。程式更新則先 `git pull`，再啟動。

## 常用操作

```powershell
# 使用自己的影片；路徑有空白也可以
.\start.cmd -Input "D:\videos\my-video.mp4"

# 只建置環境與資源，暫不處理影片
.\start.cmd -SetupOnly

# 指定 CPU（預設會檢查可用 GPU）
.\start.cmd -Device cpu

# 只讀檢查，不安裝、不下載、不產生影片
.\start.cmd -CheckOnly

# 指定新的輸出檔名，完成後不開啟播放器
.\start.cmd -Input "D:\videos\my-video.mp4" -Output "D:\results\my-result.mp4" -NoOpen
```

影片預設存到 `outputs/demo/quickstart-時間.mp4`；每次另存新檔，不覆蓋舊成果。旁邊的 CSV／JSON 保存分數、事件與動作顯示資訊。若系統沒有可用的預設播放器，會顯示輸出位置。

預設使用保留的 `ur-sample-human-s1-001`，流程為 Tiny-YOLO → AlphaPose → ST-GCN 512D → Scaler → Autoencoder。畫面中的七類動作來自另一個預訓練分類頭，AE 異常警示獨立計算。`PENDING` 是尚在收集窗口，`UNKNOWN` 是當下缺少有效判定。

## Git 與自動下載的分工

Git 保存程式、設定與檢查值，啟動程式自動取得其他必要資源：

| 資源 | 取得方式 |
| --- | --- |
| Python 3.11.16、固定版本 Torch／Torchvision 與套件 | 專案內 uv 自動安裝 |
| 完整 baseline／wide AE run、雙肩候選必要紀錄 | [公開 GitHub Release](https://github.com/p77151087-jpg/Anomaly-Fall-Detection/releases/tag/teammate-runtime-v1)，下載後核對逐檔 SHA256 |
| 固定版本 Human 程式與四份預訓練資源 | 原有下載器及 [上游 Issue 117 的社群鏡像](https://github.com/GajuuzZ/Human-Falling-Detect-Tracks/issues/117) |
| Train `adl-05`、Validation `fall-02` 兩段 RGB | [UR Fall 官方頁](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) |

模型 Release 不包含原始影像或展示影片。預訓練權重與資料也核對固定檢查值；來源限制保留於 [模型來源](model-sources.md)。上游 checkout、權重、原始資料、快取與環境不加入本專案 Git。

自動建置不下載 Test、不訓練、不校準，也不評估 Test。安裝檢查與範例展示不代表新的辨識成績。

## 若遇到問題

- **網路中斷：** 資源下載最多自動重試 3 次，已完成的資源會重用；仍未完成時，網路恢復後再執行同一個命令。
- **既有環境不相容：** 程式保留它並停止。首次使用以新的 clone 為宜，不直接複製另一台電腦的 `.venv`。
- **資源 hash 不同或上游程式被改過：** 程式保留檔案並指出路徑，不會自動蓋掉研究成果。可使用另一份乾淨的 clone。
- **GPU 不可用：** 自動模式改用 CPU，也可加 `-Device cpu`。CPU 處理影片可能較慢。
- **輸出已存在：** 換新的輸出檔名，或省略 `-Output` 使用自動名稱。

## 修改程式或重新訓練

環境採 editable 安裝，修改 `src/` 就會使用新程式。要重建特徵與 AE，才需要完整資料；進階操作使用原有 CLI：

```powershell
.\.venv\Scripts\python.exe -m fall_detect --help
```

重新訓練使用新的 run ID 與 cache，保留固定 split、Train 僅正常 ADL 的規則。不要直接修改既有 run 的設定、Scaler、門檻與 hash。研究操作不會在 `start.cmd` 自動執行。
