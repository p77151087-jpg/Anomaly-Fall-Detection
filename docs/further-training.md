# 接續改善訓練（2026-09-17）

使用者要求繼續提升辨識。這輪固定三個 AE 實驗，沿用同一批真實 Pose/ST-GCN 特徵、固定 split、品質政策與事件規則。各組從 seed 42 重新初始化訓練，並非將上一版的 Epoch 78 checkpoint 接著跑；原 run 全部保留。

## 訓練依據與固定範圍

正常 Train 的 4,180 個有效窗中，UR Fall 只有 346（8.28%），Le2i 有 3,834（91.72%）。原本按窗口等機率抽樣，長影片帶來較多次參數更新。因此增加可選的 `autoencoder.sampling: sequences`：每個有效窗口權重是其來源影片有效窗數的倒數，使 57 段有特徵的正常影片具有相同**預期**抽樣機會。五段無特徵來源不虛構資料，仍保留在資料統計中。

均衡模式每回合有放回抽 4,180 次，可能抽到重複窗口，不代表新增 4,180 筆資料。正常 Train 的資料集比例改為 UR 19/57、Le2i 38/57 的預期抽樣比率；每回合實際數量由已保存 seed 驅動。預設 `windows` 的原有 shuffle 行為維持不變。

Scaler 仍由全部正常 Train 窗口 fit；三組與上一版 Scaler 檔案 hash 完全相同。Validation Loss 都是全部 286 個正常驗證窗口的平均，選 checkpoint 的依據不變。均衡模式 Train Loss 是該回合抽到的窗口平均，程式 Loss 畫面會提示此差異。

| run | 抽樣 | AE 壓縮維度 | 學習率 | 最多回合 / patience |
|---|---|---:|---:|---:|
| le2i-long-002（同設定重跑） | 窗口等機率 | 32 | 0.0003 | 250 / 30 |
| le2i-seqbalanced-001 | 影片均衡 | 32 | 0.0003 | 250 / 30 |
| le2i-seqcompact-001 | 影片均衡 | 16 | 0.0003 | 250 / 30 |

隱藏層仍為 128，batch size 32；Pose/ST-GCN 不更新。這輪不增加第四組候選，不調整品質／時間／事件門檻政策以改變 unknown 數量。

## 校準與比較規則

每組正常 Validation MSE 最低的 checkpoint 各自依規格做 F1 校準；另建立以 `balanced_sequence_accuracy` 校準的獨立 run，AE/scaler/Loss 原檔沿用，不假裝重訓。平衡指標是 `0.5 × (TP / 全部 Fall + TN / 全部 ADL)`，unknown 留在分母。

開始前鎖定：三個平衡校準版本與舊 `le2i-timing-balanced-001` 比較，先取較高平衡指標，再依整體正確判定率、全部跌倒召回、較少 FP 排序；完全同分保留先前基準。此為在同一已知 Validation 上探索設定與模型選擇，數字可能樂觀，**不是獨立 Test 或新場景泛化結果**。本輪不讀 Test 特徵評分，也不另跑 Test。

每個新 run 都有完整實際 `training-history.csv`、最佳權重、來源 hash、抽樣方式、正常 Train 各來源窗口數與環境資訊。舊 run 不覆寫。

## 操作

雙擊「查看Loss曲線.cmd」，在選單切換上述 run。也可執行：

```powershell
.\.venv\Scripts\python.exe -m fall_detect loss --run runs/le2i-seqbalanced-001 --port 0
```

重現訓練時使用尚未存在的新 run ID（不必重新抽取 Pose/ST-GCN 特徵）：

```powershell
.\.venv\Scripts\python.exe -m fall_detect train --config configs/le2i-seqbalanced-001.yaml --run-id le2i-seqbalanced-next --show-loss
# 訓練完成後，關閉保留畫面的服務，再校準。
.\.venv\Scripts\python.exe -m fall_detect calibrate --run runs/le2i-seqbalanced-next
.\.venv\Scripts\python.exe -m fall_detect calibrate --run runs/le2i-seqbalanced-next --objective balanced_sequence_accuracy --output-run runs/le2i-seqbalanced-next-balanced
```

設定檔的完整既定來源與參數保存在 configs；原始資料、預訓練權重、cache 與 run 都不加入 Git。Loss 是特徵重建 MSE，不能轉換成跌倒機率。

## 證據

- [訓練前診斷](validation/further-training-diagnosis.json)
- [候選與選擇規則鎖定](validation/further-training-lock.json)
- [相同特徵／Scaler 檢查](validation/further-training-input-checks.json)
- [抽樣、Train/Test 隔離與校準風險測試](validation/pytest-further-training-risk.xml)
- [全套測試](validation/pytest-further-training-full.xml)

三組均已完成，完整結果如下。

## 實際完成結果

成功完成三組共 **577 回合**（250＋172＋155）。最佳正常 Validation Loss 由上一版 0.204255 降至影片均衡組的 0.197624，但這次 **整體正確判定率沒有提升**。

| 訓練版本 | 完成回合 | 最佳回合 | 最佳正常 Validation Loss |
|---|---:|---:|---:|
| le2i-timing-001 | 88 | 78 | 0.204255 |
| le2i-long-002 | 250 | 237 | 0.203127 |
| le2i-seqbalanced-001 | 172 | 142 | 0.197624 |
| le2i-seqcompact-001 | 155 | 125 | 0.285058 |

所有新模型依規格 F1 校準時，結果都是 TP20／FP10／TN0／FN0、unknown8/38，整體正確判定20/38＝52.6%。各自平衡門檻的結果：

| 平衡校準版本 | 整體正確判定（含 unknown 分母） | 跌倒有警示 | 正常誤報 | unknown |
|---|---:|---:|---:|---:|
| le2i-timing-balanced-001 | 21/38（55.3%） | 12/26 | 1/12 | 8/38 |
| le2i-long-002-balanced | 21/38（55.3%） | 12/26 | 1/12 | 8/38 |
| le2i-seqbalanced-001-balanced | 21/38（55.3%） | 12/26 | 1/12 | 8/38 |
| le2i-seqcompact-001-balanced | 20/38（52.6%） | 10/26 | 0/12 | 8/38 |

預先鎖定的平衡指標選到 compact：60.8974%，比舊平衡版60.5769%僅高0.3205個百分點。這個指標平均兩類正確率，**不等於整體準確率**：compact少1段正常誤報，卻少2段跌倒警示，整體正確判定反而由21段降為20段。因此保留此客觀選擇結果，但不宣稱compact比較準，也不以它取代使用者目前的整體準確率基準。所有結果來自同一38段Validation，尚無新Test成績。

## Windows 存檔中斷與恢復

le2i-long-001在發布165回合後，原子替換training-progress.json發生WinError5而退出，未完成可用checkpoint。與Windows讀取/替換競爭一致，但未直接確認持有鎖定的程序。失敗紀錄保留、狀態標failed；不把部分曲線當成已完成模型。

已為Windows錯誤5/32/33增加有界重試（最多9次嘗試、合計等待最多約1.06秒），舊完整檔在重試期間仍可讀取，永久權限／磁碟錯誤不隱藏。訓練例外與中斷會留下失敗狀態，Loss畫面不再被舊running快照蓋過。新增測試驗證暫存清理、永久錯誤、部分訓練與不能覆寫既有run。

le2i-long-002採完全相同配置與seed重跑250回合；前165回合的Train／Validation Loss與失敗紀錄逐值差0。此為技術失敗重跑，不是額外一組超參數候選。原始鎖定文件不變，另存[恢復紀錄](validation/further-training-recovery.json)。

## 最後驗證與限制

- **100 passed／1 skipped／11 warnings，19.01秒**。唯一skip仍是原Human缺四個外部資源；CUDA deterministic warn-only、舊套件棄用與CPU backend警告未隱藏。ruff、format、compileall通過。
- 兩段真實Validation影片用compact平衡版重新執行Pose→ST-GCN→AE，正常Home02/video47輸出359幀、UR fall-02輸出110幀，均可完整解碼。48＋7個共同可見有效端點的分數差0；UR狀態與完整事件相同。Le2i有4點受cache標註間隔影響而狀態不同、1點因當下低品質隱藏分數，不宣稱全狀態相同。
- 正常影片無警示；fall-02首次警示2.633秒，2秒抽幀仍顯示normal。已查看實際抽幀，沒有將序列TP當作事件開始時間正確或即時偵測成功。
- Loss瀏覽器確認失敗run顯示中斷、重跑run完成250回合、均衡run完成172回合與Epoch142最佳0.197624，並顯示影片均衡抽樣提示。
- 舊五個凍結run的全部hash不變，新三組F1／平衡校準的AE、Scaler、Loss CSV配對hash相同；每組相同Train特徵、Scaler、正常Validation來源均核對。
- 本輪使用既有真實特徵重新訓練AE；沒有新增資料、未微調Pose/ST-GCN，沒有消除原有8段unknown。完整CPU實驗、新場景泛化與事件起始／延遲均未驗證。
- 這輪證據支持「增加回合、均衡抽樣與縮小AE尚未提高本批整體正確率」。不能據此斷言任何更長訓練皆無效，也不能用更低Loss宣稱辨識成功。

證據：[完整數字與凍結hash](validation/further-training-results.json)、[真實影片整合](validation/further-training-integration.json)、[存檔重試回歸](validation/pytest-training-lock-recovery.xml)、[最後全套測試](validation/pytest-further-training-final.xml)、[最後檢查](validation/further-training-final-checks.json)。
