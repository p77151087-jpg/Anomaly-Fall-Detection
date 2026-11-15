# 雙肩可選 demo：原 RGB 至畫面的 23 支 Validation 驗收

實驗 `ur-shoulder-demo-20261112-001` 已完成。先由原影片實跑 fall-19、adl-35，兩支通過後才執行其餘 21 支。固定原 wide 與既有雙肩候選，沒有重訓、重新校準、讀取真實 Test 特徵／影像或新增 Test 評估。原預設 demo 保留；雙肩模式由 CLI 明確選用。正式 Notebook、五分鐘影片及公開上傳繼續暫緩。

## 實際修改

- `python -m fall_detect demo` 新增 `--shoulder-candidate`。候選必須綁定原 wide 的 run、特徵 recipe、split、正常標定檔與完整設定 SHA；不能混用另一 AE 或門檻。
- 原 RGB 仍實際經過 detector、AlphaPose、追蹤、ST-GCN、Train-only Scaler 與凍結 AE。新分支直接觀察同一條因果採樣序列，不讀舊 Pose／特徵來代替本次 RGB 推論。
- AE 門檻 **0.5341113805770874**；雙肩門檻 **0.18564587694466197**，沿用先前 24 Train ADL＋8 Validation ADL 的正常 K3 上界，沒有在本輪重算。30 幀 isotropic、stride 1、六格差分、原信心／合法窗口／追蹤／過期遮罩保留。
- AE 與雙肩各自形成 K3，然後同 track 取 OR。顯示的 `event_score` 是兩個獨立 K3 除以各自門檻後的最大值，判定嚴格大於 1；`raw_score` 是可用瞬時分數的正規化最大值，不拿它混合累積 K3。兩者都不是跌倒機率。
- sample 畫面標題為 `AE + SHOULDER EVENTS`，保留 `ALERT RECORDED` 與 `Current: ...` 的區別。左側七類 ACTION 仍是真實原分類頭的獨立對照。
- 額外保存本次新 Pose NPZ／JSON、512D NPZ、逐窗分支／合併 CSV、原 wide 事件 CSV。一般 demo 的既有輸出欄位及未指定新選項時的行為維持。

## Pilot 與完整結果

Pilot 的兩片原來源 SHA 相同、真實權重 strict 載入及 forward 均通過。fall-19 首次事件／當前可見警示 **1.500 秒**，連續 **7 幀**，1.733333 秒因端點信心關閉；adl-35 **0 事件、0 當前警示幀**。383 個輸出影格核對通過後才開始其餘 21 支。

| 版本／證據 | 整體 Accuracy（含 unknown） | 可判定 Accuracy | Fall F1 | Precision | Recall | TP / TN / FP / FN / U |
|---|---:|---:|---:|---:|---:|---|
| 目前舊 demo 預設 `ur-sample-human-s1-001`（歷史） | 20/23＝86.96% | 20/23 | 0.888889 | 12/12 | 12/15 | 12 / 8 / 0 / 3 / 0 |
| 原 wide AE 參照（歷史） | 22/23＝95.65% | 22/23 | 0.965517 | 14/14 | 14/15 | 14 / 8 / 0 / 1 / 0 |
| 既有 wide＋雙肩候選（快取重播） | 23/23＝100% | 23/23 | 1.000000 | 15/15 | 15/15 | 15 / 8 / 0 / 0 / 0 |
| **本輪原 RGB＋可選 demo 實跑** | **23/23＝100%** | **23/23** | **1.000000** | **15/15** | **15/15** | **15 / 8 / 0 / 0 / 0** |

新模式矩陣（列：ADL／Fall；欄：normal／suspected_fall／unknown）為 `[[8,0,0],[0,15,0]]`。二元可判定分母 23，整體分母也為 23，因影片 unknown 為 0。原 wide 的矩陣為 `[[8,0,0],[1,14,0]]`。相對原 wide 多救回一支 fall-19，Accuracy +4.35 個百分點；相對已完成的雙肩候選**沒有再提高 Accuracy**，本輪新增的是完整 RGB 推論及實際畫面一致性證據。

合法唯一端點仍 **2,287/2,710＝84.39%**，未放寬品質。共重新讀取 **3,373 個原始 RGB 觀測**，輸出 **3,377 個因果時間格點**、**2,917 個 track-window 列**、**2,363 個有限 512D 特徵**；其中 554 個窗口列因 track_gap／低信心／無人物不可評分。track-window 列數、特徵列數與唯一時間端點的分母不同，不能混用。影片 unknown0 不代表每個窗口都可用。

## 判對與警示時間分開

| Fall | 原 wide 首事件（秒） | 新模式首事件（秒） | 新模式當前警示幀 |
|---|---:|---:|---:|
| fall-02 | 2.533333 | 2.100000 | 29 |
| fall-06 | 1.566667 | 1.333333 | 52 |
| fall-07 | 3.766667 | 3.300000 | 51 |
| fall-08 | 2.533333 | 1.633333 | 13 |
| fall-09 | 5.900000 | 4.566667 | 16 |
| fall-11 | 3.100000 | 2.333333 | 27 |
| fall-13 | 2.300000 | 1.966667 | 14 |
| fall-14 | 1.533333 | 1.300000 | 21 |
| fall-16 | 1.433333 | 1.400000 | 13 |
| fall-18 | 1.433333 | 1.300000 | 4 |
| fall-19 | 無 | **1.500000** | **7** |
| fall-22 | 1.100000 | 1.100000 | 17 |
| fall-25 | 2.466667 | 2.133333 | 7 |
| fall-29 | 2.566667 | 2.466667 | 6 |
| fall-30 | 1.866667 | 1.566667 | 4 |

原 14 支 TP、8 支 TN 全保留；13/14 原 TP 的首次事件提前，fall-22 持平。本輪 23 個合併事件包含同片多次事件，**不是 23 支跌倒影片**。共 281 個時間格點有當前警示。每片起訖、峰值及展示格點決策均與先前雙肩候選一致。

fall-19 的原 AE 仍未越線，新分支在原影格 45／採樣 1.500 秒警示，屬使用者既有 transition 姿態階段；當時七類頭仍顯示 WALKING，不替代異常模型判定。首次警示的整體骨架本人標註仍為 `wrong_or_drift`：信心通過不代表幾何正確，也沒有因為救回影片就宣稱骨架已修好。

既有六段粗時段參照維持事件起點命中 5/6、當前品質通過警示 6/6、正常前段早報 0。這些不是教師成績公式或精確人工時間真值。fall-09 首事件 4.566667 秒仍略早於舊粗 phase 的 4.604 秒，不能把 23/23 判對稱為全數時機正確。

人工三片的首次時間均早於既有最早落地界限：fall-18 1.300 vs 1.301 秒、fall-09 4.566667 vs 4.705 秒、fall-30 1.566667 vs 1.635 秒。fall-18 僅差 1 ms，遠不足以宣稱精確的提早量；標註是時刻可能範圍。其餘影片未補做人工精確時間標註。

## 驗證與保存

- **synthetic**：71 項風險測試通過，涵蓋可選 CLI、兩條獨立 K3、嚴格門檻、合法窗口與缺失重置、held source／因果來源、候選 SHA／run／quality／split 綁定、五個新增 sidecar 防覆寫、原串流／畫面流程。測試中的 synthetic Test 字樣不涉及真實 UR Test。
- **real_feature_smoke**：本次 23 支 RGB 實跑保存後的 Pose／bbox NPZ 與原快取逐值相同；全部 2,363×512 特徵有限且與原特徵逐值相同。detector、Pose、ST-GCN 的 strict 與 forward 證據逐片保存。這是同一批完整實跑的特徵檢查，沒有另外重跑模型。
- **real_end_to_end**：23 支從原 RGB 重新執行到成品。獨立 CSV-only rolling minimum／OR 核對 2,917 窗與 23 事件；逐片重新計算矩陣、分母、Accuracy／F1。3,377 個原編碼影格逐幀核對右面板、當前狀態 glyph 與事件歷史紅字；再另存 23 支 H.264/yuv420p，相同 3,377 幀再次核對文字／色彩／尺寸／FPS／幀數。
- 分支分數相對原快取最大差 **1.92948015e-7**，所有缺失遮罩、決策、事件起訖與當前警示格點一致；不能把這種浮點差異稱為模型改善。原面板整體最大 MAE 4.02253，低於事前 8.0 上限，另以四種文字模板判別及紅字顏色核對避免僅靠平均誤差。
- 215 個有界原成果／run／47 支 Train/Validation 快取／預訓練權重 SHA 保全。五份改動前 source／文件原 bytes 保存於 QA `prior-source`。真實 CLI 重複輸出被拒絕，兩支 pilot 22 檔 SHA 不變；只讀 `verify` 再跑全 23 支通過，沒有重新推論或改寫已完成驗收。
- 初版 pilot checker 與 contract 保留；在完整階段前另存 v2，加入當前文字四分類及歷史紅字判別，重新檢查既有兩片後才放行 21 片。沒有重跑／調參或修改舊候選 lock。
- Graph 使用 Tier 2，相關來源 coverage 為 no_recorded_issue／metadata_match；這是 best effort。run／cache／outputs／weights 設計排除，直接讀取有界原檔與 SHA；部分 trace 邊為 heuristic，不宣稱完整 graph 證明。

**結論：雙肩模式通過本批 Validation，可作為明確選用的 demo；舊預設保留。** 新模式 Test 未知。Validation 已多次用於研究及正常門檻標定，不是全新盲測；歷史已查看 Test 也沒有恢復獨立性。沒有自行換算課程分數、發布成果、重訓或降低品質門檻。

## 檔案與直接執行

- [23 支可播放成果頁](../outputs/ur-shoulder-demo-20261112-001/gallery/index.html)、[fall-19 H.264](../outputs/ur-shoulder-demo-20261112-001/gallery/fall-19.mp4)、[adl-35 H.264](../outputs/ur-shoulder-demo-20261112-001/gallery/adl-35.mp4)、[實際解碼首警示圖](../outputs/ur-shoulder-demo-20261112-001/gallery/fall-19-first-alert.png)。成果頁提供片段切換、事件跳轉、半速及逐格；此輪以解碼像素驗收，沒有宣稱完成真瀏覽器 UI 操作驗收。
- [全部影片結果 CSV](../outputs/ur-shoulder-demo-20261112-001/validation-sequences.csv)、[完整驗收 JSON](../outputs/ur-shoulder-demo-20261112-001/validation-acceptance.json)、[pilot v2](../outputs/ur-shoulder-demo-20261112-001/pilot-acceptance-v2.json)、[鎖定來源](../outputs/ur-shoulder-demo-20261112-001/contract-v2.json)、[H.264 核對](../outputs/ur-shoulder-demo-20261112-001/gallery/verification.json)。逐影片 CSV／窗口／事件與新 Pose／特徵位於同 outputs 的 `pilot`、`validation`。
- [synthetic XML](validation/ur-shoulder-demo-20261112-001/synthetic-risk.xml)、[真實特徵證據](validation/ur-shoulder-demo-20261112-001/real-feature-smoke.json)、[防覆寫](validation/ur-shoulder-demo-20261112-001/overwrite-refusal.json)、[graph 證據](validation/ur-shoulder-demo-20261112-001/graph-evidence.json)。
- [獨立 CSV 指標／人工時序再核](validation/ur-shoulder-demo-20261112-001/independent-post-check.json)、[交付檔案核對](validation/ur-shoulder-demo-20261112-001/delivery-check.json)。成果頁 JavaScript 已通過 `node --check`。

在專案根目錄使用 PowerShell：

```powershell
# 只讀核對已完成 23 片，不推論、不訓練、不評估 Test
.\.venv\Scripts\python.exe -B -X utf8 scripts/validate_shoulder_demo.py verify

# 開啟本機成果頁
Start-Process (Resolve-Path 'outputs/ur-shoulder-demo-20261112-001/gallery/index.html').Path

# 手動產生新的可選 demo；new-output.mp4 與所有 sidecar 必須尚不存在
.\.venv\Scripts\python.exe -B -X utf8 -m fall_detect demo --run runs/ur-isotropic-wide-20261012-001 --input data/raw/urfall/fall-19 --output outputs/shoulder-demo-manual/new-output.mp4 --shoulder-candidate outputs/ur-fall19-shoulder-descent-20261107-001 --device cuda --overlay sample --no-display
```

省略 `--shoulder-candidate` 即沿用原 AE-only 流程。已完成的 `pilot`／`validation` 命令僅重用並核對既有成果；缺少任何完整檔時不會蓋掉半成品。訓練與校準的舊命令不屬本次操作。
