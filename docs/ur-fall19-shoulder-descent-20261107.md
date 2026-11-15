# fall-19 已在 Validation 由雙肩下移分支救回

`ur-fall19-shoulder-descent-20261107-001` 在同一批 23 支 Validation 達 **23/23**。fall-19 首次事件在 **1.500 秒**，對應原影格 45，同 track 的當前 AE 警示共 **7 幀**。原 wide 的 14 支跌倒 TP、8 支正常 TN 及其原有可見警示端點全部保留。這組候選通過事前的辨識及展示檢查。 [事前規則](ur-fall19-shoulder-descent-20261107-plan.md) · [並排影片](../outputs/ur-fall19-shoulder-descent-20261107-001/visual-review/fall-19-wide-vs-shoulder.mp4)

## 這次抓到了什麼訊號

既有圖像顯示 fall-19 出框前雙肩一起往下移，部分下肢點已不可靠。將兩側肩點各自的垂直位移和六個格點前的同一 track 比較，取較小位移除以前一端人物框高度，負值記為零。這項觀測只在原合法窗口內使用，當前與過去骨架和兩肩都須達原信心要求。兩分支各自用嚴格 `>` 及 K3，通過後才合併事件。原 AE 的分數和門檻照用。

新分支門檻 **0.1856458769** 來自 24 支正常 Train 和 8 支正常 Validation 各片最高 K3 的最大值（`adl-38`）。fall-19 的雙肩分支峰值 **0.4811579114**。在第一個警示格點，兩肩至少下移 **62.95 像素**，過去人物框高度 **212.31 像素**。三個相鄰合法窗已越線，K3 **0.210684**。同一時刻 wide K3 **0.091116**，仍低於其 **0.534111** 門檻。 [正常界線](../outputs/ur-fall19-shoulder-descent-20261107-001/normal-calibration.json) · [逐窗證據](../outputs/ur-fall19-shoulder-descent-20261107-001/validation-windows.csv)

可見警示從格點 1.500 到 1.700 秒，涵蓋原影格 45–51，位於人工第 33–61 幀的轉換階段。第 45 原影格的來源時間為 1.468 秒，警示使用因果格點 1.500 秒。較早 11 個合法窗的雙肩 K3 峰值 0.120154 仍低於門檻。1.733333 秒起因信心條件關閉此事件，之後的歷史紅字仍保留。首次警示幀通過原信心門檻，既有人工作業仍標整體骨架 `wrong_or_drift`；警示已救回，骨架錯位及出框後的觀測缺失仍需如實呈現。

![解碼後的首次警示，右側在同一原影格顯示 Current: alert](../outputs/ur-fall19-shoulder-descent-20261107-001/visual-review/first-visible-alert.png)

畫面兩側使用同一原 RGB、Pose 及封存七類 `ACTION`，第一個警示格點的分類都是 `WALKING`。右側 AE 的 `ALERT RECORDED` 與 `Current: alert` 另由雙肩異常分支觸發。

## 其他影片與展示驗收

| 固定 Validation | 原 wide | wide＋雙肩分支 |
|---|---:|---:|
| 正常判對 | 8/8 | 8/8 |
| 跌倒判對 | 14/15 | 15/15 |
| 整體 Accuracy | 22/23＝95.65% | 23/23＝100% |
| Fall F1 | 0.965517 | 1.000000 |
| 正常誤報／unknown | 0／0 | 0／0 |
| 合法唯一端點 | 2,287/2,710 | 2,287/2,710 |
| 舊六段事件開始命中 | 4/6 | 5/6 |
| 舊六段有當前良好骨架警示 | 2/6 | 6/6 |
| 舊六段正常前段早報 | 0 | 0 |

原 14 支 TP 中有 13 支首次事件提早，事件總數 14→23。fall-18、fall-09、fall-30 的首次事件分別由 1.433→1.300、5.900→4.567、1.867→1.567 秒，三者早於各自人工最早落地界限。fall-18 只比最早界限早約 1 毫秒，fall-09 落在開始時刻的不確定範圍內。這些時間是既有人工邊界的對照，其他提前案例和新增事件仍有時序真值限制。 [23 支逐片結果](../outputs/ur-fall19-shoulder-descent-20261107-001/validation-comparison.csv) · [完整事件](../outputs/ur-fall19-shoulder-descent-20261107-001/validation-events.csv)

這次依已看過的 fall-19 診斷選定訊號，32 支正常影片也用於固定新分支界線。23/23 表示這組 Validation 的影片級結果，尚無新資料上的泛化證據。新候選與原 demo 路徑各自保存，訓練 0 epochs，Test 評估 0 次。

## 驗證與重現

28 項 synthetic 測試通過。獨立重算 47 支影片的 7,646 個肩部分支窗口、2,917 個合併窗口、23 個事件、23 個影片結果和 3,377 個展示格點。fall-19 的原影像並排 H.264 短片解碼 100 幀，逐格核對事件歷史與當前狀態，首警示圖取自實際解碼影格。170 份實驗輸入和 107 份畫面輸入的 SHA 前後一致，重建實驗及短片皆實測拒絕覆寫。這是既有真實特徵與分數加新分支的評估，畫面證據為既有 RGB 與 Pose 到編碼影格的重播。 [驗證紀錄](validation/ur-fall19-shoulder-descent-20261107-001/verification.json)

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe -B -X utf8 scripts/fall19_shoulder_descent.py verify
```
