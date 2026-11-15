# 接續訓練：單獨比較條件式旋轉骨架

2026-09-17，新增 `runs/openpose18-rotation-001`。已完成真實特徵上的 AE 訓練、Validation 門檻校準與一支真實 Validation 影片整合驗證。**本輪未執行 Test；以下數字是用於選模／選門檻的 Validation 結果，不能當獨立測試成績。**

## 在程式查看

雙擊根目錄「查看Loss曲線.cmd」，選擇 **openpose18-rotation-001**。也可執行：

```powershell
.\.venv\Scripts\python.exe -m fall_detect loss --run runs/openpose18-rotation-001 --port 0
```

圖中實跑 100 Epoch，最佳正常 Validation Loss **0.4996334114985075**、Epoch **95**；同回合 Train Loss 0.05607065775921579。第100回合 Train／Validation 為0.053857／0.503671。使用既有真實骨架特徵快取，因此本次 AE 重訓不需重新處理70段影片；真實原圖推論另由下方小型demo驗證。

## 本輪控制的條件

原基準 `openpose18-baseline-001` 與本輪均使用同一份 UR Fall 原始70段split，Train24正常、Validation8正常＋15跌倒；沒有新增Le2i。AE架構、seed42、Adam、learning rate0.001、batch32、最多100回合及early stopping預設相同。唯一預定的前處理改動為先前已實作的條件式0/±90度真實影像pose推論；沿用同組真實Pose／ST-GCN權重、18點／256D結構及品質規則。

訓練前核對47段Train／Validation快取及所有既有run凍結hash。實際Train19/24段、346個有效特徵；正常Validation7/8段、128個特徵。原基準Train為351窗。旋轉可能改變骨架、追蹤與可用窗集合，不保證每段都增加有效特徵。兩個run各自只用Train fit scaler，因此原始MSE不能直接當作辨識能力排名。

## 相同23段 Validation 的結果

兩個run各依既定Validation規則選擇門檻；下表包含unknown，不因無法判定而縮小分母。

| 指標 | 原基準 | 本輪條件式旋轉 |
| --- | ---: | ---: |
| 正常影片誤報 | 6/8 | 6/8 |
| 正常影片正確無警示 | 0/8 | 0/8 |
| 正常影片unknown | 2/8 | 2/8 |
| 跌倒影片有警示 | 8/15 | 9/15 |
| 跌倒影片unknown | 7/15 | 6/15 |
| 全部影片unknown | 9/23 | 8/23 |
| 有效窗口coverage | 38.02% | 39.28% |
| 校準門檻 | 0.19389973580837247 | 0.2031607925891876 |

這輪只多一段Validation跌倒影片能警示，**正常誤報沒有改善**；不能說旋轉已解決問題。條件式F1從0.7273變成0.7500是校準集指標，且排除了unknown，不能寫成75%測試準確率。

Le2i擴充run使用另一份38段Validation，不能把其校準分數直接當成此表相同分母的比較。Le2i四段正常Validation無有效特徵的阻塞仍未修復，本輪也沒有宣稱修復。

## 實際整合驗證

- [Validation fall-02 展示影片](../outputs/real_end_to_end/rotation-validation-only/fall-02.mp4)：110幀完整重新解碼，30fps輸出；7個共同有效端點raw score差0，狀態及events相同。警示從1.9667秒確認，2.8秒因track_gap結束。未驗證人工跌倒起點或警示延遲。
- 真實抽幀仍可見重複骨架及低品質／追蹤斷裂，沒有把unknown改成正常或補假骨架。
- 相關測試11 passed、5.77秒：[pytest紀錄](validation/pytest-rotation-training.xml)。未修改核心程式，因此未重跑上回76項完整測試。
- CUDA訓練記錄deterministic為warn_only；本輪收到CuBLAS提醒，沿用已鎖定預設，未宣稱GPU重訓逐位元可重現。
- 原基準、Le2i與本輪凍結產物均核對。沒有重新選擇前兩個run的模型／門檻，也沒有新增Test結果。

證據：[開始前設定鎖定](validation/rotation-training-lock.json)、[訓練與校準比較](validation/rotation-training-results.json)、[真實整合驗證](validation/rotation-training-integration.json)。本輪屬探索性對照；歷史Test已看過，後續改善仍應保留新的未查看測試資料。

接下來要改善的重點仍是正常Validation的骨架與追蹤可用性，以及正常活動誤報；單純繼續增加Epoch並沒有提供解決這些問題的證據。
