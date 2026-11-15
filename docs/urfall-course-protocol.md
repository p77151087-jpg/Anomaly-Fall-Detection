# UR Fall 課程實驗協定

2026-09-18。CA-01/02 實際執行；來源與未定事項見 [課程調整計畫](course-alignment-plan.md)。

## 資料與方法

沿用既有 UR Fall camera 0 RGB 原始序列切分：Train 24 ADL／0 Fall；Validation 8 ADL／15 Fall；Test 8 ADL／15 Fall。共 70 序列，seed 42；沒有重新切分，不保證人物隔離。歷史 Test 已反覆查看，這次結果是固定切分的工程對照，不是全新獨立測試。

YOLO11m Pose（COCO17 映射 OpenPose18）→凍結 Kinetics ST-GCN 的 256D 特徵→Scaler→正常資料 AE→重建 MSE。沒有七類分類或跌倒機率。預訓練模型並非從零開始的無監督訓練，權重來源、授權、實際 SHA256 保存在各 run 的 sources.json。

本次嚴格核對並重用 70 支既有真實特徵與骨架快取，另建 course-ur-yolo11 索引。每支原始來源 hash、切分角色、recipe、特徵陣列及窗口紀錄一致。讀取 Test 快取做完整性相等核對，沒有用它產生候選或選模。證據：[快取重用](validation/course-cache-reuse.json)。

Scaler 僅擬合 UR Train 24 支 ADL 的 659 個有效窗口。AE 亦僅用這些窗口；正常 Validation 8 支、202 個窗口的 MSE 選 checkpoint。完整 Validation 23 支校準門檻及選模型。資料標籤、正常特徵 hash、取樣數量可由各 run 的 sources.json 核對。

## 訓練前鎖定

[機讀鎖定紀錄](validation/course-experiment-lock.json) 在新訓練前寫入。三個候選：

| Run | AE latent | 訓練取樣 |
| --- | --- | --- |
| course-ur-baseline-001 | 32 | 全部窗口等權 |
| course-ur-balanced-001 | 32 | 每個序列等機率；序列內窗口等機率、有放回 |
| course-ur-compact-001 | 16 | 每個序列等機率 |

全部最多 100 epochs、patience 10、seed 42、hidden128、batch32、lr0.001。正常 Validation MSE 選 checkpoint；門檻最大化 Validation balanced sequence accuracy = (TP/全部Fall + TN/全部ADL)/2，同值選較少 FP、再選較高門檻。嚴格 `score > threshold`。

候選選擇依序：Validation operational accuracy、Fall F1、全體 Fall recall、較少 FP、較早列出的候選。恰好三個候選即停止；Test 後不追加訓練、不調門檻、不改選。採用的取樣改善針對 Train 每序列 9–55 個窗口差異；latent16 是限制重建能力的假設，不保證有效。

30 fps、30 幀窗口、stride5、連續三個有效窗口確認事件，品質與追蹤規則完全沿用。名義最早確認時間為第39格／1.30秒；實際需滿足品質與連續性。畫面分數最多保留 2×5/30 秒，品質失效立即 unknown。這些都是工程預設，未宣稱教師強制。

## 統計分母

影片中有任何確認事件為陽性；有合法連續三窗分數但無事件為陰性；沒有合法 event score 為 unknown。影片層級判對不代表起跌時間判對；未取得可靠 onset 核對前不報事件延遲、提前預警準確率。

- 二元 Accuracy=(TP+TN)/n_evaluable；Precision=TP/(TP+FP)；Recall=TP/(TP+FN)；F1=2TP/(2TP+FP+FN)。unknown 不在這組指標分母。
- 另列 2×3 表、n_total/n_evaluable/n_unknown、coverage，以及 operational accuracy=(TP+TN)/n_total、全體跌倒召回=TP/全部Fall。unknown 不改填正常。
- 空分母依既有契約顯示未定義；有可判定樣本但無陽性預測時 precision/F1 的零值約定沿用現有程式。
- Validation 只作開發選擇，Test 只作本輪凍結後對照。混合資料模型另列，不與 UR-only 混稱。

老師尚未明定官方切分、評估單位、unknown、外部訓練資料與公式尺度，保持待確認；不自行換算課程分數。UR 90%／F1 0.87 僅列課程參考，沒有宣稱直接同協定重現。

CUDA 訓練曾輸出 CuBLAS 決定性警告：本次沒有預設 CUBLAS_WORKSPACE_CONFIG。已保存 seed、環境、實際模型與 hash；不承諾重新訓練能逐位元相同，不為消除警告而重跑已鎖定的候選。
