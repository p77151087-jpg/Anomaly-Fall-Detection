# 下一步：保留全域動態的正常近鄰側分支（事前計畫）

2026-10-12，單一候選 `ur-motion-novelty-20261012-001`。此階段依使用者「繼續下一步」及既有正常資料方法研究授權執行，並非延長已結束的六 SVM／兩 AE 網格。教師方向仍為正常學習與異常偵測。

## 診斷依據與假設

前輪所有候選仍漏 fall-19；其 AI 粗時段有合法骨架及 K3，單純降低 wide 門檻須誤報八正常 Validation。`human_preprocess` 的逐幀 isotropic 置中及縮放，數學上移除每幀整體平移與尺度變化。前輪 raw XY 診斷也使用這個正規化，未檢驗被移除的全域軌跡。這支持另測未丟棄軌跡的表示，不支持直接宣稱它是漏報原因或盲目自監督微調。

舊 `ur-human-displacement-001` 共用窗口尺度已失敗；本次保留 wide 的真實 ST-GCN 輸入、權重、512D 與既有 AE，不重跑該前處理或重訓舊 AE。新增的正常近鄰分支獨立讀原像素骨架，避免改變預訓練模型的輸入分布。

## 鎖定的唯一方法

- 原 30 幀／stride1，沿用相同 Pose、track、resampler、品質及 K3。只對原合法窗口生成新表示，不補缺失、不跨 track、不用未來。
- 每幀 13 點像素 XY 外框中心 c(t) 與最大邊長 s(t)。按時間保留 30×3：`[(cx(t)-cx(0))/s(0), (cy(t)-cy(0))/s(0), log(s(t)/s(0))]`，共 90D。不使用 Fall 標籤、置信度加權、絕對鏡頭位置、分類頭或手定下降規則。
- 只用原 24 Train ADL 的全部合法窗 fit 90D Scaler 及近鄰 bank。距離為標準化表示的 Euclidean／sqrt(90)，分數為最近 5 個 Train 窗口距離平均，越大越異常。固定 k=5；沒有搜索 k、距離、權重、容量、seed 或閾值規則。
- Train 距離定標時排除整支查詢來源影片，避免重疊窗口與自鄰居使分數接近零；這些分數只用定標。Validation 的近鄰只來自完整 Train bank。原始 24 支不增強，不增加獨立影片。
- wide 分支沿用凍結模型 B=1 MSE。兩路各除以原 Train 分數 p95（近鄰用上述 leave-video-out 分數），最終 `max(AE/p95_AE, motion/p95_motion)`。兩路 p95 及 Scaler 不讀 Validation。分數不是機率。融合方式事先固定，不於結果後改成加權或 OR。
- 唯一正式候選為融合模型。兩路分數另存作診斷，沒有各自選一個最佳模型的追加候選。

## 隔離、pilot 與校準

另建私有 cache、設定、run、method recipe 和 90D representation recipe；原 512D feature recipe 原樣保留，90D 不冒充 ST-GCN 特徵。原 70 支 split 保留，Test 23 支 not_prepared／無路徑，入口拒絕 Test；不解碼 Test 特徵、不評估，不依歷史 Test 改選。

固定 pilot：Train adl-05／adl-06、Validation fall-19，各第一個合法窗口。核對嚴格真實 ST-GCN 512D、原快取 parity、有限 90D、時間／track／quality 一致及 cache 隔離，通過才生成全部 47 片表示並 fit。記錄原觀測與衍生表示，不覆寫舊結果。沒有新 Pose／SSL 或神經網路 epochs，近鄰 bank 擬合是真實正常模型工作。

完整 23 Validation 門檻掃描 finite unique K3 sequence scores 加最低值下界，strict `>`；排序含 unknown 整體 Accuracy、Fall F1、全體 Fall Recall、少 FP，最後較高門檻。舊 wide 門檻與結果不改寫，所有分母／unknown／事件另列。

## 採用與停止

辨識採用必須上述排名嚴格優於 wide 22/23／F1 .9655，unknown 不增加、窗口 coverage 不降低、不增加 FP 或其他 Fall FN。展示部署另外要求六 AI 粗時段事件命中至少原 4/6、粗正常早報不增加，救回的 fall-19 有粗時段內當前良好 Pose 警示；粗時段不是人工 GT，這些是工程展示檢查，不是教師計分公式。僅辨識通過而展示未通過可保存研究候選，不能擅換 demo。

只 fit 一個候選、一次 Validation 校準；不因失敗調 k、融合、事件或品質。持平、合法窗較多或個案提前不稱 Accuracy 提升。結果後停止並保存失敗候選；原 demo 預設保留，沒有新增 Test 授權。正式 Notebook／五分鐘影片仍暫緩、不公開上傳。

## 驗證要求

synthetic：置中縮放資訊反例、軌跡的平移／固定縮放不變性與時間順序、退化／非有限拒絕、Train-only 排除影片、融合及分數方向、缺失／K3／Test scope、不可覆寫。

real_feature_smoke：固定 genuine pilot、全部 47 真實觀測重建及 bank fit、23 Validation。獨立以 NumPy 重算 Scaler 與距離／融合、sklearn 重算 CSV 指標與分母、手動事件機重算每窗和完整事件、來源 SHA 與新舊 recipe／cache。real_end_to_end 不冒稱已跑；本階段不重新抽 RGB Pose 或製作 demo。

原 run、raw、external retirement、快取、權重、設定及輸出先 hash 再核對；原契約修改前副本保存。Graph Tier2 探索與 coverage，ignored cache／weights 用精確讀取與 hash 補核對，best effort 不當完整性證明。
