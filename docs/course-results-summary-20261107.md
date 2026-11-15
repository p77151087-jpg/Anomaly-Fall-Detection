# UR-FALL 跌倒異常偵測｜期末成果摘要

2026-11-07 本機整理。以原 24 支正常 UR-FALL Train ADL 擬合 Scaler 與 AE，從 ST-GCN 骨架特徵的重建誤差偵測異常。同一追蹤對象須有三個連續合法窗口越過門檻才記錄事件。影片級比較使用固定的 23 支 Validation，含 8 支正常與 15 支跌倒。

## 辨識結果

| 角色 | 版本 | 整體 Accuracy | Fall F1 | 跌倒判對／漏報 | 正常誤報／unknown |
|---|---|---:|---:|---:|---:|
| 最高 Validation 研究參照 | `ur-isotropic-wide-20261012-001` | **22/23（95.65%）** | **0.9655** | 14／1 | 0／0 |
| 目前 demo 預設 | `ur-sample-human-s1-001` | 20/23（86.96%） | 0.8889 | 12／3 | 0／0 |
| 殘差聚合對照，未採用 | 凍結 wide＋Top‑26 | 21/23（91.30%） | 0.9286 | 13／2 | 0／0 |

三組皆可判定 23 支。wide 的成績用於研究比較，demo 的畫面以實際載入版本標示。Validation 曾用於候選比較與門檻校準。 [wide 評估](../outputs/isotropic-refinement-20261012/ur-isotropic-wide-20261012-001/metrics.json) · [demo 評估](../outputs/human-isotropic-validation/ur-sample-human-s1-001/metrics.json) · [Top‑26 報告](ur-fall19-residual-aggregation-20261107.md)

## 三段展示證據

| 影片 | demo 首次 AE 事件 | wide 首次 AE 事件 | wide 當前警示／歷史紅字幀數 | 短片 |
|---|---:|---:|---:|---|
| fall-18 | 無 | 1.433333 秒 | 0／22（共 65 幀） | [並排](../outputs/ur-validation-display-20261105-001/videos/fall-18-compare.mp4) |
| fall-09 | 5.900000 秒 | 5.900000 秒 | 8／8（共 82 幀） | [並排](../outputs/ur-validation-display-20261105-001/videos/fall-09-compare.mp4) |
| fall-30 | 無 | 1.866667 秒 | 0／14（共 70 幀） | [並排](../outputs/ur-validation-display-20261105-001/videos/fall-30-compare.mp4) |

三段 wide 的歷史面板起點與 CSV 事件一致。fall-18、fall-30 在事件當下因骨架品質顯示 `Current: unavailable`，歷史面板仍記錄事件。七類 `ACTION` 是獨立的分類輸出。三次 wide 首次事件均落在人工標註的落地時刻不確定區間，落地前預警仍未確認。 [畫面核對](ur-validation-display-20261105.md) · [人工時序對照](ur-validation-human-timing-20261105.md)

## fall-19 漏報與結案依據

轉換階段有 **21/29 個合法窗口**，原始誤差中位數 **0.091116** 與正常 Validation 的 **0.094487** 重疊。固定 21 個連續合法窗做對照後，8 支正常 Validation 仍各有一段的 K3 峰值高於 fall-19。倒地階段有 **0/38 個合法端點**。兩段分別反映分數分離不足和觀測中斷。 [分數與窗口診斷](ur-fall19-score-overlap-20261107.md)

人工修正六個可見骨架錯點後仍無事件。Top‑26 在早期片段 **0/9 個 K3 端點越線**，另使 fall-18 變成漏報。本階段以 wide 為研究參照，三段現成短片及 fall-19 失敗案例可作為後續正式 Notebook 與五分鐘影片的素材。 [六點控制](ur-fall19-whole-coordinate-20261013.md) · [早期逐窗](../outputs/ur-fall19-residual-aggregation-20261107-001/fall19-early.csv)
