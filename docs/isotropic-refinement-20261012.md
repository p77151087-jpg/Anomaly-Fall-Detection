# Test／Validation 差距診斷與等比例特徵 AE 改善（2026-10-12）

後續授權更新：本頁三候選Train／Validation實驗完成後，使用者明確要求只評估大容量wide一次Test、不依結果調參。已完成Test Accuracy18/23＝78.26%、Fall F1 .8000、TP10／TN8／FP0／FN5／U0，原門檻／模型／split／採用guard保持不變。[後續Test正式結果](ur-isotropic-wide-test-20261012.md)。以下「本輪不跑Test／新候選Test未知」保留為當時有限實驗的紀錄，不改寫原lock／結果／停止條件。

使用者授權找出差距並繼續使用 Train／Validation 改善，明確禁止以 Test 調參。本輪不新增任何候選的 Test 評估，不修改切分、舊 run、模型、門檻、原始資料或骨架品質規則。正式 Notebook／五分鐘影片仍等待辨識品質。

## 診斷完成、候選已鎖定

已查看的凍結 `ur-human-isotropic-001`：Validation TP13／TN8／FP0／FN2，21/23=91.30%、F1=.9286；Test TP11／TN8／FP0／FN4，19/23=82.61%、F1=.8462。兩者 unknown 均0，二元／整體分母各23；差8.70百分點是多漏報2支。每錯1支為4.35百分點；尚無證據將差距全部歸因單一原因。Test 只沿用先前保存的描述，不將其漏報身分、窗口或分數用於本輪候選選擇。

重新推論 Train／Validation 真實快取，證實正常 Validation 的平均重建誤差 .115134856，是乾淨 Train .056816514 的2.03倍。這是正常重建泛化落差，不足以單獨證明模型過擬合，更不證明 Test 全部下降由此造成。已保存的完整門檻掃描最高整體 Accuracy 仍21/23，單純降低門檻不能改善準確率。

Validation fall-18 K3=.449466884，門檻=.452907205，差約0.76%；fall-19 K3=.171946585，僅23/71有效端點、特徵較接近正常 Train 活動。原本人工抽幀已發現跌倒時部分身體在畫面外／骨架中斷。fall-19 的可用特徵最近正常 Train 距離中位數=.18156，正常 Validation 各影片距離中位數=.11007–.45332，表示已有特徵重疊；最近鄰只是描述，沒有改成分類器。fall-18 可用特徵距離中位數=.66627，不能把所有漏報都說成相同原因。

上述距離是影片可用窗口的分布，包含跌倒前正常動作；不能單憑最近鄰距離證明「跌倒動作本身已被當成正常」或指定某支Train影片為漏報原因。後段骨架中斷與低分數是已觀察現象，其對此模型漏報的完整因果貢獻仍未量化。

待驗證假設：新等比例特徵下 AE 容量或去噪能改善正常泛化／異常分離。Validation 已多次選模，可能偏樂觀；23支的小樣本也可能放大差距。精確貢獻尚未識別。stride1 的相鄰窗口高度重疊，不能把數千窗當數千支獨立影片。沒有正式骨架及跌倒時間真值，影片級判對不等於警示時間正確。

## 本輪事前契約

三組均使用同一 genuine Human512D 等比例特徵、原24 Train ADL／3638窗、原23 Validation、原split。Scaler／AE只用Train，seed42、sequence sampling、wd=.001、每組最多100epochs／patience10；乾淨正常Validation Loss選checkpoint，完整Validation balanced sequence Accuracy選門檻，保留strict > 與K3，以及 .2／9／.8品質規則。

| 新run | hidden／latent | Train輸入noise std |
|---|---:|---:|
| ur-isotropic-compact-20261012-001 | 64／8 | 0 |
| ur-isotropic-wide-20261012-001 | 256／64 | 0 |
| ur-isotropic-denoise-20261012-001 | 128／32 | .1 |

比較目前最高Validation 21/23；候選須嚴格提高全體Accuracy、F1不下降、無新增誤報／unknown、舊六案例至少4段粗略命中／0早報，且每支新救回漏報須有舊時段內當前良好骨架的警示，才可選為改善候選。按Accuracy／F1／全體召回／少FP排序，平手保留較早候選。舊預設仍 `ur-sample-human-s1-001`；既有 isotropic 未通過的採用規則不改寫。三組跑完即停止，不追加seed、容量、品質、K或門檻目標；即使成功也不跑Test。

訓練前保存舊run、設定、相關完整快取、manifest、4組真實權重、核心source及前次Test輸出的實際hash。保護函式直接拒絕Test特徵；hash保全的讀取不參與訓練或選模。完整index保留70支原始切分的metadata，不重新切分；Train／校準只讀允許的序列。

診斷：`validation/isotropic-refinement-20261012-diagnosis.json`。候選鎖定：`validation/isotropic-refinement-20261012-lock.json`。工作流程：`scripts/human_isotropic_refinement.py`，沒有Test操作入口。

## 執行與驗收

三候選實際完成 **121 epochs**，未追加試驗或 Test。結果如下；全部有23個可判定序列、unknown0，所以可判定二元Accuracy與包含unknown全體Accuracy的分母均23，數字相同。Fall F1為可判定二元指標；全體跌倒召回以15支Fall為分母。

| run | epochs | 最佳正常Val Loss | 凍結門檻 | TP／TN／FP／FN | Accuracy | Fall F1 | 全體Fall召回 |
|---|---:|---:|---:|---|---:|---:|---:|
| 既有 isotropic 參照 | 49（前輪） | .115134855 | .452907205 | 13／8／0／2 | 21/23＝91.30% | 26/28＝.9286 | 13/15＝86.67% |
| compact | 59 | .173296404 | .680558562 | 14／7／1／1 | 21/23＝91.30% | 28/30＝.9333 | 14/15＝93.33% |
| wide | 34 | .115616366 | .534111381 | 14／8／0／1 | **22/23＝95.65%** | **28/29＝.9655** | 14/15＝93.33% |
| denoise | 28 | .118674403 | .615851045 | 13／8／0／2 | 21/23＝91.30% | 26/28＝.9286 | 13/15＝86.67% |

**wide 的影片級 Validation Accuracy 確實提高4.35百分點，救回 fall-18、無新增誤報，剩下 fall-19 漏報。** fall-18 K3 .547446549 > 新門檻 .534111381，第一個事件1433.333ms。正常Loss沒有低於既有參照，因此本輪不支持「降低正常Loss」這項改善機制；只證實不同容量改變此Validation案例與正常活動的分數排序／分離。這一支分數距門檻僅約2.50%，仍不代表穩健泛化。

compact救回fall-18但誤報adl-37，Accuracy沒提高；denoise亦未提高。wide有4段舊粗略時段命中／0早報，但新救回fall-18在其時段內**當前良好骨架警示為0幀**，不通過事前採用guard；fall-09首次事件5900ms也落在既有粗略時段外。整體命中數相同掩蓋了個別時間案例變化，因此逐案例紀錄全部保留。舊預設仍 `ur-sample-human-s1-001`，wide另列為目前最高影片級Validation研究候選，沒有追溯改寫舊guard、沒有宣稱警示時間合格。

這輪候選**均未跑Test**，它們的Test Accuracy／F1未知。已查看的舊isotropic Test仍82.61%／.8462，未改動。保存結果中4個Test漏報均有合法K3分數、但未跨凍結門檻，影片unknown0；不是把缺失影片當正常或漏算造成的分母問題。Test有效窗口覆蓋率81.07%、Val84.39%，這個小差異不足以證明全部漏報由骨架品質造成。候選假設及設定來自Train／Validation診斷，不回用Test漏報分數。歷史Test已看過、Validation反覆選模，均不能宣稱新的盲測或直接等同未明定協定的課程分數。

## 驗證與保全

- `synthetic`：與變更風險相符的 **26 passed／0 skipped**，測試Test隔離、實際AE訓練／校準、Scaler、快取、K3、unknown、等比例前處理與不覆寫。XML：`validation/pytest-isotropic-refinement-20261012.xml`。初版一項新測試錯寫synthetic Validation筆數4，實際為2，修正後通過，保留失敗XML。沙箱pytest暫存目錄失敗後使用授權測試環境執行。
- `real_feature_smoke`：既有真實Pose／ST-GCN快取上，三組實際訓練、校準與完整Validation評估；sklearn從逐序列CSV獨立重算Accuracy／F1／混淆矩陣，獨立從raw scores重建69支次K3最高分、事件有無及嚴格門檻，全部一致。23支逐序列、窗口、事件均保留，沒有把15個或更多事件當15支影片。
- 三Scaler與僅用24 Train ADL／3638窗重新擬合的mean／scale逐值完全一致；AE Train來源hash與正常Val checkpoint來源一致。預訓練Human特徵曾使用Le2i跌倒，只有本AE為Train正常學習，不能聲稱整個流程完全無監督。
- `real_end_to_end`：本輪沒有重新從原影片抽取Pose／ST-GCN，沒有新demo或影片端到端成績。既有端到端證據只適用其原來的方法與案例；不用本輪功能測試替代辨識效果。
- 1039個既有檔案實際hash全部不變：舊run／設定、相關完整快取、manifest、4組真實權重、核心Python及前次Test輸出。三新run各有新的checkpoint、完整設定、Loss曲線、Scaler、門檻與來源hash。核心程式、品質／時間／事件規則沒有修改。
- 已實際確認完成後 `diagnose`／`lock`／`training` 拒絕覆寫；`status` 為只讀hash重驗。Ruff通過。比較圖已目視確認。驗收腳本初版對無track排程端點的空track欄位轉整數失敗，修正為確認其無score再略過；之後從全部CSV逐值重驗通過，不變更模型或判斷。
- CUDA訓練沿用seed42／deterministic warn_only，實際出現CuBLAS環境設定警告，不保證跨硬體／重訓逐位元相同。凍結模型與實際推論輸出hash已保存；未修改系統Python／套件／權重。

## 成果與操作

三個新run：`runs/ur-isotropic-{compact,wide,denoise}-20261012-001/`；Validation輸出：`outputs/isotropic-refinement-20261012/<run>/metrics.json`、`sequences.csv`、`windows.csv`、`events.csv`；比較圖：`outputs/isotropic-refinement-20261012/validation-comparison.png`。候選、逐案例時間與採用結果：`validation/isotropic-refinement-20261012-validation.json`；獨立驗收：`validation/isotropic-refinement-20261012-verification.json`；實際訓練log亦保留。

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe scripts/human_isotropic_refinement.py status
.\.venv\Scripts\python.exe scripts/verify_isotropic_refinement.py
```

已完成本輪有限候選與驗收，沒有外部執行阻塞。仍未完成：最高新候選Test泛化驗證（本輪依指示不執行）、穩定可見警示、教師指標單位／split可比性確認，以及暫停中的正式Notebook／五分鐘影片。下一輪可先用Train／Validation標出跌倒前、中、落地時段及關節可見性，再針對fall-19可用特徵與fall-18當前骨架失效做姿態／時間上下文pilot；這是尚未執行方向，必須另做新的有限候選契約，不能追加到已完成本輪或靠降低品質門檻假稱改善。
