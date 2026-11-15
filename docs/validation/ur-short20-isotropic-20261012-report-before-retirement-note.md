# UR-only：20 幀＋保留身體比例實驗（2026-10-12）

**實際完成單一候選訓練與完整 Validation；沒有提升，不採用。** 新候選 `ur-short20-isotropic-20261012-001` 共39 epochs，最佳Epoch29，Validation 20/23＝86.96%、Fall F1 24/27＝.8889，低於30幀等比例最高研究候選22/23＝95.65%／28/29＝.9655，Accuracy下降8.70百分點。原demo預設 `ur-sample-human-s1-001` 保留；其既有86.96%與最高候選95.65%不混稱為已部署成績。本輪沒有恢復／下載外部資料、沒有Test、沒有額外候選。

## 方法、鎖定與實作

[訓練前計畫](ur-short20-isotropic-20261012-plan.md)及[不可覆寫lock](validation/ur-short20-isotropic-20261012-lock.json)固定20幀、stride1、30FPS、short_20、per_frame_isotropic；原AlphaPose／Human TSSTG真實權重、13→14點、分類頭前雙流512D方法不變。AE256/64、seed42、lr.001、wd.001、noise0、batch32、sequence sampling、max100、patience10、min_delta1e-6均沿用wide。只有原24支UR Train ADL，無shift／鏡像／外部資料。品質.2／9真實點／.8、因果重採樣與缺失、strict >／K3不變。舊20幀per_frame結果不是本輪per_frame_isotropic結果。

核心 `prepare/train --train-validation-only` 新增明確opt-in。新index保留原70支split及hash，47支Train／Validation全部prepared，Test23支均not_prepared且無pose／feature路徑；不偽造prepared、不讀Test特徵、不混30幀Test快取。訓練檢查全部Train／Validation含Fall均準備完成，缺片／smoke／舊Test路徑都拒絕。預設完整訓練仍需完整prepared；scoped run禁止Test evaluate。核心API／CLI及風險測試一起更新。

固定10支pilot通過全部10檢查。真實ST-GCN strict load及20幀完整forward／重接head parity通過；特徵512D有限、時間戳因果、品質及完整端點排程通過，30等比例／20per_frame／20等比例recipe hash三者不同。正常平均coverage比值1.002633、六粗時段合法K3總數比值.969072，均過事前gate。固定樣本及逐片證據見[pilot](validation/ur-short20-isotropic-20261012-pilot.json)；gate不測Accuracy。

完整47支features全部由相同觀測Pose及真實凍結ST-GCN以20幀重新計算，沒有沿用30幀特徵。Pose array hash逐片與來源相同、原來源身份及權重hash重驗，Pose本輪沒有新RGB forward；ST-GCN實際CUDA forward有執行。正式Train24ADL 3,888窗、正常Validation8支1,515窗、Validation23支2,623窗，合計6,511窗。原split未重切，Train僅ADL。

正式AE CUDA訓練39回合，正常Val最佳Epoch29／MSE .11289029724330399，依該checkpoint校準原23支Validation，balanced_sequence_accuracy目標、FP平手及较高門檻规则沿用wide。凍結門檻 **0.4238913059234619**；沒有以Test挑epoch／門檻。兩個模型Scaler不同，較低正常Loss不當成辨識改善。

## 正式Validation比較與分母

單位為原始影片sequence。兩者n_total=23、n_evaluable=23、unknown=0，所以可判定二元Accuracy與包含unknown的整體Accuracy各自分母均23。Fall為positive；Precision分母為预测Fall，Recall分母為可判定真Fall；全體Fall召回另以全部15支Fall計。

| 指標 | 30幀等比例wide基準 | 20幀等比例候選 |
|---|---:|---:|
| 可判定二元Accuracy | 22/23＝95.65% | 20/23＝86.96% |
| 包含unknown整體Accuracy | 22/23＝95.65% | 20/23＝86.96% |
| Fall F1（可判定） | 28/(28+0+1)＝.9655 | 24/(24+0+3)＝.8889 |
| Precision（可判定） | 14/14＝100% | 12/12＝100% |
| Recall（可判定） | 14/15＝93.33% | 12/15＝80% |
| 全體Fall召回 | 14/15＝93.33% | 12/15＝80% |
| TP／TN／FP／FN／U | 14／8／0／1／0 | 12／8／0／3／0 |
| ADL誤報（全部ADL分母） | 0/8 | 0/8 |
| 漏報（全部Fall分母） | 1/15：fall-19 | 3/15：fall-19／18／30 |
| unknown（全部影片分母） | 0/23 | 0/23 |
| 影片coverage | 23/23＝100% | 23/23＝100% |
| 窗口coverage | 2287/2710＝84.3911% | 2521/2940＝85.7483% |

2×2（列ADL／Fall、欄normal／suspected_fall）：基準 `[[8,0],[1,14]]`，候選 `[[8,0],[3,12]]`。2×3加unknown欄：基準 `[[8,0,0],[1,14,0]]`，候選 `[[8,0,0],[3,12,0]]`。窗口分母是各自窗長後全部預定端點；短窗暖機少10格，因此窗口總數不同，增加有效窗不等同每幀可判定或增加獨立影片。

## fall-19：品質、时间窗與分數

原粗略跌倒phase1468–2369ms。30幀實際span966.67ms、20幀633.33ms（30FPS下T-1個間隔）。相同Pose hash，整片有效端點23/71→33/81，track_gap37→27、low_confidence11→21；品質規則未弱化。新增10個合法端點為sample index [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]，時間700.00–1000.00ms，全部在粗略正常前段；共享23個、移除0個。精確差異見[端點診斷](../outputs/ur-short20-isotropic-20261012/fall19-window-difference.json)。

**可能跌倒phase內仍是27個預定端點、9個有效端點、9個合法K3，沒有增加。** 基準phase最高raw .18390694、K3 0.134636492，門檻0.534111381；候選raw 0.293740064、K3 0.235246241，門檻0.423891306，仍未跨門檻，無事件／無時段內良好骨架警示。分數屬各自Scaler及AE，不以raw数值較大當成改善。

因此不能將漏報直接歸因於没有骨架。短窗增加了前段可用窗口，但沒有增加此粗phase的合法K3，也没有救回fall-19；「減少正常歷史稀釋／裁切無效窗可改善辨識」在這個鎖定候選沒有得到Accuracy改善的支持。並未量化所有漏報原因的因果贡献；AI粗时段、confidence與可見性診斷不是人工真值。

## 警示時間與影片判對分開

完整[逐影片比較CSV](../outputs/ur-short20-isotropic-20261012/sequence-comparison.csv)包含23支預測／coverage／首次警示。以下秒數是影片來源時間的實際K3確認時間，**不是人工onset延遲**。12支共同TP中10支首次事件較早、2支相同；fall-18／30則由有事件退步為沒有事件。沒有以刪減警示或剪輯挑成功片段作成績。

| Validation Fall | 基準首次事件（秒） | 候選首次事件（秒） |
|---|---:|---:|
| fall-22 | 1.1000 | 1.0667 |
| fall-19 | 無 | 無 |
| fall-16 | 1.4333 | 1.3667 |
| fall-18 | 1.4333 | 無 |
| fall-08 | 2.5333 | 2.5000 |
| fall-13 | 2.3000 | 2.1333 |
| fall-29 | 2.5667 | 2.5667 |
| fall-14 | 1.5333 | 1.4667 |
| fall-09 | 5.9000 | 5.6333 |
| fall-07 | 3.7667 | 3.5333 |
| fall-11 | 3.1000 | 3.0667 |
| fall-25 | 2.4667 | 2.2000 |
| fall-06 | 1.5667 | 1.4667 |
| fall-02 | 2.5333 | 2.5333 |
| fall-30 | 1.8667 | 無 |

六個舊粗時段事件起點命中由4降至2，仍0早報；失去fall-18／30命中。fall-09事件5.9000→5.6333秒，仍超出舊phase4.604–5.506秒，影片TP不能稱警示时间合格。候選只fall-29／25有粗phase事件；當前良好骨架警示端點分別3／2。所有六片[逐時段結果](validation/ur-short20-isotropic-20261012-validation.json)保留。新的14個事件不等同14支TP，12支影片計TP。

事前採用guard未通過：Accuracy/F1下降、兩支基準判對片退步、粗時段命中减少；無FP／unknown增加及coverage增加不能抵銷。**不採用、原預設保留，單一候選實驗結束，不追加調參。**

## 驗證、保全及限制

- synthetic：33相關風險測試全過；全套首次271 passed／1历史預設Human權重路徑skip／2本機HTTP沙箱權限失敗，兩項以允許loopback環境重測2 passed。總274項中273通過、1 skip，保留原失敗XML與log，不偽稱首次全綠。新使用的真實sample Human／20幀ST-GCN strict與forward實際通過，skip不代表本輪權重缺失。[風險XML](validation/pytest-ur-short20-isotropic-risk.xml)／[全套XML](validation/pytest-ur-short20-isotropic-full.xml)／[loopback重測](validation/pytest-ur-short20-isotropic-loopback.xml)。
- real_feature_smoke：固定10片genuine20／512D gate、完整47片真實ST-GCN重新抽取、39epoch AE、23Val凍結評估；獨立sklearn重算Accuracy／Precision／Recall／F1／兩矩陣與分母、每窗K3／状态及14完整事件開始／結束／峰值／原因全部相同。Train-only Scaler逐值相同，Train24與正常Val8 hash、原23Val門檻來源及最佳正常Val checkpoint核對通過。[獨立驗收](validation/ur-short20-isotropic-20261012-verification.json)。
- real_end_to_end：本輪未跑。没有新RGB Pose擷取／demo／Test；已有Pose沿用hash保全，不能把新ST-GCN抽取与功能測試称完整端到端性能驗收。Test全部23支not_prepared，沒有新Test特徵／評分；歷史Test已查看，不稱全新盲測。
- 1031個舊run／設定／TrainVal相關完整cache／split／原权重／既有結果檔hash不變。原raw身份於prepare重新hash驗證，Pose原array相同；舊30幀cache／外部封存資料未覆寫。拒絕4個完成後stage mutation；新run有自己的recipe **92719437ce9edcfd09c6695ac156c1ffba8cfea0f6ba6bdfaba9caa21dafb842**、config、scaler、checkpoint、threshold、Loss及來源hash。
- 初版流程於訓練前修正CLI prepare派送；首次獨立驗收讀CSV漏處理BOM，修正為utf-8-sig後全通過。原source／lock／失敗log及新hash鏈分開保留，沒有重訓、重校準或改選擇／停止條件。[派送修復](validation/ur-short20-isotropic-20261012-dispatch-repair.json)／[驗收器修復](validation/ur-short20-isotropic-20261012-verifier-repair.json)。
- Graph服務恢復，Tier2先查search／trace／snippet及coverage；初代新脚本239–242语法parse_partial直接读取並修正。後續src及實驗／驗收／測試路徑coverage metadata_match／no_recorded_issue；ignored cache與upstream以精确source／hash／實測補核對。coverage為best effort，不是完整正確性證明。
- Validation只有23支且反覆使用、自訂split未保證人物隔離；預訓練Pose／Human ST-GCN不是UR-only從零學習，Human动作權重既有Le2i来源如實保留。完整人工骨架／onset GT、教師評估協定、跨場景泛化／完整CPU實驗仍未確認。CuBLAS deterministic warn_only限制保留，不承諾跨硬體重訓逐位元相同。
- 正式成果Notebook與五分鐘影片继续暫緩，沒有公開上傳、不自行換算課程分數。

![Validation比較與fall-19分數](../outputs/ur-short20-isotropic-20261012/validation-comparison.png)

## 成果與可執行命令

模型：`runs/ur-short20-isotropic-20261012-001`。正式47片cache：`data/cache/ur-short20-isotropic-20261012-001`；pilot另置real_feature_smoke。正式[metrics](../outputs/ur-short20-isotropic-20261012/ur-short20-isotropic-20261012-001/metrics.json)、sequence-results／window-results／events／2×3／混淆矩陣及Loss曲線均保留。鎖定／診斷／pilot／prepared／validation／verification與实际log位於`docs/validation/ur-short20-isotropic-20261012-*`。

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe scripts/ur_short20_isotropic.py status
.\.venv\Scripts\python.exe scripts/verify_ur_short20_isotropic.py
.\.venv\Scripts\python.exe -m fall_detect loss --run runs/ur-short20-isotropic-20261012-001
```

前兩條只讀重驗，不跑Test；Loss命令顯示39回合真實紀錄。流程依序lock／pilot／prepare／training已完成，重跑皆拒絕覆寫／追加候選。若只重驗風險測試：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_train_validation_scope.py tests/test_human_context.py tests/test_human_isotropic.py tests/test_training_cache.py tests/test_events_metrics.py --basetemp .pytest_cache/short20-review -q
```

核心新模式的可用介面為 `python -m fall_detect prepare --config <新設定> --train-validation-only --reuse-poses-index <來源index>`，再 `python -m fall_detect train --config <新設定> --train-validation-only --index <該index> --run-id <新ID>`；本輪已结束，不以此另加候選。
