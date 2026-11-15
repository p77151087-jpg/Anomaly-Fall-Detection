# UR-only 與 Train 資料增強（2026-10-12）

使用者撤回外部正常訓練資料，改以UR-FALL及Train原影片的增強版本繼續改善。本次外部raw／cache已退出原使用路徑，移至`data/retired/external-normal-20261012`；201個移動檔案、1,430個原實驗受檢檔案hash保全。永久刪除0，舊混合模型／設定／結果不覆寫、不列主課程成果。原外部驗證工具依歷史路徑運作，不再適用；改用`scripts/retire_external_normal.py status`核對封存映射。

外部來源不參與後續Scaler／AE／checkpoint／門檻／評估。原UR70支Train24ADL／Val8ADL15Fall／Test8ADL15Fall及歷史Test已查看的限制保留。資料增強僅原Train24正常序列，不將Val／Test變形後搬進Train；沒有新增原始影片、人物或獨立場景。預訓練Pose／ST-GCN的既有來源並非UR-only從零訓練，本次UR-only指專案AE訓練資料來源。

## 可行性與有限實驗

空間整體平移在現有逐幀置中／等比例縮放下會被抵消；固定5%影像寬高平移僅用於診斷，不能以浮點微差冒稱新訓練資訊。30幀窗口已stride1逐格取樣，時間shift主要是既有相鄰合法窗口，沒有新增觀測。鏡像已在上輪試過，Val21/23，沒有提高最高Accuracy。

本輪另立單一`ur-joint-shift-20261012-001`，事前固定256/64容量與原seed42／max100／patience10／sequence sampling／wd.001／noise0。只對每個原Train正常30幀窗口的可靠、全窗在影像內關節作單一固定XY偏移，每軸最多可見身體跨度1%；同一窗口各影格使用相同關節偏移，低confidence／出框點不動、confidence／時間／品質不變。由真實凍結ST-GCN重抽特徵，剔除與原特徵無可測變化的增強列，原始列保留。

事前gate：48原始特徵首尾parity、全部record品質／時間相同、至少90%原Train列產生可測變化。Scaler／AE僅Train原＋增強矩陣；原正常Val8支選checkpoint，原Val23支校準門檻；Val／Test不增強、不重切。仍以原最高22/23／F1 .9655及舊警示guard作參照，只完成這一候選，不追加容量／shift幅度／seed／品質／門檻掃描；不跑Test、不依Test調參。

## 階段進度

- 封存完成：外部原使用路徑不存在，201檔移動後hash不變，1,430原檔保全；[封存證據](validation/external-normal-retired-20261012.json)。
- synthetic風險測試完成：36 passed，0 skipped；[XML](validation/pytest-ur-train-shift-20261012.xml)。
- 真實Train增強抽取完成：24原Train／3,638窗全部產生可測改變，另存3,638增強列；原＋增強7,276列，沒有新增原影片。48筆原特徵parity差0，全部record／品質／時間相同，嚴格真實ST-GCN權重及forward有執行。
- 單一AE完成29epochs，Validation TP13／TN8／FP0／FN2／影片unknown0，二元及含unknown全體Accuracy皆21/23＝91.30%，Fall F1 26/28＝.9286，全體Fall召回13/15＝86.67%。漏報fall-19／18，低於原最高22/23／.9655，未採用，原預設保留。
- 獨立驗收完成：只用24UR Train原＋增強7,276列的Scaler逐值吻合；AE訓練來源hash／count、原8正常Val checkpoint、原23Val CSV混淆矩陣／K3／事件重算一致；1,435舊檔保全，8次覆寫／重啟外部資料／追加訓練操作拒絕，1,542受檢檔前後hash不變，無新Test。

官方對資料增強的額外限制仍未明定，將參數／來源／切分與結果如實揭露。這是正常資料AE，不用跌倒資料重訓／微調分類頭，也不把AE分數當跌倒機率。取樣時段／可見性v2保留，不當完整人工GT或精確警示延遲驗收；正式Notebook／影片仍暫停。

## 實測結果及限制

| UR Validation模型 | Train原影片 | 訓練列 | TP／TN／FP／FN／U | Accuracy | Fall F1 |
|---|---:|---:|---|---:|---:|
| 原最高ur-isotropic-wide-20261012-001 | 24 | 3,638 | 14／8／0／1／0 | 22/23＝95.65% | .9655 |
| 本輪ur-joint-shift-20261012-001 | 24 | 7,276 | 13／8／0／2／0 | 21/23＝91.30% | .9286 |

新AE256/64門檻 .5084131956100464，正常Val選定checkpoint Loss .1164740054（本模型Scaler單位，不能直接與不同Scaler比較）；沒有Test調整門檻。新2×2矩陣列ADL／Fall、欄normal／suspected_fall為`[[8,0],[2,13]]`，含unknown的2×3為`[[8,0,0],[2,13,0]]`。可判定與全體Accuracy分母各23；影片coverage1.0，但原窗口coverage .843911439，不能說每幀都可判定。

對48個原Train首尾窗口的整體5%寬高平移，正規化tensor最大差 .00000190735，真實ST-GCN特徵最大差 .000450969；數學上位移被置中消除，實際浮點運算留微差，未作為新訓練資料。1%各關節位移最大特徵差 .3315766，屬可測變化，但本輪沒有改善Accuracy。此處測的是**已觀測骨架座標**平移，不能冒稱已評估原RGB影像位移後重新Pose推論的影響。

舊6段粗時段命中由參照4降至3、早報0，沒有救回原最高參照的fall-19；採用guard未通過。仍缺正式時段GT及穩定可見警示驗收，影片級判對不等於警示時間正確。保留原預設ur-sample-human-s1-001，既有Val20/23＝86.96%／.8889與最高候選95.65%不混稱為已部署成績。

本輪finite one-candidate已結束，不追加幅度／容量／seed或Test。資料增強在方法上可行，但不是所有UR影片都能加入Train：即使來源同為UR，原Val／Test的變形副本仍帶有原影片資訊，不可挪用；參照[scikit-learn資料洩漏說明](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)。窗口加倍不是獨立人物／場景加倍，也未解決出框關節或提高泛化的證據。教師是否另有增強限制尚未明定，不自行換算課程分數。

## 驗收與操作

synthetic證據為36相關風險測試；real_feature_smoke為原UR觀測骨架經真實ST-GCN的明示座標增強、29epoch AE訓練／校準、原23Val獨立核對。沒有新原RGB Pose抽取、demo或real_end_to_end／新Test成績。Graph index／search／coverage仍Transport closed，已先嘗試，再以相關source補核對，未取得新generation／完整coverage。

首次獨立驗收因讀取不存在的頂層count欄位失敗，檢查器改讀核心實際記錄的`training_sampling.train_normal_feature_counts`後全數通過；保存原檢查器source及失敗log，新驗證log另存。未更改資料／模型／Scaler／門檻／選模／成績，未重訓。[檢查器修復](validation/ur-train-shift-20261012-verifier-repair.json)。

[訓練前鎖定](validation/ur-train-shift-20261012-lock.json)｜[原／增強矩陣與hash](validation/ur-train-shift-20261012-inputs.json)｜[正式Validation結果](validation/ur-train-shift-20261012-validation.json)｜[獨立驗收](validation/ur-train-shift-20261012-verification.json)｜[最終QA](validation/ur-train-shift-20261012-final-qa.json)。模型另存`runs/ur-joint-shift-20261012-001`，CSV在`outputs/ur-train-shift-20261012/ur-joint-shift-20261012-001`；原raw／cache／來源hash及舊run保留。封存與新cache均已被Git忽略。

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe scripts/retire_external_normal.py status
.\.venv\Scripts\python.exe scripts/ur_train_shift.py status
.\.venv\Scripts\python.exe scripts/verify_ur_train_shift.py
```

上述皆不執行Test。完成後archive／freeze／derive／training／verify --write均拒絕覆寫；舊外部acquire／prepare／training亦拒絕重啟。本輪沒有外部阻塞，graph限制以源碼補核對；完整人工GT／教師協定仍待確認，Notebook／正式影片繼續暫停。
