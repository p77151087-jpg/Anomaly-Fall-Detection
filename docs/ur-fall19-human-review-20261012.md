# fall-19 人工自述標註匯入與凍結分數診斷

2026-10-12，使用者提供 `C:/Users/Huayi/Downloads/fall-19-human-self-reported.json`。已實際匯入100幀自述標註，來源manifest、frame ID／時間／RGB hash、確認欄位與完整標籤全部通過。原檔逐bytes保存，SHA256為 `f7dbda11fcf42dbcf4f2a203ea0d0c22b70d7fa0b0c7915051be8cc540b88f3c`；副本在前輪證據目錄的 `human-annotations/<SHA>/original-export.json`。確認欄位是本人自述，程式不獨立證明身份、點位或時間正確性。

本階段只作Validation診斷，不訓練、不校準、不改Pose／品質／K3／split，也沒有Test特徵讀取或評估。舊AI記錄、初始未覆核模板、歷史lock及報告原樣保留；它們的「0/100人工確認」是當時狀態，本次新增紀錄為100幀自述已收到、時間語義及兩幀標籤待釐清。

## 收到的標註

| 項目 | 原始標註區段 |
| --- | --- |
| 姿態 | upright f1–32；transition f33–61；ground f62–100 |
| 可見性 | full f1–43；partial f44–100 |
| 骨架吻合 | matches_person f1–43；wrong_or_drift f44–53／56–58／60–100；missing f54–55／59 |
| 備註 | 跌倒過程有遮擋、出框 |
| onset欄位 | f33–61＝1067–2002ms |
| impact欄位 | f62–100＝2035–3303ms |

兩個時間欄位恰好等於完整transition／ground區段，可能是整段姿態階段，也可能是邊界的不確定範圍，已向使用者詢問。**不自行把33或62改成精確onset／impact**，不計算正式時間Accuracy或精確延遲。第1、2幀原快取没有骨架卻標matches_person，也已詢問；保留原匯出，不自行修改。這兩幀屬前段暖機，未改下列transition／ground診斷。

## 因果對齊與發現

新增 `evaluation/annotation_diagnosis.py` 及 `scripts/analyze_fall19_annotation.py`，讀取已驗證匯入和前輪凍結review-data。對每個窗口取 `source timestamp <= available_at_ms` 的最後實際觀測幀，與該幀人工標籤對照，避免把較晚RGB的標註搬到較早網格。結果描述的是**最後來源幀的人工吻合判斷**，不是完整20／30幀窗口皆有人工作真值。

依此，三路在使用者transition階段均有21/29合法窗口＝72.41%；其中11個最後來源幀被標matches_person，10個標wrong_or_drift；19個最後來源幀達原信心品質。使用者ground階段三路均0/38合法窗口；无有效分數，最高分數列null而非0，也不當正常。38是因果網格排程端點數，不能混稱39個ground原影格。

| 凍結方法 | transition最高Raw／K3 | 最後来源幀matches_person的最高Raw／K3 | 原門檻 |
| --- | ---: | ---: | ---: |
| wide30 isotropic AE | .183907／.134636 | .126250／.069902 | .534111 |
| 20幀 isotropic AE | .323840／.235246 | .214085／.102102 | .423891 |
| wide＋motion近鄰 | 1.680733／1.230448 | 1.153804／.638835 | 4.881265 |

三路所有transition原分數、K3均無跨門檻。即使最後來源幀被標為骨架吻合，分數也低；不能把整支漏報全歸於沒有骨架或K3擋住了已跨門檻的跌倒分數。這不是新的21個獨立樣本，也不是新模型成績；相鄰窗口高度重疊。

第44–51幀共8幀，被使用者標wrong_or_drift，但原confidence規則仍判當前good。這是人工自述與現有品質判定的實際不一致，支持進一步檢查「信心足夠仍可能錯點」；沒有完整逐關節人工座標，不能換算Pose準確率。後段没有合法窗口，與使用者partial／wrong／missing標籤對應。兩個問題可同時存在：早段有可用觀測但低分，後段觀測品質受限。**尚未做修正Pose的控制實驗，不能證明修正會提升Accuracy，也未證實唯一漏報因果。**

舊AI粗phase的9合法窗口與本次人工姿態區段的21合法窗口是不同區間，分開保留。不能把區段變大造成的窗口增加稱辨識改善。

## 驗證、保全與限制

synthetic：45相關cases全過（9個新因果來源／錯點信心／空分數／track／來源隔離案例＋原36標註驗證案例）。lint／format通過；未重新宣稱全套回歸。

real_feature_smoke：獨立用逐窗口掃描的最大過去timestamp對照production bisect，223個窗口的來源／標籤／分數逐項吻合，各phase排程／合法數／標籤分布／最高Raw與K3／跨門檻數重算通過。下載原檔與匯入副本SHA完全相同；前輪245個非進度契約成果hash不變、兩份進度／驗收修改前副本保存。完成`--save`拒絕覆寫；診斷CSV與JSON另存。Graph Tier2、coverage及相關exact snippets另存，best effort；原raw/cache/run為excluded，資料對照來自已保存的凍結review-data。

real_end_to_end未跑，没有新Pose／ST-GCN推論／訓練／部署／Test。Validation成績原樣：wide與motion22/23＝95.65%／F1 .9655、20iso20/23＝86.96%／F1 .8889；影片unknown0不代表倒地時段可判定。fall19仍FN，沒有Accuracy或警示時間改善，demo仍原預設。新方法Test未知、歷史Test不當新盲測；Notebook／五分鐘影片仍暫緩，不公開上傳。

## 成果與只讀命令

- `outputs/ur-fall19-human-review-20261012-001/diagnosis.json`
- `human-frame-diagnosis.csv`：100原幀標註、實際Pose有無／信心、人工錯點與good衝突。
- `human-window-diagnosis.csv`：223原窗口與最後實際來源幀／標籤／凍結分數。
- `docs/validation/ur-fall19-human-review-20261012-001`：lock／診斷／原契約／獨立核對／synthetic／拒絕覆寫；原匯出與驗證在前輪human-annotations雜湊目錄。

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe -X utf8 scripts/analyze_fall19_annotation.py
```

命令只讀重算並核對已保存的診斷，不重新訓練或跑Test。原匯入已完成，重複import及`--save`會拒絕覆寫。時間欄位意義與f1／2修正須依使用者回答另存補註，無須重做100幀；精確時間仍不足時保留不確定性，不強填GT或盲目啟動新候選。
