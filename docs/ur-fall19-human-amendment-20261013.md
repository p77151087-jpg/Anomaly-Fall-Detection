# fall-19 人工補註完成：姿態階段與缺失骨架

2026-10-13，使用者已回答前輪兩項澄清：時間欄位的33–61及62–100是「整段姿態階段」；第1、2幀改為「沒有骨架」。本次已實際另存補註、產生修正版標註及重算診斷，不需重新匯出或重做100幀紀錄。

新階段為 `ur-fall19-human-amendment-20261013-001`。原匯出、前輪待澄清報告、全部凍結模型／分數／快取與歷史hash保留；前輪文件中的「待釐清」是當時狀態，由本補註接續解決。這是Validation診斷資料整理，沒有模型擬合、門檻校準、Pose重抽或Test評估。

## 實際保存與時間語義

`human-amendment.json`保存本次人類回答及兩個明確修正；`derived-annotation.json`為獨立schema v2診斷標註。原schema v1及其嚴格匯入驗證器不修改；先驗證原檔來源與完整性，再由新補註驗證器檢查來源SHA、本人回答、完整連續姿態區段、僅f1／2的授權修正與兩幀快取確實無Pose。v2不能冒充原v1匯出，不回寫舊覆核頁或原匯出。

| 項目 | 補註後的記錄 |
| --- | --- |
| upright姿態 | f1–32，0–1034ms |
| transition姿態 | f33–61，1067–2002ms，整段姿態區間 |
| ground姿態 | f62–100，2035–3303ms，整段姿態區間 |
| 精確onset／impact時點或不確定邊界 | 未提供，兩欄為null；不由區段起點推算 |
| matches_person | f3–43，共41原幀 |
| missing | f1／2／54／55／59，共5原幀 |
| wrong_or_drift | f44–53／56–58／60–100，共54原幀 |

其他逐幀標籤、來源frame ID／時間／RGB SHA、備註及本人自述全部保留。修正不產生逐關節座標真值，也不獨立驗證標註者身份。精確警示延遲及正式時間Accuracy仍不可計算；不能把phase邊界當已確認的跌倒開始或落地瞬間。

| 保存檔案 | SHA256 |
| --- | --- |
| 原匯出original-export.json | `f7dbda11fcf42dbcf4f2a203ea0d0c22b70d7fa0b0c7915051be8cc540b88f3c` |
| 新human-amendment.json | `e7ca46828d42a5b6b6142df0a60349e360152610aab89877ba34f5f6f1564ff2` |
| 新derived-annotation.json | `412e95ec2eac7b2c573297f67c48da1c6d13523606394e4be8cb90a60118b0df` |

## 重算結果與已支持的診斷

100幀CSV僅2個人工pose_check儲存格改變。223窗口CSV與前輪逐bytes相同，三方法的全部phase摘要、Raw／K3分數、原門檻、品質及事件狀態均未改變。最後來源幀按 `source timestamp <= available_at_ms` 對齊，禁止使用未來RGB；其人工標籤只代表端點最後來源觀測，不代表完整20／30幀窗口的骨架都正確。

三路transition階段仍21/29合法＝72.41%，其中11個合法端點最後來源幀為matches_person、10個為wrong_or_drift。ground階段仍0/38合法，沒有可判定分數，最高值為null；38是窗口端點數，與39個原ground影格分母分開。

| 凍結方法 | transition最高Raw／K3 | matches_person端點最高Raw／K3 | 原門檻 |
| --- | ---: | ---: | ---: |
| wide30 isotropic AE | .183907／.134636 | .126250／.069902 | .534111 |
| 20幀 isotropic AE | .323840／.235246 | .214085／.102102 | .423891 |
| wide＋motion近鄰 | 1.680733／1.230448 | 1.153804／.638835 | 4.881265 |

三路Raw／K3跨門檻均0，fall-19仍沒有警示。這支持「前段仍有合法窗口但異常分數低」，不能把漏報全部歸於沒有骨架；在這些凍結分數下，也不是K3擋住已跨門檻的Raw。後段合法觀測缺失是另一個已觀察限制。

第44–51幀共8幀仍是「人工自述錯點、原confidence規則當前good」。這支持品質信心不等於人物點位吻合，後續品質對照有診斷依據；**沒有修正Pose的控制實驗，尚未證實修正骨架會改善Accuracy，也未證實唯一漏報原因**。原AI粗phase9合法與本人姿態區段21合法是不同範圍，不算新增辨識改善。

## 辨識、採用與限制

本次沒有新增全Validation評估或訓練，以下為保留的歷史正式結果，不能稱為補註後的新模型成績。

| 版本 | 整體Accuracy（分母23） | Fall F1 | 定位 |
| --- | ---: | ---: | --- |
| ur-isotropic-wide-20261012-001 | 22/23＝95.65% | .9655 | 原最高Validation比較基準 |
| ur-motion-novelty-20261012-001 | 22/23＝95.65% | .9655 | 與wide持平、未採用 |
| ur-short20-isotropic-20261012-001 | 20/23＝86.96% | .8889 | 20幀isotropic，未採用 |
| ur-sample-human-s1-001 | 20/23＝86.96% | .8889 | 目前demo預設，保留 |

wide／motion歷史混淆矩陣TP14、TN8、FP0、FN1(fall-19)、影片unknown0；可判定二元與含unknown整體Accuracy各為22/23，Precision1、Recall14/15。影片unknown0不表示ground階段可判定；精確時間GT未知，影片判對與警示時間正確分開。六SVM／兩AE比較已停止，結果見[方法比較報告](ur-normal-methods-20261012.md)，不因本補註擴大網格。

沒有Accuracy或警示時間改善、沒有新模型採用，demo不換。新方法Test維持未準備／未知，沒有讀Test特徵或新增Test評估，歷史已查看Test不改稱新盲測。外部資料封存、正式Notebook／五分鐘影片仍暫緩，不公開上傳、不換算課程成績。

## 驗證與可直接執行命令

- **synthetic**：58相關cases通過（13新補註風險＋9原診斷＋36原標註驗證）；來源／自述／Test與AI隔離、非phase邊界、未覆核、未授權第三幀修改、現有Pose被誤稱missing、精確時點不得自動補值、輸入不變與分數不變有測試。相關source lint／format通過，未宣稱全套回歸。
- **real_feature_smoke**：另寫標準庫獨立核對，100原RGB SHA、100來源及CSV、僅2人工標籤差異、223最大過去timestamp對照及凍結分數／門檻／狀態逐值通過；phase排程／合法數／最高值／跨門檻數重算一致。223窗口CSV bytes不變，下載原檔與保存原檔SHA相同。264個承接不可變成果hash不變、兩份修改前進度／驗收原bytes另存。實際重複`--save`被拒絕且原成果未改。
- **real_end_to_end**：未新增。沒有新RGB→Pose→ST-GCN推論、AE訓練或模型部署；只使用原真實凍結觀測及分數作診斷。

Graph以Tier2查詢、exact snippets／trace、相關coverage驗證並另存。graph是best effort，無recorded issue不是完整性證明；outputs／raw／cache／run刻意excluded，精確輸出、RGB及來源hash改用直接檔案核對。初次帶file_pattern查詢沒有命中，後續精確name查詢取得3個新符號；沒有把0命中宣稱函式不存在。

新script的outbound trace將同名diagnose及部分exists／update誤配到歷史source或其他物件（heuristic邊）；不採用這些關係，直接核對新script明確import及實際呼叫的annotation_diagnosis.diagnose。coverage沒有recorded gap不能排除這種同名解析限制。

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe -X utf8 scripts/amend_fall19_annotation.py
.\.venv\Scripts\python.exe -X utf8 docs/validation/ur-fall19-human-amendment-20261013-001/independent_check.py
```

兩命令只讀重算並核對已保存成果；`--save`已執行，重複使用會拒絕覆寫。成果在 `outputs/ur-fall19-human-amendment-20261013-001`：human-amendment／derived-annotation／diagnosis JSON及100幀／223窗口CSV；驗證、lock、原契約副本、synthetic XML及graph證據在同名 `docs/validation` 目錄。**這次兩項補註都已完成，現在無須再人工匯出或重做100筆。**
