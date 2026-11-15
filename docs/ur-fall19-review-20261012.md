# fall-19 完整影格診斷與人工覆核工具

2026-10-12，接續已結案的正常資料方法比較與全域動態近鄰。已實際生成原 Validation fall-19 全部 **100 幀** RGB／既有 Pose／窗口／分數對照並目視；沒有新增候選、訓練、校準或 Test 評估。這是可執行的漏報診斷工作，**不是辨識改善**。

## 實際完成

- 新增 `scripts/fall19_review.py` 與本機 HTML、`evaluation/frame_review.py` 因果對齊及人工自述標註驗證、36 個新 synthetic cases、獨立 `scripts/verify_fall19_review.py`。
- 新包 `outputs/ur-fall19-review-20261012-001`：100 張原 PNG bytes 副本、100 張既有 Pose 診斷疊圖、全片／密集區間接觸表、逐幀 CSV、三路窗口 CSV、source manifest／hash、未覆核標註模板與互動頁面。原模型／資料／split／快取／門檻不修改，沒有把疊圖變成新合法窗口。
- 新 lock、來源 hash、舊成果保全清單與五份修改前契約副本在 `docs/validation/ur-fall19-review-20261012-001`。完成後 `build` 拒絕重寫；人工匯入保留原 JSON bytes，另存驗證結果，同一內容拒絕重寫。
- 只使用原 fall-19 Validation RGB、既有 Pose、30／20 幀真實快取與正式 Validation CSV。兩 AE 嚴格重載，56 個原 AE 窗口分數逐值一致；另核對 23 個既有融合分數。沒有新 Pose／ST-GCN forward、擬合、補缺失或讀 Test 特徵。

## 逐幀與窗口發現

100 幀使用同步 CSV 的原時間 0–3303ms，沒有估算 timestamp。當前 Pose：49 幀達原信心品質、46 幀低品質、5 幀沒有 Pose；95 筆既有 Pose 均 track 1。這個品質只檢查信心／有限數值，不證明點位符合人物。

| 凍結方法 | fall-19 合法／排程窗 | 覆蓋率 | 最後合法端點 | 原粗區間最高 Raw／K3 | 自身門檻 |
| --- | ---: | ---: | ---: | ---: | ---: |
| wide 30 幀 isotropic AE | 23/71 | 32.39% | 1766.67ms | .183907／.134636 | .534111 |
| 20 幀 isotropic AE | 33/81 | 40.74% | 1766.67ms | .293740／.235246 | .423891 |
| wide＋動態正常近鄰 | 23/71 | 32.39% | 1766.67ms | 1.680733／1.230448 | 4.881265 |

上表粗區間沿用舊 AI 1468–2369ms，不是人工真值。三路區間內都有 **9 合法窗口及合法 K3**，全部 Raw 亦未跨自身門檻，事件皆 0。20 幀新增的 10 個窗口全部在較早端點 700–1000ms，沒有延長最後合法時間；不能把總合法窗增加當救回跌倒或準確率提高。不同方法分數單位不同，不直接比較數值大小，均非跌倒機率。

第 54 原影格是 1768ms、當前沒有 Pose；因果網格最近端點是 1766.67ms，其窗口仍合法，30／20 幀 good-frame ratio 分別 .9333／.9。它使用過去觀測，沒有偷用未來；**當前骨架和窗口品質不能混為一談**。第 55 幀 1801ms 最近端點 1800ms 已為 track_gap／unknown；之後再無合法窗，30 幀合計 37 track_gap／11 low_confidence，20 幀合計 27／21。後段有些 Pose 仍存在但可靠點不足，不能說後段每幀都沒骨架。

完整原圖接觸表 5 頁全數目視，另看 1400–2600ms 密集 Pose 表及 f41／47／54／58 原解析度影格。AI 觀察：f37–44 明顯前傾和下沉；f45–51 向前下方移動、模糊增加；f52–57 頭／肢體接近或超出下緣；f58–100 可見背部及一手維持低位，多數身體出框。完整落地接觸不清楚，精確 onset／impact 未判定。記錄另存 `ai-visual-diagnostic.json`，`human_confirmed=false`，不填入人工模板、不改寫舊粗時段。

已由輸出支持：下沉前段有合法窗但低分，後段品質失效；本輪縮短至20幀沒有延長跌倒附近可用窗口，原 K3 並非單獨攔住一個已跨門檻的 fall-19 分數。AI 畫面支持「後段缺失與出框同時出現」的觀察，尚未有人工時間／關節真值或改 Pose 控制實驗，**不能量化出框的因果影響，不能認定它是唯一漏報原因**。正常動作稀釋、表示不足、部分高信心錯點及正常／跌倒観測相似仍是假設。

UR 官方說明同步資料包含 frame、時間、加速度；另有深度特徵姿態欄，不能將現有同步 CSV 當 RGB onset 標註。本輪未下載或解碼包含其他 split 的深度特徵檔。[UR Fall Detection Dataset 官方資料說明](https://fenix.ur.edu.pl/~mkepski/ds/uf.html)

## 歷史成績與採用

本輪只獨立核對三份既有 Validation CSV，沒有重新校準。全體及可判定 Accuracy 的分母均 23，影片 unknown 均 0/23；逐窗口 unknown 另列品質，不能因影片判定成功而忽略。

| 版本 | 全體 Accuracy | Fall F1 | TP/TN/FP/FN/U | 本輪角色 |
| --- | --- | ---: | --- | --- |
| demo：ur-sample-human-s1-001 | 20/23＝86.96% | .8889 | 12/8/0/3/0 | 既有展示預設，未部署新模型 |
| wide：ur-isotropic-wide-20261012-001 | 22/23＝95.65% | .9655 | 14/8/0/1/0 | 最高 Validation 研究參照 |
| ur-short20-isotropic-20261012-001 | 20/23＝86.96% | .8889 | 12/8/0/3/0 | 已訓練39epochs的20iso；不是舊20 per_frame |
| 最佳新 SVM06／GRU AE | 21/23＝91.30% | .9333 | 14/7/1/1/0 | 前輪完成並停止，未採用 |
| ur-motion-novelty-20261012-001 | 22/23＝95.65% | .9655 | 14/8/0/1/0 | 前轮持平、未採用 |

wide／motion Precision14/14＝1，Fall Recall14/15＝.9333，F1=28/29；二元矩陣 `[[8,0],[1,14]]`，含unknown `[[8,0,0],[1,14,0]]`。20iso Precision12/12＝1，Recall12/15＝.8，F1=24/27；矩陣 `[[8,0],[3,12]]`，含unknown `[[8,0,0],[3,12,0]]`。本輪三版本的 fall-19 都 FN、無警示，其他影片沒有改變；沒有 Accuracy、F1 或警示時間改善、没有新方法可採用，預設保持原版本。

六 SVM／兩 AE 的全部成功與失敗候選、獨立七類對照及自監督停止依據見[方法比較報告](ur-normal-methods-20261012.md)；全域動態候選見[動態近鄰報告](ur-motion-novelty-20261012.md)。本轮没有延長已停止網格，七類分數不替代異常模型結果。

## 覆核、證據與限制

本機頁面支援前後幀、影格滑桿／ID、逐幀或已查看區間的姿態／可見性／骨架一致性三項標示，以及 onset／impact 不確定影格邊界。模板 100 幀全部 unreviewed，邊界 null。輸出人工自述標註須完整覆核、填 reviewer、明確本人確認；未達条件實際操作會被拒絕。程式不能獨立驗證身份或標註正確性，匯入只能作 Validation 診斷，從不自動成為 Train 或門檻更新。目前 **人工確認 0/100，無人工匯出或匯入**，正式時間 Accuracy 未產生。

synthetic：66 個相關 cases 全過，包含36個新對齊／缺失／錯來源／不確認／AI來源／邊界／排序／完整標註檢查，以及原時間窗／timestamp／事件 cases。新 source lint／format 通過。只跑與本輪風險相關測試，沒有宣稱重新執行全套。

real_feature_smoke：100 RGB 完整解碼及原 SHA／時間核對、95 Pose／track／信心獨立對照，300 因果模型影格對齊、223 窗口 K3／方向／缺失狀態獨立重建，三份原23片 CSV 的 Accuracy／F1／分母／矩陣及 fall-19 無事件核對，全部通過。舊成果24966項 hash全部相同，五份原契約副本吻合，詳見 `verification.json`；新產物 hash、實際 UI檢查、預設 build／未確認 import 拒絕各自另存。Test 舊檔只作 bytes SHA 保全，沒有特徵解碼或成績選模。

Graph Tier2 查詢／trace／snippets及coverage保存，best effort、excluded raw/cache/run直接讀精確檔補驗，未宣稱graph完整。初次瀏覽器啟動 timeout 與沙箱 loopback Win10013／ERR_CONNECTION_TIMED_OUT 保留；允許本機連線的只讀127.0.0.1服務重測後頁面正常，f54合法／當前缺Pose、f55 unknown與未確認匯出拒絕均實際核對。

real_end_to_end 未跑，没有新RGB Pose／ST-GCN forward／demo或Test。新方法 Test未知，歷史 Test 已查看不作新盲測；小 Validation 反覆研究、人物隔離未證、預訓練來源限制保留。正式 Notebook／五分鐘影片仍暫緩、舊成品保留、不公開上傳、不自行換算課程成績。

## 可直接執行

頁面可直接用瀏覽器開啟 `outputs/ur-fall19-review-20261012-001/index.html`，資料嵌於本機 JS，不需安裝或網路。此次已開啟的本機頁面為 `http://127.0.0.1:8800/index.html`，伺服器停止後用下列命令另開8799：

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe -X utf8 scripts/fall19_review.py serve --port 8799
# 瀏覽 http://127.0.0.1:8799/index.html；Ctrl+C停止唯讀服務
```

只讀查核與相關測試：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/fall19_review.py status
.\.venv\Scripts\python.exe -X utf8 scripts/verify_fall19_review.py
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_frame_review.py tests/test_skeleton_timing.py tests/test_dense_schedule.py tests/test_events_metrics.py tests/test_timestamp_tolerance.py --basetemp=.artifacts/fall19-review-local-check -q
```

日後由本人覆核並將頁面匯出 JSON 存至專案內後，例：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/fall19_review.py import --annotation outputs/ur-fall19-review-20261012-001/fall-19-human-self-reported.json
```

上例檔案目前不存在，未確認草稿／AI／未覆核模板會被拒絕。標註只寫入獨立診斷目錄；不訓練、不校準、不重跑Test。下一研究的依據是先補人工 onset／impact及姿態吻合覆核，再決定是否值得立一個有對照的品質／觀測候選；此處未把待辦當已完成。
