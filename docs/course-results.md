# 期末專案本輪成果

2026-09-18。CA-01 至 CA-05 的本機可執行項目已完成。辨識品質仍不足；公開上傳未執行，官方評估細節仍待教師確認。

## 實際完成

1. Train／Validation 診斷：47 支原始序列，分數重疊、誤報／漏報、骨架／追蹤、窗口及 unknown，區分觀察與假設。見 [診斷](urfall-diagnosis.md) 與 [協定](urfall-course-protocol.md)。
2. 新建三個 UR-only run，共 297 epochs；沒有覆蓋20個舊 frozen run。Train僅24支UR ADL／659窗，正常Validation8支／202窗；固定原切分、recipe、品質與事件規則。70支真實快取逐項核對後另建索引，沒有重跑既有資料準備。
3. 執行成果Notebook，共7個程式儲存格、0個cell error，內嵌Loss及兩種混淆矩陣；正式Test／Validation的逐序列、窗口、事件、unknown、分母及hash核對通過。另以乾淨kernel再次執行，保存獨立檢查副本。Notebook執行只讀輸出，不啟動訓練或Test推論。
4. 中文展示／英文debug兩模式。大字狀態、狀態色骨架、人物標示、原影片時間、原始AE MSE、連續三窗最小分數、門檻、新鮮度；unknown時不顯示舊數字。中文使用本機微軟正黑體，不散布字型檔。
5. 完整五分鐘影片：1280×720、30fps、9,000影格、H264，有中文合成旁白／無旁白兩版，均有中文說明與字幕。正常、跌倒、誤報及漏報的Validation片段完整原速播放，結束後明示定格分析。沒有公開上傳。

## 事前選擇與正式成績

候選在訓練前鎖定；正常Validation MSE選checkpoint，完整Validation balanced accuracy選門檻，再依operational accuracy、F1、全體跌倒召回、較少FP、候選順序選模型。原始機讀證據：[候選鎖定](validation/course-experiment-lock.json)、[Test前凍結](validation/course-before-test-freeze.json)。

| UR-only候選 | epochs / best | Validation正確 | Validation F1 | 門檻 |
| --- | --- | --- | --- | --- |
| course-ur-baseline-001 | 100 / 99 | 15/23＝65.22% | 66.67% | 0.485934 |
| **course-ur-balanced-001** | **100 / 94** | **16/23＝69.57%** | **75.86%** | **0.335563** |
| course-ur-compact-001 | 97 / 87 | 14/23＝60.87% | 60.87% | 0.674850 |

僅選定版本做本輪Test；其他候選沒有追加Test選擇。較小latent沒有改善Validation；序列均衡取樣版本較佳，但這不是單因素普遍有效性的證明。

正式UR Test（23支）為TP7／TN4／FP4／FN7／unknown1：

| 指標 | 實際值 | 分母 |
| --- | --- | --- |
| Accuracy，可判定 | 50.00% | 11/22 |
| Precision，可判定 | 63.64% | 7/11 |
| Recall，可判定Fall | 50.00% | 7/14 |
| F1，可判定 | 56.00% | 14/(14+4+7) |
| operational accuracy，全體 | 47.83% | 11/23 |
| 全體Fall recall | 46.67% | 7/15 |
| Sequence coverage | 95.65% | 22/23 |
| unknown | 1支Fall | 保留在全體23支分母 |
| Window coverage | 69.39% | 有效／全部排程端點 |

2×3結果（列是真實、欄是預測）：

| 真實 | Normal | Suspected fall | unknown |
| --- | --- | --- | --- |
| ADL | 4 | 4 | 0 |
| Fall | 7 | 7 | 1 |

unknown為fall-24，僅2個有效特徵窗口，無法組成3個連續有效窗；不改填為正常。這是Test凍結後的描述，不拿來改本輪品質門檻。

既有混合模型在相同UR Test為TP3／TN7／FP1／FN11／U1，可判定Accuracy45.45%、F1 33.33%、全體accuracy43.48%。新版本多抓到4支跌倒，也多3支正常誤報；**全體僅多判對1支**。不能把F1改善包裝成高可靠度。混合模型的Le2i／CAUCA成績另列歷史報告，不代替UR課程成果。

## 驗證與實際限制

- 完整測試：**136 passed、1 skipped**，27.08秒。[JUnit](validation/pytest-course-alignment.xml)。skip僅舊Human缺四個權重，不影響本輪真實YOLO11m／Kinetics路徑。首次沙箱測試44項因pytest暫存目錄權限無法setup，原紀錄另存；授權後完整重跑通過。
- Ruff檢查、格式檢查及compileall通過。新測試檢查unknown分母、數字被竄改／檔案缺失時Notebook拒絕繼續、unknown不保留舊分數、renderer不修改推論輸入。既有切分／Scaler／AE隔離、真實權重、品質／時序／事件等測試一併通過。
- 四支Validation真實新推論，共628影格／每種模式；debug與presentation的CSV及事件CSV逐位元相同。67個可直接比對的有效窗口分數與快取差0，adl-20另有1個有效窗口因當前畫面品質遮罩，不硬算入67個比較分母。unknown洩漏舊分數0次；快取事件與新推論事件一致。[整合證據](validation/course-demo-integration.json)。
- 4種真實狀態抽幀已視覺查核；影片13段抽幀查核版面。五分鐘有聲／無聲版均完整解碼通過，9,000影格及30fps驗證，兩版視訊流hash完全相同。旁白300秒，無數位削波；為本機合成，原音檔保留，旁白速度調整以配合段落；**展示影片播放速度沒有改動**。[最終驗證](validation/course-final-verification.json)。
- Notebook已核對hash，7個cell成功且2張圖內嵌。再次乾淨kernel執行成功；第二次沙箱執行僅IPython歷史資料庫無法寫入，未影響cell結果，原Notebook未被覆蓋。[執行證據](validation/course-notebook-execution.json)。
- 本輪真實結果、real_feature_smoke資源驗證與synthetic測試分開。重用cache時只有hash／內容驗證，不宣稱重新做了70支pose forward；本輪4支demo才是新的完整真實推論證據。
- 歷史Test已查看、不保證人物隔離、預訓練來源並非全流程無監督；序列TP不證明起跌時間準確，沒有可靠onset延遲驗收。CUDA的CuBLAS決定性警告已記錄，不承諾重新訓練逐位元相同。
- 未達UR課程參考Accuracy90%／F1 0.87，且官方切分、評估單位、unknown、外部訓練及公式尺度尚未確認，因此不直接換算課程分數。原Human缺資源仍是歷史路徑阻塞；本輪沒有因此停留在synthetic。

## 成果位置

| 成果 | 路徑 |
| --- | --- |
| 已執行Notebook | `notebooks/course-results.ipynb` |
| 免kernel閱讀版 | `notebooks/course-results.html` |
| 正式Test輸出 | `outputs/course-ur-test/` |
| 正式Validation輸出 | `outputs/course-ur-validation/` |
| 中文demo及原始紀錄 | `outputs/course-demo/*-presentation.mp4/.csv/.json` |
| 五分鐘有旁白影片 | `outputs/course-film/UR_Fall_Final_Project_5min.mp4` |
| 五分鐘無旁白影片 | `outputs/course-film/UR_Fall_Final_Project_5min_silent.mp4` |
| 字幕與分鏡 | `outputs/course-film/subtitles.zh-TW.srt`、`storyboard.json` |
| 逐段展示腳本 | [course-demo-script.md](course-demo-script.md) |
| 模型／設定／來源 | `runs/course-ur-balanced-001/`，另保留baseline/compact |
| 驗收與hash | `docs/validation/course-*.json` |

## 可直接執行

在PowerShell執行；無須修改系統Python：

```powershell
Set-Location -LiteralPath 'C:\成大專案\fall-detection-anomaly'

# 只讀檢查正式數字、模型與20個歷史run
.\.venv\Scripts\python.exe scripts/verify_course.py

# 乾淨kernel重新執行Notebook；另存帶時間戳的副本
.\.venv\Scripts\python.exe scripts/verify_course.py notebook

# 用選定run製作新中文demo，時間戳避免覆蓋
$courseStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
.\.venv\Scripts\python.exe -m fall_detect demo --run runs/course-ur-balanced-001 --input data/raw/urfall/fall-02 --output "outputs/course-demo/manual-$courseStamp.mp4" --overlay presentation --device cuda --no-display

# 查看此run的Loss（開啟本機瀏覽器；Ctrl+C結束）
.\.venv\Scripts\python.exe -m fall_detect loss --run runs/course-ur-balanced-001
```

換成`--overlay debug`可查看英文原因碼與毫秒age。中文字型若不在預設位置，可傳`--font`指向本機中文字型。`.venv`附加依賴列於`requirements/course-results.txt`。

本輪實際建立流程依序執行了`course_alignment.py diagnose/lock/prepare/train/test`（初次diagnose由等價`.venv`診斷程式產出同格式證據）、`build_course_results.py evidence/notebook`、四支CLI demo、`verify_course.py integration`、`build_course_film.py plan`、`course_narration.ps1`及`build_course_film.py render`。建立腳本故意拒絕覆寫已存在run、鎖定紀錄與影片；不要為重現而刪除舊成果。本轮停止條件已達成，新增模型實驗需另立候選協定。
