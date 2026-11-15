# 跌倒異常偵測

## 從 Git 抓下來直接使用（Windows x64）

安裝 Git 後，在 PowerShell 執行：

```powershell
git clone https://github.com/p77151087-jpg/Anomaly-Fall-Detection.git
cd Anomaly-Fall-Detection
.\start.cmd
```

第一次會自動建立專案專用 Python 環境、安裝套件、下載既有 AE 模型、預訓練權重與兩段範例，再產生並開啟示範影片。之後直接執行 `start.cmd`，既有環境與資源會重用，不需手動啟用 venv 或先重新訓練。首次需連網，CUDA 套件下載量較大；自動選擇可用 NVIDIA GPU，否則使用 CPU。

自己的影片：`.\start.cmd -Input "D:\videos\my-video.mp4"`。只建置：`.\start.cmd -SetupOnly`。詳細選項見[組員使用指南](docs/teammate-setup.md)。下方保留歷次研究紀錄與手動安裝方式。

**最新UR-only方向（2026-10-12）：外部輸入已退出；Train資料增強可行，但本輪未提高Accuracy。** 原外部raw／cache封存data/retired，原模型／結果／hash保留。僅24UR Train ADL產生1%關節位置擾動，3,638原＋3,638增強列、無新增原影片；單一AE實跑29epochs，Val21/23＝91.30%／F1 .9286，低於原最高95.65%／.9655，不採用。36風險測試、48原特徵parity差0、Train-only Scaler／23支K3、1,435舊檔保全與8次拒絕通過，無新Test。[結果、限制與只讀操作](docs/ur-only-shift-20261012.md)。以下外部資料結果為已退出的歷史比較。

**最新新增真實正常資料（2026-10-12）：32支下載、21支可用，Accuracy未提高。** Train24／3,638窗增至45／7,469，真實Pose／ST-GCN抽取13,270新RGB幀。47支Train／Val有340圖AI取樣可見性及15段跌倒不確定區間，尚非人工完整GT。固定2AE實跑95epochs，皆Val21/23＝91.30%／F1 .9333，低於原最高95.65%／.9655，不採用。36風險測試、Train-only Scaler／46支次K3、1,160舊檔hash與10次覆寫拒絕通過，無新Test。[來源、結果、標註與命令](docs/external-normal-20261012.md)｜[影像標註檢視](outputs/external-normal-20261012-report/annotation-review.html)。

**最新Train／Validation工作（2026-10-12）：資料量影響有實測，Accuracy仍未提高。** 4個Train子集學習曲線與2個Train鏡像AE共312新epochs，正常Val共同尺度MSE隨8→16→24原影片下降；鏡像2組Val91.30%／F1 .9286。另2個Train殘差profile新增epochs0，Val95.65%／.9655平手；粗略時間案例有改善，非正式延遲驗收。37風險測試、48原特徵parity差0、1110舊檔hash與Val獨立核對通過；原預設保留、所有新候選未跑Test。24原始影片不因7276衍生列就變成更多獨立影片。[完整診斷、結果及操作](docs/normal-data-support-20261012.md)。

**最新指定候選Test（2026-10-12）：大容量Accuracy78.26%，Fall F1 .8000。** 使用者明確要求後，只評估`ur-isotropic-wide-20261012-001`一次，固定Validation門檻與原23支Test：TP10／TN8／FP0／FN5／影片unknown0，兩種Accuracy均18/23。沒有依Test重訓／改門檻／改選或重切；Validation95.65%未延續至此Test，原預設及舊guard保留。13風險測試、獨立CSV／K3重算及1103舊檔hash通過。[成績、限制與操作](docs/ur-isotropic-wide-test-20261012.md)。以下「候選未跑Test」為先前實驗階段歷史紀錄。

**最新Train／Validation改善（2026-10-12）：Validation提高到95.65%，F1 .9655。** 固定三候選實跑121epochs；`ur-isotropic-wide-20261012-001` TP14／TN8／FP0／FN1／unknown0，兩種Accuracy均22/23，比原最高21/23多救回fall-18。新增TP當前良好骨架警示guard未過，原預設保留；所有新候選未跑Test，不宣稱Test改善。26相關測試、獨立CSV／K3重算及1039舊檔hash驗證通過。[診斷、成績、限制與只讀命令](docs/isotropic-refinement-20261012.md)。

**最新Test結果（2026-10-12）：最高Validation候選的Accuracy為82.61%，F1 .8462。** 使用者指定`ur-human-isotropic-001`後，固定原門檻及23支Test：TP11／TN8／FP0／FN4／unknown0，兩種Accuracy分母皆23；四支漏報、沒有正常誤報。模型／Scaler／門檻／split與舊成果hash不變，17相關測試及CSV獨立重算通過。Validation91.30%不能當成Test成績；歷史Test已查看限制保留，正式Notebook／五分鐘影片仍暫停。[正式成績、錯誤案例與命令](docs/ur-human-isotropic-test-20261012.md)。

**最新展示：大字UNKNOWN已改為sample式七類動作。** Human的`--overlay auto`現在使用獨立原30次觀測分類，支援isotropic AE；低品質用小字提示，追蹤消失不硬填動作，AE事件另列歷史／時間。232 tests passed／1歷史路徑skip，六片888幀AE CSV／事件逐位元不變，674筆七類分數重算差0，舊41run／8764cache保留。這次沒有提高Accuracy或重訓，七類仍可能錯判。[新版短片、限制及操作](docs/sample-display.md)。以下歷次「最新」段落屬歷史紀錄，顯示預設以本段為準。

**最新已實跑：影片級Validation提高至91.30%。** 依可見關節診斷新增保留像素比例的前處理，`ur-human-isotropic-001`完成49epochs：TP13／TN8／FP0／FN2／unknown0、Accuracy21/23=91.30%、F1=.9286；fall-09可見AE警示5.9→5.0秒。但新增TP fall-30仍無可見警示，未通過事前展示guard，所以最高影片級候選另存、原預設及七類展示保留。215 tests passed／1歷史路徑skip，6片888幀／559分數一致，40舊run／8450cache hash保留；無新Test或正式Notebook／五分鐘影片，不宣稱課程全部達標。[結果、短片、限制與操作](docs/human-isotropic.md)。

以下保留歷次實驗，成績及預設請以上方最新紀錄為準。

**最新骨架／正常活動優化完成：** 12片翻轉骨架pilot未達品質門檻；正常Train特徵型態抽樣AE完成33epochs，Validation仍20/23=86.96%、F1=.8889，保留`ur-sample-human-s1-001`。210 tests passed／1歷史路徑skip，723幀／526分數一致，原七類＋AE展示保留；沒有新Test，Accuracy90%仍未達。[正常活動抽查、實驗結果與操作](docs/human-quality.md)。

**最新準確度實驗已完成：** 兩種Train正常殘差加權MSE候選分別19/23=82.61%、20/23=86.96%，沒有提高Accuracy；保留原`ur-sample-human-s1-001`（F1=.8889）。沿用AE，新增訓練epochs=0，Train-only權重與Validation門檻各自凍結。203 tests passed／1歷史路徑skip，三支診斷影片555幀／381分數完全吻合，無新Test；Notebook／五分鐘影片仍暫停。[結果、限制與可執行命令](docs/human-residual.md)。以下保留歷次工作紀錄。

**新版七類動作展示已完成：** 畫面左側ACTION顯示STANDING／WALKING／SITTING／LYING／STANDING UP／SITTING DOWN／FALLING，右側保留獨立AE警示。CLI預設auto，最佳Human模型自動啟用；骨架不足仍UNKNOWN。四支真實影片503幀、313窗分類分數吻合，原AE分數與事件完全不變；191 tests passed／1歷史路徑skip，未宣稱AE Accuracy提升。[直接看影片與操作](docs/seven-actions.md)。

**最新優化已實跑：** 20／15幀短窗及保留全身位移三個新候選共147epochs，Validation分別20/23、19/23、19/23，未提高Accuracy，保留`ur-sample-human-s1-001`（86.96%、F1=.8889）。新候選雖補到一支跌倒，卻新增兩支漏報，且部分窗口事件完全未在畫面顯示；不能當成實用改善。174 tests passed／1歷史路徑skip，真實影片503幀／313可比窗分數一致，無新Test；正式Notebook／五分鐘影片仍等待品質達標。[本輪結果、限制、短片與命令](docs/human-displacement.md)。

**前輪AE優化：** 四組容量／去噪候選共170epochs，Validation均20/23，沒有辨識提升。[歷史結果](docs/human-ae-refinement.md)。

**最新sample同源流程已實跑：** AlphaPose＋512D雙流ST-GCN＋正常AE，選定`ur-sample-human-s1-001`。固定Validation由17/23=73.91%提升至20/23=86.96%，F1=.8889、unknown0，少3次誤報；仍未達90%，沒有新Test。警示時機與骨架仍有限制，正式Notebook／影片暫停。[實測結果、短片與操作](docs/sample-human-reproduction.md)。

Python 專案位於 `C:\成大專案\fall-detection-anomaly`。

**目前優先：先改善UR Fall辨識效果，再更新正式Notebook及期末影片。** 目標Accuracy≥90%、F1≥0.87；目前尚未達成。既有展示保留為歷史紀錄。[目前改善工作](docs/accuracy-first-plan.md)。

**展示文字已依最新回饋簡化。** CLI `demo`現在預設`--overlay simple`：PENDING／NORMAL／FALL ALERT／UNKNOWN，大字狀態與骨架，不顯示跳動的追蹤編號或分數面板，保留更新時間。`--overlay debug`保留分數與編號；`--overlay presentation`保留舊中文畫面。STANDING／LYING不是現有AE的分類輸出，沒有以改字偽造動作辨識。四支Validation影片前後CSV／事件完全一致，[影片與驗證](docs/simple-demo.md)。

**2026-09-18：期末課程調整 CA-01 至 CA-05 本機完成。** 保留 ST-GCN 特徵＋正常資料 AE；UR-only三候選共297回合，選定`course-ur-balanced-001`。UR Test可判定Accuracy **50%（11/22）**、F1 **56%**；全體accuracy **47.8%（11/23）**，unknown1，品質仍不足。Notebook已執行、中文demo前後推論一致、五分鐘有聲／無聲影片完成，136 tests passed／1原Human skip，20個舊run保留。公開上傳未執行。[成果、檔案與操作命令](docs/course-results.md)｜[Notebook](notebooks/course-results.ipynb)｜[閱讀版](notebooks/course-results.html)｜[展示腳本](docs/course-demo-script.md)。以下保留歷史實驗。

**最新訓練強化已完成：`caucafall-yolo11-balanced-001`。** 四組新增521回合，依 Validation 選定每支正常影片平均抽樣版，Val **74.1%→81.0%**。固定 Test 的 CAUCAFall **18/20→20/20（100%）**，但三資料集合計仍 **57/81（70.4%）**；少2次誤報也多漏2支跌倒，**整體95%仍未達成**。20支測試全對不能推廣到所有新場景。程式 Loss 可查看144回合／最佳124；4支真實展示與25項相關測試通過，原模型保留。[本輪結果、取捨及使用方式](docs/refinement-training.md)。

**前輪骨架／時間改善：`caucafall-yolo11-jitter-001`。** 官方預訓練 YOLO11 骨架＋時間處理修正，保留凍結 ST-GCN／正常資料 AE 方法。相同 81 支 Test 整體 **59.3%→70.4%**、CAUCAFall **75%→90%**、整段無法判定 **10→1**。[前輪操作、真實成績與限制](docs/target95-improvement.md)。以下亦保留歷次實驗紀錄。

**程式內查看 Loss：雙擊根目錄的「查看Loss曲線.cmd」**，即可在本機畫面切換既有訓練紀錄、查看 Train／Validation 曲線與最佳回合。新訓練加 `--show-loss` 可同步更新。[完整操作方式](docs/loss-viewer.md)。

**新一輪訓練已完成**：`openpose18-rotation-001` 實跑100回合，最佳Epoch95，補足相同UR Fall資料下的骨架設定對照。Validation跌倒警示8/15→9/15、正常誤報仍6/8；本輪未執行Test。Loss選單可直接查看，[訓練紀錄](docs/rotation-training.md)。

**Le2i擴充已完成**：161段資料、6,404個有效特徵，run `le2i-extension-001` 實跑79 epochs並完成固定Test與兩支展示。正常Train來源24→62段（實際可用19→45段）。相同UR Fall Test的誤報7/8→6/8，跌倒警示仍7/15；Le2i Test跌倒15/22警示、正常11/16誤報，品質仍不足。69 tests passed、1 skipped。直接使用見[擴充操作指南](docs/le2i-extension.md)，完整結果與限制見[Le2i結果](docs/le2i-results.md)。原基準保留。

**2026-09-17 第二條路已實際完成：Lightweight OpenPose 18 點骨架 → 原始 ST-GCN Kinetics 256D 特徵 → 正常資料 Autoencoder → 連續三窗異常警示。** 兩個真實權重已嚴格載入，70 段資料完成處理，正式 run `openpose18-baseline-001` 已訓練、校準及測試，並產出真實展示影片。

**流程可執行，但辨識效果不足：** Test 共23段，8段無法判定；15段跌倒中7段觸發警示，8段正常活動中7段誤報。這是實際基準結果，不能宣稱可靠跌倒偵測。詳見 [結果摘要](docs/results-summary.md)、[目前進度](docs/current-progress.md) 及 [驗收結果](docs/acceptance-results.md)。測試54 passed、1 skipped（僅原Human缺權重）。

**原基準請使用 [第二條路操作指南](docs/openpose18-usage.md) 與 `--config configs/openpose18.yaml`。** 下方舊版指令保留作 Human 方案的操作紀錄；不加 `--config` 仍選 Human，不會自動或靜默替換模型。原 Human 的四個權重資源仍缺少，但不影響第二條路。

## 目前環境直接執行

在專案根目錄 PowerShell 執行，不需啟用 venv：

```powershell
.\.venv\Scripts\python.exe -m fall_detect doctor --config configs/default.yaml --mode core --device cpu
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m fall_detect smoke-test --config configs/default.yaml --mode synthetic
```

synthetic 命令會建立新的 `outputs/synthetic/<時間>/`，含手製特徵、AE 訓練紀錄、scaler、校準門檻、評估報告與可觀看的 `synthetic-demo.mp4`。影片與數字只用來檢查程式，不能當作跌倒辨識成果。已完成的範例在 `outputs/synthetic/first-integration/`。

## 安裝與重建環境

此電腦已建立 `.python/`、`.tools/`、`.venv/`。以下是相同隔離方式的重建步驟；保留既有環境時不必重跑。Windows 系统憑證透過 `--system-certs` 載入，保持 TLS 驗證。

```powershell
python -m venv .tools
.\.tools\Scripts\python.exe -m pip install uv==0.12.13
$env:UV_PYTHON_INSTALL_DIR = Join-Path (Get-Location) '.python'
.\.tools\Scripts\uv.exe --system-certs python install 3.11.16
.\.tools\Scripts\uv.exe --system-certs venv --python 3.11.16 .venv
.\.tools\Scripts\uv.exe --system-certs pip install --python .venv/Scripts/python.exe torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.\.tools\Scripts\uv.exe --system-certs pip install --python .venv/Scripts/python.exe -r requirements/core-tested.txt -r requirements/pose-download-tested.txt pytest==9.1.1 ruff==0.16.7
.\.tools\Scripts\uv.exe --system-certs pip install --python .venv/Scripts/python.exe --no-deps -e .
```

CUDA 組合依 [PyTorch 官方版本安裝表](https://pytorch.org/get-started/previous-versions/#v280) 選擇，已在 RTX 5060 Laptop GPU 實測。`--device cpu` 在此 CUDA wheel 的 CPU 執行路徑已測；若安裝 CPU-only wheel，可將官方 index 換成 `https://download.pytorch.org/whl/cpu`，該獨立 wheel 組合尚未在本機另建環境驗證。完整套件版本見 `requirements/windows-cu128.freeze.txt`。

## 資料與模型取得

```powershell
.\.venv\Scripts\python.exe -m fall_detect fetch-assets --models --dataset urfall-cam0-rgb --limit-per-class 1
.\.venv\Scripts\python.exe -m fall_detect smoke-test --mode real
```

下載器從固定的 [Human README](https://github.com/GajuuzZ/Human-Falling-Detect-Tracks/blob/7ed2faa4d6147dfd576f58869b6c25545208af35/README.md) 和 [UR Fall 官方頁](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) 取得連結。模型下載若失敗會回傳非零狀態並保留報告；不會以隨機初始化繼續。`--limit-per-class 1` 根據已保存的完整 split 選 Train ADL／Validation Fall，不下載 Test 作互動調整。

目前缺少：

| 角色 | 來源 README 項目 | 放置位置 |
| --- | --- | --- |
| 人物偵測權重 | Tiny-YOLO oneclass `.pth` | `weights/pretrained/detector.pth` |
| 人物偵測設定 | Tiny-YOLO oneclass `.cfg` | `weights/pretrained/yolov3-tiny-onecls.cfg` |
| AlphaPose 權重 | SPPE FastPose `resnet50` | `weights/pretrained/fast_res50_256x192.pth` |
| 動作特徵權重 | ST-GCN `tsstg` | `weights/pretrained/tsstg-model.pth` |

手動取得後放入上述位置，執行 `doctor --mode core` 註冊實際 SHA256，再執行 `smoke-test --mode real`。完整 state dict 的 keys/shapes、受限 `weights_only=True` 載入、`strict=True` 及真實 forward 通過後才標記 usable。來源與檢查值存於 `weights/manifest.json`。本地計算 hash 是實驗識別，不代表上游提供的預期檢查值。

上游程式固定為 commit `7ed2faa4d6147dfd576f58869b6c25545208af35`，放在忽略 Git 的 `third_party/human/`。若只需取得程式執行結構測試：

```powershell
git clone https://github.com/GajuuzZ/Human-Falling-Detect-Tracks.git third_party/human
git -C third_party/human checkout --detach 7ed2faa4d6147dfd576f58869b6c25545208af35
```

## 完整實驗順序

先讓 real smoke 產生兩個真實來源的有限 512D 特徵，再執行後續完整資料處理。real smoke 尚未訓練 AE，不會輸出正常／跌倒正式分類。

```powershell
.\.venv\Scripts\python.exe -m fall_detect fetch-assets --dataset urfall-cam0-rgb
.\.venv\Scripts\python.exe -m fall_detect doctor --mode full
.\.venv\Scripts\python.exe -m fall_detect prepare --config configs/default.yaml
.\.venv\Scripts\python.exe -m fall_detect train --config configs/default.yaml --run-id baseline-001
.\.venv\Scripts\python.exe -m fall_detect calibrate --run runs/baseline-001
.\.venv\Scripts\python.exe -m fall_detect evaluate --run runs/baseline-001 --split test
.\.venv\Scripts\python.exe -m fall_detect demo --run runs/baseline-001 --input data/raw/urfall/fall-02 --output outputs/demo/fall-02.mp4 --no-display
```

`demo --input` 亦可指向一般 MP4／AVI。影像資料夾以原始 frame ID 排序；同層 `*-data.csv` 唯一時使用其 timestamp，否則採明確記錄的 fps fallback。影片以固定 target fps 的因果時間格點輸出，時長誤差最多一個格點；處理速度另存，與播放 fps 分開。

## 命令與產物

| 命令 | 主要產物／失敗條件 |
| --- | --- |
| `loss` | 本機互動 Loss 畫面、選擇 run、每2秒刷新、SVG／CSV下載；`--run`指定初始紀錄 |
| `doctor --mode core/full` | `outputs/doctor-*.json`；full 需要完整資料、backend 與指定 device 的真實權重驗證 |
| `fetch-assets` | 原始 ZIP／PNG／同步 CSV、來源目錄、下載報告；任一資源失敗返回 2 |
| `prepare --inventory-only` | 不依賴模型，解碼與檢查資料；未達所選協定的完整序列數返回 2，保留全體缺失紀錄 |
| `prepare` | 分開保存 pose／feature NPZ+JSON，內容與 recipe 相符時重用；每段完成後保存 index，可接續 |
| `train` | 新 run，最佳 AE、正常 train scaler、設定、來源、環境、訓練 CSV／曲線；正式訓練要求所選協定的完整序列（原版70、擴充161） |
| `calibrate` | validation 候選表、float64 門檻與 frozen artifact hash；兩類都至少一個可評估序列 |
| `evaluate` | 新 evaluation 目錄，metrics、sequence/window/event CSV、2×3 表、混淆矩陣及說明；不改模型 |
| `demo` | MP4 或 AVI、同名逐幀 CSV／JSON、`*-events.csv`；需要真實且已校準的 run |
| `smoke-test --mode synthetic/real` | 模式分開的報告；缺少真實權重時 real 返回 2 |

所有命令都有 `--help`。未知設定、維度／hash 不合、重複 run ID、已存在的輸出影片會明確報錯。`calibrate`／`evaluate` 使用 run 的已保存設定；不要加另一份 config。改品質、時間格點或 backend 必須重新 prepare、train、calibrate 並使用新 run。

## 方法與限制

- AE 只使用所選協定的正常訓練序列；門檻用包含正常／跌倒標籤的 validation 校準。
- 第二條路的上游ST-GCN採監督式Kinetics預訓練；原Human方案則以Le2i七類動作訓練。不能說整套系統完全無監督。
- 主要單位是原始 sequence，自訂切分未保證人物隔離。高重建誤差只提示「疑似跌倒」，不是跌倒機率。
- K=3 同一人物的相鄰有效 windows 全部嚴格 `score > threshold` 才觸發。相等不觸發；警示時間不往前回填。
- 無人、缺窗或數值退化保留 unknown；沒有合法 event score 的 sequence 不算 TN。報告同時呈現 coverage、2×3 結果與 operational accuracy。
- CUDA完整流程已驗證；CPU只完成真實影像與骨架窗forward，未完成整段驗證，也不承諾即時。後續依需求新增本機Loss檢視頁；攝影機、雲端與通知不在第一版範圍。

詳見 [架構](docs/architecture.md)、[資料協定](docs/data-protocol.md)、[模型來源](docs/model-sources.md)、[環境](docs/environment.md) 與 [結果狀態](docs/results-summary.md)。


報告用[Loss圖與圖說](docs/loss-figures.md)已補齊；後續[擴充方向與來源](docs/next-expansion.md)另列。


### 最新改善：低 FPS 修正與訓練結果

已完成時間重採樣修正與88回合真實訓練。Loss選 `le2i-timing-001`；另有降低誤報但增加漏報的 `le2i-timing-balanced-001`。操作、Validation完整分母與限制見[改善結果](docs/timing-improvement.md)。本輪未重跑Test，不將校準集成績當作泛化成績。


### 接續訓練結果

又完成三組AE共577回合，Loss畫面可選 `le2i-seqbalanced-001`（172回合、最佳Epoch142）。這輪正常Validation Loss下降，但整體正確判定率仍未超過55.3%；保留所有模型與誤報/漏報取捨。[完整訓練比較](docs/further-training.md)。


### CAUCAFall 擴充訓練（2026-09-17）

已下載官方100支完整AVI（約135MB），依人物6/2/2切分，30支新正常影片加入AE訓練。`caucafall-extension-001`完成97回合；CAUCAFall Test75.0%（15/20），合併Test59.3%（48/81），皆將unknown保留分母。原38支Validation仍55.3%，不等於整體升到75%。

[完整結果、來源及重跑命令](docs/caucafall-training.md)。程式內查看Loss：`.venv\Scripts\python.exe -m fall_detect loss --run runs/caucafall-extension-001 --port 0`。
