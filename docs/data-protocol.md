# UR Fall camera 0 RGB：自訂 sequence v1

來源：[UR Fall 官方頁](https://fenix.ur.edu.pl/~mkepski/ds/uf.html)。資料有 40 ADL、30 Fall；本專案只使用 camera 0 RGB。資料授權為 CC BY-NC-SA 4.0、供非商業學術用途，引用 Kwolek & Kępski (2014), *Human fall detection on embedded platform using depth maps and wireless accelerometer*。

`data/manifests/split-v1.json` 是已保存的完整 70 序列清單。seed=42，官方 ID 排序後用同一個 `numpy.random.default_rng(42)` 先 permutation ADL、再 Fall；ADL 按 24/8/8、Fall 按 0/15/15 分 Train/Validation/Test。NumPy 2.4.6 及 split hash 保存在 manifest。重跑沿用已保存 ID，不按下載了幾段重新切分。

Train／validation／test 原始序列無交集，Train 僅 ADL，所有衍生窗沿用來源 split。這是自訂序列隔離，沒有人物或 session metadata 可證明 subject-independent。Test 不用來挑 epoch、門檻、K、品質或前處理。原協定正式 train 要求所有70段prepared，目前已完成。下表保留最初兩段smoke的來源核對；完整基準見validation/openpose18-dataset-report.json。Le2i擴充使用另一個161段協定，見本文末節。

| 本次來源 | 圖片數 | ID | 同步 CSV 實際結構 | 檢查 |
| --- | ---: | --- | --- | --- |
| adl-05 | 180 | 1…180 | 前兩欄 frame ID、timestamp_ms | 全部解碼、ID 對齊、單調、無中間缺圖 |
| fall-02 | 110 | 1…110 | frame ID、timestamp_ms、accelerometer value | 全部解碼、ID 對齊、單調、無中間缺圖 |

不用 CSV 列號當 RGB index。一般 PNG／JPG 根據檔名末端整數排序，重複 frame ID 拒絕；缺失圖片建立 source_missing 記錄，保留時間缺口。若缺失 frame ID 也缺少 timestamp，以明確 fps fallback 標記推估；若與之後來源時間衝突則拒絕，要求明確時間資料。來源損壞、非單調時間、尺寸改變會報錯並保留序列失敗狀態。

重採樣時間軸從來源第一個 timestamp 開始，target_fps=30。每個 tick 只取當時／之前的最新觀測；可短暫重複並記錄 repeated，超過一個 target 間隔或明確缺圖時保留 missing，不用未來骨架回填。端點固定 `sample_index=29+j*5`。不足 30 samples 為 warming_up；window coverage 分母為零時輸出 null。

原Human配方是13點中至少9點confidence≥0.2；目前OpenPose配方則是18點中至少12點，才算合格影格。一窗至少 80% 合格；任一完整骨架／track 缺失按 invalidate_window 無效，不能把缺失的 20% 當成品質合格。品質與填值不因 Test 成績變更。

官方逐幀 label：-1=未躺地、0=跌倒過渡、1=躺地；0 保留，不轉正常或任意排除。這些是姿態狀態，不能從 label=1 推斷跌倒起始；本次未使用 extracted-features posture CSV 計分，其 RGB 對齊未驗證。若日後使用，需另外建立與確認幀／事件協定。深度特徵欄位不作骨架模型輸入。

主要 ground truth 是整段 Fall=1／ADL=0。每 track 的合法連續 K=3 個 raw scores 取最小值成 event_score；每 sequence 跨 track 取最大 event_score。無合法 event score 為 unknown。`score > threshold` 嚴格比較，相等不觸發；警示只在第 K 個高分窗可用時確認，不回填到第一個窗口。

Validation 兩類都至少一個可評估序列才能校準。候選是唯一有限 sequence scores 加最小值向下的 float64 nextafter；最大 score 表示全不觸發。選最高 fall F1、再較少 FP、再較高 threshold。F1 候選與 streaming 同一比較函式。

二元 metrics 使用可判定序列，同時附 n_total/n_evaluable/n_unknown、sequence coverage、ADL/Fall 分組 coverage、2×3 表與 operational accuracy=(TP+TN)/全部序列。end_to_end_fall_recall=TP/全部 Fall。unknown 不變 TN。window coverage 是至少一個人物有效 raw score 的預定端點數／所有預定端點；與人物數無關。若序列時間軸遺失，整體 window coverage 為 null 並報 n_missing_timelines，禁止缩小分母冒稱高 coverage。

序列 TP 可能來自跌倒前誤報；CSV 保存實際警示時間供人工檢視，不把 sequence F1 當作 event onset／latency 成績或課程論文重現。

## 2026-09-17 原始同步缺口的明訂處理

完整下載為70段、11,936張RGB。第一輪inventory有69段通過，adl-37失敗：RGB編號1..350，但官方同步CSV只含1..330，末時間11117 ms。沒有刪除20張圖片，也沒有更動原始CSV。

第二條路設定 `data.allow_missing_timestamp_tail: true`、fallback_fps=30。僅在來源最後一個已知時間戳之後允許逐幀增加1000/30 ms；331..350明確標示 timestamp_estimated=true。已有的1..330完全保留原時間戳；CSV中間或前端的時間缺口仍報錯，不利用未來时间回填。Human預設仍嚴格拒絕這類缺口。

這項策略加入recipe fingerprint以避免混用舊快取；dataset-report列出estimated_rgb_timestamp_ids與fallback_fps，pose cache逐幀保存timestamp_estimated。相關測試覆蓋缺口的預設拒絕、尾端推估、來源時間保留、內部缺口拒絕與fingerprint隔離。


## Le2i 擴充協定

`configs/openpose18-le2i.yaml` 使用 `urfall-plus-le2i-cs-sequence-v1`，保留UR Fall所有原始分組。Le2i依固定OmniFall cross-subject清單，僅38段全片無fall/fallen的Train來源可訓練；驗證原有19段中的兩組重複內容四段全部排除，保留4正常／11跌倒；Test保留16正常／22跌倒。混合總數161，Train/Validation/Test為62/38/61；不能宣稱整個混合集都跨人物。

所有190個Le2i原始影片有實際SHA256；跨split同內容拒絕，同split重複組整組排除。先決定原始影片split，後擷取窗口。已知標註區間的union保留空白；未完整落在已知區間的窗口記為unlabelled_interval，不使用於AE或正式計分，仍保留在預定端點分母。沒有把無標註片段當正常。示範一般影片不讀標籤，與這項標註範圍限制有明確差別。

正式評估同時輸出混合結果與by_dataset，保留unknown。UR Fall Test已看過，只能做歷史比較；新Le2i Test在設定、AE、門檻固定後才計分。影片完整性要求實際解碼到宣告frame count；明確limit的smoke除外。完整來源、去重、旋轉選擇及操作見[Le2i擴充紀錄](le2i-extension.md)。


## CAUCAFall 擴充協定

`urfall-le2i-caucafall-subject-sequence-v1` 以凍結的 161 序列清單為基礎，另加 70 支官方完整 AVI。CAUCAFall 十人以 seed42 分 6/2/2，人員不跨 Train/Validation/Test；30 支 Train ADL，Validation/Test 各 10 ADL+10 Fall；30 支 Train 人物的 Fall 整段排除。保留原 161 序列切分與 hash，整個合併資料集不可宣稱人物獨立（UR Fall 原協定仍只有序列隔離）。

原作者活動資料夾提供整段標籤；本次不套用第三方切段，也不把倒地姿態推算為跌倒開始時間。原始 AVI 實測全為 720x480/20 FPS，與來源頁所述攝影機能力不同；依檔案 PTS/FPS 處理。來源 byte ranges/ZIP CRC/SHA256、事前鎖定與實驗結果見 [CAUCAFall 實驗](caucafall-training.md)。
