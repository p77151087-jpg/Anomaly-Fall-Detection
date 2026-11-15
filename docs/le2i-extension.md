# Le2i 正常資料擴充實驗

狀態：**已完成真實資料擴充、79 epochs訓練、Validation校準、固定Test與兩支展示**。新run `runs/le2i-extension-001`，原基準保留；誤報仍高，詳見[實際結果與限制](le2i-results.md)。

**Loss 已整合進程式畫面**：雙擊根目錄「查看Loss曲線.cmd」，選擇 `le2i-extension-001`，即可查看79回合曲線。[操作說明](loss-viewer.md)。

## 直接使用已訓練模型

在專案根目錄PowerShell執行；本機模型已訓練好，不需重新下載或訓練：

```powershell
cd 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe -m fall_detect demo --run runs/le2i-extension-001 --input 'C:\你的資料夾\影片.mp4' --output outputs/demo/my-video-le2i.mp4 --no-display
```

將input替換成自己的影片路徑；output請用尚未存在的檔名。完成會得到MP4、逐幀CSV、事件CSV和JSON。`normal`表示當下未超過門檻，`suspected_fall`是疑似異常警示，`warming_up`是收集時間窗，`unknown`表示骨架或追蹤不足，不能當正常。這版誤報仍高，請搭配實際畫面檢查。

已產生的[UR Fall示範](../outputs/real_end_to_end/le2i-extension/urfall-validation-fall-02.mp4)與[Le2i示範](../outputs/real_end_to_end/le2i-extension/le2i-validation-coffee42.mp4)可直接觀看。Le2i示範也包含跌倒前的誤報，不只展示成功片段。

## 資料與協定

原始影片來自 [Le2i / ImViA 官方資料頁](https://search-data.ubfc.fr/imvia/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html)，ZIP實際9,608,701,260 bytes（SHA256 `01df91e32a034d9897d1f9cd5e0211c70ab54289d27e8490db5a699384c17fc8`），CC BY-NC-SA 3.0。保留原始AVI，不轉碼、不改影片時間。下載器最多6個明確byte ranges並行，核對HTTP Content-Range及完成長度；解壓時驗ZIP CRC，保存實際SHA256。

標註及人物切分來自 [OmniFall](https://huggingface.co/datasets/simplexsigil2/omnifall)，固定revision `83572a37b9e3081df8c06a56874b1d1f2a19386c`；四份CSV各自保存並核對SHA256。OmniFall套件0.2.0僅下載原始碼查驗來源，沒有安裝或執行其自動訓練流程。

獨立協定 `urfall-plus-le2i-cs-sequence-v1`：原始影片先決定split，再形成窗口。任何片段標為fall(1)或fallen(2)，該原始影片歸為Fall；Train分組中所有95段此類影片整段排除，不擷取其前半段當正常。其餘38段正常來源影片可加入Train，原先19段Validation、38段Test。實際核對發現兩組Validation重複影片，且各副本時間標註不同，因此四段整組排除；最終15段Validation、38段Test。

| 分組 | UR Fall（不變） | 新增Le2i | 合計 |
| --- | --- | --- | --- |
| Train | 24正常 | 38正常 | 62正常 |
| Validation | 8正常＋15跌倒 | 4正常＋11跌倒 | 38段 |
| Test | 8正常＋15跌倒 | 16正常＋22跌倒 | 61段 |

Le2i依作者的cross-subject切分；原始影片、受試者與相同檔案hash不能跨組。所有190段來源先計算SHA256；跨split的相同內容直接拒絕，同split重複組整組排除，不挑選有利副本。實際排除Home_01/video_2、video_3與Home_02/video_49、video_50；Train另有重複跌倒組已在95段跌倒排除清單內。UR Fall維持原工程切分，整個混合協定不能宣稱完全跨人物。UR Fall Test已在前次實驗查看，此次只能作歷史比較；Le2i Test是新保留評估。設定、模型與門檻不能用任一Test成績反覆挑選。

部分影片有未標註的前段或中間缺口；**未標註不等於正常**。只有整個窗口落在連續已知標註時間內，才可擷取特徵；其餘記為 `unlabelled_interval`，保留在預定窗口coverage分母。此protocol依標註區間限制可評估範圍，不能把它的coverage與未作相同限制的研究直接比較。新影片demo無需標註，也不使用標籤推論。

## 骨架候選與選擇

只用Train adl-05、Validation adl-37/fall-02/fall-30做初步候選比較，不使用Test。384像素、第3 refinement僅增加1個有效窗；0/±90度真實影像推論增加fall-02的2個有效窗。選用的條件式旋轉在四段取得30/28/7/1個有效端點，與全旋轉相同。數據見 `docs/validation/le2i-pose-pilot.json`、固定選擇紀錄 `docs/validation/le2i-pose-selection.json`；這是特徵可用性，沒有AE分類成績。

旋轉輸入仍使用同一組嚴格載入的真實權重，將關節還原回原圖，抑制重複人物，保留整組實際觀測；沒有平均合成關節、補假點或以預測骨架填缺口。追蹤仍使用固定上游的關節相似度匹配。`rotation_policy: low_quality` 在原向已有品質合格人物時略過兩個旋轉方向，以限制處理成本；這不是多人／遮擋問題的完整解法。已查看六張真實抽幀：站立較完整，躺倒後仍有同人重複骨架、部分身體分裂成多個track及家具誤偵測；不得稱追蹤問題已解決。

旋轉與修正階段都進入recipe fingerprint；預設不旋轉的歷史recipe hash保持相容。新設定重新建立特徵、scaler、AE與門檻，不能直接套舊AE。資料及骨架同時改動的結果不能用來單獨歸因「更多資料」或「旋轉」的效果，需另做消融才可分辨。

## 重建實驗（平常使用不必重跑）

在專案根目錄PowerShell使用既有.venv；範例run `le2i-extension-001` 已存在，重新訓練須將所有run-id/路徑換成新的相同名稱，不能覆蓋舊run：

```powershell
.\.venv\Scripts\python.exe -m fall_detect fetch-assets --config configs/openpose18-le2i.yaml --dataset le2i-rgb
.\.venv\Scripts\python.exe -m fall_detect prepare --config configs/openpose18-le2i.yaml --inventory-only
.\.venv\Scripts\python.exe -m fall_detect prepare --config configs/openpose18-le2i.yaml --device cuda
.\.venv\Scripts\python.exe -m fall_detect train --config configs/openpose18-le2i.yaml --run-id le2i-extension-001
.\.venv\Scripts\python.exe -m fall_detect calibrate --run runs/le2i-extension-001
.\.venv\Scripts\python.exe -m fall_detect evaluate --run runs/le2i-extension-001 --output outputs/real_end_to_end/le2i-extension/test
```

如已有相同recipe的完整UR Fall快取，prepare可加 `--reuse-index data/cache/openpose18-selective/dataset-index.json`。程式核對權重／前處理recipe、來源內容、split與陣列hash後，重建獨立擴充快取；不變更原檔。任何不相符都拒絕重用。

`prepare --sequence-ids ...`僅用於有界feature smoke；不完整index不能訓練正式run。資料、權重、快取、run及影片由Git忽略；來源、協定及驗收摘要保留。來源阻塞或無效影片不會改成synthetic成功。


## 使用時要知道的事

新增的是正常活動訓練來源。Pose與ST-GCN仍使用取得的真實預訓練權重且凍結；重新學習的是Scaler與Autoencoder，並由Validation重新選警示門檻。這不是把更多跌倒影片塞進AE，也不是重新訓練骨架網路。

所有161段已完整解碼，合計54,342影格；原UR Fall 11,936、Le2i 42,406。Le2i未使用音訊，解碼器部分MP3音訊header提示不影響逐張RGB完整性檢查；實際161段影格數與宣告數一致。UR Fall舊有20張尾端時間估計仍明確保留，Le2i時間均有影片PTS：60段25 fps、31段約24.000384 fps，依來源時間因果重採樣至30 fps，沒有重標影片速度。91/91宣告影格數與完整解碼一致，見`docs/validation/le2i-video-frame-counts.json`。測試69 passed、1 skipped（舊Human缺權重）。

原本兩個替代方案權重已在本機，因此此擴充不需要先補齊舊Human的四個檔案。資料來源或hash不符會停止；沒有有效骨架的窗口保留unknown，不拿假骨架補成成功。

PowerShell顯示`epoch ...`代表AE訓練進度；`source frame ...`代表骨架擷取。後者通常遠比AE訓練耗時。中斷後可以重跑prepare，完成的每段快取會經核對後重用。train不覆蓋既有run，重做實驗請使用新run-id；校準與評估使用該run已保存的設定。
