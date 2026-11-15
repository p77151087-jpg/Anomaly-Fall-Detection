# CAUCAFall 原始影片擴充

2026-09-17 完成。CAUCAFall 持出測試 75.0%，合併測試 59.3%；完整數字與限制見下方。

來源：[原作者 Mendeley v4](https://data.mendeley.com/datasets/7w7fccy7ky/4)，DOI 10.17632/7w7fccy7ky.4，CC BY 4.0，作者 Jose Camilo Eraso、Elena Muñoz、Mariela Muñoz、Jesus Pinto。100 支完整 AVI 原始位元組共 135,044,978 bytes；官方整包 8,326,349,365 bytes 包含圖片與標註。使用 HTTP Range 與 ZIP 目錄只取原始 AVI；核對固定 archive size/ETag、每支 ZIP CRC，保存實際 SHA256。未下載整個 archive，故其官方公布 SHA256 僅為來源資訊，不宣稱已驗證整包 hash。

先檢查的 Zenodo 173,504,611-byte 第三方包含切段與重複命名版本，保留供來源稽核，**不進入此次訓練或評估**。不重組片段，不生成替代影片。

## 事前固定協定

- 保留原 UR Fall + Le2i 的 161 支影片分組、來源與標註。新 protocol `urfall-le2i-caucafall-subject-sequence-v1`。
- CAUCAFall 依人物、seed 42：Train subjects 6/7/1/8/4/3；Validation 5/10；Test 2/9。
- 30 支 Train ADL；整段排除 30 支 Train Fall，不取其中站立片段訓練。Validation/Test 各 10 ADL + 10 Fall。
- 合併後 231 序列：Train 92 ADL；Validation 22 ADL + 36 Fall；Test 34 ADL + 47 Fall。
- 使用原作者活動資料夾作整段標籤，不聲稱有逐幀跌倒起點或可驗證的事件延遲。
- `configs/openpose18-caucafall.yaml`：相同嚴格真實 OpenPose18/ST-GCN 權重、來源 FPS 因果重採樣、12/18 關節及 80% 品質；有效特徵不足保留 unknown。
- AE 採規格預設 128/32、Adam 0.001、batch 32、100 epochs 上限、patience 10、windows sampling。只 Train ADL fit scaler/AE；正常 Validation 選 checkpoint。
- 主結果依規格採 Validation sequence fall F1 門檻；另存同 AE 的 balanced_sequence_accuracy 門檻，列為取捨對照。兩者在讀取 Test 分數前凍結；不按 Test 成績挑選模型。
- 舊模型不覆寫。先做新資料真實 feature smoke，再完整準備與訓練。Test 成績依資料集分列；既有 UR/Le2i Test 曾在歷史實驗評估，新增 CAUCAFall Test 為本次首次持出評估。

## 重跑命令

```powershell
.\.venv\Scripts\python.exe -m fall_detect fetch-assets --dataset caucafall-rgb --config configs/openpose18-caucafall.yaml
.\.venv\Scripts\python.exe -m fall_detect prepare --config configs/openpose18-caucafall.yaml --reuse-index data/cache/le2i-source-interval/dataset-index.json
.\.venv\Scripts\python.exe -m fall_detect train --config configs/openpose18-caucafall.yaml --run-id caucafall-extension-001
.\.venv\Scripts\python.exe -m fall_detect calibrate --run runs/caucafall-extension-001
.\.venv\Scripts\python.exe -m fall_detect calibrate --run runs/caucafall-extension-001 --objective balanced_sequence_accuracy --output-run runs/caucafall-extension-001-balanced
.\.venv\Scripts\python.exe -m fall_detect evaluate --run runs/caucafall-extension-001
.\.venv\Scripts\python.exe -m fall_detect loss --run runs/caucafall-extension-001 --port 0
```

## 20 FPS 影格追蹤修正

3 支 smoke 初版全為零特徵，原因是 AVI 實際 20 FPS，其 50 ms 間距有 50.00000000000003 等浮點誤差，舊追蹤每次遇到此誤差就重新編號。新增顯式 pose.track_gap_tolerance_ms=1e-7；預設仍為 0，歷史 recipe hash 不變。未降低骨架品質門檻，超過容差的真實間斷仍重設。新 recipe 31790b41169e814324b6f66011ddefb926181f2ccdf2771785c9a30e749d60b8。

重用舊影片前驗證只變此項、權重相同、每一個已保存時間戳的重設決策相同，161 支全部相同；若不同則重新推論，不修改舊快取。新三支影片重新推論後 Hop S1 / FallForward S5 / Walk S5 得到 49/19/62 個真實 256D 特徵，Pose/ST-GCN strict load/forward 通過。修正只用 Train/Validation smoke 發現，尚未讀取 Test 分數。44 項相關風險測試通過。原始與修訂的事前實驗鎖均保留。

## 完成結果（2026-09-17）

run `caucafall-extension-001` 已完成 97 epochs，最佳 Epoch 87，正常 Validation MSE **0.1982771373**。只使用 Train ADL 更新；5276 窗來自87/92支有效正常影片，其中 CAUCAFall 30/30支、1096窗。正常 Validation 21/22支、637窗。預設 F1 與替代平衡校準均只看 Validation，這次選到相同門檻 **0.4199422299861908**。替代 run 沿用同一 AE/Scaler/Loss，沒有第二次訓練。兩版本均在 Test 計分前凍結。

### 固定 Test：主要整段準確率

準確率 =（正確正常 + 正確跌倒）/ 所有原始序列；unknown 計入分母。TP為偵測到跌倒、FN為有結果但漏報、FP為正常誤報，unknown另列。

| Test來源 | 正確/總數 | 準確率 | TP | FN | FP | TN | unknown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CAUCAFall（新持出人物2/9） | 15/20 | **75.0%** | 6 | 3 | 1 | 9 | 1 |
| Le2i | 25/38 | 65.8% | 13 | 8 | 4 | 12 | 1 |
| UR Fall | 8/23 | 34.8% | 3 | 5 | 2 | 5 | 8 |
| 合計 | 48/81 | **59.3%** | 22 | 16 | 7 | 26 | 10 |

CAUCAFall：正常10支正確9支、誤報1支；跌倒10支警示6支、漏報3支、無法判定1支，含unknown的跌倒召回率 **60.0%**。整體跌倒召回率 **22/47=46.8%**。只在有結果的影片上計算的條件準確率為整體48/71=67.6%、CAUCAFall15/19=78.9%，不可拿來隱藏unknown。

### 與舊資料的同組比較

| 相同61支 UR/Le2i Test | 準確率 | 偵測跌倒/37 | 正常誤報/24 | unknown |
| --- | ---: | ---: | ---: | ---: |
| 舊 timing F1 | 26/61=42.6% | 26/37 | 23/24 | 9 |
| 舊 timing balanced | 31/61=50.8% | 17/37 | 9/24 | 9 |
| 新 CAUCAFall F1（平衡版相同） | 33/61=54.1% | 16/37 | 6/24 | 9 |

相較先前平衡版，同組 Test 多判對2支（+3.28百分點）、少3支正常誤報，但少偵測1支跌倒。UR/Le2i Test 已在歷史開發中多次被評估，這個比較不能當成全新獨立泛化證据；本次兩版門檻均先凍結，不依 Test 選模型。

原38支 Validation仍 **21/38=55.3%**，與先前平衡版整體正確率相同（新TP13/FP2/TN8/FN7/unknown8；舊TP12/FP1/TN9/FN8/unknown8）。加入CAUCAFall的20支Validation後為38/58=65.5%，其中CAUCAFall17/20=85%。因分母與資料變了，不把65.5%或75%直接當作原55.3%指標的提升。

### 真實展示及驗證

兩支先前選定的Validation原始影片重跑全流程：WalkS5輸出338幀、FallForwardS5輸出299幀，皆完成解碼驗證。共同可見有效端點62與18個，快取/streaming raw score差均0，狀態及events一致；Fall有1個已算分端點因當下骨架低信心而隱藏，明確保留。Fall首次警示5.4667秒只是影片內警示時間，未有原始跌倒起點標註，不能宣稱事件延遲已驗證。

- [正常展示](../outputs/caucafall-demo-walk.mp4)
- [跌倒展示](../outputs/caucafall-demo-fall.mp4)
- [Loss程式畫面](http://127.0.0.1:8766/?run=caucafall-extension-001)：97回合、最佳87、資料數量及實際曲線已在瀏覽器檢查。
- [結果JSON](validation/caucafall-results.json)、[端到端驗證](validation/caucafall-integration.json)、[完整特徵稽核](validation/caucafall-preparation.json)、[來源選擇](validation/caucafall-source-selection.json)。
- 全套 **117 passed / 1 skipped / 11 warnings（22.52秒）**；skip為原Human模型資源缺失，替代真實權重測試已通過。ruff、format、compileall通過。舊五組run及新兩組凍結hash不變。

本輪已完成，依使用者要求先提供成績，未接續新增其他資料或調參。辨識仍有明顯漏報與unknown，不能宣稱可靠跌倒偵測。資料為真人演示跌倒，非真实意外跌倒驗證；全流程CPU及新場景泛化未驗證。
