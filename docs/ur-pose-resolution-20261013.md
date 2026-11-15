# 骨架品質解析度對照實驗：fall-19未改善

2026-10-13。唯一事前固定候選 `ur-pose-resolution-20261013-001` 已完成真實Pose／ST-GCN特徵重建、AE訓練、Validation校準、23支影片評估及獨立核對。**不採用**：fall-19仍漏報，整體Accuracy由22/23降為16/23，其他影片及覆蓋率退步。全程僅Train／Validation；新方法Test未知。這是一次失敗的解析度控制實驗，不是辨識改善。

## 實際改動與事前契約

只把原AlphaPose人物crop從256×192改為512×384，對應真實17×128×96熱圖；保留相同FastPose ResNet50、Detector及ST-GCN真實權重，strict載入無缺鍵／多鍵、無隨機fallback。解碼使用實際crop與熱圖尺寸。其餘人物偵測384／confidence .45、tracking、track proposals、quality .2／9實測關節／80%好幀，30幀／stride1／per_frame_isotropic／512D、bounded_jitter及K3 strict `>`均沿用wide。

原24支UR Train ADL無增強、無外部資料、無Fall搬入；新Scaler僅Train擬合。AE256/64、seed42、lr .001／wd .001、batch32、sequence sampling、noise0、max100／patience10／min_delta1e-6；正常Validation8支選checkpoint，完整Validation23支以原balanced_sequence_accuracy校準一次，unknown保留於目標分母，較少FP／較高threshold解平手。設定、來源、計畫與程式在新推論前鎖定；結果之後未新增尺寸、模型、種子或品質候選。

[不可變事前計畫](ur-pose-resolution-20261013-plan.md)；`configs/ur-pose-resolution-20261013-001.yaml`。新recipe：

```text
16a7359ec150ed2ac7a742f3a5ce1a8583a5128c64c47ab7b87f9d70ef00ba4b
```

新cache、manifest、run與輸出完全隔離。配置繼承的通用`evaluation.split=test`未被本研究入口使用；入口明確遍歷原Validation並拒絕Test的raw／feature存取。原70支split不改，47支Train／Validation準備完成，23支Test保持not_prepared且無feature／Pose路徑。Test只讀舊bytes做保全hash，沒有解碼或新評估；不利用歷史Test選模。

## 真實執行與訓練

固定完整pilot：Train adl-05／adl-24、Validation adl-04／fall-19，合法512D特徵分別148／38／118／20，共324。三套真實權重、有限熱圖／512D、分類頭parity、來源時間／品質／recipe隔離通過。pilot是技術gate，不用分數或好幀增加選尺寸。

完整47支重建3,419特徵：Train24支2,094，Validation23支1,325（正常8支638）。四支完整pilot經來源／recipe／hash驗證後複用，其他43支新抽取。基準Train為3,638、Validation2,363；原影片數未增加。新fall-22沒有合法特徵，仍保留於評估。

AE實際32 epochs，最低正常Val MSE於第22 epoch：`.142064319427204`；獨立重載checkpoint MSE為`.14206431877323453`，容差內一致。沒有追加回合。新門檻`.3137603998184204`，基準`.5341113805770874`；各用自己的Train Scaler、AE與Validation校準值。兩個recipe的Loss／MSE尺度不同，不能用數字上升或下降宣稱準確率改善；分數也不是跌倒機率。

CUDA裝置為RTX5060 Laptop GPU。訓練保留CuBLAS確定性警告：seed42固定，但未設定CUBLAS_WORKSPACE_CONFIG，不能宣稱GPU逐位可重現。未為消除警告追加重訓；正式checkpoint與實際分數已保存並獨立重載核對。

首次報告因缺`protocol_id`中斷，模型／一次門檻校準已完成。失敗log與部分輸出保留在證據目錄；新增`resume_ur_pose_resolution_validation.py`僅補原協定metadata，複用凍結門檻。已鎖原程式、模型、Scaler、整個run所有hash不變；新增訓練0、重校準0。沒有掩蓋此次工程失敗。

## Validation辨識比較與分母

| 指標 | 原wide 256×192 | 新512×384 |
|---|---:|---:|
| 含unknown整體Accuracy | 22/23＝95.65% | 16/23＝69.57% |
| 可判定二元Accuracy | 22/23＝95.65% | 16/22＝72.73% |
| 可判定Fall F1 | 28/(28+0+1)＝.9655 | 18/(18+1+5)＝.7500 |
| Fall Precision | 14/14＝1.0000 | 9/10＝.9000 |
| 可判定Fall Recall | 14/15＝.9333 | 9/14＝.6429 |
| 全體Fall Recall（含unknown） | 14/15＝.9333 | 9/15＝.6000 |
| TP／TN／FP／FN | 14／8／0／1 | 9／7／1／5 |
| unknown | 0/23 | 1/23（Fall） |
| 可判定影片coverage | 23/23 | 22/23 |
| 合法唯一窗口端點coverage | 2287/2710＝84.39% | 1325/2710＝48.89% |
| 全track窗口紀錄／事件 | 2917／14 | 2799／12 |

Accuracy下降26.09個百分點；F1下降.2155。窗口coverage分母是唯一排程端點，不是track紀錄或影片數。F1／Precision／二元Recall依可判定影片；整體Accuracy與全體Fall Recall保留unknown的影響。不能只報較高的16/22。

混淆表欄位依序為Normal／Suspected fall／Unknown，行為ADL／Fall：

| 模型／真值 | Normal | Suspected fall | Unknown |
|---|---:|---:|---:|
| wide ADL（8） | 8 | 0 | 0 |
| wide Fall（15） | 1 | 14 | 0 |
| 新ADL（8） | 7 | 1 | 0 |
| 新Fall（15） | 5 | 9 | 1 |

新增FP為adl-35；FN為fall-19／13／29／07／25；unknown為fall-22（no_legal_event_score，0合法特徵）。原TP退步五支，其中四支變FN、一支變unknown；不是把unknown當正常。原正常ADL其他七支仍判對。

版本分開：demo仍`ur-sample-human-s1-001`（歷史Val20/23／F1 .8889）；研究最高參照wide22/23／.9655；已完成20幀isotropic `ur-short20-isotropic-20261012-001` 為20/23／.8889；motion融合22/23持平；前六SVM／兩AE最高21/23／.9333。本輪沒有重跑這些網格或改寫舊結果。[歷史方法比較](ur-normal-methods-20261012.md)。

## fall-19：有效觀測與低分並存

人工補註是完整姿態階段：upright f1–32、transition f33–61（1067–2002ms）、ground f62–100（2035–3303ms）；精確onset／impact仍null。窗口採當時最後已觀測原幀對齊，不取未來影格。下表分母為該phase排程端點，因此ground39原幀對應38排程端點。

| fall-19項目 | 原wide | 新512×384 |
|---|---:|---:|
| 全片合法窗 | 23/71 | 20/71 |
| transition合法窗 | 21/29 | 18/29 |
| ground合法窗 | 0/38 | 0/38 |
| 最後合法窗口可用時間 | 1766.67ms | 1666.67ms |
| transition最高Raw MSE | .18390694 | .21962853 |
| transition最高K3 MSE | .13463649 | .17581737 |
| 各自門檻 | .53411138 | .31376040 |
| transition／ground事件 | 0／0 | 0／0 |
| 原100幀有Pose | 95/100 | 93/100 |
| confidence規則good | 49/100 | 42/100 |
| transition confidence good | 19/29原幀 | 12/29原幀 |

有連續合法窗且可形成K3，新Raw與K3最高值仍都低於自己的門檻：**不是單純沒有骨架，也不是只有K3把跨門檻Raw擋掉**。新fall-19最高K3低於六支正常Val影片的K3最高值；在這套固定分數中以單一門檻救回它會把這六支正常影片判異常。這只是現有分數排序的診斷，沒有另選門檻或跑追加候選。

後段ground無合法分數，缺值保留null而非0；放大crop未補回觀測，最後合法時間反而提前100ms。原人工指出的f44–51錯點但confidence good證據仍成立；舊pose_check僅屬原Pose，新Pose沒有人工點位真值，不能拿舊標籤算新骨架正確率。並排RGB／Pose的AI觀察顯示部分點仍錯位或低信心，落地後身體多在畫面下緣外；AI記錄另存，未充作人工確認。

已支持的結論是：**同權重提高crop解析度這個候選未改善fall-19，並使合法覆蓋率與辨識退步**。原低分、後段缺有效觀測兩種現象並存。未證實唯一漏報原因，也未證明所有骨架修正都無效。原分辨率訓練尺度與放大推論不匹配、局部點位／出框、512D表示或重建誤差分離不足仍屬待驗證假設。本輪七類分類不作異常成績，沒有新增分類評估。

![fall-19凍結分數、門檻與人工姿態階段](../outputs/ur-pose-resolution-20261013-001/figures/fall19-score-comparison.png)

## 影片判對與警示時間分開

原23支首次警示、prediction與coverage逐片保存在`video-comparison.csv`，完整事件保存於兩套`events.csv`。新12事件包含正常adl-35誤報，不能把事件數當成正確跌倒數。

六片既有AI粗時段的事件起點命中由4/6降為2/6；有當前good骨架警示的影片數仍2/6，但由原fall-29／25變為fall-09／18，並非原影片全保留。粗時段提前事件0→1。這是診斷／工程展示guard，非教師指定計分或精確時間GT。

| 影片 | 原首次警示ms | 新首次警示ms | AI粗phase內事件起點：原→新 |
|---|---:|---:|---|
| fall-09 | 5900.00 | 4800.00 | 0→1 |
| fall-19 | 無 | 無 | 0→0 |
| fall-29 | 2566.67 | 無 | 1→0 |
| fall-25 | 2466.67 | 無 | 1→0 |
| fall-18 | 1433.33 | 1233.33 | 1→1 |
| fall-30 | 1866.67 | 1100.00 | 1→0（新增粗提前） |

fall-09局部時序進入粗phase、fall-18較早且當前good，均如實記錄；fall-30仍影片TP但新警示早於粗phase。其他TP首次警示提前／延後也不能在缺精確人工時間GT時稱延遲改善。[粗事件對照CSV](../outputs/ur-pose-resolution-20261013-001/coarse-event-comparison.csv)。

## 驗證、採用與限制

- **synthetic**：40項相關風險測試通過（16新／24原），覆蓋單一設定、recipe/cache、來源／phase／track因果、Test／Fall Train拒絕、Train-only Scaler、正常Val checkpoint、Validation校準、K3及缺失／事件規則。相關ruff check與format check通過。
- **real_feature_smoke**：真實RGB、strict Pose／ST-GCN pilot與47支cache、24正常Train AE、23Val正式CSV已實際完成。獨立核對Train Scaler完全相同、正常Val checkpoint，重載兩AE共3,688分數、5,716窗與26事件全部欄位、23支各自指標／分母／混淆表、23個門檻候選排序及8,643逐原幀Pose對照通過。
- **保全／防覆寫**：25,138舊asset SHA全部不變；五份修改前契約原副本吻合。六次實際重複stage拒絕，238新run／cache／輸出檔before／after不變。報告repair前後整run不變。原始資料、權重、快取、human匯出、舊設定及歷史Test保留。
- **real_end_to_end**：本輪未跑Test／demo完整部署驗收；明確不將完整Train／Val研究流程充作原Test E2E契約通過。Test23未準備，影像／特徵解碼0、新Test評估0。
- Graph Tier2先search／trace／snippet，再coverage；沒有錄得parse gap，但屬best effort。raw/cache/run/output/upstream等設計排除，已讀精確來源或核hash；部分heuristic邊誤連歷史副本，未據此作完整caller／唯一因果宣稱。

事前採用條件均未滿足，`decision.json`保存逐條結果。最高wide比較基準與原demo都保留，候選停止。原Validation很小、反覆選模、自訂序列切分非人物隔離、預訓練分類器含Fall資訊的限制保留；不宣稱新獨立盲測、不換算課程分數。新版正式Notebook／五分鐘影片繼續暫緩，沒有公開上傳。

本輪不需要重做100筆人工標註。若未來要驗證某個新Pose是否點位真的正確，才需要對固定少量並排樣本新增人工判斷；在沒有新候選與事前契約前不要求全面重標。

## 成果與可直接執行的命令

核心新增`src/fall_detect/evaluation/pose_comparison.py`；實驗入口`scripts/ur_pose_resolution.py`、報告repair入口`scripts/resume_ur_pose_resolution_validation.py`、獨立只讀驗證`scripts/verify_ur_pose_resolution.py`、可視對照`scripts/render_pose_resolution.py`及`tests/test_pose_resolution.py`。未改原模型／品質／事件核心。

- run：`runs/ur-pose-resolution-20261013-001`，checkpoint／scaler／threshold／history／sources／frozen hashes。
- cache：`data/cache/ur-pose-resolution-20261013-001`，新47支features／poses及原split的Test未準備狀態。
- 結果：`outputs/ur-pose-resolution-20261013-001`，baseline／candidate正式CSV、metrics、事件、混淆圖，逐幀Pose／逐片影片／粗事件比較與分數圖。
- 證據：`docs/validation/ur-pose-resolution-20261013-001`，plan lock／before／pilot／prepare／training／validation／verification／decision／rejection／graph／synthetic XML／原契約／失敗log／report repair／AI可視檢查。

PowerShell只讀狀態與獨立重算；後者會重載Train／Val與舊asset hash，耗時數分鐘，**不新增訓練、校準或Test評估**：

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
& .\.venv\Scripts\python.exe -X utf8 scripts/ur_pose_resolution.py status
& .\.venv\Scripts\python.exe -X utf8 scripts/verify_ur_pose_resolution.py
Get-Content -LiteralPath outputs/ur-pose-resolution-20261013-001/candidate/metrics.json -Encoding utf8
Import-Csv -LiteralPath outputs/ur-pose-resolution-20261013-001/video-comparison.csv
```

本輪已實際執行`inventory → pilot → prepare → train → validation → report-only resume → verify --save`。完成stage再次執行會拒絕，不能在相同目錄重訓／重校準／覆蓋。原採用規則及計畫hash不改寫。
