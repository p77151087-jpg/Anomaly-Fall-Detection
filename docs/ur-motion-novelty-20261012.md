# UR-only全域動態正常近鄰側分支（2026-10-12）

**已完成單一候選的真實正常資料擬合與23支Validation；Accuracy與時間均未改善，不採用。** 原wide仍22/23＝95.65%／Fall F1 .9655，demo仍`ur-sample-human-s1-001`；沒有重訓舊run或擴充前輪網格。新方法Test未知、未準備／未評估。

## 實際方法與鎖定

先閱讀根契約、目前結果、前輪方法比較及舊位移實驗，Graph Tier2探索human_preprocess、window／cache／事件／模型依賴再查coverage。原isotropic逐幀置中及縮放會移除全域平移／尺度變化；前輪raw XY診斷也已移除這些資訊。這是數學資訊差異，尚不是漏報原因證明。舊per_window曾19/23且失敗，本次不重跑它，也不改ST-GCN預訓練輸入。

事前[計畫](ur-motion-novelty-20261012-plan.md)固定一個`ur-motion-novelty-20261012-001`：原30幀isotropic 512D與wide256/64 AE保持凍結，原30×13×3像素骨架另生按時間排序的30×3中心位移／log尺度軌跡（90D），以第一幀中心及最大邊長定錨。沒有Fall標籤／七類分類／手定下降規則、鏡像／噪音增強或新Pose。

原24 Train ADL、3638合法窗只fit90D Scaler及5近鄰bank。異常分數為標準化90D中最近5個Train窗的平均Euclidean距離／sqrt90。Train定標排除整支來源影片，避免自鄰居及重疊窗口；Validation只查完整Train bank。wide Train分數p95=0.109420693666，motion整影片排除Train p95=0.954710487269，融合固定為兩路各除Train p95後取max。分數非機率；沒有依Val改k或權重。bank實際擬合完成，新增神經網路epochs=0，不冒稱重訓AE。

新method recipe `22d580e0a4f315c1c088f2c8bc22f54ca850ae2ecf7cb1c472d0d0005e0202f9`、motion recipe `7e300103e7ddb54d2c059e81818823fb8fca52a104d9165c2ad4e61ee68ab52d`；原512D feature recipe `713ee59a0853f20641cb7f664df2920359baa0734807acfa5cf86fb87a4e2f5b`保持原樣。90D不是ST-GCN512D快取，原Pose／feature bytes複製到獨立目錄，衍生原窗口／90D／來源列另存NPZ與JSON。原70split保留，Train24 ADL、Val8 ADL／15Fall，Test23not_prepared且無路徑，Test入口實測拒絕。

固定adl05／adl06／fall19各第一合法窗口pilot通過：真實Human ST-GCN strict載入、512D有限與原cache最大差0、head parity通過；90D有限、track／品質／時間相同。通過後重建全部47片共6001個原合法窗口；原始影片數沒有增加。Pose沿用觀測，沒有新RGB Pose抽取。

## Validation正式比較與分母

門檻只用完整Validation依含unknown整体Accuracy→Fall F1→全體Fall Recall→少FP→較高門檻選，strict >、K3不變；新門檻4.88126480178。原wide門檻.5341113805770874保留，分數單位不同不能直接比較數值。

| 候選 | 含unknown Accuracy | 可判定Accuracy | Fall F1 | Precision | Recall | TP/TN/FP/FN/U |
| --- | --- | --- | ---: | ---: | ---: | --- |
| wide參照 | 22/23＝95.65% | 22/23＝95.65% | .9655 | 1.0000 | .9333 | 14/8/0/1/0 |
| 動態正常近鄰融合 | 22/23＝95.65% | 22/23＝95.65% | 0.9655 | 1.0000 | 0.9333 | 14/8/0/1/0 |

兩種Accuracy分母均23，unknown0/23；Precision14/14、Recall及全體Fall Recall14/15、F1=28/29。2×2 `[[8,0],[1,14]]`，包含unknown2×3 `[[8,0,0],[1,14,0]]`。FN仍fall-19，FP0；其他影片預測無改變。合法端點coverage2287/2710＝84.3911%，原分母不縮；CSV2917列包含多track。影片可判定23/23不代表逐幀皆可判定。

## 漏報與事件診斷

fall-19粗phase仍9合法窗口／9合法K3、7當前良好Pose端點。phase motion距離最高0.783931464，正常Val motion p95=1.0189979；除Train定標後motion最高0.821119569、AE最高1.68073274、融合最高1.68073274，均未達門檻4.8812648。同融合分數降門檻救回fall-19至少誤報8支正常Val。這個候選没有補到漏報，不把它歸因於完全沒骨架。

23影片首次警示時間改變0支；2917窗口狀態改變0列，原14及新14事件的track／確認起點／最後活動／結束時間／原因逐項相同。六粗phase事件命中仍4/6、當前良好Pose警示影片仍2/6、粗正常前段早報0。fall09首次仍5.9秒，沒有本輪時序改善。事件peak分數因融合單位改變，不要求數值與舊MSE相同。fall18／30的正式TP保留，當前良好Pose可見警示不足另列展示限制。

已證實：正規化數學上移除全域運動；新側分支實際保留這些訊息但本次辨識／時序未改善。**未證實**：全域運動損失是fall-19漏報的主要原因、Pose正確或時段真值完整。低品質／出框、前段正常稀釋、正常與跌倒觀測相似及表示不足仍是假設；不能從單一失敗候選推論所有動態方法無效。前輪自監督支持條件仍未取得新證據，本輪沒有盲目微調。

![fall19兩路與融合分數](../outputs/ur-motion-novelty-20261012-001/fall19-motion-scores.png)

## 採用、驗證與限制

事前辨識排名要求嚴格勝過wide，持平不採用；新候選未通過。原研究參照及demo預設保留，展示檢查與教師成績分開。一次候選／一次Validation校準已停止，没有追加k／seed／門檻或品質事件調參。

synthetic：60相關風險測試全過，包括新的12項動態資訊、退化／有限、排除來源影片與分數融合測試，及既有cache／Train-only／Validation校準／K3／scopedTest。首次49passed／11setup errors源於沙箱pytest預設暫存目錄無法建立；換專案內獨立暫存目錄後同一60案例全過，初次XML保留，不把環境錯誤隱藏或稱首次全綠。沒有修改模型或實驗設定來通過測試。

real_feature_smoke：嚴格真實3窗pilot、47片觀測重建、正常bank擬合、23Val正式CSV。獨立NumPy重算90D／Train Scaler／最近5距離（與production scipy.cdist差容限1e-12），模型重載及strict分數逐值相同；18003個兩路／融合分數及3638個排除整影片Train分數通過。sklearn獨立核對23CSV指標／矩陣／分母，獨立手動事件機核對2917窗與14完整事件，門檻網格與最高排序吻合；所有Test入口被拒絕。

24620舊檔hash全部不變，原五契約副本hash吻合；包括raw／外部封存／權重／舊run／前輪完整方法與augmentation證據。原契約進度合法新增，本輪修改前版本另存，未改寫歷史lock或報告。近鄰bank、Scaler、分數、門檻、正式CSV及新來源/hash均另存。

real_end_to_end未跑，沒有新RGB Pose、demo接線或Test；新模型未接舊AE展示loader，未部署。Test未知且歷史Test已查看，不能稱新盲測；Val反覆選模／只有23片、非人物隔離、Human既有含Fall預訓練來源限制保留。AI粗phase與Pose confidence非人工GT、相鄰重疊窗非獨立影片，近鄰bank受正常動作覆蓋與推論計算量限制。正式Notebook／五分鐘影片繼續暫緩、舊成品保留、不公開上傳、不換算課程分數。

## 檔案與只讀命令

run `runs/ur-motion-novelty-20261012-001`；私有cache `data/cache/ur-motion-novelty-20261012-001`（原Pose／512D與motion分開）；正式metrics／sequence-window-events CSV與對照在`outputs/ur-motion-novelty-20261012-001`。事前lock／pilot／保全／獨立核對／風險XML／圖／最終QA在`docs/validation/ur-motion-novelty-20261012-001`。

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe -X utf8 scripts/ur_motion_novelty.py status
.\.venv\Scripts\python.exe -X utf8 scripts/verify_motion_novelty.py
Import-Csv outputs/ur-motion-novelty-20261012-001/candidate-comparison.csv | Format-Table run, overall_accuracy, fall_f1, FP, FN, unknown
```

status及verify為只讀，不重訓／不跑Test；inventory／prepare／fit及report完成後拒絕覆寫。完整人工onset／可見骨架標註仍缺，下一研究應先核對實際跌倒觀測與缺失時段，避免持續在同一小Validation上盲目搜索。
