# 可見關節診斷與身體比例對照（2026-09-18）

本輪依使用者「好，按你的做調整」已完成診斷、實作、49回合訓練與真實影片核對。新候選 **Validation影片級Accuracy由20/23=86.96%提高到21/23=91.30%，F1由.8889提高到.9286**。但新補fall-30只有窗口事件、沒有可見警示，未通過事前採用條件；原預設模型保留，新候選另存，不能宣稱已完整達到課程要求。

## 診斷與限制

已查看50張取樣原始影格、同時間骨架接觸表，以及9張640×480關鍵影格。另存跌倒起始／接地時間範圍、可見與出框部位、33個帶容許半徑的粗略可見關節區域。這是AI目視工程標註，非官方GT、非獨立盲測；曾查看骨架覆蓋，不能用來聲稱正式關節精度。未評估的關節不視為可見，不補造遮擋／出框座標，不投入訓練。

三支影片的最後合格當前Pose為fall-19 1668ms、fall-18 1368ms、fall-30 1635ms。總計出框關節觀測21／10／34次，其中信心≥.2為1／6／17次；這是關節觀測次數，不是影格數。下降時動態模糊，接地後頭部／上身部分出框。這能證實觀測受限，不能證實所有漏報只因骨架。

程式與數值確認原 `per_frame` 分別以每幀X／Y範圍縮放，所有非退化骨架外框變成1:1；原像素寬高比例消失。低信心座標也參與範圍計算。這些是已確認的機制；「保留比例能提高跌倒分離」仍是假設。此次不追加低信心插值或品質放寬。

標註：[可見區域與時間範圍](validation/human-keyframes-annotations.json)；[逐關節／幾何診斷](validation/human-keyframes-diagnosis.json)；`outputs/human-keyframes/`原始／Pose／review接觸表。新時間範圍與舊工程時段分開保存，選模仍用原鎖定時段，沒有事後移動事件正解。

## 事前規則

僅 `ur-human-isotropic-001` 一個候選。每幀依像素骨架外框中心平移，X／Y共同除以最大像素邊長；保留比例與角度。原觀測、信心、track、30幀／stride1、品質.2/9/.8、K3、真實512D ST-GCN全部不變。原 `per_frame`／`per_window`運算及預設hash保留，新模式獨立recipe/cache/run。此模式與預訓練輸入分布不同，七類頭不直接套用，demo auto沿用既有fallback顯示AE；原模型的七類展示不变。

8支Train／Validation pilot須保留相同Pose arrays、窗口排程和有效性、合法K3，且真正512D特徵有變化才訓練。固定Train24ADL Scaler／AE128/32、sequence sampling、seed42、wd.001、noise0、最多100epochs/patience10；正常Validation選checkpoint，全Validation校準。

只在整體普通Accuracy嚴格提高、F1不降、unknown不增、舊粗略時段命中不減／早報不增、新補FN於原時段有當前合格Pose的可見AE警示時採用，平手保留基準。只跑此一候選，不追加seed／窗口／confidence／threshold搜尋。本診斷實驗不追加Test；历史Test已看、Validation反覆使用不獨立。正式Notebook／五分鐘影片依使用者要求仍待辨識品質達標。

40舊frozen runs、8450舊cache檔案、設定與manifest已另存hash；三個修改前核心檔案亦封存。[執行前快照](validation/human-isotropic-before.json)。34項相關測試已通過，僅表示方法實作與相容性，不代表辨識提升。

## 實際結果

8片`real_feature_smoke` pilot通過：Pose arrays及窗口排程完全一致，真正512D特徵有變化，每片保有合法K3。完整準備流程另存70序列的新特徵；Test部分只是前處理，沒有計算Test異常分數／指標或參與選擇。70序列觀測Pose arrays、來源hash和窗口有效／無效排程已逐一核對一致。

Train24ADL／3638有效窗訓練49epochs，正常Validation最佳Loss=.1151348554，epoch39；Scaler重算與凍結結果完全一致。這是新特徵空間的Loss，不能直接用Loss下降宣稱辨識改善；辨識提升依下面正式Validation逐影片輸出。

| Validation，8ADL＋15Fall | 原基準，預設保留 | 保留比例候選，另存 |
| --- | --- | --- |
| TP／TN／FP／FN／unknown | 12／8／0／3／0 | 13／8／0／2／0 |
| 可判定二元Accuracy | 20/23＝86.96% | 21/23＝91.30% |
| 包含unknown全體Accuracy | 20/23＝86.96% | 21/23＝91.30% |
| 可判定二元F1 | .8889 | .9286 |
| 全體跌倒召回 | 12/15＝80% | 13/15＝86.67% |
| 窗口coverage | 84.39% | 84.39% |

兩者unknown0，可判定／全體Accuracy分母均23；並不等於每一影格可判定。新候選門檻.4529072046（嚴格大於），最高合法K3：fall-19=.1719465852、fall-18=.4494668841、fall-30=.5397668481。剩餘漏報fall-19／18，未因接近門檻而追加調整。舊困難正常adl-35=.4172827601、沒有誤報。這是窗口重建誤差，非跌倒機率。

六個既有粗略時段：事件命中2→4、早報仍0。fall-09首次可見警示5.9→5.0秒，原時段內僅5.000／5.033／5.067秒三張畫面，之後會因品質不足中斷，不能說警示已穩定。fall-30於1.8667秒形成K3事件，但當前Pose不合格，**全片可見FALL ALERT為0幀**，仍有UNKNOWN。因此候選未通過「每個新補FN須有時段內可見警示」的事前規則，保留為最高影片級Validation候選，未取代原預設。新AI目視時間範圍沒有拿來移動舊驗收時段。

證據：[事前規則](validation/human-isotropic-lock.json)、[pilot](validation/human-isotropic-pilot.json)、[Validation與選模](validation/human-isotropic-validation.json)。原分數及新候選的metrics／sequences／windows／events均位於`outputs/human-isotropic-validation/<run>/`；新run為`runs/ur-human-isotropic-001/`。

## 驗證與成果

完整 **215 passed／1 skipped／12 warnings**，34項相關測試通過；skip僅歷史預設資源路徑，新Human真實權重有執行。原`per_frame`與`per_window`前處理對封存程式逐元素一致；新模式保留幾何比例、輸入不變、無未來資訊、confidence不升高、pose reuse與recipe隔離通過。舊模型fingerprint不變，未重写舊實驗source lock；後續核對用本輪命令。

六支`real_end_to_end`診斷影片共 **888幀／559個可比分數差0**，狀態／事件與快取一致，H264完整解碼並目視。原基準fall-09的AE／事件／ACTION CSV與改版前逐位元一致。保留比例候選使用`simple` AE狀態，原七類頭只適用原前處理，沒有將不相符特徵接回七類頭或以其成績替代AE。[整合證據](validation/human-isotropic-integration.json)。

40舊run、8450舊cache、設定／manifest已完整hash核對；重跑lock／pilot／training的拒絕均實測。9張關鍵影格／33個粗略可見區域均已記錄原始來源hash，未送入Scaler／AE訓練。完整標註仍不足，目視區域不應換算成官方關節精度。[最終驗證](validation/human-isotropic-verification.json)、[pytest](validation/pytest-human-isotropic-full.xml)。

- 新候選真實短片：`outputs/human-isotropic-streaming/ur-human-isotropic-001/{adl-35,fall-09,fall-19,fall-18,fall-30}-preview.mp4`。
- 同時刻對照：`outputs/human-keyframes/fall09-5seconds-comparison.jpg`，左原AE NORMAL，右新AE FALL ALERT；不是用原七類FALLING當AE警示。
- 原模型對照：`outputs/human-isotropic-streaming/ur-sample-human-s1-001/fall-09-preview.mp4`。

本輪有限候選已完成並依停止條件結束，無執行資源阻塞。仍未完成：新候選可靠的fall-30可見警示、fall-19／18漏報改善、完整獨立GT、未接觸的新測試證據，以及教師評估單位／切分確認。歷史Test已看、反覆Validation的偏差仍存在；**91.30%是本工程固定Validation的影片級數字，不是Test成績、課程正式得分或每影格準確率**。無新Test／正式Notebook／五分鐘影片／公開上傳，舊成果全部保留。

## 操作

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe scripts/human_isotropic.py status
.\.venv\Scripts\python.exe scripts/verify_human_isotropic.py
.\.venv\Scripts\python.exe scripts/human_isotropic.py verify --deep
.\.venv\Scripts\python.exe -m fall_detect demo --run runs/ur-human-isotropic-001 --input data/raw/urfall/fall-09 --output outputs/isotropic-review.mp4 --overlay auto
```

前三條只核對既有證據；deep逐一hash8450舊cache，耗時較長。最後一條另存新候選診斷短片，若輸出已存在會拒絕覆寫。原預設展示仍可指定`--run runs/ur-sample-human-s1-001`使用七類ACTION＋AE雙欄。
