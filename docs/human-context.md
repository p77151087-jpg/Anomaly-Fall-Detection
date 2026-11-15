# Human 短時間窗實驗（2026-09-18）

使用者要求繼續優化。本輪承接四組AE仍20/23的診斷，檢查跌倒早期是否被一秒窗口中的正常歷史稀釋；這是待驗證假設。固定既有UR原始序列切分，Test不參與診斷、選模或門檻。正式Notebook／五分鐘影片仍以辨識品質優先。

## 實作與相容性

新增顯式 `skeleton.temporal_context: short_20 / short_15`，分別搭配20／15幀；原 `original_30` 保持預設及歷史recipe hash。空間布局仍Human13→14、凍結雙流512D、無分類頭分數。時間長度在recipe hash中隔離，不能混用快取或模型。原骨架觀測逐值／hash保留，不填補缺失、不合成新影格。

上游模型卷積及池化可以處理可變T，真實權重15／20幀已做CPU forward與分類頭前特徵一致性測試；但上游訓練30幀，短窗存在時間分布差異，不能當成重現同學原實驗。相同品質仍要求至少9個真實關節信心≥.2、窗口80%好影格，K3不變。短窗容許的低品質影格數隨比例縮小，不保證覆蓋增加。

修改前已保存34個舊run、所有舊config／manifest、來源70序列共280個pose／feature陣列與metadata hash，以及5個修改檔案原始碼。舊實驗腳本若鎖定當時原始碼會拒絕新的實作版本；不覆寫舊鎖。舊模型及設定可由Scorer／CLI繼續使用，本輪只讀驗收入口為`human_context.py status`。

## 訓練前有限候選

僅 `ur-human-context20-001`、`ur-human-context15-001`，每組只訓練固定128／32 AE一次。沿用sequence sampling、wd=.001、noise=0、seed42、max100epochs／patience10。Scaler／AE只用24支Train ADL；正常Validation Loss選checkpoint，完整Validation balanced sequence accuracy選門檻。原30幀基準保留比較。

Pilot共10支：Train adl-05、adl-24；Validation adl-04、adl-35、fall-09、19、18、30、25、29。先實際重抽ST-GCN特徵，骨架重用已保存的真實觀測。

事前gate：10支皆有合法連續K3分數、ADL平均有效端點coverage至少原90%、六個粗略跌倒區間皆有至少一個合法K3窗口、區間總合法窗口至少原80%、原pose array hash不变。常數零分只用來計數合法K3窗口，不能用來報分類成績。通過者才完整準備70支特徵並訓練；Test只作固定特徵抽取，選定前不評分。

選擇：候選不得降低Accuracy／F1或增加unknown；六個粗略時間案例的命中數至少基準2、早報不增加。任何原漏報fall-18／19／30若變成TP，還必須在事前區間內有新事件，避免只因正常前段誤警而得到影片級TP。通過者依全體Accuracy、F1、全體跌倒召回、較少FP排序；相同成績保留基準／較早候選。

新增粗略時段已人工看過原影格：fall-18正常截止400ms、下降區間800–1601ms；fall-30正常截止800ms、下降區間1334–1902ms。其餘四段沿用前輪。截止與下降間的空隙不計時間對錯；這不是人工逐影格真值，也不是正式警示時間Accuracy。圖在`outputs/human-context-diagnosis/`。

停止：兩個窗口、至多200epochs，不追加大小、seed、AE或校準目標。僅當Validation全體與可判定Accuracy皆≥.90、F1≥.87且時間檢查通過，才對選定凍結模型做一次Test工程對照。歷史Test已查看、Validation已重複使用，結果均不能當作全新獨立驗證。未達目標不製作新版正式Notebook／影片。

## 階段驗收

程式風險測試44 passed，10.56秒：短窗明確啟用、原fingerprint不變、空間前處理與motion數值、原始觀測不可變、因果性、品質比例、追蹤缺口、cache隔離、真實Human權重forward、原30幀相容。這是功能證據，未代表Accuracy改善。

以下是當時訓練前設計與命令紀錄；Pilot、訓練、真實影片核對均已完成，結果見後文。相關XML：`validation/pytest-human-context-related.xml`。目前原始碼另有新正規化實驗，舊source lock保持原樣；不要重跑下列已完成階段，改用最新`human_displacement.py status`或`verify_human_optimization.py`核對。

```powershell
.\.venv\Scripts\python.exe scripts/human_context.py lock
.\.venv\Scripts\python.exe scripts/human_context.py pilot
.\.venv\Scripts\python.exe scripts/human_context.py training
.\.venv\Scripts\python.exe scripts/human_context.py status
```

### Pilot完成

兩候選10支真實特徵pilot皆通過，原pose array hash完全相同。20幀ADL平均coverage比為1.004310、六區間合法窗比為0.969072；15幀分別為1.005941／0.907216。所有10支維持可形成K3，6個跌倒區間皆有合法觀測。這是`real_feature_smoke`，不是Accuracy。已依事前gate開始完整準備及各一次固定AE訓練；沒有追加候選。

### 短窗實驗完成，未採用

20幀34epochs：TP12/TN8/FP0/FN3/U0，Accuracy20/23=86.96%、F1=.8889。15幀83epochs：TP12/TN7/FP1/FN3/U0，Accuracy19/23=82.61%、F1=.8571；adl-35新增誤報，三支原漏報仍未抓到。共117epochs，依規則保留原30幀基準，未達90%，Test gate已實際拒絕。15幀六段粗略時段命中2→3不能抵銷整體Accuracy下降，不採用。

完整169 passed／1舊路徑skip，27.81秒；新Human真實權重測試有執行。三支真實影像重跑共234幀、105個可比窗口，最大分數差0、事件與可見狀態一致，H264完整解碼；原30幀fall-29推論／事件CSV與舊版逐位元相同。舊34run、來源280個cache檔案及設定hash保留。完整證據`validation/human-context-verification.json`。本階段沒有Test或正式Notebook／影片。

短窗候選已停止。新確認的逐影格正規化會消除全身位移，另立一個不同的[共用窗口尺度實驗](human-displacement.md)，不延伸短窗網格，也沒有使用Test決定新方向。最新原始碼下舊實驗腳本的source lock會拒絕；請使用`human_displacement.py status`核對目前檔案與結果。
