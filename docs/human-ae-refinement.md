# Human 特徵的正常 AE 優化（2026-09-18）

使用者要求繼續優化，延續 `ur-sample-human-s1-001`，本輪訓練前先完成 Train／Validation 診斷並鎖定四個候選。舊 run、設定、骨架與特徵不覆寫；不使用 Test 選模。

## 診斷與假設

基準 Validation TP12／TN8／FP0／FN3／unknown0，Accuracy20/23=86.96%、F1=.8889。漏報的 fall-19、fall-18、fall-30 皆有可判定的正常窗口，所以影片級 unknown 為0；不代表整段皆可判定。

| 漏報 | 最大 K3 event score | 有效端點／排程端點 | 已證實限制 |
|---|---:|---:|---|
| fall-19 | .124607 | 23/71 | 跌倒時部分身體離開畫面，低信心與觀測中斷；後段骨架不足 |
| fall-18 | .242297 | 16/36 | 第47影格起無骨架觀測；倒地後部分身體在畫面外 |
| fall-30 | .325666 | 26/41 | 一個raw score達.799992，但未連續三窗超過.688018門檻；接著品質／追蹤中斷 |

基準已保存的門檻候選，最高全體 Accuracy 仍20/23；單純降門檻不足以提高此數字。三個原始骨架抽幀已人工檢視，見 `outputs/human-optimization-diagnosis/`。沒有關節真值，不能報骨架Accuracy。前輪 fall-25／fall-29僅1／2個可見警示影格，是當前影格品質遮罩造成；窗口80%合格與當前骨架合格是不同條件。

待驗證假設：減小AE瓶頸可能保留早期下降動作的重建誤差；增加容量或去噪可能降低正常活動泛化誤差，讓Validation門檻更有效。這些不是已證實的漏報原因，也不能修復相機外的身體。保留UNKNOWN與K3，不用舊骨架補出假的新觀測。

## 訓練前固定候選與停止條件

全部重用同一Human512D特徵。Train24支ADL、3638窗；Scaler／AE只用Train正常資料，乾淨正常Validation Loss選checkpoint，完整Validation選門檻。固定seed42、sequence sampling、weight decay .001、至多100epochs、patience10、原始split、30幀／stride1、品質與K3。預訓練特徵曾學過Le2i跌倒；不是全流程無監督。

| 新run | hidden／latent | input noise std | 目的 |
|---|---|---|---|
| ur-human-ae-compact-001 | 64／8 | 0 | 限制整體重建容量 |
| ur-human-ae-bottleneck-001 | 128／16 | 0 | 單獨縮小瓶頸 |
| ur-human-ae-wide-001 | 256／64 | 0 | 改善正常活動重建的對照 |
| ur-human-ae-denoise-001 | 128／32 | .1 | noisy Train輸入、乾淨目標；Validation保持乾淨 |

門檻仍使用balanced sequence accuracy；這不等於課程Accuracy。候選必須維持四個粗略時間案例至少2段區間命中、0次早報、Accuracy／F1不下降及unknown不增加；再按全體Accuracy、F1、全體跌倒召回、少誤報選擇，相同時保留基準／較早候選。此時間檢查不是正式警示時間成績。

本輪只跑四組，總計至多400epochs，不追加seed／模型組合。Validation全體與可判定Accuracy皆≥90%、F1≥.87且時間檢查通過，才對選定凍結模型做一次Test工程對照；未達即停止本輪。歷史Test已查看、Validation已反覆使用，均不能重新宣稱獨立測試。正式Notebook與五分鐘影片仍等待辨識品質。

## 執行紀錄

已完成四候選，共 **170 epochs**；全部都是 TP12／TN8／FP0／FN3／unknown0。可判定二元Accuracy與含unknown全體Accuracy分母均23，皆 **20/23=86.96%**，F1 **24/(24+0+3)=.8889**，全體跌倒召回12/15=80%。四個粗略時段案例均維持2段命中、0次早報。**這輪沒有辨識改善**，相同成績依事前規則保留 `ur-sample-human-s1-001`。

| run | 實際epochs | 最佳正常Validation Loss | 凍結門檻 | Accuracy／F1 |
|---|---:|---:|---:|---|
| 原Human stride1基準 | 38（前輪） | .137535 | .688018 | 86.96%／.8889 |
| compact | 41 | .214978 | 1.151504 | 86.96%／.8889 |
| bottleneck | 38 | .138654 | .628895 | 86.96%／.8889 |
| wide | 38 | .135039 | .582585 | 86.96%／.8889 |
| denoise | 53 | .132722 | .637285 | 86.96%／.8889 |

去噪模型的正常Loss較低，沒有轉化成額外跌倒召回；不能把訓練Loss下降當成辨識進步。新run保存在 `runs/ur-human-ae-{compact,bottleneck,wide,denoise}-001/`，有各自完整設定、Loss、checkpoint、scaler、門檻與hash。原run不變。

## 驗證與證據

- `real_end_to_end`：使用既有真實Human抽取的特徵，實際訓練／校準四組，再重新計算23支Validation的所有正式指標，相符。此次沒有重新抽取骨架或重跑影片，前輪真實影像端到端證據繼續適用於未變的基準；不把單元測試稱為真實辨識。
- `synthetic` 風險測試：訓練隔離、cache、sampling、去噪、校準、K3／unknown及中斷保護等 **23 passed、0 skipped，11.23秒**。XML：`validation/pytest-human-ae-refinement.xml`。核心訓練、姿態及事件程式沒有修改，因此執行相關測試，未重跑整套160項。
- 30個舊run、4個新run及既有設定／權重／來源index hash不變。四組新Scaler與只用Train24支ADL、3638窗重新擬合的mean／scale逐值完全相同；訓練來源hash與Train ADL一致。
- 實際執行並通過「再次lock拒絕」、「完成後training拒絕」、「90% gate未達拒絕Test」；未產生本輪before-test／test證據。此輪無新增Test，也没有依Test選模。
- 診斷：`validation/human-ae-refinement-diagnosis.json`；訓練前契約：`validation/human-ae-refinement-lock.json`；結果：`validation/human-ae-refinement-validation.json`；逐項核對：`validation/human-ae-refinement-verification.json`。原始骨架抽幀與分數／品質圖在 `outputs/human-optimization-diagnosis/`；圖中空白包含暖機或缺少合法分數，不表示該整支影片都是unknown。

## 下一步及未完成事項

本輪有限候選已跑完，沒有外部權重、套件或硬體阻塞。Accuracy90%仍未達；老師尚未確認的逐幀／逐影片與split協定亦仍不確定，不能把本Validation数字直接等同課程成績。歷史Test已看過、Validation已多次參與選模，不能宣稱獨立泛化成績。

下一輪優先檢查Human姿態估測在下降動作中的裁切／方向與時間上下文，先做少量Train／Validation pilot，再決定是否重建recipe與訓練。判斷門檻應是「保留新的真實骨架觀測與合法窗口，同時不惡化正常案例」，不能靠放寬品質、補造畫面外骨架或延長顯示來宣稱Accuracy提升。這是尚未執行的方向；現有結果只證明本輪四組AE設定沒有改善，不代表所有方法都已試盡。

現有短促警示的原因已診斷，這輪未修改顯示或推論；目前畫面仍保留當下UNKNOWN。正式Notebook、五分鐘影片及公開上傳未更新。需要額外人工標註協助時，最有用的是標出Validation跌倒起迄與可見關節，供姿態與警示時間驗證；無須以新拍影片替換既有UR測試資料。

## 操作

```powershell
.\.venv\Scripts\python.exe scripts/human_ae_refinement.py lock
.\.venv\Scripts\python.exe scripts/human_ae_refinement.py training
.\.venv\Scripts\python.exe scripts/human_ae_refinement.py status
```

完成後`lock`／`training`會拒絕覆寫，`status`為只讀重驗hash。Test gate未通過時`test`會直接拒絕且不讀取Test特徵。

最終圖表目視與Ruff／Graph紀錄：`validation/human-ae-refinement-visual-review.json`。已檢視三張骨架抽幀及[漏報品質／分數圖](../outputs/human-optimization-diagnosis/missed-cases-reviewed.png)，保留初版圖表；新圖標題明示空白只是無合法分數，並非把暖機誤稱unknown。
