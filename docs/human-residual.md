# Train 正常重建誤差加權實驗（2026-09-18）

本輪已實作並校準兩個候選，**沒有提高 Accuracy**。依事前選擇規則保留 `ur-sample-human-s1-001`；沒有追加 Test，正式 Notebook 與五分鐘影片仍依使用者要求暫停。

## 診斷、假設與有限候選

固定 UR 原始序列切分，Train 24 支正常影片、3638 個有效窗口。已量到的現象：AE 各特徵維度的 Train 平均平方重建誤差最大／最小約 56 倍；位置分支均值 0.042503、動作分支 0.086715。這說明原始平均 MSE 對不同維度的貢獻不均，**不證明它是漏報原因**。

看候選 Validation 前保存[診斷](validation/human-residual-diagnosis.json)、[原始檔案快照](validation/human-residual-before.json)、[兩候選與選擇／停止規則](validation/human-residual-lock.json)，再凍結[實作及設定 hash](validation/human-residual-implementation.json)。候選只有：

1. `train_channel_mse`：以每維 Train ADL 平均平方誤差的倒數加權。
2. `train_stream_mse`：前 256 維位置、後 256 維動作各用一個 Train ADL 誤差尺度。

尺度下限固定為 `max(1e-12, 0.1 × 正值通道誤差中位數)`，權重平均正規化為 1；分數為 `mean(weight * squared_reconstruction_error)`，仍是 AE 加權重建誤差，**不是跌倒機率**。權重使用 Train 全部有效正常窗口估計，沒有用 Validation／Test 擬合。

本輪沿用原 AE 最佳正常 Validation checkpoint、Scaler 和 38 回合訓練歷史；**新增 AE 訓練回合為 0**，不可將複製的 Loss 曲線說成重新訓練。新 run 明確記錄 `training_reused_from`、`additional_epochs: 0` 及 Train-only profile。特徵 recipe、原始骨架、30 幀／stride1、品質要求與連續 3 窗規則不變。

各新 run 只以 Validation 的 balanced sequence accuracy 校準門檻，平手取較少 FP、較高門檻。模型按整體 Accuracy、F1、全體跌倒召回、較少 FP 選擇，平手保留舊模型。另要求 Accuracy／F1 不退步、unknown／粗略早報不增加、粗略時段命中至少原基準 2 支；若補到原本漏報，必須在事前粗略跌倒區間內有目前骨架品質合格的可見 AE 警示。

## 實際結果

固定 Validation 為 8 ADL＋15 Fall，共 23 支。以下三者全部可判定，unknown=0，所以條件式二元 Accuracy 與包含 unknown 的整體 Accuracy 分母皆為 23；二元 F1 不包含 unknown，全體跌倒召回分母為 15。

| 版本 | TP／TN／FP／FN／U | 兩種 Accuracy | 二元 F1 | 全體跌倒召回 | 門檻 | 採用 |
| --- | --- | --- | --- | --- | --- | --- |
| 原 MSE `ur-sample-human-s1-001` | 12／8／0／3／0 | 20/23＝86.96% | 0.8889 | 12/15＝80% | 0.68801838 | 保留 |
| 逐維加權 `ur-human-residual-channel-001` | 11／8／0／4／0 | 19/23＝82.61% | 0.8462 | 11/15＝73.33% | 0.49582237 | 不採用：退步 |
| 雙分支加權 `ur-human-residual-stream-001` | 12／8／0／3／0 | 20/23＝86.96% | 0.8889 | 12/15＝80% | 0.57507318 | 不採用：平手 |

所有候選仍漏報 `fall-19`、`fall-18`、`fall-30`；逐維加權另外漏報 `fall-16`。三者均無誤報，六支粗略時段的事件命中／可見警示命中皆 2 支，早報 0。這些人工粗略時段只用於工程檢查，不是正式逐影格標註或警示時間 Accuracy。

三種校準皆由正常 `adl-35` 的最高合法 K3 分數形成門檻；降低此門檻會影響誤報取捨。原三個漏報在雙分支加權下的「最高合法 K3／門檻」仍僅約 0.191、0.395、0.479，沒有越過門檻。比值只供診斷，不能當機率。逐維加權還把 `fall-16` 的比值從 1.207 降至 0.845，解釋新增漏報。

整體有效窗口 coverage 約 84.39%，影片 coverage=100% 不等於每一影格可判定。原先已觀察的身體出框、骨架品質下降與短暫警示限制仍在；本輪沒有改善姿態估計，不能宣稱骨架更穩定。

## 證據與限制

最終完整回歸 **203 passed／1 skipped／12 warnings**，見[XML](validation/pytest-human-residual-final.xml)。skip是歷史預設Human資源路徑，新鏡像Human權重測試有實跑。首次完整回歸為199 passed／2 failed／1 skipped，原因是舊單元測試使用`object.__new__(Scorer)`繞過初始化，少了`residual_weights`；修正該測試並增加加權CPU／CUDA覆蓋後全部通過，沒有改動已鎖定的計分程式，初次XML保留。Ruff與9檔格式檢查通過。

真實串流共555幀、381個可比分數差0；三者ACTION CSV相同，原MSE AE CSV／事件也與舊版逐位元相同。每片可見警示8幀、第一幀皆5.9秒；H264全部解碼，三份接觸表已目視確認4.70秒FALLING／AE NORMAL、5.33秒UNKNOWN、5.93秒FALLING／AE FALL ALERT，沒有呈現時機改善。

- [正式 Validation、逐影片預測、六段時間檢查及 hash](validation/human-residual-validation.json)。獨立 CSV／JSON 匯出在 `outputs/human-residual-validation/<run>/`。
- [最終保存／測試／拒絕重跑與 Test gate 的實測紀錄](validation/human-residual-verification.json)。計分及選擇程式維持執行前凍結hash；整合驗證器後續僅移除未使用import並增加已保存輸出hash檢查，最初執行版本另存來源快照，唯讀核對已重跑通過。
- [真實串流整合](validation/human-residual-integration.json)：三個 run 各重跑同一支 Validation `fall-09`，與真實快取特徵的 AE 分數／狀態／事件核對；七類 ACTION CSV 必須完全相同。診斷短片保存在 `outputs/human-residual-streaming/`，不是較佳版本或正式期末影片。
- synthetic 單元測試驗證 Train／Validation 隔離、權重規則、拒絕改寫 frozen run、profile hash 防竄改、CPU／GPU 批次及串流一致性。這些功能測試不計入辨識成績。
- 此輪沒有新增特徵抽取 recipe 或獨立 `real_feature_smoke` pilot；沿用已嚴格載入真實權重的特徵，正式快取評估與重新影像推論均列 `real_end_to_end`。
- 原 37 個 run、1165 個 cache 檔案、設定、manifest、權重及舊影片 hash 保留；新兩 run 不覆盖舊結果。
- 90% 普通 Accuracy 尚未達到，不能以 balanced accuracy=90% 代替。新 Human 模型沒有 Test 成績；歷史另一模型曾查看的 Test 仍需揭露，反覆使用同一 Validation 也使其不再是獨立泛化證據。教師評分單位／切分未明，不能自行換算課程分數。

下一步較值得調查的是正常困難姿態與跌倒的特徵可分性，以及可見下降期間的骨架品質；本輪結果只排除了這兩個固定加權方法的改善，不代表所有方法已達上限。若能提供完整入鏡的坐下、起身、躺下等正常連續影片及跌倒時間標註，可另建來源清楚的訓練／診斷資料；不能用現有 Test 補訓後再宣稱獨立成績。尚未執行這個後續方案，沒有外部資源阻塞。

## 操作

在專案根目錄執行；預設核對命令不重新訓練、不重校準、不查看 Test：

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
.\.venv\Scripts\python.exe scripts/human_residual.py status
.\.venv\Scripts\python.exe scripts/verify_human_residual.py
.\.venv\Scripts\python.exe -m fall_detect demo --run runs/ur-sample-human-s1-001 --input data/raw/urfall/fall-09 --output outputs/residual-review-best.mp4 --overlay auto
```

最後一條另產生保留模型的七類／AE 並列影片；輸出已存在時會拒絕覆寫。完整實驗執行過 `freeze`、`run`，已完成便不可重跑覆蓋。未達門檻時 `human_residual.py test` 必須拒絕，不能自行降低 gate。
