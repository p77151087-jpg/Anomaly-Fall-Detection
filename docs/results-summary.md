# 原第二條路基準：真實實驗結果

最新Le2i擴充已完成，見[擴充版結果](le2i-results.md)。以下保留原run `openpose18-baseline-001` 的結果，不與不同Test樣本數直接比較。

2026-09-17，正式 run：`runs/openpose18-baseline-001`。**真實端到端實驗已完成，但第一版辨識效果不足。** 以下來自實際 UR Fall camera0 RGB 的固定 Test，不是 synthetic 或只挑展示影片計分。

## Test 結果

原始指標：[openpose18-test-metrics.json](validation/openpose18-test-metrics.json)。全部23段Test：

| 真實類別 | 判為正常 | 疑似跌倒 | 無法判定 unknown | 總數 |
| --- | ---: | ---: | ---: | ---: |
| 正常活動 ADL | 0 | 7 | 1 | 8 |
| 跌倒 Fall | 1 | 7 | 7 | 15 |

- 全部跌倒的端到端召回率：**7/15 = 46.67%**；其餘1段判正常、7段unknown。
- 正常活動誤報：**7/8**；剩餘1段unknown，沒有正常活動被正確判為正常。
- 可判定序列coverage：15/23 = 65.22%；有效時間窗coverage：39.89%。沒有缺少整段時間軸而縮小分母。
- 將unknown計入全部輸入的operational accuracy：7/23 = 30.43%。
- **只在15段可判定序列內**：TP=7、FP=7、FN=1、TN=0；accuracy=46.67%、precision=50.00%、recall=87.50%、F1=63.64%。87.50%不能當作全部15段跌倒的召回率，必須同時呈現unknown及端到端46.67%。

完整逐序列／逐窗口／事件CSV、2×3表及混淆矩陣在 `outputs/real_end_to_end/openpose18/test-evaluation/`。混淆矩陣只畫可判定15段，上表保留全部23段。

## 方法與隔離

Lightweight OpenPose產生原生18點，原始ST-GCN Kinetics400 checkpoint嚴格載入、凍結，抽分類頭前256D特徵；AE為256→128→32→128→256。沒有補成512D、沒有用分類分數冒充特徵。

固定seed42、先以原始序列切分：ADL Train/Validation/Test=24/8/8，Fall=0/15/15。全部70段共11,936張RGB完整解碼，擷取777個有效特徵窗；無有效窗的序列保留並可成為unknown。Scaler與AE參數只使用正常Train有效特徵；正常Validation選最佳checkpoint，含跌倒標籤的Validation選門檻。預訓練特徵模型本身採監督式Kinetics動作訓練，不能把整套方法稱為完全無監督。

AE實跑100 epochs，最佳正常Validation loss=0.49676624350831844。Validation門檻為0.19389973580837247，同人物連續三個有效窗嚴格大於門檻才警示。校準／評估／展示固定逐筆AE計分，避免批次形狀造成門檻邊界浮點差異。

Test評估後沒有以Test調門檻、改權重或覆蓋此run。最後展示的EOF修正只令事件結束時間使用實際來源結尾，沒有重選模型或改Test分數；凍結檔案hash再次核對不變，見 [整合證據](validation/openpose18-final-integration.json)。

## 展示與限制

可直接觀看：

- [fall-02 展示](../outputs/real_end_to_end/openpose18/fall-02-final.mp4)：Validation，110影格；約1.97秒確認疑似跌倒，約2.47秒追蹤中斷，後續保留unknown。
- [adl-05 展示](../outputs/real_end_to_end/openpose18/adl-05-final.mp4)：Train，179影格；不是獨立泛化證據。
- [MP4輸入展示](../outputs/real_end_to_end/openpose18/fall-02-mp4-final.mp4)：同一fall-02真實圖片另編成30fps MP4，驗證影片檔輸入；衍生時間軸不作正式Test評估。

三支影片完整重新解碼。兩段Train/Validation展示的可比有效端點分數與快取計分完全一致，狀態及事件一致，範圍見 [batch/streaming證據](validation/openpose18-batch-streaming.json)。示範片能觸發不代表整體辨識良好。

抽幀看到躺倒時漏點、同人物被分裂追蹤；這與低coverage相符，但尚未做消融實驗證明所有漏報原因。訓練與正常Validation loss也有明顯差距。後續改善應先用Train/Validation檢查骨架、追蹤、時間窗與特徵，另建run。本次Test已查看，後續反覆參考它會有選擇偏差，需保留此基準並用新保留資料確認改善。

資料例外：adl-37有350張RGB、只提供前330張同步時間；僅最後20張以30fps因果延伸，明確標記estimated並納入recipe fingerprint，不補中間缺口。見 [dataset report](validation/openpose18-dataset-report.json)。

CPU已實跑真實影像pose及真實骨架ST-GCN測試，但很慢；完整CPU 120幀smoke曾中止，沒有宣稱CPU完整smoke或即時效能。正式實驗使用CUDA。未做人工跌倒起點與延遲評估，未宣稱重現原論文或已符合老師所有評分要求。

## 重現識別與證據分類

- recipe：`lightweight_openpose18_stgcn_kinetics_v1`
- recipe hash：`58eae3cc93c17dec6a53f628311b1d9ae4bf9b355555275fed3e8ad18896885e`
- split hash：`0b9a0f0cf3609c478afcbbe028130c7d519e26b573135fbd739c074c68bc93c5`
- 權重來源、實際SHA256、upstream commit：[模型來源](model-sources.md)／`weights/openpose18-manifest.json`。模型、資料與run留本機並由Git忽略。

| 證據種類 | 本次內容 | 可支持的結論 |
| --- | --- | --- |
| synthetic | outputs/synthetic/openpose18-validation，手製256D、6 epoch全流程 | 程式流程可執行，不是UR Fall成績 |
| real_feature_smoke | outputs/real_feature_smoke/openpose18-cuda-v2，adl-05／fall-02強制實際forward | 真實pose與256D可取得；classification=not_performed |
| real_end_to_end | 本頁固定Test、真實run及三支影片 | 真實端到端已執行；效果不足，不能宣稱可靠偵測 |

操作見 [使用指南](openpose18-usage.md)。原Human四個檔案仍缺，舊blocked紀錄不適用於此替代run。
