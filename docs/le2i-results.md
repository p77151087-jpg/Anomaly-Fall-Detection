# Le2i擴充：真實訓練與測試結果

2026-09-17，正式run `runs/le2i-extension-001`。**資料擴充、重訓、校準、固定Test與展示均已實際完成；誤報仍高，不能宣稱可靠跌倒偵測。** 原 `openpose18-baseline-001` 的模型、門檻及其他凍結檔案hash全部保持不變。

## 實際增加了什麼

官方Le2i完整ZIP及190段原始AVI已取得，標註固定OmniFall revision。95段Train含fall/fallen影片整段排除；兩組Validation重複內容、不同標註的四段整組排除。最後新增38正常Train／15 Validation／38 Test；保留UR Fall原分組，混合161段（62／38／61）。原始資料完整解碼54,342影格，來源、實際SHA256、排除原因與split保存在[來源](validation/le2i-sources.json)、[切分](validation/le2i-split.json)及[資料核對](validation/le2i-dataset-report.json)。

全部161段prepared，共6,404個256D特徵。62段正常Train中45段有可用特徵，共3,766窗：UR Fall 19/24段、346窗，Le2i 26/38段、3,420窗。舊基準為19段、351窗。沒有有效特徵的影片仍留在清單與評估分母；不能把62段都說成有效訓練樣本。詳見[prepare證據](validation/le2i-preparation.json)。

同一組真實Pose／ST-GCN權重保持凍結；只重新擬合Train正常Scaler與AE。骨架新增原向品質不足時的±90度真實影像推論，仍有重複骨架、track分裂和家具誤偵測。Pilot僅Train/Validation，設定選擇先於新Test固定，見[選擇紀錄](validation/le2i-pose-selection.json)、[實驗設定](validation/le2i-experiment-lock.json)。

## 同一資料集的比較

| 實驗與Test | 跌倒影片中有警示 | 正常影片誤報 | unknown／全部 |
| --- | ---: | ---: | ---: |
| 原基準，UR Fall | 7/15（46.67%） | 7/8 | 8/23 |
| 擴充版，UR Fall | 7/15（46.67%） | 6/8 | 8/23 |
| 擴充版，新增Le2i | 15/22（68.18%） | 11/16 | 10/38 |

相同UR Fall測試只有1段正常影片由誤報變正確，跌倒召回和unknown數沒改變。不同資料集的比例不能直接當作改善幅度；UR Fall Test在原實驗已查看，只作歷史比較。資料和骨架同時變更，未作消融，不能單獨歸因更多資料或旋轉。

## 擴充版全部61段Test

| 真實類別 | 判正常 | 疑似跌倒 | unknown | 合計 |
| --- | ---: | ---: | ---: | ---: |
| ADL | 1 | 17 | 6 | 24 |
| Fall | 3 | 22 | 12 | 37 |

全部跌倒的端到端召回22/37＝59.46%；正常17/24誤報，另6段unknown。序列coverage 43/61＝70.49%，有效端點coverage 53.08%，沒有缺失timeline。將unknown計入全部输入的operational accuracy為23/61＝37.70%。

只在43段可判定序列內：TP22、FP17、FN3、TN1，accuracy53.49%、precision56.41%、recall88.00%、F1 68.75%。**88%不是全部跌倒影片的召回率。** 原始[metrics](validation/le2i-test-metrics.json)含by_dataset；逐序列、窗口、事件、2×3表及混淆矩陣在 `outputs/real_end_to_end/le2i-extension/test/`。混淆矩陣只畫可判定43段。

## 訓練、校準及限制

AE實跑79 epochs，依正常Validation最低loss保存模型，best loss 0.26481568650342524；以既定Validation sequence F1規則選threshold＝0.15086834132671353，固定同track K=3、strict >。Test未參與Scaler、AE或門檻選擇；固定後執行一次評估，沒有看Test重挑設定。來源與凍結hash見[訓練校準](validation/le2i-training-calibration.json)。

**Le2i四段正常Validation皆無有效特徵**，已標註範圍內是track_gap，另有unlabelled_interval；正常Validation實際只剩UR Fall的7段、128窗。這是跨場景驗證的明顯不足，沒有拿Test正常影片補進去。

Le2i未完整落在已知標註區間的窗口仍算unknown；連標註小數捨入造成的毫秒以下空白也沒有自動橋接。這項保守限制會降低coverage，不能與未採相同標註範圍限制的研究直接比較。一般影片demo不讀標籤，因此其完整事件時間不一定與受標註範圍限制的cache相同。

## 真實展示與一致性

- [UR Fall Validation fall-02](../outputs/real_end_to_end/le2i-extension/urfall-validation-fall-02.mp4)：110影格，約1.967秒確認、2.800秒track_gap結束，之後仍有unknown。
- [Le2i Validation Coffee_room_01/video_42](../outputs/real_end_to_end/le2i-extension/le2i-validation-coffee42.mp4)：原生AVI輸入，輸出190影格，包含normal、警示及unknown。

兩支輸出都完整重新解碼；已查看六張正式抽幀、訓練曲線及混淆矩陣。UR Fall的7個、Le2i的28個共同可見有效端點，快取和真實重新推論的raw score差均為0；UR Fall狀態及events完全一致。Le2i的共同狀態相同，但第二個cache事件因標註空白於5.633秒結束，無標籤demo則到來源末6.320秒；不能宣稱其全部events一致。詳見[整合證據](validation/le2i-final-integration.json)。

Le2i示範第一個警示約1.967秒，早於標註跌倒起點3.904秒，屬起身階段的誤報；另有4.967秒警示。這說明sequence TP可能來自跌倒前誤報，不能把整段分類成績當作正確起點或延遲評估，也不能只挑影片第二次警示展示成成功。

本機兩段展示處理速度約23.69及41.82輸出影格/秒，僅短片實測，不是即時效能保證。完整實驗使用CUDA；CPU整段效能仍未驗證。

## 驗證與重現

69 passed、1 skipped（僅原Human缺四個資源），[pytest XML](validation/pytest-le2i.xml)。包含去重、跨組拒絕、標註空白、旋轉座標、條件推論、cache重用及不完整影片拒絕；舊模型CPU/CUDA真實權重測試也實跑。ruff check、format及compileall通過。新舊run在評估及展示後再次核對凍結hash不變。

- recipe hash：`bc21ccb5ccb7fe5be7309efa4ba37fadecc31e047d11b3f04b575d250ca9fc9e`
- split hash：`a6454c4389e4a8ef7b161920088dd290489dd9e361b24b1afab56507bf11e4b8`
- 來源ZIP SHA256：`01df91e32a034d9897d1f9cd5e0211c70ab54289d27e8490db5a699384c17fc8`

synthetic測試只驗程式；pilot屬real_feature_smoke、沒有分類成績；本頁正式run/Test/示範屬real_end_to_end，三者分開。下一步若繼續研究，應先修復正常Validation骨架可用性與追蹤，重新訂立驗證協定並保留新的未查看測試資料。操作見[擴充指南](le2i-extension.md)。


報告用[Loss圖與圖說](loss-figures.md)已補齊；後續[擴充方向與來源](next-expansion.md)另列。
