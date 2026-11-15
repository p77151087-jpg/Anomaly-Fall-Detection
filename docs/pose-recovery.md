# UR Fall 骨架觀測恢復實驗

2026-09-18。使用者要求繼續改善辨識。維持原始序列split、Train ADL Scaler／AE、Validation選模及門檻；歷史Test已查看，不能重新宣稱獨立測試。Notebook與正式影片暫不更新。

## 診斷與假設

[Train／Validation 47支診斷](validation/continuity-diagnosis.json)確認：多個track_id不能直接等同人物配對錯誤；fall-19有42/100幀完全沒有Pose，49/100幀沒有滿足原品質要求的Pose。fall-02有30/110幀沒有合格Pose，而相鄰「各一個Pose」影格之間沒有ID切換。單純合併ID不足以恢復缺失的真實觀測。診斷中的單觀測ID統一僅用於比較可用窗，不用於訓練或推論，且排除多Pose影格，並非理論性能上限。

待驗證假設：更高影像解析度，以及僅在原推論沒有合格骨架時降低人物偵測門檻，可恢復倒地期間的實際觀測。關節可信度0.2、12個真實關節、窗口80%好幀、30幀／stride5、K3皆不變；不插值關節、不強制延用人ID、不降低骨架品質條件。

## 事前候選與停止條件

兩個Pose候選：960輸入／人物門檻0.25；960輸入／原結果無合格Pose時，對同一張影像以0.10人物門檻再推論。原640配方保留，新fallback設定預設0（停用）且舊recipe hash不變。兩者不加旋轉，承接前輪旋轉候選失敗證據。

pilot使用Train adl-05、adl-24、adl-38及Validation adl-04、adl-20、fall-02、fall-18、fall-19、fall-29。事前gate：四支Fall有效特徵窗合計至少增30%，ADL至少保留90%，原可判定序列不得變unknown。若兩者均通過，按Fall有效窗、ADL有效窗、先列候選選擇，並目視核對新恢復的Fall觀測確實落在人物上。這些是骨架候選的選擇規則，不是Accuracy成績。

通過才prepare完整新cache（Test僅擷取固定特徵），以三個有限AE候選訓練：sequence/32、sequence/16、window/32，最多100epochs、patience10。只用Train ADL擬合，正常Validation選checkpoint；全Validation以balanced accuracy校準，候選按全體Accuracy、F1、全體Fall recall、較少FP、先列候選選擇。Validation未同時達Accuracy90%、含unknown全體Accuracy90%、F1 0.87時，不評分Test。停止此有限候選組，不追加網格追分。

## 操作與證據

```powershell
.\.venv\Scripts\python.exe scripts/pose_recovery.py verify
```

`lock`、`pilot`、`training`皆保留既有產出，完成後不覆蓋；training需要pilot gate及視覺核對通過。實際結果在進度與驗收頁追加，不能將本計畫寫成已達標。

## 實測結果

兩個候選各完成9支Train／Validation真實擷取。960候選ADL有效窗121→126；960加低偵測門檻121→127；兩者Fall均27→28（+3.7%），未達事前+30% gate。fall-18／19／29皆沒有更多有效特徵窗。依停止條件，沒有完整重做70支、沒有AE訓練、沒有Test評分。證據屬`real_feature_smoke`，不是Accuracy改善。

[pilot原始輸出](validation/pose-recovery-pilot.json)、[結束時驗證](validation/pose-recovery-verification.json)保留。當時程式已按lock hash核對；config.py／yolo.py快照保存在`validation/pose-recovery-source/`，後續AE新增設定不覆蓋快照。`verify`檢查舊run及來源、歷史程式快照；重新執行pilot／training則要求當前程式符合原lock。
