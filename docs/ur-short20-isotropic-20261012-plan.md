# UR-only 20 幀＋保留身體比例：訓練前單一候選契約

2026-10-12。機讀鎖定：[lock](validation/ur-short20-isotropic-20261012-lock.json)。本計畫於正式 AE 訓練前建立，精確設定與 hash 以該 lock 為準，結果另記；不以 Test 選擇設定。

只跑 `ur-short20-isotropic-20261012-001`。30 幀等比例 wide `ur-isotropic-wide-20261012-001` 為比較基準，Validation 22/23、F1 28/29，預設仍 `ur-sample-human-s1-001`。舊 per_frame 的 20 幀結果不作本輪結果。

固定 20 幀、stride 1、30 FPS、`short_20`、`per_frame_isotropic`；保留真實 AlphaPose／Human TSSTG 權重與分類頭前雙流 512D。原 24 支 UR Train ADL、無增強／鏡像／外部來源。AE 256/64、seed42、Adam lr.001／wd.001／noise0／batch32、sequence sampling、max100、patience10、min_delta1e-6，皆沿用 wide。品質 .2／9 真實點／.8、因果缺失政策、strict >／K3 不變。Scaler／AE 僅 Train；checkpoint 僅正常 Validation；原 23 支 Validation 的 balanced_sequence_accuracy 校準及比較不變。

假設：短窗可能減少跌倒前正常動作稀釋及裁切造成的無效窗。這不是已證實的漏報原因。基準 fall-19 粗時段 1468–2369ms 有 9 個合法 K3 分數，最高 .13463649 < 門檻 .53411138，整片 23/71 有效端點。粗時段為既有 AI 診斷，不是人工確認 onset／end。

固定 pilot：Train adl-05／06，正常 Validation adl-01／04，Validation fall-09／19／29／25／18／30。先真實權重嚴格載入並 forward，驗證 finite512、相同觀測 Pose hash、因果時間戳、完整合法／無效窗口排程及快取隔離。每片須有合法 K3；正常平均 coverage 至少基準90%，六粗時段各至少1合法K3且總數至少基準80%。未通過就停止、不訓練。此 gate 不測辨識成績。

採用須全部成立：整體 Accuracy 嚴格高於22/23、F1不低於28/29、無新增 FP／unknown、不降低窗口 coverage、不使基準判對影片退步；六舊粗時段至少4段事件命中、0早報、不失去基準個別時段／當前良好骨架警示；救回 fall-19 必須在其舊時段內有當前良好骨架警示。未通過保留預設；持平、Loss降低、有效窗增加不當作 Accuracy 改善。單一候選沒有排序平手選擇，持平保留基準。

停止條件：pilot／finite／來源hash失敗即停止；AE最多100回合及patience10；完成一候選Validation後停止，無臨時增容量、種子、品質、K、正規化或校準目標。Test不準備、不讀特徵、不評估，不混30幀快取。核心新增顯式 `prepare/train --train-validation-only`；index保持完整原70支split，Test status=not_prepared且沒有pose／feature路徑。預設完整模式仍要求原完整prepared條件；scoped run禁止Test evaluate。

33項相關風險測試已通過，pilot10項檢查全過。先前 CLI prepare 派送錯誤已於訓練前修復；原source／lock不覆寫，修復hash鏈另存[修復證據](validation/ur-short20-isotropic-20261012-dispatch-repair.json)，沒有更改選擇／停止條件。synthetic、real_feature_smoke及未跑的real_end_to_end分開；歷史Test已查看、Validation反覆選模、人物隔離／教師協定及人工時間GT未確認。正式Notebook／五分鐘影片繼續暫緩，不公開上傳、不換算課程分數。
