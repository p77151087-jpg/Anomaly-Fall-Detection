# fall-19逐幀診斷與人工覆核：事前界線

2026-10-12，使用者要求繼續下一步。前輪動態近鄰已完成且持平，本階段不延長模型網格；只檢查原Validation fall-19，完整100支來源影格及既有骨架、30／20幀窗口、wide／20iso／motion分數與K3。

- 保留raw、快取、權重、run、門檻、品質／track／split；不重新擬合、校準、補缺失或評估Test，不恢復外部資料。
- 來源時間／frame ID與原RGB SHA精確對齊，完整序列列出目前Pose、track／信心／可靠點出框數，以及窗口有效性和時間。窗口終點與來源影格時間用因果關係對照，不能把未來窗口分數搬到較早影格。
- 逐幀覆核頁面的原影像不受品質遮罩，低信心骨架只是診斷疊圖，不能改成合法AE窗口。畫面列出暖機／track gap／低品質和原始分數，不偽造警示。
- 官方同步CSV是時間與感測同步，不是onset真值。官方網站的深度姿態標籤與RGB人工onset亦不能無條件混同。本輪不下載／解碼包含Test的深度特徵檔。
- 固定全片以及1400–2600ms的密集接觸表供視覺觀察；AI視覺觀察另保存`ai_visual_diagnostic`且`human_confirmed=false`。不據此宣稱人工GT或正式時間Accuracy。
- 人工覆核表預設所有幀`unreviewed`、onset／落地邊界null；支援逐幀標示站立／過渡／倒地／不確定，與骨架可見性／追蹤一致性分開。明確填寫reviewer及確認之後才可匯出標註；人工身份無法靠程式獨立驗證，保留自述限制。
- 新標註只可作Validation診斷，從不成為Train資料或自動修改門檻。匯入驗證schema、來源SHA、完整frame ID、排序／時間、null／未確認狀態、onset區間與可見性；拒絕冒用Test、AI假扮human及覆寫。原始匯出／來源與匯入驗證另存，舊粗時段不改寫。
- 人工標註未完成時仍完成可執行的診斷、機器一致性驗證及UI，明確列出仍待人工核對的原因。若新的AI觀察支持缺失或稀釋，仍只能列假設，不盲目追加訓練。
- synthetic為時間對齐／缺失／範圍／標註驗證測試；real_feature_smoke為原RGB完整解碼與既有Pose／窗口／分數覆核；没有新RGB Pose／ST-GCN forward或real_end_to_end辨識驗收。舊Validation22/23／.9655不變，Test未知、Notebook／五分鐘影片暫緩。

輸出獨立`outputs/ur-fall19-review-20261012-001`；證據`docs/validation/ur-fall19-review-20261012-001`。先保全與鎖定，再生成覆核包，不覆寫完成目錄。
