# 23支Validation警示與骨架品質盤點：執行範圍

紀錄ID：`ur-validation-alert-quality-20261013-001`。這是既有成果的只讀診斷與人工覆核材料整理，沒有新模型候選、訓練、推論、校準或採用決策。

主版本為`ur-isotropic-wide-20261012-001`，30幀／stride1／per_frame_isotropic／512D／256-64 AE／固定門檻0.5341113805770874／嚴格大於／K3。demo預設`ur-sample-human-s1-001`的既有Validation影片判定另列，不混稱同一模型，不共用數值門檻。

只讀既有23支Validation的正式sequences/windows/events/metrics、對應原Pose快取和特徵JSON metadata。特徵NPZ只核hash，不載入特徵；AE/Scaler/權重只核檔案hash，不載入模型。index可能含歷史Test metadata，程式必須先以split篩選23支Validation，才解析快取路徑；不讀Test快取／影像／成績，不新增Train衍生資料。原始影像只讀選出的三支Validation，覆核頁保存原PNG bytes及來源SHA。

逐片分開記錄：影片TP/TN/FP/FN/unknown、唯一合法窗口端點／所有排程端點、事件首次／最後active／關閉時間、原影格骨架缺失及confidence品質、事件時同track因果採樣骨架品質門檻、AI粗略phase對照。品質通過不是骨架正確率；active窗口端點門檻不是實際RGB畫面可見警示的逐幀驗收。原六片coarse guard與後來15片AI取樣phase各用其原定義，不互相替換。

人工來源分開：fall-19既有本人姿態phase（33–61 transition、62–100 ground）繼續作phase診斷，不改為精確開始／落地時刻，也不重做100筆或已完成的三幀關節覆核。其他影片若沒有人工時間界限，標成尚未確認；AI標註不冒充人工GT。

少量覆核固定為三種已有疑點各一段：fall-18（TP但覆蓋低／原guard無good警示）、fall-09（TP但首次警示超出原coarse phase，15片取樣phase較寬，需人工釐清）、fall-30（短事件且非常接近AI possible phase末端，來源時間與30Hz格點不能混同）。這是看過既有結果後的診斷取樣，非盲測、非統計代表樣本。只需三段各填一筆開始／落地最早最晚範圍或不可確認，及整段骨架品質；不逐幀／逐joint標註。一次匯出JSON，頁面不預填AI答案、不展示AE分數，不代填本人確認。若片段不含關鍵時刻，須填片段外／不可確認，不能硬標。

風險驗證涵蓋split與路徑防護、unique endpoint分母／多track、unknown與warmup、strict >及K3連續／缺失重置／事件、因果原幀映射、phase來源隔離、人工範圍驗證、輸出防覆寫及舊hash。synthetic與既有真實結果重核分開；本轮不跑real_feature_smoke或新RGB至AE的real_end_to_end。

輸出獨立新outputs及docs/validation目錄；build存在即拒絕，verify只讀。修改current-progress及acceptance前封存bytes。停止於完整23片盤點、獨立核對、三段人工頁與文件完成；不降低既有品質或時間驗收、不換demo、不重訓、不新增Test，Notebook／五分鐘影片／公開上傳仍暫緩。
