# 三段Validation實際警示畫面：快取重播與像素核對

ID：`ur-validation-display-20261105-001`。使用者授權fall-18、fall-09、fall-30的demo預設`ur-sample-human-s1-001`與wide參照`ur-isotropic-wide-20261012-001`對照短片。本輪僅把既有真實RGB／Pose／分數／人工界限送入原顯示流程，不訓練、不執行Pose／ST-GCN／AE／分類頭新推論、不校準、不讀Test，不改正式預設或既有結果。

## 固定來源與顯示契約

- 原23支Validation正式CSV分別取`outputs/human-isotropic-validation/ur-sample-human-s1-001/`與`outputs/isotropic-refinement-20261012/ur-isotropic-wide-20261012-001/`。各自門檻、strict >、K3、quality、score hold皆沿用凍結config。只用所選三支的既有Pose快取；先驗原split及來源，再載入。
- 已核三片兩模型Pose NPZ bytes／來源hash及逐點／bbox／track／原時間一致。共同七類顯示快取來自`outputs/sample-display/ur-human-isotropic-001/{sid}-actions.csv`；其原30/per_frame分類獨立於AE，並核對歷史30個實際觀測、當前原幀、品質、分數方向／argmax及來源metadata。fall-09與demo自己的既有action log逐欄完全一致。這是顯示資料複用，不挪用該run的AE分數或門檻。
- 使用原`draw_sample`（auto在這類真實Human模型選sample）與原`EventHistory`，保留左ACTION／右AE EVENT HISTORY／Current提示。七類FALLING不是AE警示；ALERT RECORDED是歷史事件，不等於當前good骨架或持續active。
- 依原`demo`順序：CausalResampler→讀該端點既有raw score→原EventState→先EventHistory.observe→原當前Pose／track／score age gate→原renderer。壞Pose／過期分數保持unknown、數值隱藏，不降低9/13與.2／80%窗口規則或補缺失。只替代模型執行為讀已保存數值，事件需與正式CSV完全一致。
- 不更動renderer原碼。原native畫面外另加模型／影片級結果、CSV active與當前狀態、人工作開始／落地不確定界限及原來源／格點時間；新增說明不當生產警示像素。顯示分數不是跌倒機率。

## 短片範圍與時間

沿用本人覆核bundle範圍：fall-18原幀1–65（0–2135ms）、fall-09原幀103–185（3403–6139ms）、fall-30原幀1–70（0–2302ms）。先重播完整既有Pose／分數歷史，再截取此時間範圍內的30Hz輸出格點；不從片段起點重設warmup／事件／七類歷史，不為湊幀補未來。每個媒體影格另列絕對grid timestamp、原source id／timestamp與media offset，原圖末幀未落在格點時不補造。

交付六個個別模型片及三個左右並排片，H.264／30fps／本機播放頁，可半速及逐幀查看。這是內部診斷短片，正式Notebook／五分鐘影片／公開上傳仍暫緩。人工作界限維持不確定範圍，不將包絡命中改稱已確認落地前警示。

## 驗證、保全與停止

1. 在新目錄封存前輪來源hash與進度／驗收原bytes；build目錄已存在即拒絕，ffmpeg用-n防覆寫，來源模型／設定／快取／原影片／hash均不改。
2. 相關synthetic測試：K3與缺失／track／stale gate、event history先於品質遮蔽、動作頭獨立、片段不重置歷史、時間因果、原renderer壞骨架不畫、輸出防覆寫。synthetic與真實快取畫面重播分開；本輪無新real_feature_smoke或RGB至模型的real_end_to_end。
3. 獨立重算正式CSV的事件、clip／格點與source映射及品質狀態，不依新trace自證；逐解碼六片與三並排片核尺寸／FPS／幀數／時間與native header文字及歷史紅字像素。原始PNG源hash保留，選定事件前／當下／後的實際畫面另存。
4. 報告每模型每片的影片TP／FN、CSV事件首次／active、畫面當前AE alert、歷史ALERT RECORDED、七類FALLING分別計數；遮蔽如實列出，不把current unknown否認影片TP，也不把歷史或七類標籤當AE的現在警示。
5. 本輪只驗既有顯示行為及一致性，不改門檻／顯示門檻／持續規則，不換demo，不新增候選或Test。完成六片＋三並排、像素／事件核對與文件即停止；失敗原檔另存，不覆寫舊成果或宣稱Accuracy提高。
