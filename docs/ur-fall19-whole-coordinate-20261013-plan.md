# 人工整副骨架覆核接續：六點固定控制

ID：`ur-fall19-whole-coordinate-20261013-001`。本次接續使用者已授權的錯點修正控制診斷，收到三幀人工JSON後另立本協定；不改寫全骨架定位或左手肘歷史lock。

- 原JSON SHA256 `9b4f3216924a04896578187249f85d7dab4d855068e961620816d516a38f7429`。僅接受原bundle、原Pose／manifest／RGB SHA、44／47／49、track1、原13關節、原時間／信心及明示human self-report。26符合原點、6錯位可見、7不可確認；自述覆核不是獨立多人關節真值。
- 單一介入：47幀joint5/6/9/10/11與49幀joint1的六個人選XY。44不改，7不可確認保留原點；不插值、不補造、不遮罩、不合併先前手肘座標、不依分數搜索位置。原confidence／bbox／track／時間／品質完全不改。
- 凍結wide30幀isotropic真實512D ST-GCN、原Train-only Scaler／AE256/64、嚴格大於原門檻 .5341113805770874、stride1／K3。無fit、訓練、checkpoint選擇、校準、新增候選或採用。原24Train ADL來源保留，本控制不需要重新學習。
- 先驗来源与synthetic風險，重算原fall-19的sham23窗並核對≤1e-6（特徵相對容差1e-5）；Train adl-05／正常Val adl-04各固定前三合法窗作不變控制，通過才重算介入。僅這三片可讀觀測／特徵，Test先拒絕、不解碼／評估。索引中的歷史Testmetadata不作選模。
- 報告raw／K3最大分數與固定門檻差距、升降窗、合法／總窗、unknown、FN／事件／首次警示及粗姿態phase覆蓋。分數越大越異常而非跌倒機率。分數上升不等於辨識改善；若仍無事件，明確結案為漏報未改善。人選局部輸入不能當正式全Val或部署改善，不改wide正式22/23、demo與舊CSV。
- 姿態phase33–61為整段transition、62–100為ground，非精確跌倒開始／落地時間真值。44全點符合與舊整體wrong差異並列保存；右踝49仍不可確認，不稱全骨架修復或已排除Pose問題。
- 新output／QA／cache獨占建立，來源／協定／程式在推論前鎖hash。舊關鍵來源／wide run／模型／既有控制與定位成果有界保全；mutable文檔先存原bytes，原final-qa不改。來源不一致、品質或時間變化、sham不符即停，不放寬驗收。
- 獨立從NPZ核XY以外不變、手算Scaler／AE MSE與因果來源／K3／事件／分母。synthetic、real_feature_smoke及限定Pose→AE/K3 real_end_to_end分開；完整RGB／全Val／Test／demo驗收未執行。Notebook／五分鐘影片／公開上傳繼續暫緩。
