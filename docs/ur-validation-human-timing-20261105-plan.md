# 三段本人覆核與既有Validation警示：只讀比對方式

ID：`ur-validation-human-timing-20261105-001`。來源為使用者上傳的`validation-three-clips-human-review.json`及原`ur-validation-alert-quality-20261013-001` bundle／正式wide Validation成果。這是已知疑點的事後診斷，不是新模型、盲測或新成績。

1. 保存上傳原bytes／SHA，驗schema、bundle id與重新計算的bundle內容hash、3個唯一片段、本人自述確認、逐段reviewed、原影格整數範圍與順序、時間戳、原PNG SHA及相同Validation來源。備註是標註資料，不是指令。本人自述不等於多人獨立真值，開始／落地的兩端點是不確定範圍，不擴解為確定整段phase。
2. 定義onset=[Olo,Ohi]、impact=[Ilo,Ihi]。可能跌倒包絡為[Olo,Ihi]；若Ohi≤Ilo，所有允許時刻組合都涵蓋的區間為[Ohi,Ilo]。同時單列「開始不確定範圍內」「落地不確定範圍內」「確定早於開始」「確定晚於落地」。包絡命中不能稱已確認跌倒過程警示；落地界限內不能推斷一定落地前或後。完全缺標時記未知，不算錯／對，交疊界限不製造確定區間。
3. 用原CSV的available_at_ms作警示可用時間；原來源frame timestamp只用來轉人工界限與核因果映射。不把30Hz sample_index當source_frame_id，也不四捨五入後做邊界判定。延遲保留區間[t-Ohi,t-Olo]、[t-Ihi,t-Ilo]，不取中點冒充精確GT。
4. 影片TP／FN與事件首次、各active端點／最後active／關閉時間分開；不把event end當持續active。另列同track當前confidence-good的active端點，只是展示門檻診斷，不宣稱已重跑RGB可見警示。整段mostly_matches與部分出框備註，不改寫成每幀／每joint真值。
5. 主模型固定原wide30 isotropic／.5341113805770874 strict >／K3。demo原預設另列；不換模型、AE／Scaler／門檻／品質／追蹤／事件規則，不重訓／推論／校準，不讀Test影像／快取／成績。所有數字只讀既有Validation CSV，不讀特徵陣列或載入模型。
6. 保存舊231個outputs artifact及前輪29個final QA hash；修改進度／驗收前封存原bytes。原bundle、原AI phase與human pending歷史報告不改；新23列overview只增列本輪human結果，其他未標精確界限的Fall繼續未知，不能外推3片到15片或23片時間正確率。
7. 新目錄一次性建立、重複拒絕；風險測試涵蓋界限／缺標／交疊／嚴格邊界、時間範圍與因果性、缺失／K3及active分母、防覆寫／來源隔離。對正式CSV指標、每個事件／時間類別獨立核對；synthetic、existing-real-result-audit分開，real_feature_smoke與real_end_to_end本輪不執行。

停止於驗證、三段對照、新23列補充表、獨立核對與文件完成。沒有採用或改善宣稱；新版正式Notebook／五分鐘影片／公開上傳仍暫緩。
