<!-- course-alignment-next -->

## fall-19座標控制實跑：分數升高、漏報未改善（2026-10-13）

`ur-pose-coordinate-diagnostic-20261013-001`已完成事前鎖定、第44–51幀track1可見左手肘8個AI暫定XY介入、原wide固定ST-GCN／Scaler／AE／.534111門檻／K3推論。sham23×512特徵完全一致；Train adl-05／正常Val adl-04各3控制窗亦一致，無fit／訓練／校準。10合法窗受影響，9個raw上升／1個下降，max raw .18390694→.20033166；max K3 .13463649→.15106279（+12.20%），仍低門檻、零事件、fall-19仍FN；合法覆蓋23/71、transition21/29、ground0/38與最後合法1766.67ms不變。分數敏感度不是辨識改善。

目前不是人工正確joint座標實驗：舊100幀人工pose_check只屬分類標籤，AI點位明確human_confirmed=false，人工座標仍pending。另建8点覆核頁，點選／不可確認／草稿保存／導覽／匯出禁用已驗；只需8點確認及單一JSON，不重做100幀。看不清不補造，confidence／bbox／track／時間不改，禁止依分數找點。收到真實人工JSON後才另hash/cache推論；無自動採用。

synthetic54passed；real_feature_smoke＝嚴格真實ST-GCN／512D有限／sham及6正常控制窗；real_end_to_end僅既有Pose觀測至AE/K3的fall-19控制推論，未做新RGB Pose全流程／完整Val重新評分／Test／demo驗收。独立NumPy Scaler＋AE forward＋手算K3核對69分數、213窗口、零事件及causal source／逐Pose未變字段；12,710舊asset hash不變，五舊契約bytes原樣封存、299舊解析度成果及原formal CSV保留。Graph Tier2及coverage正常，cache/run/raw設計排除以原檔核對、best effort限制照記。

正式最高Val仍wide22/23／Fall F1 .9655；demo仍ur-sample-human-s1-001；20幀及512解析度候選另列舊結果，不混稱。這次未提升Accuracy、不採用、Test未知且未載入／評估。Notebook／五分鐘影片／公開上傳仍暫緩。[報告與命令](ur-pose-coordinate-diagnostic-20261013.md)｜[事前契約](ur-pose-coordinate-diagnostic-20261013-plan.md)。

## 骨架品質解析度對照結案：fall-19未改善（2026-10-13）

唯一512×384同AlphaPose權重候選實際完成47片新Pose／ST-GCN、3,419特徵（Train24支2,094／Val23支1,325）、AE32epochs／正常Val最佳22，門檻一次校準 .3137603998。Val TP9TN7FP1FN5U1，整體Accuracy16/23＝69.57%、可判定16/22＝72.73%、Fall F1 .7500、全Fall Recall9/15；低於wide22/23／.9655。新增adl-35 FP，原fall-07／13／25／29變FN、fall-22無合法特徵變unknown；不採用、原demo ur-sample-human-s1-001保留，未追加候選。

fall-19仍FN／零事件，transition合法21/29→18/29、ground仍0/38、最後合法1766.67→1666.67ms；新Raw .21963／K3 .17582仍低自身門檻，不能全歸沒有骨架或只有K3。全Val唯一端點coverage2287/2710→1325/2710（84.39→48.89%）；六AI粗phase事件4/6→2/6，可見good警示影片數仍2/6但原兩片退步，粗早報0→1。個別fall-09時序改善另記，不稱Accuracy或整體時序提升；舊人工pose_check只屬原Pose，新Pose無人工正確率、姿態phase非精確事件GT。

40相關synthetic／真實512D與strict權重／獨立3,688分數、5,716窗、26事件、CSV分母／矩陣、Train Scaler與正常Val checkpoint、23門檻候選／8,643原幀對照全過；25,138舊assets hash不變、五契約原副本保存、六實際防覆寫拒絕通過。首次報告protocol_id缺失的log／部分輸出保留，新增resume只補metadata，整run hash不變、重訓／重校準0。CuBLAS確定性警告保留；Graph Tier2 best effort及排除檔直接核對如實列示。

Test23保持not_prepared無路徑、影像／特徵解碼0、新評估0；新Test未知，未宣稱Test／demo real_end_to_end。Notebook／五分鐘影片／公開上傳暫緩，不需重做100人工筆。[實際結果、限制與只讀命令](ur-pose-resolution-20261013.md)｜[事前契約](ur-pose-resolution-20261013-plan.md)；同名outputs/run/cache與docs/validation保存全部成功及失敗。

## 骨架解析度完整特徵與AE訓練完成（2026-10-13）

47支Train／Validation已重建3,419筆真實512D特徵，固定四片同recipe完整pilot經hash驗證複用。單一AE實際32epochs、正常Validation第22epoch最低MSE .1420643194；原Test23未準備、無路徑。CUDA CuBLAS確定性警告原樣保留，不宣稱GPU逐位可重現。門檻一次校準完成 .3137603998；報告輸出因缺protocol_id中斷，失敗部分另存，不修改已鎖原程式或重訓／重校準，只以新增resume程式補報告metadata。正式成績與獨立驗收完成後另記，不將Loss當辨識改善。

## 骨架解析度pilot通過，完整47片重建中（2026-10-13）

已保全25,138舊檔及五份修改前契約；三套原Detector／AlphaPose／ST-GCN嚴格真實權重hash相同。固定四片512×384 pilot已實際完成，17×128×96有限熱圖與512D／原分類頭parity通過，共324合法特徵；fall-19為20合法窗，未當成品質或Accuracy提升。technical gate通過，正在以同recipe驗證複用四片並重建其餘43片；AE／正式Validation尚未執行，Test不解碼／不評估。

## 骨架品質單一解析度對照開始（2026-10-13）

使用者已授權僅Train／Validation，單一 `ur-pose-resolution-20261013-001` 已事前固定：相同AlphaPose真實權重256×192→512×384人物crop，其他人物偵測／tracking／quality、30isotropic 512D、AE256/64／seed42／max100／patience10及K3沿用wide。40相關synthetic風險測試已通過；舊成果hash與五契約副本保存正在執行，technical pilot／完整47片新Pose與特徵／AE／Validation尚未完成，不填成績。

新recipe/cache/run/manifest隔離，Test23保持未準備、禁止特徵／影像解碼與評估；原人工pose_check只標原Pose、phase不當精確時間GT。單一候選結果無論好壞停止，不追加尺寸或其他網格，demo／Notebook／影片目前保留／暫緩。[事前規則](ur-pose-resolution-20261013-plan.md)。

## fall-19人工補註已完成：階段語義／缺Pose修正（2026-10-13）

本人已澄清33–61與62–100是整段姿態階段，第1、2幀改為沒有骨架。原匯出bytes／hash保留，新human-amendment及schema v2診斷標註另存；精確onset／impact仍未提供、欄位null，不由phase起點強填時間真值。前輪「待釐清」是歷史狀態，本節接續解決，無需重做100幀或再次匯出。

100幀CSV僅2個pose_check改變；223窗口CSV逐bytes、三路方法摘要及凍結分數／門檻／事件全部不變。transition仍21/29合法（11 matches／10 wrong端點）、ground0/38合法分數null；f44–51仍8個人工wrong但confidence good。前段低分與後段缺有效觀測並存，尚未做修Pose控制實驗，不證明唯一原因或辨識改善。

58相關synthetic passed（13新／45原），獨立100原RGB SHA／來源／CSV、223因果最後來源窗口及分數／品質／跨門檻、僅2人工儲存格差異核對通過；264承接不可變成果hash不變、兩份修改前契約原副本保存，實際save拒絕覆寫、相關lint／format通過。Graph Tier2及best effort coverage另存，ignored精確成果直接核對。

無訓練／校準／Pose重抽／real_end_to_end／新Test；wide及motion歷史22/23／F1 .9655、20iso20/23／.8889，fall19仍FN、沒有Accuracy或時間改善，demo仍ur-sample-human-s1-001。新方法Test未知，Notebook／影片暫緩、不公開。[補註報告及只讀核對命令](ur-fall19-human-amendment-20261013.md)。

## fall-19人工自述已匯入，凍結分數對照完成（2026-10-12）

使用者JSON100幀來源manifest／frame-time／RGB SHA／確認欄位通過，原bytes另存SHA f7dbda11…88f3c。姿態upright1–32、transition33–61、ground62–100；partial44–100，骨架44起多wrong／少數missing。時間欄位恰等完整階段、f1／2無Pose卻標matches兩項待使用者釐清，原檔不改，不自行補精確onset／impact；收到100幀自述不等於完整獨立時間／關節GT。以下舊工具報告0/100為匯入前的歷史狀態。

因果最後實際來源幀對齊：三路transition21/29合法，11端點人工matches、10wrong；ground0/38合法且score null。matches端點wide Raw最高.12625／K3 .06990<門檻.53411；20iso .21409／.10210<.42389，motion亦未跨。f44–51共8幀被人標wrong但confidence判good，新增品質不一致證據；有骨架低分與後段品質失效並存，未證明修復會救回或唯一因果。舊粗phase9與人工階段21分開，不當覆蓋／Accuracy提升。

新增純診斷module／可只讀重算入口與100frame／223window CSV；45相關synthetic（9新＋36原）全過，223對齊／標籤／分數／phase統計独立核對，245前輪非進度成果hash不變、两原契約副本保存，完成save拒絕。無Train擬合／校準／新Pose／ST-GCN／real_end_to_end／Test，舊模型／分數與22/23／.9655不變，demo保留、新方法Test未知。Notebook／影片仍暫緩。[人工匯入診斷與限制](ur-fall19-human-review-20261012.md)。

## fall-19完整逐幀診斷／人工覆核工具完成（2026-10-12）

沒有追加模型／訓練／校準／Test；已生成100原RGB（0–3303ms）與既有Pose、30wide／20iso／motion三路窗口分數的本機互動覆核包。全部100原幀接觸表、密集Pose及代表原解析度影格已AI目視，AI紀錄另存、human_confirmed=false，人工仍0/100、onset／impact未確認。頁面支援原影格ID、逐幀品質、三項人工覆核、不確定時間邊界及JSON匯出／嚴格匯入；未確認匯出實際被拒絕，未造人工GT。

當前Pose49good／46低品質／5缺失，95筆全track1；30wide23/71合法、20iso33/81，新增10窗皆較早700–1000ms，最後合法端點都1766.67ms，原粗phase9合法K3且全部Raw未過自身門檻。f54當前缺Pose但先前合法窗口仍normal，f55轉track_gap／unknown；AI原圖顯示下沉後多數身體落到下緣，出框與後段失效同時出現，但不證明唯一因果或人工時間真值。不能把漏報全部歸於沒有骨架，也不能說縮窗已改善跌倒期覆蓋。

66相關synthetic cases全過；獨立100原圖SHA／時間、95Pose／track／信心、300因果對齊、223窗口K3及三份原23Val CSV指標／分母／矩陣／fall19零事件通過，24966舊assets hash不變、五契約原副本吻合。兩AE56原分數重載逐值吻合、另核23融合分數，完成build及未確認import拒絕。初次CUA timeout／沙箱loopback限制另存，允許本機連線服務重測UI通過。Graph Tier2／coverage best effort、ignored精確檔補驗；相關source lint／format通過。

辨識無改：wide及motion22/23＝95.65%／F1 .9655，20iso20/23／.8889；fall19仍FN，三路無警示。沒有採用新方法，demo仍ur-sample-human-s1-001。證據synthetic／real_feature_smoke分開，無新RGB Pose／ST-GCN／real_end_to_end或Test，新方法Test未知；Notebook／五分鐘影片暫緩、不公開上傳。下一依據仍需本人核對時間與Pose吻合，未把AI觀察當人工GT。[完整診斷及命令](ur-fall19-review-20261012.md)｜[本機覆核頁](../outputs/ur-fall19-review-20261012-001/index.html)｜`docs/validation/ur-fall19-review-20261012-001/verification.json`。

## 接續全域動態近鄰已結案：Accuracy與時序持平（2026-10-12）

單一 `ur-motion-novelty-20261012-001` 已實際擬合：保留wide真實30幀isotropic 512D／256-64 AE，原像素骨架另生30×3中心位移／log尺度的90D因果表示；原24Train ADL／3638窗fit Scaler與5近鄰bank，整支來源影片排除的Train分數定標，兩路各除Train p95後固定取max。新神經網路epochs0，近鄰正常模型fit完成，非重跑舊per_window或追加前輪SVM／AE網格。

23Validation TP14／TN8／FP0／FN1(fall-19)／U0，整體與可判定Accuracy22/23＝95.65%、F1 .9655、Precision1／Recall14/15，與wide持平，不採用、demo仍ur-sample-human-s1-001。窗口2287/2710＝84.3911%，全部2917窗口狀態、14事件track／時間／原因、23首次警示時間均不變；粗phase事件4/6、良好Pose警示影片2/6、早報0，不稱時序或Accuracy改善。fall19phase9合法K3／7當前良好Pose；動態距離最高.78393仍低於正常Val p95=1.01900，融合phase最高1.68073<門檻4.88126，降低門檻救回仍需8正常FP。已證實補回全域運動資訊於本候選無改善，未證明真正漏報原因、Pose正確或人工時間真值。

固定3窗genuine strict ST-GCN pilot差0後重建47片／6001原合法窗；90D與原512D／Pose快取隔離。60風險synthetic全過：首次49passed／11預設sandbox暫存setup errors，改專案獨立basetemp後同60案例全過，原XML保留。獨立重算18003分支／融合分數、3638整影片排除Train分數、Scaler／Validation校準、23CSV指標／矩陣／分母及2917窗／14事件通過；24620舊assets hash不變、原五契約副本保存。新recipe／run／來源hash／防覆寫另存，未微調／改Pose／K3／補缺失／添加候選。

證據為synthetic／real_feature_smoke；real_end_to_end未跑，新Test23未準備無路徑、Test未知，沒有RGB Pose重抽、demo部署、Notebook／五分鐘影片或公開上傳。原小Validation反覆選模、非人物隔離、預訓練含Fall及AI粗時段限制保留。[完整報告與只讀命令](ur-motion-novelty-20261012.md)｜[事前計畫](ur-motion-novelty-20261012-plan.md)｜`docs/validation/ur-motion-novelty-20261012-001/verification.json`。

## 本輪方法比較結案：無辨識提升，原模型保留（2026-10-12）

六SVM＋去噪／GRU兩AE正式完成並停止，共62新AE epochs；最高新模型SVM06與GRU皆21/23／F1 .9333，低於wide22/23／.9655，依固定順序選SVM06只作新方法比較代表，**不採用、demo仍ur-sample-human-s1-001**。粗時序改善另列：SVM06 fall-09首次事件5.9→5.0333秒、六粗phase事件命中4→5，phase良好Pose可見影片數仍2/6，非人工onset延遲。所有新方法Test未知。

47相關風險測試全過；全套首次294 passed／1歷史權重路徑skip／2 loopback Win10013，這兩項於允許本機連線環境重測2 passed；同一297案例去重後296 passed／1 skipped、無未解失敗，原失敗XML/log保留。ruff全src/tests/scripts與compileall通過，新程式format檢查通過，模型比較／fall19分數圖與教師示例／基準／Sample AE頁已目視。既有Human缺資源skip屬歷史預設路徑，與本輪真實sample權重嚴格載入三窗差0分開。

獨立模型／CSV／事件核對通過；47,630分數、184影片結果及175事件驗證完備，23,925舊assets不變，原五契約副本hash吻合。最終文件核對修正了時序輸入數的文字誤植：Train3449／正常Val1364／全Val2174，原30iso正常Val1427；原報告／generator另存，模型、CSV、門檻未改／未重訓。時序原端點2112/2710，不能混用2174多track特徵行。Graph Tier2查詢／trace／snippets及新舊相關coverage取得當前generation；best effort，ignored原cache/weights/附件以精確原檔／source／hash補驗，不宣稱graph完整。

自監督未實訓：事前原骨架與512表示對照rule未取得支持（連續支持0）；停止該選配階段，不當表示已正確或所有漏報已解釋。synthetic／real_feature_smoke分開，real_end_to_end未跑。原資料與外部封存保留、無新Test或重切，Notebook／五分鐘影片不更新、不公開。完整[方法比較報告](ur-normal-methods-20261012.md)、[事前規則](ur-normal-methods-20261012-plan.md)、`docs/validation/ur-normal-methods-20261012-001/verification.json`與`final-qa.json`提供可直接執行的只讀核對。

## 方法比較第二階段完成：去噪／時序AE共62epochs，未超過wide（2026-10-12）

在六SVM與診斷完成後另鎖兩個seed42／max100／patience10候選。去噪AE512-256-64-256-512、Train-only標準化輸入Gaussian .1／clean target，40epochs最佳30／正常Val MSE .11541825；Validation TP14TN6FP2FN1U0，20/23／F1 .9032。時序AE為512→128→GRU64、context+position→GRU128→512，8個同影片同track連續合法30幀特徵；缺失／gap重置、無padding／未來，全部8步重建MSE訓練、最後一步MSE計異常。22epochs最佳12／正常Val MSE .26024893，TP14TN7FP1FN1U0，21/23／F1 .9333。正常Train24影片不變；時序Train3449輸入不是新增影片，端點分母2710不縮，coverage降至77.93%。兩候選停止，均漏fall-19、皆誤報adl-35，去噪另誤報adl-37；無Test。

47相關synthetic通過；獨立重載八模型／Scaler後47,630分數逐值吻合，8×23 CSV指標／矩陣／分母、逐窗口K3狀態與175完整事件、門檻校準、最佳正常Val checkpoint、时序因果／histories全部重算一致；23,925舊assets hash已全部重驗相同。fall-19粗phase有9合法窗／9合法K3、7個當前良好Pose端點；wide raw最高.1839／K3 .1346，無單窗跨門檻，不是K3單獨擋住，降低門檻救回會誤報全部8正常Val。原七類頭在同粗phaseFALLING0；fall-09有13個FALLING但wide未警示端點，phase兩單窗過門檻、K3仍未過。七類與AE使用不同正規化，各自獨立，不當AE成績。

觀測骨架XY與pooled512的Train鄰近性受控診斷另有事前rule：fall-19原骨架距離.348–.872，全部未超過正常Val raw p95=1.8447；pooled p95=.8230，未找到至少3連續高品質「原骨架異常但表示正常」端點。沒有取得支持本輪自監督微調假設的對照證據，停止該選配階段，不盲目微調；不宣稱表示正確或已證明漏報因果。粗AI時間／骨架信心不是人工GT。新方法的phase事件可提前但Accuracy下降，不換demo；正式Notebook／影片暫緩。

## 方法比較第一階段完成：六組One-Class SVM（2026-10-12）

教師54頁簡報與Sample10頁PDF已重新讀取；根規格／AGENTS／課程計畫區分教師方向、使用者授權與歷史工程預設，修改前契約封存。23,925既有run／raw／cache／設定／權重／輸出hash快照保存。沒有其他訓練程序；原最高wide、20幀與joint-shift均已完成，未重訓。新47片Train／Val cache逐bytes複製隔離，Test23片not_prepared且無路徑；真實ST-GCN嚴格載入，固定兩Train／fall-19三窗口重算512D，與原cache最大差0；Train-only scaler與wide逐值相同。

六組nu=.01/.05/.10 × gamma=scale/.001正式擬合完成，均Train24 ADL／3638窗；各自負decision_function分數與Validation門檻、同品質K3。前五組TP14/TN6/FP2/FN1/U0，20/23／F1 .9032；最後nu=.10/gamma=.001為TP14/TN7/FP1/FN1/U0，21/23／F1 .9333，最佳但低於wide22/23／.9655，均漏fall-19。新校準按含unknown Accuracy→F1→全體Fall Recall→較少FP，固定候選順序；歷史balanced校準另列敏感度不作追加選模。網格停止，無新Test。40相關synthetic通過；正式真實特徵擬合／Validation為real_feature_smoke，未跑新RGB／demo／real_end_to_end。前置Pose cache API缺參數失敗尚未訓練，失敗cache／source另存，修正後smoke通過。兩AE在診斷後另鎖，不以本段規劃當成已完成。

成果：`docs/validation/ur-normal-methods-20261012-001`、`outputs/ur-normal-methods-20261012-001`、六個獨立`ur-ocsvm-XX-20261012-001`run。原demo保留，新方法Test未知，外部封存／Notebook／影片暫緩。

## 最新完成：UR-only 20幀＋等比例單一候選39epochs，Validation退步（2026-10-12）

補充完整封存查核：原retirement status因本輪合法改動3核心source而拒絕；3份原source反向重建後逐位元吻合舊hash並另存，舊source／lock不變。新 `scripts/verify_short20_retirement.py` 已重驗1,430歷史檔（201外部封存檔）hash全部相同、外部active路徑不存在，沒有恢復／下載。只做歷史bytes SHA保全、不解碼Test特徵或讀分數選模。[封存驗證](validation/ur-short20-isotropic-20261012-retirement-verification.json)。

先鎖定一個20幀／stride1／per_frame_isotropic、AE256/64／seed42／max100／patience10候選，原24UR Train ADL無增強／外部資料。固定10片genuine512D pilot全過，完整47支Train／Validation真實ST-GCN重建6,511窗；Pose沿用相同觀測hash、本輪無新RGB Pose forward。Train3,888窗、正常Val1,515窗；AE39epochs、最佳29／正常Val MSE .1128902972、門檻 .4238913059，完成正常Val checkpoint及原23Val校準／評估。

候選TP12／TN8／FP0／FN3／U0，兩種Accuracy20/23＝86.96%、Fall F1 .8889，低於最高wide22/23＝95.65%／.9655；fall-19仍漏報，fall-18／30退步為漏報，不採用、原demo預設保留。窗口coverage84.3911%→85.7483%不當Accuracy改善。fall-19有效窗23/71→33/81，新增10窗全在粗略正常前段；粗跌倒phase仍9個合法K3，候選最高.2352462低於門檻，不能歸因沒有骨架。六舊粗時段事件命中4→2／0早報，不當人工時間GT。

核心新增明確、有測試的prepare／train `--train-validation-only`：原70split不變、47片完整prepared、Test23片not_prepared且無cache路徑，拒絕混舊Test、缺Validation、smoke冒充及scoped Test evaluate。Test特徵讀取／評分0。33相關測試全過；全套271passed／1歷史路徑skip／2loopback權限失敗，兩項於允許本機連線環境重測通過，合計273通過／1skip。獨立CSV指標／分母／每窗K3與狀態／14完整事件、Train-only Scaler逐值／來源hash及最佳正常Val checkpoint全部一致，1,031舊檔hash不變、完成stage4次覆寫拒絕。

CLI派送及驗收CSV BOM修復保留原source／lock／失敗log及精確hash鏈，無重訓／重校準／改規則。Graph恢復，Tier2與coverage實測，早期新腳本parse gap已直接讀取修復；忽略的cache／upstream以源碼／hash／實測補足。證據synthetic與real_feature_smoke分開，real_end_to_end未跑；歷史Test已查看、反覆Validation／教師協定／人工GT及泛化限制保留。Notebook／五分鐘影片仍暫緩、未公開上傳。有限候選已結束，不追加調參。[完整結果、分母、時間與命令](ur-short20-isotropic-20261012.md)｜[訓練前契約](ur-short20-isotropic-20261012-plan.md)｜[獨立驗收](validation/ur-short20-isotropic-20261012-verification.json)。

## 最新完成：外部資料退出，UR Train限定增強29epochs，Accuracy未提高（2026-10-12）

依使用者撤回外部訓練指示，外部raw／cache移出原路徑並封存在data/retired；201移動檔與1,430原檔hash保留，混合run僅歷史比較、永久刪除0。原Train24ADL／3,638窗小幅關節位移經真實ST-GCN產生3,638增強列，合計7,276，沒有新原影片／外部來源，Val／Test不搬／不變。48原特徵parity差0、時間／品質不變，整體骨架平移被isotropic抵消至浮點微差，未當新訓練樣本。

有限單一256/64 AE實際29epochs，Val TP13／TN8／FP0／FN2／U0、二元及全體Accuracy21/23＝91.30%、Fall F1 .9286，仍漏fall-19／18，低於最高22/23／.9655；舊粗時段命中4→3，不採用、原預設保留。36相關synthetic全過、Train-only Scaler逐值／來源hash、原23Val CSV／K3／事件獨立重算通過；1,435舊檔保全、8次覆寫／追加／外部重啟拒絕，1,542受檢檔前後相同。驗收器count欄位修復封存舊source／失敗log，不改模型結果；graph／coverage不可用以源碼補核對。無新Test，證據real_feature_smoke、沒有新原RGB抽取／demo；正常資料增強可執行但本輪未改善Accuracy。Notebook／影片暫停。[本輪成果、分母、限制與命令](ur-only-shift-20261012.md)。以下外部資料為已退出的歷史紀錄。

## 最新完成：新增32支真實正常影片與取樣標註，Accuracy未提高（2026-10-12）

資料階段：UMAFall8＋Ursul24支實際下載，保存官方來源／授權／CRC／SHA；13,270原RGB幀嚴格真實Pose／ST-GCN抽取。裁去片頭／圖表並排除11支零合法窗口，21新正常來源／3,831窗加入原Train24／3,638窗，合計45支／7,469窗。外部僅Train，不搬原留出資料，UR70支split保留。

標註階段：完成v2取樣版，47支Train／Val、340原圖、15段Val跌倒起迄不確定區間與7區域可見性；106,808行Human13自動診斷另存，不當GT。Train無跌倒、Test未標註，尚未人工複核／完整逐幀左右座標GT，失敗v1保留。[影像與標註檢視](../outputs/external-normal-20261012-report/annotation-review.html)。

實驗階段：事前固定128／256混合正常AE實際68＋27＝95epochs，皆TP14／TN7／FP1／FN1／U0，二元與全體Accuracy21/23＝91.30%、Fall F1 .9333，低於原最高22/23＝95.65%／.9655。FP adl-35、FN fall-19，未採用／換預設，正常Loss略降不當Accuracy提升；有限候選已停止，沒有新Test特徵讀取／評分。

驗收階段：36相關synthetic全過，真實45 Train ADL Scaler逐值、46支次CSV／K3／事件及新時段診斷核對；1,160舊檔hash保全、10次覆寫／追加操作拒絕。Windows ZIP路徑修復封存原source／失敗XML，不改模型結果；graph／coverage不可用揭露。結果圖目視、340原圖hash吻合。新增RGB抽取為限定real_end_to_end，AE評估為real_feature_smoke，非新UR demo／Test。完整人工GT、泛化與警示時間仍缺；SAFER較大來源需登入／條款未取得；Notebook／影片暫停。[本輪成果、成績、限制、命令](external-normal-20261012.md)。

## 最新完成：資料量實測及4個候選，Accuracy未提高（2026-10-12）

原Train24影片／3638窗，不重疊有效端點135。4個Train子集學習曲線228epochs，共同尺度正常Val MSE隨8→16→24影片下降，支持資料覆蓋可能有關，不量化Test因果。2固定Train鏡像AE84epochs皆21/23／F1 .9286；完成後另固定2個Train正常殘差score，0新epochs，皆22/23／F1 .9655，未提高最高Val Accuracy，原預設保留。殘差粗略命中4→5、fall-09事件5.9→5.1秒，不冒稱正式警示時間。合計312新epochs；37風險測試、48原特徵parity差0、各46支次CSV／K3重算、1110舊檔與派生來源hash保全通過。設定註記修復及graph連線限制已揭露，核心不變；所有新候選未跑Test，Notebook／影片仍暫停。[完整診斷、結果、限制與操作](normal-data-support-20261012.md)。

## 最新完成：指定大容量候選Test，Accuracy78.26%／F1 .8000（2026-10-12）

使用者在三候選實驗完成後明確要求只評估大容量`ur-isotropic-wide-20261012-001`一次，固定Validation門檻.5341113805770874與原23支Test。TP10／TN8／FP0／FN5／影片unknown0，可判定及全體Accuracy均18/23＝78.26%、Fall F1＝20/25＝.8000、全體Fall召回10/15＝66.67%；漏報fall-27／23／21／20／03。Validation95.65%未延續至Test，不能宣稱泛化改善。未重訓／重切／校準／改選或修改舊採用guard，原預設保留。

13項相關風險測試、sklearn獨立Accuracy／F1／混淆矩陣與23支K3／事件核對通過，1103既有檔案hash不變。證據為真實快取的凍結AE評估，沒有新原影片Pose／ST-GCN重抽或demo。歷史Test已看，不是新盲測；正式Notebook／五分鐘影片仍暫停。見[本次Test結果與驗收](ur-isotropic-wide-test-20261012.md)。以下「未跑新候選Test」為當時實驗範圍，指定wide的後續授權以本段為準。

## 最新完成：Train／Validation 限定三候選，Validation95.65%／F1 .9655（2026-10-12）

Test 比 Validation 多2個漏報、少8.70百分點；兩邊FP／影片unknown皆0。Train／Validation診斷確認正常Val平均重建誤差是乾淨Train的2.03倍、既有門檻掃描最高21/23。事前固定compact／wide／denoise三個等比例特徵AE後，實際完成59／34／28＝121epochs。wide為TP14／TN8／FP0／FN1／U0，兩種Accuracy均22/23＝95.65%、Fall F1 .9655，較原最高Val多救回fall-18，剩fall-19；compact新增誤報、denoise無改善。wide的新TP在舊時段內當前良好骨架警示0幀，fall-09事件亦變晚，未通過採用guard，原預設不切換。26相關測試、sklearn與69支次K3／事件獨立重算通過，1039舊檔／41凍結run／真實權重hash不變。所有新候選未跑Test，舊Test82.61%／.8462保留；不能宣稱泛化或警示時間改善。見[本輪實作、成績與驗收](isotropic-refinement-20261012.md)。

## 最新完成：最高Validation候選已跑Test，Accuracy82.61%／F1 .8462（2026-10-12）

依使用者明確要求，固定`ur-human-isotropic-001`及Validation門檻.4529072046279907，評估原8ADL＋15Fall Test，不重訓／重切／調參。Test TP11／TN8／FP0／FN4／U0，可判定及全體Accuracy均19/23＝82.61%、F1＝22/26＝.8462、跌倒召回11/15＝73.33%；漏報fall-27／23／21／20。原Validation21/23＝91.30%不是Test成績，此次Test低於課程參考90%／.87，官方協定可比性仍未確認。

使用既有真實Pose／ST-GCN快取重新CUDA計算凍結AE分數，未重新逐影片擷取骨架；與synthetic風險測試分開。17相關測試通過，sklearn逐CSV重算及窗口K3嚴格門檻重建全部吻合；41旧run、92個Test相關快取、真實權重與核心程式hash保留，模型／Scaler／門檻／split不變。混淆矩陣已目視。結果另存`outputs/ur-human-isotropic-test-20261012-001`，既有guard與原保留模型不因這次Test改寫。歷史Test已查看、反覆Validation、未保證人物隔離及警示時間未驗證的限制保留。

Test工作已完成、無外部阻塞；正式Notebook／五分鐘影片仍暫停，未公開上傳。[正式結果、漏報、驗證與操作](ur-human-isotropic-test-20261012.md)。以下歷次「沒有新Test」為當時紀錄。

## 最新完成：依sample移除大字UNKNOWN（2026-09-18）

Human的auto現用sample：獨立原30次實際觀測七類forward，品質不足仍顯示分類並以小字提示；缺人不硬填動作，右側保存真實AE事件歷史／時間，當前AE仍可能unavailable。支援isotropic AE而不誤套其前處理；舊顯式actions／simple等保留。

完整232 passed／1歷史權重路徑skip／12warnings，39相關測試、lint／format通過。六片888幀的AE分數／狀態／事件CSV與改版前逐位元相同；674筆七類與快取Pose的原模型full forward分數差0，含81筆當前低品質分類；H264完整解碼並目視。41舊run、8764cache、44設定、5manifest、37舊媒體hash核對不變，覆寫拒絕實測。全體驗證完成，無外部阻塞。

移除大字UNKNOWN不是辨識改善：AE原基準仍20/23=86.96%、最高影片級候選21/23=91.30%，二元與含unknown整體分母均23；歷史Test／反覆Validation限制保留。fall-30的1.8667秒事件現在留在歷史欄，當前Pose品質不變，不回頭改舊guard或換預設；七類仍有STANDING UP誤判及正常adl-35單幀FALLING。沒有新訓練／Test、正式Notebook／五分鐘影片或上傳。本輪約5.70–8.68處理FPS，離線輸出30FPS，不宣稱即時效能。[完整契約、六支短片與操作](sample-display.md)。

## 最新完成：Validation提高至91.30%，可見警示條件仍未通過（2026-09-18）

三支漏報50張取樣影格／9張全解析度影格與33個粗略可見關節區域已目視另存，非官方GT。確認出框／模糊與逐幀獨立XY縮放丟失像素寬高比後，事前鎖定單一`per_frame_isotropic`候選，8片pilot通過並實際訓練49epochs。新`ur-human-isotropic-001` Validation TP13／TN8／FP0／FN2／U0，普通及全體Accuracy均21/23=91.30%、F1=.9286、全體跌倒召回13/15=86.67%；剩餘漏報fall-19／18。

六段舊粗略時段命中2→4、早報0，fall-09首次可見AE警示5.9→5.0秒，但會中斷。新補fall-30在1.8667秒有K3事件、全片0幀可見警示，因此未通過事前展示guard，**最高影片級Validation候選另存，預設仍保留原20/23模型**。不把91.30%宣稱Test成績或課程全部达標。

34相關測試及完整215 passed／1歷史路徑skip；六片888幀／559分數差0，狀態事件一致、H264解碼目視，原AE／ACTION CSV逐位元不變。40舊run、8450cache及設定manifest hash全部核對，70序列原Pose／排程未變，Train24ADL／3638窗Scaler重算吻合。無新Test評分／正式Notebook／五分鐘影片／上傳，無外部資源阻塞，歷史Test與反覆Validation限制仍在。[完整結果、短片與命令](human-isotropic.md)。

## 最新完成：骨架翻轉／正常活動抽樣，Accuracy未提高（2026-09-18）

正常地面活動與漏報診斷完成：Train已含相近地面運動／沙發側躺，不能稱缺少所有躺下動作。另交付五份原始影像接觸表與人工抽查CSV，非完整逐幀GT。翻轉AlphaPose的12片real_feature_smoke已跑完，三漏報跌倒區間合格Pose端點31→31、正常coverage比值.96584，未通過事前gate，不訓練該候選。

Train-only KMeans8型態抽樣AE實際33epochs，Validation仍TP12／TN8／FP0／FN3／unknown0、20/23=86.96%、F1=.8889；全體與可判定Accuracy分母都23，全體跌倒召回12/15=80%。依平手規則保留`ur-sample-human-s1-001`，三漏報不變；六支粗略時段命中仍2、早報0。39舊run、1165原cache、設定與權重hash保留。

26項相關測試與完整210 passed／1歷史路徑skip；Train-only Scaler和分群assignment／權重重算吻合。四支真實串流723幀、526個可比分數差0，狀態／事件一致，原模型AE CSV與事件逐位元不變；有對照的ACTION CSV不變，H264完整解碼並目視。fall-09仍5.9秒、fall-30仍漏報，沒有警示時機改善。沒有新Test、Notebook／五分鐘影片或上傳，無外部資源阻塞；新外部影片及正式逐幀標註尚未取得，90%目標仍未達。[完整結果、檔案與核對命令](human-quality.md)。

## 最新：兩種 Train 殘差加權完成，Accuracy 未提高（2026-09-18）

依使用者繼續優化準確度的指示，先診斷Train ADL各維重建誤差約56倍差異，事前鎖定逐維／雙分支兩候選，再以Train24支ADL、3638窗估計權重、Validation校準。沿用已凍結AE／Scaler及38回合歷史，新增AE epochs=0，兩個新run另存。逐維加權19/23=82.61%、F1=.8462；雙分支20/23=86.96%、F1=.8889，沒有改善。保留`ur-sample-human-s1-001`：TP12/TN8/FP0/FN3/U0，全體與可判定Accuracy分母皆23，跌倒召回12/15=80%。

原fall-19／18／30仍漏報，逐維加權另漏fall-16；六段粗略時段事件／可見警示命中皆2、早報0。實際重跑三版本fall-09共555幀、381個可比分數差0、狀態／事件一致，ACTION CSV逐位元相同，原MSE的AE CSV／事件也與舊版相同；警示皆5.9秒，H264完整解碼並目視，不能宣稱警示時機改善。37舊run、1165cache及設定／權重／舊影片hash保留。

完整回歸203 passed／1歷史權重路徑skip，CPU／CUDA原始及加權分數批次／串流4項通過。首次完整回歸的2項失敗來自舊測試跳過Scorer初始化，已補上測試欄位並擴充覆蓋，未更動凍結計分程式；初次失敗XML保留。新Human真實權重有執行。沒有追加Test、正式Notebook／五分鐘影片；沒有外部資源阻塞，Accuracy90%仍未達、歷史Test已查看／重複Validation限制不變。[完整結果、證據與命令](human-residual.md)。

## 七類動作展示完成（2026-09-18）

依使用者「幫我改」授權，原30幀Human分類頭已接回，畫面左側ACTION顯示七類動作，右側ANOMALY (AE)獨立呈現原AE結果；骨架不足／人物消失／過期仍UNKNOWN。CLI預設auto：相符Human使用actions，其他模型保留simple，不套用不相符分類頭。原512D特徵、AE、門檻、K3與事件未變，不把分類結果算成AE改善。

四支Validation503幀、313個分類窗口真實重跑，七類分數與凍結特徵重算差0，AE推論／事件CSV和改版前逐位元相同，H264全片解碼並目視。fall-09在4.70秒FALLING／AE NORMAL，兩者清楚分開；fall-30仍有STANDING UP誤判與後段UNKNOWN，不宣稱七類可靠性。37舊run、1165cache、原設定／權重／影片hash保留。32項相關測試及完整191 passed／1歷史路徑skip，最新15項分類測試亦通過。[新版影片、限制及操作](seven-actions.md)｜[整合證據](validation/action-display-integration.json)。

AE Validation仍20/23=86.96%、F1=.8889，沒有新訓練／Test或課程成績提升。正式Notebook與五分鐘影片继续暫停；此七類展示請用`scripts/verify_action_display.py`核對，舊實驗source lock不改寫。

## 本輪優化已完成：147epochs，Accuracy未提高（2026-09-18）

實際完成20幀／15幀／共用30幀座標尺度三候選，分別34／83／30epochs。固定Validation Accuracy分別20/23=86.96%、19/23=82.61%、19/23=82.61%，F1=.8889／.8571／.8462，全部不採用；保留`ur-sample-human-s1-001`：TP12／TN8／FP0／FN3／unknown0、F1=.8889，跌倒召回12/15=80%。可判定與包含unknown整體Accuracy分母均23，不等於每一影格可判定。普通Accuracy仍未達90%，不能以balanced accuracy90%替代。

共用尺度候選補到fall-30卻新增fall-29／11漏報。六段粗略事件時段命中2→3，但fall-30新事件完全被目前Pose品質遮罩，影片0幀FALL ALERT；fall-09提前至4.833秒的事件同樣被遮罩，畫面仍5.9秒才警示。未宣稱辨識或展示警示改善；三個候選依事前停止條件結束，無新Test，正式Notebook／五分鐘影片依使用者要求繼續暫停。

完整174 passed／1歷史路徑skip；新Human真實權重測試有執行。最新四片503幀、313個可比窗口分數差0，事件／可見狀態吻合、H264解碼與目視核對，原30幀CSV／事件與舊版逐位元相同。36個舊run及843個舊cache檔案hash保留，四組Scaler重算只用Train ADL、Validation全數重算一致。Lint通過；一個純換行format提示保留，避免修改訓練前鎖定原始碼bytes。無外部資源阻塞，歷史Test與反覆Validation限制仍在。[完整成果、分母、驗證及命令](human-displacement.md)｜[机器驗證](validation/human-displacement-verification.json)｜[短窗結果](human-context.md)。


## Human短時間窗事前設計紀錄（已完成，結果見上方）

已依上輪漏報診斷實作顯式15／20幀因果窗口，固定品質與K3；原30幀hash不變。44項相關風險測試通過，兩候選／pilot gate／選模與停止條件已在訓練前鎖定；尚未宣稱辨識改善。34個舊run與來源cache保留。[本輪契約與進度](human-context.md)。


## Human AE 四組優化已完成，Accuracy未再提高（2026-09-18）

四組容量／去噪候選事前鎖定，實際完成41／38／38／53epochs，共170回合。全部Validation仍TP12／TN8／FP0／FN3／unknown0：可判定與全體Accuracy均20/23=86.96%，F1=.8889，跌倒召回12/15=80%。依平手規則保留`ur-sample-human-s1-001`，沒有把較低正常Loss當成辨識進步。

三支漏報fall-19／18／30已核對原始影像、骨架與分數：跌倒時部分身體出框，品質下降／觀測中斷；此前合法event score低於門檻。這輪未改姿態、品質、K3或畫面，短促警示仍是限制；下一輪應先驗證骨架／時間上下文pilot，尚未實作。

23項相关風險測試全過（0skip，11.23秒）；30舊run、4新run與設定／來源hash保留，四組Scaler與Train24支ADL、3638窗重算完全一致。三個重跑／Test gate拒絕均已實測。無新Test、無新Notebook／五分鐘影片、無外部資源阻塞；90%目標仍未達。歷史Test與反覆Validation限制保留。[本輪結果、圖表、驗證及操作](human-ae-refinement.md)。


## Sample 流程已完成實作與比較（2026-09-18）

已取得同源Human公開鏡像權重並嚴格載入；新增預設關閉的追蹤框裁切、保留13→14點前處理與分類頭前512D雙流特徵，接固定UR正常Train-only Scaler／AE。两候選72／38epochs共110回合，選定`ur-sample-human-s1-001`：Validation TP12／TN8／FP0／FN3／unknown0，**20/23=86.96%、F1=.8889**；較舊17/23少3誤報。所有23支均可判定，條件式／全體Accuracy分母皆23。

Accuracy仍未達90%，Test gate已拒絕，沒有新Test或正式Notebook／五分鐘影片。四段粗略時段命中0→2、早報1→0；fall-09晚至5.9秒，fall-25畫面只警示一影格，不能冒稱時機與骨架已穩定。舊Test已看過，反覆使用Validation亦非獨立評估。

完整回歸160 passed／1舊權重路徑skip；新鏡像權重測試實跑。四支真實診斷影片518幀／333窗分數差0、事件／可見狀態相符、H264完整解碼；fall-29為追加的已知時段命中個案，其可見警示僅2影格。舊28個凍結run與原設定／來源index保留。詳細資料、來源限制、各階段證據與操作見[sample流程實測](sample-human-reproduction.md)。

## 2026-09-11 18:23 進度

- M0：CLI help 與 core doctor 已通過；Python 3.11.16、Torch 2.8.0+cu128；CPU 與 RTX 5060 GPU 算術可用，mp4v／MJPG 編碼後解碼成功。Python 下載使用 Windows 系統憑證，TLS 驗證未停用。
- M1：四個 Google Drive 模型資源均下載失敗（gdown 無法取得公開檔案）；真實模型驗證為 `blocked_external_asset`。原始 ST-GCN 次選亦需另備 OpenPose18 相容模型和 Kinetics 權重，目前未取得完整相符組合，未切換 recipe。
- M2：固定 70 序列 split 已保存。smoke 僅取 Train adl-05（180 PNG）及 Validation fall-02（110 PNG），290 張皆可解碼，frame ID 可對齊來源 timestamp；未拿 Test 調整。
- M3/M4：已實作 AE、scaler、校準、事件、coverage、sequence 指標與影片輸出；第一輪 synthetic 整合成功產出六 epoch 訓練、校準、測試報告與可解碼的 85-frame MP4。數字僅屬手製特徵測試，不是 UR Fall 成績。
- 目前進行 M5 風險測試與缺陷修正，尚未宣稱所有驗收通過。

命令／證據：

```powershell
.\.venv\Scripts\python.exe -m fall_detect --help
.\.venv\Scripts\python.exe -m fall_detect doctor --mode core
.\.venv\Scripts\python.exe -m fall_detect smoke-test --mode synthetic --output outputs/synthetic/first-integration
.\.venv\Scripts\python.exe -m fall_detect smoke-test --mode real
```

`outputs/doctor-core.json`、`outputs/asset-fetch-report.json`、`outputs/real_feature_smoke/smoke-report.json`、`outputs/synthetic/first-integration/smoke-report.json`。

## 交付驗證

最終測試證據：`docs/validation/pytest.xml`（40 passed、1 skipped），`docs/validation/cli-checks.json` 與逐命令 log。`docs/acceptance-results.md` 對規格第15節逐項標示證據範圍。最終 synthetic 成果在 `outputs/synthetic/final-validation/`；未取得的正式 run 不會出現在 runs/baseline-001。

修正了：缺圖保留完整時間軸、缺失timeline不縮coverage分母、tensor座標clamp、畫面unknown之後不復用舊分數、最佳validation checkpoint與early-stop的min_delta、cache fingerprint包含時間fallback及上游commit。骨架／模型圖、批次／串流、checkpoint与來源隔離驗證均有測試。

目前下一個可執行命令：

```powershell
.\.venv\Scripts\python.exe -m fall_detect doctor --mode core --device cpu
```

取得四個缺少資源並放在README指定位置後：

```powershell
.\.venv\Scripts\python.exe -m fall_detect smoke-test --mode real
```

只有此命令在兩個真實序列都產出有效512D特徵後，才接續README的完整下載→prepare→train→calibrate→evaluate→demo。這次不把synthetic的數字或影片當成真實跌倒偵測成功。

## 2026-09-17 替代路徑進行中

使用者已選擇第二條路。已取得原始 ST-GCN 官方 Drive 資料夾中的 st_gcn.kinetics.pt，以及 Lightweight OpenPose 官方 README 指向的 checkpoint_iter_370000.pth。兩者受限載入成功，state_dict 分別 172 / 426 個 key，與完整模型鍵名、尺寸匹配。Lightweight 模型完整載入三個 refinement stage，推論取作者 demo 的第一個 refinement stage。

正在實作獨立 OpenPose18 / 256D recipe、CPU/CUDA 真實 feature smoke、快取與展示介接。尚未宣稱真實端到端完成；原 Human recipe 保留。上游固定 commit：ST-GCN 221c0e152054b8da593774c0d483e59befdb9061、Lightweight OpenPose d23c284b09acf27a163e1febd511e7482cac25ed。

### 2026-09-17 真實特徵與風險測試通過

- 原始 ST-GCN / Lightweight OpenPose 權重 strict load 成功；GPU 實際重算 smoke：adl-05 18 個有效窗、fall-02 5 個有效窗，全部256D且有限值。`outputs/real_feature_smoke/openpose18-cuda-v2/smoke-report.json` 記錄實際forward旗標與strict報告；分類為 not_performed。
- 新增非正方形／缺點mask／前處理與上游parity、18點cache、256D cache、recipe隔離、時間缺口及CPU/CUDA真實權重測試。48 passed、1 skipped（skip僅原Human缺少四個權重），`docs/validation/pytest-openpose18.xml`。
- CPU的oneDNN深度卷積路徑非常慢；局部改用PyTorch native CPU kernel，保留參數／裝置不變。CPU/CUDA均實際執行真實影像pose和真實骨架窗ST-GCN；CPU完整120幀smoke因耗時中止，不宣稱CPU全序列smoke通過。
- 已查看兩段真實骨架的六張抽幀。站立／坐下關節合理；fall-02躺倒後有漏點、tracking碎裂，保留unknown。預覽 `outputs/real_feature_smoke/openpose18-pose-preview.jpg` 僅pose，沒有AE分類。
- 完整70段資料正在下載；下载改為最多4段並行，各段獨立HTTP session，保留原zip/hash/atomic publication與固定split。完成後才prepare→train→calibrate→evaluate，不以Test選設定。

### 完整資料檢查的修正

第一輪完整inventory為69/70；adl-37最後20張RGB缺少官方同步時間。已按規格允許的FPS fallback加入顯式、僅尾端的時間推估，原時間不動、所有估計影格保留旗標、recipe hash更新；中間缺口仍拒絕。資料風險測試通過後啟動完整GPU prepare，並重新跑全資料inventory。

### 正式訓練前的分數一致性修正

校準／評估原先一次傳多個特徵到AE，demo每次一個特徵；浮點GEMM的批次形狀可能造成門檻邊界差異。現在Scorer固定以B=1計分（訓練仍使用minibatch32），保留原來strict >比較、不加容差繞過門檻。新增CPU/CUDA逐筆與整批輸入bitwise一致及score==threshold不觸發測試。來源文件另外保存training_source_code_hash與inference_scoring_batch_size，區分擷取快取時與訓練時的程式版本。此修正在Test評估前完成。


### 2026-09-17 第二條路交付

完整70段prepare完成，777個有效256D窗；正式run openpose18-baseline-001實跑100 epochs並以Validation校準，接著只做固定Test評估。實際結果：23段中8 unknown，15段跌倒中7段警示，8段正常中7段誤報。所有跌倒的端到端召回46.67%，不能只報排除unknown後的87.50%。完整結果及來源hash見results-summary.md。

已完成Train adl-05、Validation fall-02及衍生30fps MP4輸入三支真實展示，全部影格179/110/110重新解碼。可比有效端點分數差0，狀態與events一致；模型及門檻等凍結hash再次核對不變。另修正展示EOF時間對齊實際來源結尾，不改Test、模型或門檻。

全套54 passed、1 skipped（僅原Human四資源缺失），68.50秒；5項streaming回歸包含非整格來源EOF時間。已查看正式訓練曲線、混淆矩陣及影片抽幀，保留躺倒後骨架碎裂與unknown。操作指南現可直接使用既有run；此為可執行但品質不足的基準，不宣稱可靠偵測。


## Le2i擴充進行中

已确认官方Le2i ZIP可取得（9,608,701,260 bytes），固定OmniFall revision 83572a37b9e3081df8c06a56874b1d1f2a19386c。依其人物切分，38段無fall/fallen標註影片可加入Train，19段Validation、38段Test；Train分組95段含fall/fallen影片排除。UR Fall原70段及split保留。

部分標註前段／中間有空白，不能視為正常；新增資料只有完整落在已知標註區間的窗口才產生特徵，空白保留unknown。新protocol/cache/run獨立，目前仍下載／實作，尚無擴充版訓練成績。

骨架候選僅使用Train adl-05及Validation adl-37/fall-02/fall-30。384像素＋第3 refinement只多1個有效窗，未見明顯改善；另測試真實影像0/±90度推論並還原原圖座標，不合成缺點。graph index_status/search/coverage因Transport closed不可用，直接讀src及必要ignored upstream補足，未假稱graph驗證通過。


### Le2i來源核對及擴充實作

官方完整ZIP下載並解壓CRC通過；190段AVI全部有固定標註。檢查內容hash發現兩組Validation重複影片，副本時間標註不同，整組四段排除；95段Train含跌倒來源仍排除。最終Le2i採38正常Train／15Validation／38Test，連同UR Fall共161段（62／38／61）。重複來源跨split直接拒絕。來源與排除原因保存data/manifests/le2i-extension。

條件式旋轉pilot有效端點30/28/7/1，僅fall-02比原版增加2窗。真實六張抽幀仍可見躺倒後重複track、斷裂及家具誤偵測；沒有宣稱骨架修復成功。選擇先於新Test評估固定，full prepare及完整影片解碼檢查進行中。新增快取重用語意/內容隔離及影片提早解碼結束拒絕測試。

Graph連線已恢復，generation 2026-09-17T03:34:38Z；本次Le2i、split、reader、pipeline、pose、evaluate與video相關路徑檢查未記錄coverage缺口，並閱讀實際來源。先前Transport closed期間的fallback紀錄仍保留。


### 完整資料與環境已通過

161/161完整解碼，UR Fall 11,936張＋Le2i 42,406張，共54,342張；Le2i時間均有影片PTS，UR adl-37既有20張尾端estimated不變。doctor full CUDA通過。全套69 passed／1 skipped（僅舊Human缺權重），15項Le2i風險測試含去重、跨組拒絕、標註空白、旋轉座標、條件推論、cache重用與不完整解碼；證據docs/validation/pytest-le2i.xml。原run凍結hash核對未變。


### 完整擴充特徵處理中

原UR Fall的新設定70/70完成，798個有效256D窗，與舊不旋轉基準777窗分開。已用語意/內容hash核對重用至新161段協定，目前Le2i持續逐段處理。另查看Le2i Validation Coffee_room_01/video_42的1、3、5秒實際骨架抽幀，沒有使用Test挑展示或改設定；分類尚未執行。


### 新AE訓練及校準完成

161/161全部prepared，共6,404個256D特徵；正常Train來源62段中45段有可用特徵，總計3,766窗（UR346＋Le2i3420）。Le2i新增38段正常來源中26段有特徵。真實Pose／ST-GCN forward均通過。run le2i-extension-001實際79 epochs，以正常Validation最低loss保存checkpoint（0.26481568650342524），依既定Validation規則校準門檻後凍結。

關鍵限制：Le2i的4段正常Validation全部零有效特徵，正常Validation實際僅UR Fall的7段、128窗。不能宣稱已驗證Le2i正常場景的低誤報，也沒有使用Test補足。


### Le2i擴充最終交付

固定Test：UR Fall誤報7/8→6/8，跌倒仍7/15警示、unknown8/23；Le2i跌倒15/22警示、正常11/16誤報、unknown10/38。混合61段中18unknown，全部跌倒召回59.46%，條件式F1 68.75%不能冒充全體準確率。模型、scaler與門檻在評估後未重選。

真實展示UR Validation fall-02（110影格）及Le2i Validation Coffee42原生AVI（190影格）完整重新解碼。UR七個、Le2i28個共同有效端點raw score差0；UR狀態/events相同。Le2i標註限制會提前結束cache事件，所以不宣稱其完整events與無標籤demo相同。第一個警示1.967秒早於標註跌倒3.904秒，是明確的早段誤報，已完整揭露。

新舊run凍結hash再次核對不變，正式曲線／混淆矩陣／六張影片抽幀已視覺檢查。兩組真實權重本次無阻塞；原Human仍缺四資源，CPU完整實驗仍未驗證。這次已完成授權範圍，品質改善仍有限；下一步需要改善正常Validation與追蹤，不能用已看過Test持續調參。


### 2026-09-17 補齊Loss圖與後續擴充查核

由原基準及Le2i真實training-history.csv匯出三張300dpi PNG及對應SVG、原CSV副本與圖說，位於docs/figures/loss及docs/loss-figures.md。原基準最佳Epoch90／0.496766；Le2i最佳Epoch69／0.264816，實跑79回合。圖中揭露Le2i四段正常Validation無有效特徵，並說明兩個run的Scaler不同，不以跨圖Loss直接推論辨識改善。

scripts/export_loss_figures.py已實際執行，逐值驗證曲線與CSV、最佳值與status相符，所有凍結檔案hash未變；PNG解碼、SVG結構與ruff通過，三張圖已視覺檢查。本次沒有重新訓練或重跑Test。CAUCAFall／UP-Fall官方來源已查核，下一步順序記錄docs/next-expansion.md；未將候選資料宣稱為已接入或已改善。

### 2026-09-17 Loss 整合進程式畫面

依使用者澄清新增 `python -m fall_detect loss` 及「查看Loss曲線.cmd」。本機瀏覽器介面可切換兩個真實run、顯示Train／Validation曲線、最新值與最佳回合、滑鼠讀值、逐回合表格及SVG／CSV下載。`train --show-loss`可開啟同步畫面；train在每個完整epoch後原子寫入training-progress.json，未改資料、模型、early stopping或門檻規則。

實際瀏覽器驗證：Le2i 79回合／最佳69，原版100回合／最佳90，兩条曲線各79點與79列表格；窄視窗曲線與中文可讀。另執行明確獨立的synthetic AE 3回合，瀏覽器自動由Epoch2更新到Epoch3及完成狀態，未產生真實辨識成績。UI驗證發現舊／synthetic index未提供特徵數量時不應當0，已修正為未提供並補回歸斷言。

全套76 passed、1 skipped、11 warnings（原Human缺四資源、舊套件及CUDA deterministic提醒），23.72秒；最後顯示修正後7項相關測試再次通過。ruff、format、compileall通過；兩個正式run全部凍結hash仍相同。證據docs/validation/loss-viewer-checks.json與pytest-loss-full.xml，操作見docs/loss-viewer.md。沒有重跑正式訓練或Test。

### 2026-09-17 接續訓練：單獨比較骨架設定

已鎖定新run openpose18-rotation-001：使用既有完整UR Fall 70段的條件式旋轉快取，維持原始split、品質門檻與AE預設。訓練只讀正常Train，正常Validation選checkpoint，Validation選門檻；本輪不評估Test。開始前核對47段Train/Validation快取、真實权重與兩個既有run凍結hash。此為補足骨架改動對照，Le2i正常Validation可用性仍待修復。固定設定見validation/rotation-training-lock.json。

本輪已完成CUDA真實AE訓練100回合，最佳Epoch95、正常Validation Loss0.4996334114985075，校準門檻0.2031607925891876。實際Train19段346窗，正常Val7段128窗。相同23段Validation：跌倒警示8/15→9/15，正常誤報仍6/8，unknown9/23→8/23；此為校準集結果，不是新Test或可靠偵測證據。

Validation fall-02真實demo110幀完整解碼，7個共同有效端點分數差0、狀態/events一致；已查看真實抽幀，重複骨架與低品質仍存在。11項訓練／Loss相關測試通過（5.77秒）；原兩個run凍結hash不變。新run已出現在Loss畫面，100回合與Epoch95最佳值均實際檢查。沒有變更核心程式，未再跑完整測試或Test評估。結果見rotation-training.md與validation/rotation-training-*.json。Le2i正常Val缺有效骨架仍是後續重點。

### 2026-09-17 改善：查出低FPS重採樣造成的假缺幀

已查明Le2i四段正常Validation皆為24.000384 FPS（間隔41.666ms），原重採樣僅允許舊觀測保留33.333ms，每5個target tick約有1個被當missing；穩定track即使存在，也無法形成30幀有效窗。先前只從track_gap分類推測骨架追蹤不足不完整，這輪用原始pose快取與時間戳定位到明確程式原因。

新增顯式source_interval重採樣政策：依影片標頭FPS保留最近已到達的真實觀測，最長max(target interval, source interval)，不用未來影格或未來時間間隔，真正缺幀與超時仍unknown；原run保留target_interval政策及舊fingerprint。讀取器與pose cache保存source_fps及是否fallback。新config openpose18-le2i-timing.yaml，cache另存。

四段Validation pilot有效特徵由0/0/0/0變為38/49/34/37，保持12關節／80%品質、原track與標註區間。新pose-only reuse僅容許hold-policy變動，驗證來源、split、權重與原pose array hash；特徵全部重新真實ST-GCN計算。full prepare進行中，尚無新模型成績。88 passed／1 skipped（原Human缺四資源）、11warnings；新12項測試含24FPS回歸、因果性、真缺幀、30FPS舊行為、hash隔離與immutable重用。鎖定見validation/improvement-timing-lock.json。


### 2026-09-17 改善完成：時間修正、真實訓練與門檻對照

161/161完整特徵重建完成；原真實Pose array與track ID經hash核對沿用，所有ST-GCN特徵重新計算。完整index與訓練snapshot相符。Train正常可用45→57/62段、3766→4180窗；正常Val可用7→11/12段、128→286窗。le2i-timing-001實際CUDA訓練88回合，最佳Epoch78／Validation MSE0.2042546279311597；預設F1門檻0.14055135846137998。

相同38段Validation：unknown18→8，但F1基準正常誤報6→10/12、跌倒警示14→20/26，不能稱為誤報改善。另做明確替代協定le2i-timing-balanced-001，沿用同一AE/scaler/Loss，只以Validation平衡全類別正確率校準，unknown仍保留分母；門檻0.4929388761520386，正常誤報1/12、正確正常9/12、跌倒警示12/26、漏警示8/26、unknown8/38。與修正後F1相比少9段誤報也少8段跌倒警示；未更換預設、未重跑Test。不是泛化或可靠警報證據。

真實demo正常Home02/video47與fall-02完整重新推論，48＋7個可比端點raw score差0。正常影片1端點因當下低品質隱藏分數，4個狀態因cache標註區間與無標註demo差異而不同，兩邊皆無警示；UR狀態／事件完全相同。抽幀已視覺檢查。Loss畫面已顯示88回合、最佳78、新資料數量與替代run沿用提示。

全套92 passed／1 skipped／11 warnings（17.76秒），ruff/format/compileall通過。舊三個run與新F1run凍結hash不變；原Human缺四資源、完整CPU／新場景仍未驗證。操作與完整取捨見timing-improvement.md，證據validation/improvement-*.json及pytest-improvement-full.xml。此輪工作已完成。


### 2026-09-17 繼續訓練：固定三組 AE 比較

依使用者要求繼續改善。Train可用4180窗中UR346窗（8.28%）、Le2i3834窗（91.72%），長短影片權重不均。固定三組：LR0.0003/最多250回合/patience30，分別原窗口抽樣、影片均衡抽樣、影片均衡且latent16；其他維持既有設定。獨立seed42重訓，並非從Epoch78接續。每組checkpoint仍僅由正常Validation MSE選取，三組F1及明確平衡門檻均保留；Validation比較會完整列出。本輪不再增加候選、不讀Test特徵評分。鎖定docs/validation/further-training-lock.json，診斷further-training-diagnosis.json。


訓練中le2i-long-001在發布165回合後遇WinError5原子替換進度JSON失敗，程序退出1，未完成checkpoint；原目錄保留並標failed。此與Windows讀取/替換競爭一致，未直接證明是哪個程序鎖住。新增只針對Windows5/32/33的有界重試（最長約1.06秒），持續錯誤仍拋出並標訓練失敗；舊完整檔與暫存清理有回歸測試。以原相同config/seed重新建立le2i-long-002，不新增第四組參數。詳見further-training-recovery.json。均衡組已完成172回合；compact仍訓練中。


### 2026-09-17 接續訓練完成：準確率未提升，如實保留比較

三組成功完成577回合：long-002 250回合／最佳237 Loss0.203127；seqbalanced-001 172回合／最佳142 Loss0.197624；seqcompact-001 155回合／最佳125 Loss0.285058。均衡組Loss下降，但F1門檻全為TP20 FP10 TN0 FN0 unknown8/38（52.6%）；平衡門檻long／seqbalanced仍TP12 FP1 TN9 FN8（55.3%），compact TP10 FP0 TN10 FN10（52.6%）。locked平衡指標選compact僅提高0.3205百分點，整體正確率反而下降，不宣稱辨識改善，也不替換原55.3%基準。未評估Test。

WinError5後同config/seed重跑long-002完成250回合，前165回合Loss逐值差0；原failed run保留。有界Windows原子替換重試與失敗狀態顯示完成。100 passed／1 skipped／11 warnings（19.01秒），ruff/format/compileall通過。兩段真實demo359/110幀、48＋7共同有效端點分數差0，UR狀態/events一致；Le2i4狀態差與1低品質遮罩如實記錄。兩張抽幀與Loss172回合/最佳142畫面已驗證。舊五組凍結hash不變，所有新衍生門檻沿用同模型/Scaler/Loss。原Human資源與完整CPU/泛化仍未驗證。完整說明docs/further-training.md，本輪固定比較已完成。


### 2026-09-17 其他影片來源再次查核

依使用者詢問查核官方公開來源：CAUCAFall v4（10人各10活動、100段AVI及標註、23FPS）、UP-Fall（17人11活動、CameraX影像ZIP；官方缺資料註記保留）、GMDCSA24作者GitHub及Zenodo v2.0（公開1.1GB ZIP，頁面提示較新版本）。蒙特婁Multiple Cameras Fall官方頁本次Access Denied，不冒稱可直接下載。已修正next-expansion.md仍把Le2i正常Validation列為零特徵的過時描述；低FPS問題前輪已修復。本次僅查核來源與公開入口，未完整下載、未接入訓練；先前577回合仍只有UR Fall＋Le2i。


### 2026-09-17 CAUCAFall 原始影片接入中

已核對第三方整理包包含切段／重複版本，未納入。改從原作者 v4 ZIP 以 Range 下載 100 支完整 AVI（135,044,978 bytes），逐支 CRC 與 SHA256；整包 8.33 GB 不下載，官方整包 hash 未實測。人物 6/2/2 事前切分，僅 30 支新 ADL 進 Train、30 支 Train Fall 全排除。原 161 序列保留，總 231 序列。相同特徵 recipe 與 AE 規格預設，舊模型保留。來源下載／風險測試進行中，尚無新準確率。詳見 [實驗協定](caucafall-training.md)。

CAUCAFall 100 支原始 AVI 下載完成；全 231 支 raw 解碼通過。首輪 smoke 揭露 20 FPS 浮點時間比較造成斷軌，顯式 1e-7 ms 容差修正並保持舊預設／fingerprint。新 smoke 真實 49/19/62 個特徵；161 舊序列逐幀 reset 決策相同，驗證後重用。44 項風險測試通過，完整 prepare 正進行；未計算新 Test 分數。

完整 prepare 231/231 通過；CAUCAFall 新 Train ADL 30/30 有效，共1096窗；合併 Train 87/92 有效、5276窗，正常Validation 21/22有效、637窗。原161支特徵數值/窗口紀錄/Pose array hash 完全一致。run caucafall-extension-001 已開始CUDA訓練；Loss即時畫面已切換並核對21回合及正確資料數量。


### 2026-09-17 CAUCAFall 本輪完成

97 epochs / best87 / Val正常MSE0.198277；F1與平衡校準同門檻0.41994223，兩者均先凍結再Test。新CAUCAFall Test 75.0%（TP6/TN9/FP1/FN3/unknown1），完整81支59.3%（TP22/TN26/FP7/FN16/unknown10）。原38支Val仍55.3%；同61支Test由舊平衡50.8%到54.1%，少3誤報也少偵測1跌倒，未宣稱全面改善。兩支Validation全新demo 338/299幀，62/18可見端點分數差0、states/events一致，Fall另1低品質端點遮罩；首次警示不是事件延遲驗證。Loss97回合與最佳87已實際瀏覽器檢查。原161序列特徵/紀錄/Pose陣列完全一致，舊五run與新兩run frozen hash不變。117 passed/1 skipped/11warnings，ruff/format/compile通過。資料、權重及快取未加入Git。本輪已完成，等待使用者看成績後再決定後續。


### 95% target: timestamp experiment in progress

Train/Validation diagnosis found integer-millisecond UR timestamps expiring real poses at 33.6667 ms. Testing an explicit 1 ms expiry tolerance with unchanged grid, observations, tracks, quality and K=3. Old defaults/hashes preserved. New cache/run; Train ADL only, Val checkpoint/threshold, no Test selection. Evidence: validation/target95-lock.json and target95-diagnosis.json. No new accuracy yet.


YOLO11 official 42,459,307-byte checkpoint acquired, actual SHA saved; 691 tensors strictly loaded into pinned ultralytics 8.3.228. Eight real-video pilot sequences pass the predefined coverage gate (5 to 6 evaluable); full fresh pose/feature extraction underway. Added a pre-Test bounded-jitter comparison: real UR capture timestamps average 33.3 ms but fluctuate up to 47 ms. Hold only existing past poses up to 50 ms, retain actual missing observations/track breaks, then retrain separately. Final candidate set locked; no more variants this round. Full suite 130 passed/1 original-Human skip.

### 2026-09-17 95% 目標本輪結果

完整 231 支全新骨架／特徵準備及時間對照完成。三組實際新增94＋99＋100＝293回合。時間容差版Val62.1%，YOLO版72.4%，YOLO＋有界抖動版74.1%，原版65.5%；依事前規則選後者，100回合／最佳95，正常Val MSE0.149559，門檻0.3282490372657776。Train正常92/92段6096窗、Val正常22/22段1026窗，衍生neck不灌入品質計數。

固定81支Test正確48→57（59.3%→70.4%），CAUCAFall15/20→18/20（75%→90%），Le2i76.3%，UR Fall43.5%；unknown10→1，總TP30／TN27／FP7／FN16。95%尚未達到。CAUCAFall10支Fall都有警示但2支ADL誤報；不是起跌時間驗證。Test後沒有調整門檻或切換候選，歷史Test重用限制已揭露。

兩支真實展示338／299幀解碼、107個有效端點分數差0、狀態與事件一致，抽幀已查看；程式Loss已顯示100回合與最佳95。132 passed／1原Human資源skip／12warnings，ruff/format/compileall通過。16個frozen run核對通過，4候選的Test前hash不變；100支原始CAUCAFall hash再確認。方法、結果、操作與未達標原因見[target95-improvement.md](target95-improvement.md)，本輪無訓練程序仍在背景執行。

## 2026-09-17 — 影片 sample 展示

- 使用 `runs/caucafall-yolo11-jitter-001` 重新執行兩次 `python -m fall_detect demo --no-display`，輸入為 CAUCAFall 驗證集 Subject 5 的 FallForwardS5.avi 與 WalkS5.avi；輸出 `outputs/sample-fall-001.mp4` 和 `outputs/sample-walk-001.mp4`，並保留 JSON、逐窗 CSV 與事件 CSV。
- 真實完整推論：跌倒影片 299 幀，第一個警示在影片 4466.67 ms；正常走路 338 幀、0 警示。此為觀看範例，不新增準確率或偵測延遲宣稱。
- 增加 `outputs/sample-viewer.html`，本機觀看網址 `http://127.0.0.1:8767/sample-viewer.html`。使用 `.tools` 中的 imageio-ffmpeg 0.6.0 將實際輸出轉為 H.264 MP4（`sample-*-001-browser.mp4`），保留原始模型輸出；轉檔完整解碼檢查通過。
- 內建瀏覽器 WebM 播放嘗試曾當機；切換 H.264 後，已觀察跌倒影片實際播放至 7.69 秒（paused=false、readyState=4）及紅色疑似跌倒警示。沒有修改模型、資料切分、門檻或推論規則。

## 2026-09-17 — 下一輪 AE 訓練改善（執行中）

Train/Validation 診斷：上一版在 100 回合上限結束、最佳為 95；92 支 Train ADL 的 6096 窗中 UR Fall 659、Le2i 3983、CAUCAFall 1454，單支可用窗數 9–302。Validation 全部 58 支可判定，但有 11 FN／4 FP，坐下及跪下也有誤報。

已於訓練前鎖定四組 2×2 實驗：200 回合上限／patience20，latent32 或16，以及 window 或 sequence 抽樣；其餘沿用 YOLO11-jitter。只用 Train ADL 訓練、正常 Validation Loss 選 checkpoint、完整 Validation 選門檻及候選。Test 不選參數，四組完成後先凍結再評估單一選定版本；若全部不優於原版則保留原版。本輪不追加候選追逐 Test。

- M1：完成 Train/Validation 診斷與事前鎖定 `validation/refinement-lock.json`，16 個既有 frozen run 已驗證。
- M2：執行 `scripts/refine_yolo_training.py`，呼叫共用 train/calibrate API，真實特徵快取不變；Loss 仍寫入程式內建 viewer。
- M3：待四組完成後進行選定模型的 Test 與真實影片整合驗證。
- M4：更新比較報告及驗收；目前尚無本輪提升成績。

### AE 強化中途檢查

相關風險測試 25 passed（抽樣／Train-Validation-Test 隔離／重載／門檻／Loss viewer）。長訓練版前 100 回合的 Train 與正常 Validation Loss 與原版逐值完全一致，證據 `validation/refinement-long-reproduction.json`。

前兩組已完成：long165回合、best145，Val43/58（TP25/TN18/FP4/FN11），與原版相同；compact131回合、best111，Val41/58（TP20/TN21/FP1/FN16），誤報減少但漏報增加。目前繼續事前鎖定的兩組 sequence 抽樣，尚未選定／執行 Test。完整 Validation 錯誤診斷保存於 `validation/refinement-validation-diagnosis.json`。


## 2026-09-17 — 期末課程對齊文件完成

已根據教師簡報第36–51張、課程網頁及同學附件查核結果，新增 `docs/course-alignment-plan.md`。根規格更新為v1.1，README與驗收文件同步連結；課程明定要求、工程選擇與待釐清事項分開。下一階段按CA-01至CA-05執行：UR Fall協定／Train-Validation診斷、UR-only基準與改善、成果Notebook、中文展示、五分鐘影片。

記錄目前YOLO11m→OpenPose18→ST-GCN256D→正常AE作為可用基準，Human512D保留歷史契約。明確區分UR10/23整體正確率、可判定部分10/22與F1=33.3%；舊Test重用限制與既有balanced校準保留。展示可將精確age移至除錯模式，但須保留更新時間與失效unknown語意。

本次只完成文件與本機附件SHA256紀錄；CA-01至CA-05皆待實作。未改程式、資料切分、模型、門檻或推論，未訓練、重跑Test、產生Notebook／新版影片或公開上傳。既有實驗結果與測試紀錄照原樣保留。

文件查核通過：五份文件UTF-8可讀、77個本機文件連結存在、兩份附件實際hash吻合、階段狀態與交叉連結一致；UR指標算式核對，並完成獨立的有界文件審查。僅文件變更，未執行模型測試。

## 2026-09-18 期末課程實作完成

CA-01至CA-05本機工作已完成，詳見course-results.md。正常Train24支659窗；3新run訓練100／100／97回合，Validation選course-ur-balanced-001。Test前凍結後只評選定run：TP7/TN4/FP4/FN7/U1，全體11/23、可判定11/22、F1 56%。之後沒有追加候選或改門檻。

Notebook7 cells全數執行並重啟kernel再檢查；4支真實demo共628影格／模式，debug與中文模式log/events一致，67個未遮罩窗口分數差0，1個窗口依既有品質遮罩。五分鐘有聲／無聲版均完整解碼，9000影格／30fps／300秒，視訊流hash一致。13段抽幀及4狀態可讀性已查核。

136 tests passed、1舊Human缺權重skip；ruff/format/compileall通過。20個舊run、舊cache index及新候選frozen檔案保持不變。首次pytest與Notebook/TTS曾遇沙箱權限，授權後完成；失敗證據保留。第二次Notebook只有IPython history無法寫入的非cell警告，所有cell仍成功。沒有訓練或推論留在背景。

教師官方split／評估單位／unknown與公式仍待確認；沒有課程分數換算、沒有公開上傳。這是可重現的工程成果，不是可靠跌倒辨識驗收。命令及成果入口見course-results.md；機讀最終驗證見validation/course-final-verification.json。
