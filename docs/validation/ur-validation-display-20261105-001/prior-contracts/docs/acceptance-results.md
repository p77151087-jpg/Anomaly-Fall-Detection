# 驗收結果

## 三段本人時刻界限已驗證、警示對照完成（2026-11-05）

- 上傳JSON原bytes／SHA另存；3/3本人自述確認、非draft、正確原bundle且重算內容hash、原影格整數界限／順序、218PNG／原source SHA／時間戳及Validation來源皆過。這是本人界限、非多人獨立精確GT；不把開始／落地兩端點當整段姿態phase。
- wide三片TP與時間分開：fall-18首次1.433333在落地1.301–1.534；fall-09首次5.900在4.705–5.973；fall-30首次1.866667在1.635–1.868。三首次可能包絡3/3、保證區間0/3、落地不確定範圍3/3；確定落地前0/3、確定落地後首次0/3，不虛稱全準時或全太晚，不外推15／23片人工時序命中率。
- 首次／後續／關閉分開：fall-09五後續active在6.000–6.133333超過最晚落地5.973；不是五影片誤報或已證明首次太晚。fall-18／30當前good各0/1、fall-09 8/8；三片整段mostly_matches／出框備註原樣保存，不把confidence當joint正確率或聲稱RGB顯示驗收。本輪demo仍18／30 FN，09 TP 5.9秒。
- 最終41 synthetic（23新＋18既有）及lint／format通過；獨立不使用新helper的Decimal原時間／23 CSV矩陣與分母／233窗K3／3事件／10active／延遲區間及五個落地後端點全一致。新增兩項測試保證Windows兩種分隔符都核封存副本，且遭改仍拒絕；文件更新後原current-doc誤查失敗／初版source及inputs另存，只修路徑、未改結果或放寬SHA。real_feature_smoke／real_end_to_end本輪未執行，沒有模型／特徵陣列載入或新模型辨識改善。
- 新outputs／QA及23列human補充表，舊23欄位逐一不變、其餘12片Fall本輪時間未知；原263有界hash、231舊outputs、前輪29artifact及兩修改前文檔bytes保全。原human pending歷史報告、bundle／AI phase不覆寫；重複build拒絕及新檔hash不變另存。Graph／coverage及heuristic／排除／best effort限制保留。
- 不訓練、推論、校準、讀取或評估Test，不換demo／模型／品質／K3；wide正式兩Accuracy22/23＝95.65%、F1 28/29＝.965517、矩陣[[8,0,0],[1,14,0]]不變。Notebook／五分鐘影片／公開上傳仍暫緩。本人覆核已完成3/3，不確定的是實際落地時刻，不是格式或漏交；無需本輪再做大量逐幀標註。

[完整界限、對照、證據及命令](ur-validation-human-timing-20261105.md)｜[只讀比對方式](ur-validation-human-timing-20261105-plan.md)｜`docs/validation/ur-validation-human-timing-20261105-001/independent-verification.json`。

## 既有23片Validation警示／品質盤點：完成、人工精確時間pending（2026-10-13）

- 全23片wide正式CSV／2917窗／14事件整理；獨立矩陣[[8,0,0],[1,14,0]]、FP0／FN fall-19／U0，兩Accuracy22/23、P14/14／R14/15／F1 28/29、unique coverage2287/2710一致。demo另列既有20/23／.888889，不混模型、未改成績／門檻／K3／採用。
- 時機診斷與影片級分開：原六片coarse仍4/6事件、2/6 good警示、粗early0；15片AI possible13/15、core6/15、possible內good11/15皆未有人類精確時間GT，不替代原guard或教師公式。active endpoint當前good≠實際RGB renderer可見警示；confidence≠骨架幾何正確。
- 三段fall-18／09／30原PNG播放器及一份JSON覆核頁實際開啟；共218張播放素材但只需三笔時間／品質，不重做fall-19 100筆或已完成joint。原phase保留transition21/29／ground0/38，FN不變。真頁0/3、未代填或勾本人；沒有把AI phase冒充human時刻。
- 18Python synthetic／22離線假DOM断言全過；獨立直接Pose3576原幀／track與bisect3377採樣、每窗K3／14完整事件及分母核對一致；218圖SHA原來源一致，實際UI三片導覽／末幀／骨架開關／正式匯出disabled、HTTP200及截圖保存。real_feature_smoke／real_end_to_end本輪未執行，不把既有輸出重核混成新辨識實驗。
- 329有界source hash及前輪48finalized artifact保全，兩修改前契約bytes封存；build與append重複拒絕，舊成果不覆蓋。新test import空行／Windows path pin初失敗、初沙箱socket拒絕／timeout及經批准127.0.0.1服務修復均留紀錄，無模型／結果修復。Graph335paths三批coverage、設計排除直接原檔核對、legacy trace／suffix及best effort限制另存。
- 沒有模型推論／訓練／校準或Test特徵／影像／成績訪問；新Test未知、原demo保留，沒有提升宣稱、Notebook／五分鐘影片／公開上傳仍暫緩。三段human self-reported時間輸入待收到後另核對。

[成果、分母、三段人工操作及命令](ur-validation-alert-quality-20261013.md)｜[只讀範圍](ur-validation-alert-quality-20261013-plan.md)｜`docs/validation/ur-validation-alert-quality-20261013-001/independent-audit.json`。

## 人工整副骨架六點控制：實跑完成、仍漏報、不採用（2026-10-13）

- 原JSON bytes及SHA另存，3幀×13＝39判斷驗來源／bundle／原Pose與RGB／manifest／同track／時間／joint mapping；26符合、6可見錯位、7不可確認。只換47的5/6/9/10/11及49的1六XY，不合併旧手肘，不補不可見／品質mask，44標註粒度衝突原樣保存。
- 新獨立protocol／recipe／cache凍結原wide30 isotropic、strict真實512D、原Train Scaler／AE、門檻.534111／K3；raw .1839069426→.1838027239、K3 .1346364915→.1472322047（+9.355%），零raw／K3過門檻、零事件／首次null、仍FN。合法23/71、unknown窗口48/71／影片0/1、transition21/29／ground0/38不變；不稱辨識／時間改善。
- 限定單片矩陣[[0,0],[1,0]]，可判定與含unknown Accuracy均0/1、Fall Recall0/1／F1 0，Precision分母0記null；不以此單片FP0推論整体正常誤報改善。原正式23列CSV独立核TP14TN8FP0FN1U0、兩Accuracy22/23、P14/14／R14/15／F1 .965517且舊SHA一致，不產生新完整Val成績。
- synthetic79＋13＝92passed；首次4項默认sandbox temp setup error原因另存、改新專案basetemp後全過；新5source lint通過。real_feature_smoke＝sham23×512完全一致／finite／嚴格原權重、Train/正常Val各3合法控制差0。限定real_end_to_end只從既有Pose至ST-GCN／AE/K3本片，不冒稱新RGB Pose／全Val／Test／demo驗收。
- 独立NumPy AE／Scaler69分數、手算因果sources／213窗口／K3／零事件、6正常控制、只有6點XY改變且品質時間不變均過；7受影響合法窗3分數升4降，其他16不變。209有界舊來源成果hash與四原契約bytes保存、兩次實際重複prepare／diagnose拒絕且當時22新檔不變。舊定位40artifact先核對，原lock／report／final-qa不覆寫；graph先查coverage、歷史副本trace與排除原檔限制記錄。
- 自述joint覆核非獨立多人GT，7不可見點特別49右踝仍未修復，姿態phase非精確事件時間；不歸因為單純無骨架／只有K3，也沒有證明已排除表示／AE／門檻因素。沒有fit／訓練／校準／真Test讀取或評估，wide正式／demo維持、Test未知、Notebook／影片／上傳暫緩。

[成果、逐點／分數比較與只讀命令](ur-fall19-whole-coordinate-20261013.md)｜[事前鎖定](ur-fall19-whole-coordinate-20261013-plan.md)。人工三幀需求已完成；下方human pending是歷史定位狀態。

## fall-19全骨架錯點定位：診斷交付完成，human pending（2026-10-13）

- 新隔離outputs／QA，先鎖44–51範圍；只讀原Validation RGB／Pose，不讀Test、不新推論／擬合／訓練／校準／事件修改。104原關節與96投影骨段表、8張並排圖、AI未確認疑點已保存；沒有替換任何座標或新辨識改善。
- 44／47／49三幀人工頁、6下肢點×3＝18必要項、全13點顯示與可選上半身；不預填修正點、不顯示AE分數、不自動把其餘點標正確。可見點XY與不可確認null分開，自含原圖／新bundle草稿隔離／單一JSON。真頁0/18未代填或勾本人確認。
- 10 synthetic風險通過＋20離線假DOM／假點UI斷言通過；首次6合成fixture錯誤修復且舊XML保留；新source lint通過。沒有以synthetic冒充real_feature_smoke／real_end_to_end，兩者本輪未執行。
- 原NPZ獨立104joint／96bone CSV的math／median、3原嵌圖bytes及原時間／track／品質／來源SHA核對通過；340既有檔＋3本輪source／template／plan lock＝343份有界盤點hash及兩修改前文檔bytes保留。兩次實際build重複拒絕、26當時新檔不變；範圍不是全專案全檔驗證。
- 真瀏覽器新頁／圖像／導覽／末幀disabled／原圖切換／無選點拒絕／未完成匯出disabled均檢查，viewport截圖保存。沙箱loopback／最初tab問題及已通過的本機8821 HTTP200紀錄另存；未弱化保護。Graph先查且coverage，設計排除成果直接核原檔，best effort限制保留。
- 髖膝踝／45腕的疑點仍待人工可見性與位置確認；若看不清不补点，不稱已查明漏報原因。原wide22/23／F1 .9655、demo保留，沒有新採用／Test；Notebook／影片／上傳暫緩。

[成果、限制與人工操作](ur-fall19-joint-localization-20261013.md)｜[事前範圍](ur-fall19-joint-localization-20261013-plan.md)。

## 人工左手肘接續對照：不採用（2026-10-13）

- 上傳JSON為annotation資料，明確自述human_confirmed=true，8幀／source／pose／proposal／影像SHA均核對；44–50七點visible、51不可確認保留原點與原品質。原檔bytes封存，人工stage `human-b01d622686b5`與舊AI／cache分開。
- 實際max raw .1839069426→.1837749630、K3 .1346364915→.1351487935（+0.38%），固定門檻 .5341113806，raw／K3過門檻0、事件0、首次時間null、fall-19仍FN／unknown影片0。合法23/71、transition21/29、ground0/38、最後合法1766.67ms不變；未改善辨識或警示時間。
- 人工點位距原0.258–1.845px，與AI下移不同；未確認AI點位，因此不能把先前AI分數升高稱人工錯點修正。僅自述單joint座標，未提供全身／多人獨立覆核／精確事件時間GT。
- 沿用原locked程式與54synthetic邊界測試；actual real_feature_smoke重跑嚴格真實ST-GCN、23×512有限／sham delta0、Train adl-05及正常Val adl-04各3控制delta0。10受影響合法窗5raw升／4降／1變化≤1e-6，其他13合法窗完全不變。
- 獨立AI／人工138分數、426窗口、零事件、原confidence／bbox／track／時間及因果來源、23/71分母全過；12,710舊檔SHA不變。55先前已封存artifact及五契約原bytes核對；人工同內容再次執行實際拒絕，原匯入JSON與AI成果均未覆寫。
- 無fit／訓練／校準、無Test特徵／影像／評估。real_end_to_end僅既有Pose到AE/K3的控制推論，完整Val正式CSV不改；wide22/23／Fall F1 .9655及demo保留，無新模型採用、Notebook／影片／上傳仍暫緩。

[人工結果、核對證據與命令](ur-pose-coordinate-human-20261013.md)。本節接續下方AI歷史結果，原lock／report／final-qa不改。

## 可見左手肘座標控制診斷（2026-10-13）

- 事前鎖定8幀同track單joint；AI暫定座標已實跑，human_confirmed=false，人工座標覆核pending。原wide原30幀isotropic／真實權重／Train Scaler／AE／.534111門檻／K3保持固定，非新模型或完整Validation候選。
- max raw .1839069426→.2003316581，max K3 .1346364915→.1510627866（+12.20%），兩者仍低門檻；raw過門檻0、K3過門檻0、事件0、fall-19仍FN，unknown影片0。合法窗23/71＝32.39%；transition21/29、ground0/38及最後合法1766.67ms不變。僅單片診斷、正式wide Acc22/23／F1 .9655不改，不稱辨識提升或正確警示時間。
- synthetic54passed；real_feature_smoke真實ST-GCN strict SHA、23×512有限、sham及Train/Val各3控制窗delta=0。有限real_end_to_end＝快取Pose到AE/K3控制推論，未做新Pose RGB／完整Val／Test／demo全流程驗收。
- 獨立69分數、213窗、零事件、完整same-source時間因果性、XY以外全部Pose字段／信心／窗口品質一致通過。10受影響合法窗9分數升／1降，13不受影響合法窗完全不變；兩個正常控制均不變。
- 12,710舊asset SHA不變；五舊契約原bytes另存，舊299解析度成果manifest以原hash／契約副本核對。新cache／recipe與舊資料隔離，prepare及diagnose同目錄重複拒絕、AI不能冒充human的實際拒絕另外保存；不fit／重訓／校準、不載入Test特徵或解碼Test影像。
- 八點覆核頁有草稿保存、不可確認點xy=null、影像點選與尺寸換算、末幀禁用next、未覆核禁止匯出；人工一次匯出單一JSON。未替使用者作人工確認或匯出，尚無human joint GT，不將AI估點當真值。
- 不採用、不改demo、Test未知；Notebook／影片／公開上傳仍暫緩。局部點位改變支持分數受此輸入影響，不證明錯點是唯一原因，也不能排除全身表示／誤差／門檻問題。

[實際結果與命令](ur-pose-coordinate-diagnostic-20261013.md)｜[事前規則](ur-pose-coordinate-diagnostic-20261013-plan.md)｜`docs/validation/ur-pose-coordinate-diagnostic-20261013-001/independent-verification-1.json`。

## 骨架品質解析度最終驗收：不採用（2026-10-13）

- 實際完成唯一512×384同權重crop控制，47 Train／Val新Pose及ST-GCN3,419特徵、原24正常Train2,094窗AE32epochs／正常Val最佳22；一次門檻 .3137603998。Test23未準備無路徑，無解碼／評估，無追加候選。
- Val整體Accuracy16/23＝69.57%、二元16/22＝72.73%、Fall F1 .7500／Precision9/10／二元Recall9/14／全體Recall9/15；矩陣ADL[7,1,0]、Fall[5,9,1]。FP adl-35；FN fall-19／13／29／07／25；Fall unknown fall-22。低於wide22/23／.9655，不採用、原demo保留。
- fall-19 transition合法21/29→18/29、ground0/38不變、最後合法提前100ms；有K3但Raw／K3仍低自身門檻，並非全片無骨架。全Val端點coverage84.39→48.89%；六AI粗phase事件4→2、good警示影片數2持平但原影片退步、粗提前0→1。影片TP與時間／展示guard分開，新Pose無人工點位GT，原人工phase非精確事件時刻。
- synthetic40passed；real_feature_smoke完成真實strict forward／完整Train-Val研究流程，獨立Train-only Scaler、正常Val checkpoint重載、3,688分數／5,716窗／26事件、23影片CSV／矩陣／分母、23門檻候選排序及8,643逐原幀對照全過。相關ruff／format通過；Test／demo real_end_to_end未跑，不混稱正式盲測。
- 25,138舊asset SHA不變、五舊契約bytes原副本保存；六stage實際重複拒絕、238新成果before／after不變。protocol_id報告失敗另存，repair只補metadata、整run hash不變且重訓／重校準0；CuBLAS確定性警告及Graph best effort／設計排除／heuristic限制原樣記錄。
- 採用criteria全未滿足、候選停止。Notebook／五分鐘影片暫緩、無公開上傳；Test未知、不依歷史Test調參／換算課程分數。本輪不需重做100人工標註。

[完整實驗及命令](ur-pose-resolution-20261013.md)｜[事前規則](ur-pose-resolution-20261013-plan.md)；`docs/validation/ur-pose-resolution-20261013-001/verification.json`、`decision.json`及同名run/cache/outputs保存證據。

## 骨架解析度完整prepare／訓練階段（2026-10-13）

- 47原Train／Validation重建完成，3,419合法512D特徵有限；四片pilot嚴格同recipe複用，原Test23保持not_prepared無特徵／Pose路徑。
- 實際AE32epochs／最低正常Val checkpoint第22epoch .1420643194；原固定seed42／max100／patience10，CuBLAS確定性警告另存。一次Validation門檻 .3137603998已保存。
- 報告metadata缺protocol_id造成輸出失敗，原log／部分輸出另存；新增resume僅補協定欄位、複用凍結門檻，未追加訓練／校準。正式比較及獨立核對尚未結案。

## 骨架解析度technical pilot驗收（2026-10-13）

- 四固定Train／Validation原序列真實推論完成，Detector／Pose／ST-GCN strict真實權重、512×384 crop→17×128×96有限熱圖、512D及分類頭parity、來源／split／hash及隔離通過；324合法特徵，fall-19為20窗，不等於辨識改善。
- 25,138舊成果hash及五契約原副本先保存，40相關synthetic通過。完整47片重建中，AE／正式23Val尚未執行，沒有Test讀取或評估、原demo不改。[事前契約](ur-pose-resolution-20261013-plan.md)；同名`docs/validation/ur-pose-resolution-20261013-001/pilot.json`保存真實證據。

## 骨架品質解析度對照：事前風險驗證（2026-10-13）

- 唯一候選已固定512×384同權重Pose crop，其他wide工程設定不變；原24Train ADL／23Validation、新recipe/cache/run/manifest，Test23禁止讀取／評估，舊成果另存保全。
- 40相關synthetic通過，含新單一變因／recipe／Test與Fall Train防護、原觀測與標籤界線、因果phase及既有cache／Train-only／Validation校準／K3風險；不當成辨識改善。
- technical pilot／完整新Pose特徵／AE訓練及23Val評估尚未完成；暫無新Accuracy或採用結論，原demo／Notebook／影片維持。事前規則見[計畫](ur-pose-resolution-20261013-plan.md)。

## fall-19人工補註與凍結對照驗收（2026-10-13）

- 本人兩項回答已另存：33–61／62–100是完整姿態階段；f1／2為missing。原JSON bytes不改，獨立schema v2只有兩個逐幀標籤改變，原時間欄位移為phase_ranges、精確onset／impact為null；無正式時間Accuracy或精確延遲。前輪待釐清狀態保留作歷史，無需重做100筆。
- 223窗口CSV bytes及三路方法摘要／分數／門檻／品質／事件不變，transition21/29合法、ground0/38合法、8幀人工wrong與confidence good不一致仍成立。支持品質檢查有依據，不證明修Pose必然提升或唯一漏報因果。
- synthetic58passed（13新補註＋45原），相關lint／format通過；獨立100原RGB SHA、100CSV／來源、223過去來源對齊與凍結分數／phase計數／最高值／門檻跨越核對通過，僅兩個人工儲存格差異。264承接不可變成果hash不變、兩份舊契約原副本保存；repeat save拒絕且未改原成果，Graph Tier2 best effort及ignored直接核對另存。
- real_feature_smoke為人工補註對照真實凍結觀測；無訓練／校準／新Pose或ST-GCN／real_end_to_end／Test讀取評估。無新Accuracy或時序改善，不採用新模型、原demo保留、新方法Test未知；Notebook／影片暫緩、不公開。

[補註報告、檔案與只讀命令](ur-fall19-human-amendment-20261013.md)；`outputs/ur-fall19-human-amendment-20261013-001`與同名`docs/validation`保存修正版及核對證據。

## fall-19人工自述匯入與對照驗收（2026-10-12）

- 匯入100原Validation幀JSON通過來源／順序／時間／SHA／自述確認與完整性；下載與保存原bytes相同，原匯出不修改、身份及GT不由程式獨立認證。時間欄位等整段姿態、f1／2無Pose卻標matches待釐清，沒有擅改人工判斷或計正式時間Accuracy。
- 因果最後來源幀對照三路transition21/29合法（11人工matches／10wrong），ground0/38合法、分數null。matches端點wide與20iso仍低門檻，沒有K3或Raw跨門檻；f44–51人工wrong但confidence good共8幀，是品質不一致證據，不推定修Pose必然改善。
- synthetic45passed（9新／36原）；223窗口來源／標籤／凍結分數與phase合法數／最高值獨立重算全過，245原非進度成果hash不變、兩份修改前契約原副本吻合；完成save拒絕覆寫、lint／format通過。新資料／報告另存，歷史初始0/100與AI粗phase原樣保留。
- real_feature_smoke為匯入與真實凍結觀測／分數對照；無擬合／校準／新Pose／ST-GCN／real_end_to_end或Test。辨識與警示沒有改善，既有wide／motion22/23／.9655、20iso20/23／.8889不變、demo不換、新方法Test未知。Notebook／影片暫緩、不公開。

[完整人工診斷與命令](ur-fall19-human-review-20261012.md)；`docs/validation/ur-fall19-human-review-20261012-001`及`outputs/ur-fall19-human-review-20261012-001`保存全部證據，匯出原檔在前輪human-annotations雜湊目錄。只須澄清兩項語義，不需重做100幀。

## fall-19完整逐幀診斷與覆核工具驗收（2026-10-12）

- 完成：100原Validation影格、95既有Pose、三路30wide／20iso／motion窗口與分數互動對照，所有原RGB bytes及來源時間保留；全片接觸表與密集／代表全解析度影格已AI目視，AI記錄獨立且human_confirmed=false。人工仍0/100，未製造時間GT。
- 診斷：49當前good／46低品質／5缺Pose；30與20最後合法端點均1766.67ms，20新增10窗皆前段。原粗phase仍9合法K3，Raw／K3皆低門檻；f54當前缺Pose但有先前合法窗口，f55轉gap／unknown。出框與後段失效同時出現是AI觀察，沒有控制實驗／人工GT，不能推定唯一漏報因果。
- synthetic：66相關passed，新36案例及原時間／事件風險；lint／format通過，未稱新的全套回歸。標註錯來源／SHA／Test／AI／未確認／排序／完整性／邊界受防護；實際未確認UI匯出、完成build及空白import拒絕。未產生人工標註。
- real_feature_smoke：兩AE56分數嚴格重載逐值一致、另核23融合分數；獨立100原圖SHA／source-time、95Pose／track／信心、300因果對齊、223K3窗口／狀態、3份23Val CSV Accuracy／F1／分母／矩陣及fall19零事件全部通過。24966舊成果hash不變、五份修改前契約副本吻合，新成果另存。
- 實際IAB頁面與f54／55／拒絕匯出核對通過；初次timeout／沙箱loopback Win10013保留，允許本機只讀127.0.0.1服務後重測正常。Graph Tier2／coverage保存best effort，raw/cache/run excluded直接核原檔，沒有宣稱全graph完整。
- 無辨識或時序改善，沒有訓練／校準／新增候選；wide／motion22/23／.9655、20iso20/23／.8889原結果不改、fall19仍FN、demo預設不換。沒有新Pose／ST-GCN／real_end_to_end或Test評估，Test只hash舊bytes不解碼，新方法Test未知。Notebook／五分鐘影片暫緩、不公開、不換算課程成績。

[完整診斷、證據與命令](ur-fall19-review-20261012.md)｜[覆核頁](../outputs/ur-fall19-review-20261012-001/index.html)；事前plan、lock、獨立verification、synthetic XML、UI與拒絕紀錄在`docs/validation/ur-fall19-review-20261012-001`。

## 接續全域動態正常近鄰最終驗收（2026-10-12）

- 辨識：唯一motion側分支＋凍結wide融合已fit，Train24ADL／3638窗；Val TP14TN8FP0FN1U0、兩種Accuracy22/23／F1 .9655，与wide持平，不採用、原demo保留。新增神經網路epochs0不冒稱重訓AE。
- 時序：2917窗口狀態、14完整事件track／起迄／原因與23首次警示皆與wide一致；粗phase4/6、良好Pose警示影片2/6、早報0，無時間改善。AI區間非人工GT，分數非機率；正式TP與展示品質分開。
- 診斷：逐幀置中縮放數學上移除全域motion；新增90D保留它，fall19phase仍9合法K3／7當前goodPose但motion最高.78393<正常Val p95 1.01900，融合1.68073<門檻4.88126；同分數救回需8ADL誤報。未支持「只補全域motion即可改善」或盲目SSL，不能把原因全部歸於沒骨架。
- synthetic：60相關passed。初次49passed／11 sandbox預設暫存目錄setup error，固定專案內新basetemp重跑相同60案例全過，原XML保留；模型設定未改。新source lint／format／compileall通過，不宣稱跑了新的全套回歸。
- real_feature_smoke：3個嚴格真實ST-GCN／head parity pilot最大差0，47片／6001因果原合法窗重建、獨立90D快取及近鄰bankfit。獨立18003分支／融合分數與3638整影片排除Train分數、Train-only Scaler、Validation門檻排序、23CSV／矩陣／分母、2917窗口K3及14事件全通過；24620舊assets hash不變、五契約原副本吻合。
- Test23未準備無路徑、入口拒絕、未知未評估；不讀Test特徵／不重切／不按歷史Test調參。real_end_to_end／RGB Pose／demo未跑，新方法未部署；無Notebook／影片／上傳。新舊recipe與實驗目录隔離，完成操作拒絕覆寫，Graph Tier2 best effort及ignored精確source／hash證據另存。

完整[實驗報告](ur-motion-novelty-20261012.md)、[事前契約](ur-motion-novelty-20261012-plan.md)；正式輸出`outputs/ur-motion-novelty-20261012-001`、run同名，證據`docs/validation/ur-motion-novelty-20261012-001`。本階段有限候選已停止，持平不算辨識改善。

## 本輪方法比較最終驗收（2026-10-12）

- 辨識：六SVM＋兩AE皆已完成，62實際新AE epochs；最高21/23＝91.30%／F1 .9333，低於wide22/23＝95.65%／.9655，不採用，demo預設保留；新Test未知。
- 時序：最佳SVM fall09事件提前至5.0333秒，粗phase命中4/6→5/6，可見phase影片仍2/6。AI粗時段不作人工GT，Accuracy下降不可稱提升；正式TP不因畫面當前品質而抹除。
- synthetic：47相關passed；全套297案例首次294passed／1skip／2 loopback沙箱Win10013，兩項重測2passed，去重後296passed／1歷史權重路徑skip，原XML／失敗log保留。lint src/tests/scripts、compileall、新source format通過。
- real_feature_smoke：三窗真實嚴格ST-GCN／head parity差0；八異常模型實際擬合與23Val正式輸出。獨立重載47,630分數、184影片／矩陣／分母、K3状态及175完整事件、Train-only Scaler／正常Val checkpoint／Validation校準、時序histories因果／缺失／track、recipe／快取隔離通過。
- 保全：23,925舊assets hash一致，五份原契約副本原hash一致；所有新run／設定／recipe／權重另存，完成stage拒絕覆寫。初次前置API失敗cache、source與報告輸入數誤植修正前版本另存，未重訓／重校準。
- 選配停止：raw骨架／512鄰近性事前診斷未支持本輪微調假設，SSC未跑；不宣稱已證明姿態正確或特徵無問題。完整人工關節／時間GT與泛化仍未知。
- real_end_to_end未跑，無新RGB Pose、demo或Test。歷史Test已查看、Validation反覆、split非人物隔離限制保留；Notebook／影片暫緩、舊成品保留、不公開上傳、不換算課程分數。

成果與命令見[方法比較報告](ur-normal-methods-20261012.md)；完整證據在`docs/validation/ur-normal-methods-20261012-001`、八模型輸出在`outputs/ur-normal-methods-20261012-001`。最終QA核對測試去重、文字輸入數、圖、六個防覆寫操作及成果hash。

## 方法比較第二階段：八候選與獨立真實驗證完成（2026-10-12）

兩AE事前鎖定後CUDA實訓40＋22epochs，checkpoint分別30／12且只依8正常Val；去噪20/23／.9032、時序21/23／.9333，未提高wide22/23／.9655，八新模型均FN fall-19。47相關synthetic全過；獨立驗證八新模型47,630分數、184影片CSV／矩陣／分母、每窗K3狀態與175完整事件、Train-only Scaler、正常Val checkpoint、Validation校準、時序缺失／track／因果及原23,925assets hash全部通過。新Test23未準備無路徑、Test入口拒絕；新方法Test未知。real_feature_smoke：真實原30iso快取／嚴格ST-GCN三窗parity／原七類獨立forward／8正式異常模型。real_end_to_end未跑；無新RGB、demo、Notebook或上傳。

fall-19非全無骨架，9合法K3且7当前良好端點，wide全phase raw仍低門檻、降門檻救回需8ADL誤報。fall-09分類頭FALLING而AE不警示可逐窗對照，但不同表示，不替代異常指標。另立事前raw骨架／512鄰近性診斷，未取得表示壓縮假設支持，正常自監督選配停止而未實訓。正式因果／人工關節及onset真值仍不足。全套回歸與最終QA另列完成結果，不把本段的相關測試通過稱辨識提升。

## 方法比較第一階段：六SVM真實擬合完成（2026-10-12）

新方法契約已按使用者授權更新，教師附件／Sample原檔hash保存；原契約另存，23,925舊assets快照待最終重驗。固定30幀isotropic／真實512D，47Train／Val快取隔離、Test23未準備無路徑；真實嚴格ST-GCN三窗口原特徵parity差0，Train24 ADL／3638窗Scaler與wide逐值相同。六固定SVM完成，最佳nu=.10／gamma=.001 Validation21/23／F1 .9333／TP14TN7FP1FN1U0；其餘20/23／.9032，均漏fall-19，網格關閉，未提升wide22/23／.9655。40相關synthetic通過；CSV與事件的獨立最終驗證尚待全階段完成。證據real_feature_smoke，無新RGB／demo／Test或real_end_to_end。新方法Test未知，原預設保留，Notebook／影片暫緩。全部候選與失敗前置cache保存於`docs/validation/ur-normal-methods-20261012-001`及各獨立run，不覆寫歷史。

## 最新：UR-only 20幀＋等比例訓練／Validation驗證完成，未提高辨識（2026-10-12）

補充：3份授權修改前核心source另存且逐位元吻合舊SHA256；1,430歷史檔及201外部封存檔映射再次hash全通過，外部active路徑不存在，未恢復／下載。舊retirement status固定當時核心hash，當前程式以新只讀 `scripts/verify_short20_retirement.py` 驗證；舊source／lock／失敗log保留。歷史Test cache只做bytes SHA、不解碼／不參與選模。[完整封存保全](validation/ur-short20-isotropic-20261012-retirement-verification.json)。

唯一事前鎖定20幀／stride1／per_frame_isotropic候選完成10片pilot、47片完整特徵及39epochs AE；原24 Train ADL3,888窗、正常Val8片1,515窗，最佳Epoch29／Loss .1128902972。真實凍結ST-GCN strict、20幀512D有限特徵及head parity通過；原Pose觀測hash不變，原權重與split保留。

原23Val：TP12／TN8／FP0／FN3／unknown0，二元可判定分母23及含unknown整體分母23的Accuracy均20/23＝86.96%、Fall F1 24/27＝.8889、Precision12/12、Recall及全體Fall召回12/15。2×2 `[[8,0],[3,12]]`，2×3 `[[8,0,0],[3,12,0]]`。相較最高wide22/23／.9655，fall-19未救回、fall-18／30退步；六粗時段命中4→2、0早報，採用guard不過、保留原預設。窗口coverage84.3911%→85.7483%不能抵銷Accuracy下降。fall-19粗phase仍9合法K3，候選.2352462<門檻.4238913，不將漏報直接歸因缺骨架；粗AI時段非人工GT，影片TP與警示時間分開。

明確Train／Validation-only模式及6新風險cases通過：完整47片prepared、原70split保留、Test23片not_prepared／無路徑，未讀Test特徵或混舊cache；預設完整模式／缺Val／smoke／偽造Test／scoped Test evaluate均受防護。33相關synthetic全過；全套首次271pass／1歷史原Human路徑skip／2沙箱loopback阻擋，兩项授權環境重測2pass，總273pass／1skip，原失敗記錄保留。本輪sample Human真實權重有strict／forward，skip不能當本轮缺件。

real_feature_smoke：獨立sklearn逐CSV核對Accuracy／P／R／F1／矩陣／分母／coverage，所有窗口K3與狀態及14事件完整區間／peak／end reason逐值吻合；Train-only Scaler逐值、24Train／8normalVal／23calibrationVal hash與最佳正常Val checkpoint吻合。1,031舊檔保全、4完成stage覆寫拒絕。派送／BOM驗收修復有原source／lock／失敗log及hash鏈，模型／校準／結果不變。

real_end_to_end未跑：無新RGB Pose／demo／Test；Pose沿用經驗證觀測，Test未準備，不能冒稱新的盲測、完整端到端性能或警示起跌延遲合格。Graph已恢復Tier2與coverage，早期新腳本parse_partial直接讀取修復；後续metadata_match／no_recorded_issue為best effort。Validation反覆使用、人物隔離未證、教師協定／人工GT／泛化未明；Notebook／影片繼續暫緩、沒有上傳或課程分數換算。[正式報告](ur-short20-isotropic-20261012.md)｜[lock](validation/ur-short20-isotropic-20261012-lock.json)｜[正式Validation](validation/ur-short20-isotropic-20261012-validation.json)｜[獨立驗證](validation/ur-short20-isotropic-20261012-verification.json)。

## 最新：外部輸入封存、UR Train限定shift實驗與驗收完成，Accuracy未提高（2026-10-12）

使用者要求UR-only，原外部raw／cache路徑已不存在，201檔移至data/retired，1,430原檔hash保全，run／設定／舊結果不覆寫。原UR Train24ADL的3,638窗各產生一個明示1%可靠關節位移版本，真實ST-GCN重抽，原＋增強7,276列，新增原影片／外部來源0。48原特徵parity差0、所有品質／時間record相同；整體平移tensor僅1.91e-6差，不當有效增強。單一AE29epochs，Val TP13／TN8／FP0／FN2／U0，二元23支及含unknown全體23支Accuracy21/23＝91.30%、F1 .9286、全體Fall召回13/15，低於最高22/23／.9655。粗時段命中4→3、0早報，不採用、原預設保留，影片TP不冒稱警示時間正確。

36相關synthetic全過／0 skipped；real_feature_smoke為嚴格真實ST-GCN增強與AE訓練／URVal核對，Scaler24Train原＋增強逐值吻合、實際hash／count／正常Val checkpoint、23支CSV指標／K3／事件重算一致。無新原RGB Pose／demo／real_end_to_end／Test；Test特徵讀取與評分0。1,435舊檔保全，8次覆寫／追加／外部重啟拒絕，1,542受檢檔前後相同。首驗收器count欄位錯誤修復保存source／失敗log，沒有重訓或改結果；graph Transport closed未取得新coverage，以相關源碼補核對。不稱課程全達標或新盲測，完整人工GT／教師協定仍未確認，Notebook／影片暫停。[完整結果與操作](ur-only-shift-20261012.md)｜[獨立驗收](validation/ur-train-shift-20261012-verification.json)｜[最終QA](validation/ur-train-shift-20261012-final-qa.json)。

## 最新：新增真實正常Train與AI取樣標註完成，Accuracy未改善（2026-10-12）

UMAFall8＋Ursul24新RGB已取得；13,270幀嚴格Pose／ST-GCN forward，21可用新來源／3,831窗，原Train24／3,638增至45／7,469，外部不加URVal分母、不改split。UMA整包MD5通過；Ursul僅Range成員CRC／長度／SHA通過，整包未下載不宣稱MD5核實。raw／cache本機保留不加入Git。

AI取樣v2涵蓋47支Train／Val、340原圖、15段Val跌倒不確定開始／穩定區間，7可見區域不分左右；106,808行Human13自動診斷另存，非GT。Train跌倒0／Test標註0，尚未人工複核或完整逐幀GT，失敗v1未覆寫。有限2候選實跑68／27＝95epochs，皆TP14／TN7／FP1／FN1／U0：二元23支Accuracy21/23＝91.30%，含unknown全體23支同值，Fall F1 28/30＝.9333，全體Fall召回14/15，窗口coverage .843911439。低於最高22/23／.9655，不採用，預設保留；Loss小幅下降不當Accuracy改善，影片TP不當時間正確。

36相關synthetic全過／0 skipped；真實Train-only Scaler／hash／最低正常ValLoss checkpoint、46支次CSV／K3／事件及新時段重算通過，1,160舊檔保留、10次覆寫拒絕前後hash不變。新正常RGB抽取限定real_end_to_end，AE訓練／URVal為real_feature_smoke；無新Test讀取／推論／調參。Windows rooted ZIP路徑修復封存原source／新舊hash、初次35過1失敗XML，全過另存新XML；graph Transport closed未刷新coverage，以相關源碼／測試補核對，不稱完整graph。

歷史Test已看、人物隔離未證、小Val反覆使用及官方協定未知保留，不宣稱新盲測／課程全達標。本輪32支與有限實驗完成，完整人工GT／精確時間仍缺；SAFER需登入同意分享聯絡資料，未取得。Notebook／五分鐘影片暫停。[完整成果與操作](external-normal-20261012.md)｜[獨立驗證](validation/external-normal-20261012-verification.json)｜[最終QA](validation/external-normal-20261012-final-qa.json)。

## 最新完成：資料量／鏡像／殘差比較，最高Accuracy未提高（2026-10-12）

4個Train子集未校準模型實際228epochs、共同Train24單位學習曲線下降，原split不變，不是新獨立Test。2個鏡像AE實際84epochs，皆TP13／TN8／FP0／FN2／U0、二元／全體21/23、F1 .9286；後續另鎖定2個Train-only殘差profile，0新epochs，皆TP14／TN8／FP0／FN1／U0、22/23、F1 .9655，未超過參照。殘差粗略phase-hit4→5／0早報、fall-09事件5900→5100ms，不等於正式時間驗收；採用guard因Accuracy未嚴格提高仍未過，原預設保留。

37相關synthetic風險測試通過／0 skipped；48真實原特徵parity差0、鏡像TrainScaler逐值、兩階段各46支次Val CSV／K3／事件核對、1110舊檔及鏡像派生來源hash通過。殘差profile只用24原Train ADL／3638窗、AE／Scaler源hash相同，新epochs0。證據real_feature_smoke，無新原影片Pose抽取或demo。strict YAML註記修復保存舊設定／source／lock、選項不變；後段graph Transport closed以完整直接源碼補核對，coverage不能刷新限制明示。全部新候選未跑Test，不拿Test調參，不宣稱Accuracy／Test改善或新的盲測；Notebook／影片暫停。[完整成果與證據](normal-data-support-20261012.md)。

## 最新：大容量候選一次凍結Test完成，Validation改善未延續（2026-10-12）

依使用者後續明確授權，以`ur-isotropic-wide-20261012-001`固定原Validation門檻.5341113805770874／strict >／K3及原8ADL＋15Fall Test，CLI evaluate exit0。TP10／TN8／FP0／FN5／影片unknown0，二元／含unknown全體Accuracy均18/23＝78.26%，可判定Fall F1＝20/25＝.8000，全體Fall召回10/15＝66.67%。與保存的舊isotropic Test相較多漏fall-03；此為結果描述，未據此調參／選模，原default／採用guard不變。

13相關synthetic風險測試通過、0 skipped；真實特徵快取上重新凍結AE推論證據為real_feature_smoke。sklearn23支CSV與獨立K3／事件重建全部一致，PNG矩陣目視8／0／5／10。1103舊檔含44凍結run、原split／門檻、核心source、真實權重、相關快取、前次Test及前輪Validation鎖定／結果／驗收hash不變。本次未重抽原影片、未改核心、未重新訓練或校準，不冒稱real_end_to_end／新的盲測／警示時間合格。課程參考辨識值未達，教師協定可比性仍未知；Notebook／影片暫停。指定評估與驗收完成，無外部阻塞。[本次正式結果](ur-isotropic-wide-test-20261012.md)｜[評估前快照](validation/ur-isotropic-wide-test-20261012-before.json)｜[獨立驗證](validation/ur-isotropic-wide-test-20261012-verification.json)。

## 最新：三個Train／Validation候選完成，影片級Validation提高、採用guard仍未過（2026-10-12）

診斷後事前鎖定三組新AE，原24Train ADL／3638窗、23Validation、split／品質／K3均保留。實跑121epochs，wide TP14／TN8／FP0／FN1／unknown0，二元／含unknown Accuracy均22/23＝95.65%，Fall F1＝28/29＝.9655，全體Fall召回14/15＝93.33%；影片級Accuracy實際增加一支，不能當Test成績。wide新增TP fall-18仍無當前良好骨架警示，fall-09首次事件5.9秒落在舊粗略時段外；事前guard未通過，原預設保留。compact／denoise均21/23，前者多1誤報。

26相關synthetic風險測試通過，另在真實Pose／特徵快取上實際訓練／校準，sklearn混淆矩陣／Accuracy／F1、69支次K3分數及事件有無逐值一致，Scaler只用Train擬合逐值一致；本輪證據為real_feature_smoke，沒有新影片Pose／ST-GCN端到端重跑。1039既有檔案、41舊凍結run、原Test輸出／切分／真實權重及核心source hash不變，新流程沒有Test入口，所有新候選Test未知。CuBLAS warn_only重現限制、未通過guard及驗收工具修復均揭露。初版新測試的synthetic分母4改成實際2，失敗XML保留。詳見[本輪成果與操作](isotropic-refinement-20261012.md)。

## 最新：最高Validation候選Test評估完成，辨識基準未達（2026-10-12）

使用者明確授權後，先保存凍結模型、原split、Validation門檻與快取hash，再以核心CLI evaluate執行`ur-human-isotropic-001`原23支Test。TP11／TN8／FP0／FN4／unknown0：二元Accuracy及包含unknown全體Accuracy均19/23＝82.61%，可判定Fall F1 .8462、全體跌倒召回11/15＝73.33%；漏報fall-27／23／21／20。沒有把91.30% Validation或七類顯示当成Test成績。

17相關風險測試通過；sklearn從逐序列CSV重算混淆矩陣／Accuracy／F1一致，窗口K3分數重建嚴格門檻與23支事件有無一致，PNG混淆矩陣目視。41舊run、92相關快取、真實權重及核心Pythonhash不變，原模型／Scaler／split／門檻凍結。無核心程式改動或新訓練，Test沒有回用校準／選模，也未追溯改寫舊guard／lock。證據為真實Pose／ST-GCN快取的重新AE評估，這輪沒有重新逐影片擷取Pose；synthetic測試只驗證程式。

結果低於課程參考Accuracy90%／F1 .87；歷史Test已看、反覆Validation、自訂序列切分未保證人物隔離、官方評估單位未確認限制仍在，沒有宣稱新獨立盲測或警示時間正確。使用者指定工作完成、無外部阻塞，正式Notebook／五分鐘影片暫停、未上傳。[完整結果與操作](ur-human-isotropic-test-20261012.md)｜[評估前快照](validation/ur-human-isotropic-test-20261012-before.json)｜[重算驗證](validation/ur-human-isotropic-test-20261012-verification.json)｜[pytest](validation/pytest-isotropic-test-20261012.xml)。

## 最新：sample式七類顯示已驗收（2026-09-18）

39相關測試通過，完整232 passed／1舊預設權重路徑skip／12warnings，真實Human權重有執行，lint／format通過。新增獨立原30觀測forward、七類主標籤、低品質小字、無人留空及AE事件歷史，auto對Human（含isotropic）啟用；不改AE unknown與評估。首次pytest暫存目錄存取setup error另存，取得存取後相同測試全過。

real_end_to_end六片888幀，AE推論／事件CSV逐位元相同（559筆分數、167筆unknown）；674筆七類完整重算七分數／argmax差0，含81筆低品質分類。每幀歷史事件計數與時間核對原CSV、無未來洩漏，H264全片解碼及六份接觸表目視。41舊run／8764cache／44設定／5manifest／37舊媒體hash保留，拒絕覆寫實測。synthetic風險測試、real_feature_smoke真權重固定輸入、real_end_to_end原片重跑證據分開。

顯示驗收通過不等於辨識提高：AE仍原20/23及另存候選21/23，unknown影片0、兩Accuracy分母均23。fall-30歷史事件1.8667秒可見不代表當前Pose合格；不回頭更改原guard或採用結論。七類仍有錯判，adl-35正常活動3.30秒單幀FALLING但AE無事件。未做新訓練／Test或正式Notebook／五分鐘影片，無外部阻塞，歷史Test／反覆Validation與教師協定未明限制不變。[契約、成品與操作](sample-display.md)｜[整合證據](validation/sample-display-integration.json)｜[完整回歸](validation/pytest-sample-display-full.xml)。

## 最新：影片級Validation改善，可見警示驗收未通過（2026-09-18）

9張全解析度／50張取樣影格與33個粗略可見區域完成，AI目視標註非官方／獨立GT，不作訓練標籤。確認原逐幀XY分開縮放消除身體比例後，單一新候選經8片pilot及49epochs實跑，Validation TP13/TN8/FP0/FN2/U0：二元及全體Accuracy21/23=91.30%、F1=.9286、全體跌倒召回13/15=86.67%。未讀Test異常分數或調參。

fall-09首次可見AE警示5.9→5.0秒，仍有中斷；新增TP fall-30只有1.8667秒窗口事件、當前Pose不足而無可見警示，依事前guard不替换預設。原`ur-sample-human-s1-001`仍20/23；新`ur-human-isotropic-001`保留為最高影片級Validation候選，91.30%不代表Test／逐影格成績或完整課程達標。

完整215 passed／1歷史路徑skip；34相關測試；真實6片888幀／559比分數差0、狀態事件吻合、H264全片解碼目視、原AE／ACTION逐位元一致。40舊run／8450cache／設定manifest hash核對、70序列Pose／排程一致、Train24ADL／3638窗Scaler重算完全吻合、拒絕重跑實測。synthetic風險測試／real_feature_smoke pilot／real_end_to_end辨識分開。無新Test／正式Notebook／五分鐘影片，無執行資源阻塞；歷史Test與反覆Validation限制不變。[完整報告與命令](human-isotropic.md)｜[結果](validation/human-isotropic-validation.json)｜[最終驗證](validation/human-isotropic-verification.json)。

## 最新：骨架翻轉與正常活動抽樣驗收完成，辨識目標未達（2026-09-18）

12片翻轉Pose pilot的三漏報跌倒時段合格端點31→31、正常coverage比值.96584，未通過門檻，沒有其AE成績。正常Train型態抽樣候選訓練33epochs，Validation仍20/23=86.96%、F1=.8889，TP12／TN8／FP0／FN3／U0；全體／可判定Accuracy分母均23、全體跌倒召回12/15=80%。平手保留原Human基準，未以較低Train Loss、七類頭或balanced accuracy90%冒稱達標。

完整210 passed／1歷史資源路徑skip，26項相關測試通過；39舊run、1165cache及原設定／權重manifest hash保留。Scaler與Train24ADL／3638窗重算一致，KMeans中心／assignment／抽樣權重也重算相同。四支real_end_to_end影片723幀、526分數差0、狀態／事件吻合，原基準AE CSV／事件逐位元相同；H264完整解碼、四份接觸表目視。fall-09仍5.9秒才警示、fall-30下降時STANDING UP且無AE警示，未改善警示時機。synthetic、real_feature_smoke、real_end_to_end分開。

正常活動抽查表與五份接觸表已交付，非完整關節／逐影格GT；沒有新外部資料或Test評分。正式Notebook／五分鐘影片暫停、無外部資源阻塞，歷史Test／反覆Validation限制保留。[報告及操作](human-quality.md)｜[正式結果](validation/human-quality-validation.json)｜[完整pytest](validation/pytest-human-quality-full.xml)。

## 最新：Train 殘差加權實作通過，辨識目標未達（2026-09-18）

兩個預先登記的Train-only加權MSE候選已完成擬合、Validation校準及逐影片重算：逐維19/23=82.61%、F1=.8462；雙分支20/23=86.96%、F1=.8889。沒有提升，平手保留原`ur-sample-human-s1-001`：TP12／TN8／FP0／FN3／unknown0。可判定二元與全體Accuracy皆以23為分母，全體跌倒召回12/15=80%；有效窗口coverage84.39%，不是逐影格皆可判定。

Scaler／AE原封沿用，新增epochs=0；誤差權重只擬合Train24ADL／3638窗，門檻只使用Validation。舊37run、1165cache、來源與hash保留。粗略六段時段事件／可見命中皆2、早報0，未補原三個漏報；逐維候選多漏fall-16，依事前規則不採用。未追加Test，沒有以七類分類頭替換AE。

synthetic隔離與契約測試、CPU／GPU批次串流檢查通過；完整203 passed／1歷史路徑skip、12警告。首次2項失敗為繞過建構式的舊測試少欄位，修正測試後通過，初次XML保留。real_end_to_end三支影片555幀／381比分數差0，窗口狀態及事件吻合、ACTION CSV逐位元相同，原MSE AE CSV／事件亦逐位元相同；完整解碼H264並目視。fall-09仍5.9秒才可見警示，功能驗收不能當成辨識改善。

Accuracy90%未達、教師評估單位／切分未明、歷史Test已查看、Validation反覆使用的限制均保留；正式Notebook／五分鐘影片暫停，無外部資源阻塞。[完整報告](human-residual.md)｜[Validation](validation/human-residual-validation.json)｜[真實影片整合](validation/human-residual-integration.json)｜[最終測試](validation/pytest-human-residual-final.xml)。

## 七類動作與AE並列展示：已通過介接與一致性驗收（2026-09-18）

原Human七類分類頭嚴格沿用已載入checkpoint，原始512D另接分類頭、argmax顯示七類動作，不經AE scaler、不替換AE。畫面ACTION／ANOMALY (AE)分欄，同人框旁並列兩種結果；PENDING／UNKNOWN及新鮮度保留。auto僅對原30幀per_frame Human啟用，明確actions遇其他recipe拒絕。

191 passed／1歷史權重路徑skip；32項相關測試與15項最終分類測試通過，新Human真實權重有實跑。四支Validation503幀、313窗七類分數差0；AE推論CSV及事件CSV與修改前逐位元相同，H264全片解碼，四支接觸表與首次FALLING已目視。37個旧run及1165個來源cache、設定／manifest／權重／舊影片hash保留。原23支AE Validation重算仍TP12/TN8/FP0/FN3/U0，Accuracy20/23=86.96%、F1=.8889，兩種Accuracy分母均23。

fall-09在4.70秒顯示FALLING時AE仍NORMAL，七類顯示未被混入AE成績；fall-30下降時仍可能STANDING UP、出框後UNKNOWN，沒有七類GT／Accuracy驗收。synthetic隔離測試與real_end_to_end影片證據分開，沒有新Test、訓練或正式五分鐘影片。[完整契約與驗證](seven-actions.md)｜[机器核對](validation/action-display-integration.json)｜[pytest](validation/pytest-action-display-full.xml)。

## 本輪優化驗收完成，辨識目標未達（2026-09-18）

兩組短窗加一組共用窗口座標尺度，共147個實際訓練epochs。Validation依序20/23、19/23、19/23，F1=.8889／.8571／.8462；均未優於原30幀，因此保留`ur-sample-human-s1-001`，普通Accuracy20/23=86.96%、F1=.8889、TP12／TN8／FP0／FN3／unknown0。可判定與包含unknown的整體Accuracy均以23為分母，全體跌倒召回12/15=80%；不能用balanced accuracy90%宣稱普通Accuracy達標。

共用尺度補到fall-30卻新增fall-29／11漏報；fall-30的事件沒有任何可見FALL ALERT影格。fall-09紀錄提前至4.833秒，畫面因目前Pose品質不足仍到5.9秒才警示。六段粗略時間檢查不是正式GT，整體guard因Accuracy／F1下降失敗。方法實作與分數一致通過，不等於辨識、警示時間或骨架品質改善。

174 passed／1舊預設權重路徑skip，46項相關風險測試通過；新Human真實权重有執行。最新四片503幀／313可比窗分數差0、事件與可見狀態相同、H264完整解碼並抽幀目視；原30幀CSV／事件逐位元相同。36舊run、843舊cache、設定／manifest／權重hash保留，四組Train-only Scaler和Validation指標重算吻合。Lint通過；format僅一個換行建議，保留原鎖定bytes。重跑及未達標Test gate拒絕實測，無新Test。synthetic反例、real_feature_smoke pilot、real_end_to_end分開保存。

沒有外部資源阻塞；歷史Test已查看、反覆Validation、教師評分單位未明限制仍揭露。正式Notebook／五分鐘影片依品質優先指示暫停，尚未宣稱課程達標。[完整報告與操作](human-displacement.md)｜[最終重算與保留證據](validation/human-displacement-verification.json)｜[完整測試XML](validation/pytest-human-displacement-full.xml)。


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

## 不合格骨架顯示修正（2026-09-18）

simple現以run品質要求檢查整副Pose，不合格時隱藏骨架／框／局部標籤，保留原UNKNOWN狀態；衍生neck不計入真實關節數，原始姿態與推論資料不變。這是顯示修正，信心合格但位置錯誤的Pose仍可能出現，未宣稱骨架或Accuracy改善。

14項顯示／串流風險測試通過，3.14秒；Ruff／格式通過。三支Validation共419幀真實重跑，推論CSV與事件CSV逐位元相同，H264全部解碼、抽幀目視確認；28個frozen run及舊影片hash保留。synthetic單元測試與real_end_to_end顯示一致性證據分開；未訓練、未追加Test。[完整說明](simple-demo.md)、[真實整合](validation/skeleton-visibility-integration.json)、[pytest](validation/pytest-skeleton-visibility-final.xml)。

## 密集計分實驗（2026-09-18，已完成；未提高Accuracy）

stride2／stride1候選各完成10支真實pilot：92／179共同窗口特徵差0、Pose array hash相同。各70支prepare、70epochs AE，共140epochs。兩個新模型Validation皆TP9/TN7/FP1/FN6/U0，可判定及全體Accuracy16/23=69.57%、F1=0.72；低於舊ur-ae-decay-001的17/23、0.80，不採用。四支粗略时间案例的跌倒区間新事件0→1、站立截止前事件1→0，只是個案工程檢查，不是正式事件時間Accuracy。[完整計畫與結果](dense-schedule.md)。

三支Validation重新影像推論419幀，284個可比特徵窗分數差0、狀態／事件一致，H264副本完整解碼且抽幀目視。fall-25在2.267–2.333秒3個影格顯示警示，隨後UNKNOWN；fall-09提早警示消失但仍漏判。`real_feature_smoke`的pilot、`real_end_to_end`的正式訓練／推論與synthetic風險測試分開，不把影片流程一致當成辨識提升。

完整153 passed／1原Human資源skip，37.08秒；Ruff及70檔格式通過。12則CuBLAS／上游相容性警告保留。26舊frozen run、2新run與source index hash核對；Validation全部重新以共用Scorer計算一致；Test gate拒絕未達標評分已實測，沒有新Test前凍結紀錄。證據：[最終pytest](validation/pytest-dense-schedule-final.xml)、[整合](validation/dense-schedule-integration.json)、[保留hash及重新計算](validation/dense-schedule-verification.json)。正式Notebook／五分鐘影片仍暫停。

## 持續優化：骨架恢復與AE泛化（2026-09-18）

兩個960候選各9支真實Train／Validation的`real_feature_smoke`已完成，Fall有效窗27→28，皆未通過+30% gate；沒有其AE或Test成績。原23個frozen run與來源index hash已核對，舊recipe不變。偵測confidence回退及衍生neck品質計數的10項相關測試通過（上游oneDNN提示1），見`validation/pytest-pose-recovery.xml`。

AE有限實驗已完成3×100epochs，見[固定計畫與實際結果](ae-generalization.md)。Validation排名第一ur-ae-decay-001：TP12/TN5/FP3/FN3/U0，可判定與全體Accuracy皆17/23=73.91%、F1=0.80；舊基準16/23、F1=0.7586。其他候選及全部分母見原始`ae-generalization-validation.json`。正常Train scaler與基準逐位元相同；四组Validation共用Scorer重算與保存指標相符。

**效果限制：** 新增影片級TP是fall-09，3.63秒人物仍站立時先警示，5.30／6.00秒倒地後UNKNOWN。影片級指標提高不等於事件時機正確，不能宣稱已改善實際跌倒警示。未達90%／0.87，不追加Test；Test gate拒絕已實測，沒有新Test凍結紀錄。[錯誤案例](validation/ae-generalization-errors.json)、[真實整合與目視核對](validation/ae-generalization-integration.json)。

兩支Validation真實重新推論產出334幀，49窗可比分數差0、未被品質遮罩的顯示狀態相符、事件相同，H264副本完整解碼。這是`real_end_to_end`流程一致性證據；兩组Pose pilot仍只標`real_feature_smoke`，synthetic隔離測試不列辨識成績。

最終完整回歸148 passed／1 skipped，85.05秒，Ruff及70檔格式檢查通過；skip只為原Human缺四項資源。12則警告涉及CuBLAS可重現性與上游相容性。初次完整測試147 passed／1 failed／1 skipped，失敗是新增去噪測試的float32誤差容限過嚴；修正後13項相關測試及完整回歸通過，兩份XML均保留。證據：[最終XML](validation/pytest-optimization-final.xml)、[保留與重算驗證](validation/ae-generalization-verification.json)。23個舊frozen run、3個新run與來源index未變；正式Notebook／五分鐘影片仍暫停。

## 簡短英文demo（2026-09-18）

已新增`simple`並設為CLI demo預設，舊debug及中文presentation保留。主畫面只用PENDING／NORMAL／FALL ALERT／UNKNOWN，不顯示ID與分數，不把normal／unknown改稱STANDING／LYING。四支Validation共628幀與舊版推論CSV、事件CSV逐位元相同，模型及舊影片hash未變；H264播放副本完整解碼，四種狀態目視確認。相關5 tests passed，Ruff與CLI help通過。[詳細驗收與操作](simple-demo.md)、[真實整合證據](validation/simple-overlay-integration.json)、[pytest](validation/pytest-simple-overlay.xml)。

範圍只涵蓋畫面及推論一致性，沒有新訓練／Test／Accuracy改善。Graph傳輸中斷，已依規則嘗試index、search、coverage，改用本次原始碼範圍核對；未宣稱graph驗證通過。

## Accuracy優先後續（2026-09-18）

- `real_feature_smoke`：8支Train／Validation的真實旋轉骨架候選已抽取；Fall有效窗27→30、ADL93→104，未通過事前Fall+30% gate。不是Accuracy驗證，也不是新`real_end_to_end`成績。[原始證據](validation/accuracy-first-rotation-pilot.json)。
- 本候選沒有AE訓練run、沒有Test評分；Notebook／影片更新暫停。既有Test可判定Accuracy11/22=50%、F1=56%，含unknown全體11/23=47.8%，仍為历史工程對照且未達目標。
- 已驗證23個既有frozen run及source index未變；三份旋轉候選config保持訓練前hash。CLI只讀檢查：`.\.venv\Scripts\python.exe scripts/accuracy_first.py status`。
- 初次pytest有6項因沙箱暫存目錄權限造成setup error，證據保留於`pytest-accuracy-rotation-sandbox.xml`；取得執行權限後相關20項通過（2項上游oneDNN提示），見`pytest-accuracy-rotation.xml`。最終程式與測試整理後的完整驗證另見下方追加紀錄。
- 同學ZIP已逐檔確認70張JPG，Notebook只有讀圖與依檔名統計，無訓練；來源與內層hash保留於`course-sample-dataset-audit.json`。參考論文補查見[Accuracy優先紀錄](accuracy-first-plan.md#參考論文補查與下一步)。

辨識改善與課程協定一致性仍未完成；這不是教師回覆前完全無法工作的阻塞，也不能保證只要取得其他權重即可達90%。

最終回歸驗證：**139 passed、1 skipped**，23.94秒，[XML](validation/pytest-accuracy-first-final.xml)。skip只涉及原Human四項缺少資源；12則警告涉及CuBLAS非完整deterministic與上游相容性，不能宣稱GPU跨次逐位元可重現。Ruff check及66個檔案格式檢查通過；已實際驗證pilot重跑拒絕及gate失敗拒絕訓練。[保留hash及防覆蓋檢查](validation/accuracy-first-rotation-verification.json)。

## 期末課程對齊（2026-09-18 本機驗收完成）

本輪完整測試136 passed／1舊Human缺資源skip；ruff、格式及compileall通過。以下為本輪新增證據，功能驗證不等於辨識可靠。詳見[完整成果與限制](course-results.md)。

| 項目 | 目前狀態 | 預定驗收證據 |
| --- | --- | --- |
| CA-01 UR 課程協定與診斷 | 已執行；課程細節仍待釐清 | urfall-course-protocol.md、urfall-diagnosis.md、validation/course-ur-diagnosis.json |
| CA-02 UR-only 基準與改善 | 已執行；辨識品質不足 | 三候選297回合；course-before-test-freeze.json；Test TP7/TN4/FP4/FN7/U1 |
| CA-03 成果 Notebook | passed | 7 cells、0 cell error、2張內嵌圖；正式輸出／分母／hash核對，乾淨kernel再執行成功；course-notebook-execution.json |
| CA-04 中文展示模式 | passed (real_end_to_end) | 4支共628影格／模式，log及事件逐位元一致；67個未遮罩窗口與cache分數差0；4狀態抽幀，course-demo-integration.json |
| CA-05 五分鐘期末影片 | passed locally／未上傳 | 有聲／無聲兩版、9000影格、30fps、300秒；完整解碼／13段抽幀，course-film-verification.json及course-demo-script.md |

二元 Accuracy／F1、含 unknown 的 operational accuracy、全體跌倒召回須並列且標明分母。既有 Test 重用限制保留，未確認官方協定前不宣稱直接達到教師基準或換算課程分數。synthetic、real_feature_smoke、real_end_to_end 證據仍分開。

## 最新：YOLO11 特徵的 AE 訓練強化已完成

四組實際165／131／144／81回合，共521回合；固定候選事前鎖定、Train僅ADL、正常Validation選checkpoint、Validation選門檻與版本。依事前規則選定 `caucafall-yolo11-balanced-001`，Test前frozen；其他三新候選未執行Test，Test後未調參或改選。

Validation43/58→47/58（81.0%）。固定Test的CAUCAFall18/20→20/20，但全體仍57/81=70.4%；TP28/TN29/FP5/FN18/unknown1，unknown留在分母。少2誤報同時多2漏報，整體95%目標未達到，亦非全新獨立Test。詳見[比較報告](refinement-training.md)及[機讀結果](validation/refinement-results.json)。

- 真實端到端：4支Validation影片共1417幀完整解碼；Walk／SitDown／Kneel沒有警示，FallForward有警示。252個有效視窗分數差0、states/events一致、0遮罩例外；[整合證據](validation/refinement-integration.json)。
- 相關風險測試：[25 passed](validation/pytest-refinement.xml)，11.20秒；ruff／compileall通過。無核心模型或推論程式修改。
- 程式Loss已核對144回合、best124／0.143568、6096個Train窗與1026個正常Val窗。4候選皆保存原始Loss、checkpoint及門檻。
- 原16個frozen run及共用特徵index hash不變，4新候選在Test／demo後的hash與凍結紀錄一致。原資料／權重／cache／runs／outputs未納入Git。

## 歷史：Le2i擴充已完成

69 passed、1 skipped（僅舊Human缺四資源），[pytest證據](validation/pytest-le2i.xml)。161段、54,342影格完整核對；190段來源hash檢查及四段Validation去重；完整prepared得到6,404個特徵。新run實際訓練79 epochs、Validation校準、61段凍結Test、兩支真實demo均已執行。實際誤報仍高，詳見[結果](le2i-results.md)。

| 擴充驗收項目 | 結果與證據範圍 |
| --- | --- |
| 來源與切分 | passed；固定官方ZIP/OmniFall revision，Train僅ADL，跨組人物/內容檢查，重複組排除 |
| 時間、完整性、cache | passed；161/161完整解碼、91段AVI宣告數匹配；真實時間重採樣，未標註空白不補正常；重用不改來源 |
| 骨架候選 | executed；只用Train/Validation，旋轉座標及條件推論測試；仍有track碎裂，未宣稱品質問題解決 |
| AE／校準／Test隔離 | passed；Train正常3,766窗、79 epochs；Validation選checkpoint及threshold，Test固定後只評估一次 |
| 跨資料集正常Validation | limitation；Le2i四段皆無有效特徵，正常Validation只有UR Fall七段128窗，沒有用Test補齊 |
| 真實Test品質 | insufficient；61段18unknown，全部跌倒22/37警示、正常17/24誤報；by_dataset完整列出 |
| 原生AVI與輸出 | passed；Validation原生AVI輸入與UR圖片輸入；輸出190/110影格完整解碼 |
| Batch／streaming | passed within scope；UR七窗raw score/狀態/events一致；Le2i28共同有效窗raw score差0，受標註範圍限制的events與無標籤demo不相同 |
| 凍結、來源、證據 | passed；新舊run於Test及展示後hash未變；synthetic／real_feature_smoke／real_end_to_end分開 |

[完整整合證據](validation/le2i-final-integration.json)、[正式Test](validation/le2i-test-metrics.json)、[資料](validation/le2i-dataset-report.json)、[訓練校準](validation/le2i-training-calibration.json)。已查看訓練曲線、混淆矩陣與真實抽幀。Le2i示範約1.967秒先誤報起身，不能把sequence TP當作跌倒起點正確。

Graph Verify最終generation 2026-09-17T04:02:26Z，10個本次src/tests路徑metadata_match、未記錄coverage缺口；ignored資料／upstream另直接讀取。此為best-effort訊號，不代表完整正確性。

## 原OpenPose基準驗收（歷史保留）

2026-09-17。第二條路已完成實作與真實實驗；**功能驗證通過，辨識品質不足**。54 passed、1 skipped、8 warnings，68.50秒：[pytest原始證據](validation/pytest-openpose18.xml)。唯一skip為原Human四個資源缺少；替代權重CPU/CUDA測試均實跑。warnings來自舊Human SciPy/NumPy與PyTorch oneDNN旗標，沒有測試失敗。

原2026-09-11 Human驗收保留於 [歷史紀錄](validation/acceptance-human-2026-09-11.md)，不再代表目前替代方案狀態。以下passed均限於證據範圍，不等於高準確率。

## 規格契約與實驗

| 驗收項目 | 狀態 | 實際證據及範圍 |
| --- | --- | --- |
| 環境、codec | passed | 隔離.venv；CPU/CUDA真實模型測試；CUDA完整流程；三支MP4完整解碼 |
| 來源、checkpoint | passed | 作者來源、固定commit、實際SHA256；Pose完整426 keys／ST-GCN完整172 keys strict load；無隨機降級 |
| 替代骨架／特徵 | passed | 原生OpenPose18、30fps、30幀、stride5；Kinetics正規化與zero-confidence mask；分類頭前256D及head parity |
| 權重凍結 | passed | eval、requires_grad false；buffers檢查，CPU/CUDA真實checkpoint與forward |
| 資料、時間 | passed with documented fallback | 70/70、11,936張；adl-37尾端20張時間估計有標記；中間缺口拒絕；測試與fingerprint |
| 原始序列切分 | passed | ADL24/8/8、Fall0/15/15；先切序列、Train僅ADL；manifest hash保存 |
| 快取、因果窗口 | passed | 18點/256D recipe隔離、尺寸/hash/空快取；因果性／track-gap測試；70段prepared、777有效窗 |
| Scaler／AE／Test隔離 | passed | 真實run僅Train正常更新，Validation正常選checkpoint；fixture驗證Test更動不影響選模 |
| 校準、門檻 | passed | 真實Validation校準；strict >、float64邊界、同track K=3；AE逐筆與批次輸入bitwise一致 |
| 事件、unknown、coverage | passed | no_person／track_gap不當normal；序列與事件一致；不縮unknown及缺timeline分母 |
| 真實Test | executed; quality insufficient | 23段、8unknown；TP7 FP7 FN1 TN0（可判定15段）；全部跌倒7/15警示、正常7/8誤報 |
| 保存、重載、凍結 | passed | 正式run重載推論；evaluate後及最後展示後凍結檔案hash不變 |
| Batch／streaming | passed within compared scope | adl-05的29個、fall-02的4個影片可比有效端點，分數差0、狀態／事件一致；另有完整synthetic窗口測試 |
| 影片輸入／輸出 | passed | 真實圖片序列179/110影格；衍生MP4檔輸入110影格；骨架、unknown、警示及CSV；headless執行 |
| EOF時間 | passed | 尾端7ms fixture回歸，finish使用實際來源末時刻；5項streaming測試含於全套54項 |
| 靜態檢查 | passed | ruff check、ruff format --check、compileall；validation/openpose18-final-checks.json保存退出狀態 |

## 可重查證據

- [完整資料檢查](validation/openpose18-dataset-report.json)、[prepare摘要](validation/openpose18-preparation.json)。
- [真實特徵smoke](validation/openpose18-real-feature-smoke.json)：classification=not_performed。smoke在尾端時間政策啟用前執行；兩段輸入皆無該缺口，正式全資料prepare另以目前recipe實跑。
- [完整doctor](validation/openpose18-doctor-full.json)、[正式整合與凍結hash](validation/openpose18-final-integration.json)。
- [Test metrics](validation/openpose18-test-metrics.json)、[batch/streaming](validation/openpose18-batch-streaming.json)、[結果與限制](results-summary.md)。
- outputs/synthetic/openpose18-validation僅手製fixture。原Human synthetic與blocked smoke也保留，不混成新方案成果。
- 已視覺檢查正式訓練曲線、混淆矩陣及真實影片1/2/3秒抽幀。躺倒後骨架碎裂及unknown可見，沒有美化成成功。

## 仍受阻或未驗證

- 原Human仍缺三個權重及一個detector結構設定，其512D真實端到端仍blocked；第二條路不用這四個檔案。
- CPU完整120幀feature smoke因速度中止；CPU真實影像及骨架窗forward已測，未驗證完整CPU實驗、CPU-only wheel或即時效能。
- 未進行官方posture CSV對齊、人工event onset／延遲評估、新場景及自行拍攝影片的泛化測試。
- 作者未提供可取得的預期checksum；保存本地實際SHA256及官方來源，不宣稱作者checksum比對。
- 課程評分要求仍以老師原文為準，沒有把此實作當作老師已批准或論文成績重現。

索引使用Verify層級；18個本次src/tests證據路徑在generation `2026-09-17T01:50:39Z` 均metadata_match、無記錄parse gap。上游、資料及輸出因gitignore不入graph，以必要原始碼讀取與實測補足。此為best-effort覆蓋訊號，非完整正確性證明。


## Loss報告圖補充

三張真實Loss圖（原基準、Le2i、並列）與PNG/SVG、CSV副本、圖說已交付。新腳本直接執行完成，曲線點與原CSV逐值一致，最佳epoch/loss與run狀態一致，凍結hash未變；三張PNG已解碼、三份SVG已解析、版面已視覺檢查、ruff通過。證據[loss-figures-checks.json](validation/loss-figures-checks.json)。本次未更動訓練／推論規則，因此未重跑完整69項模型測試，也沒有新增Test評估成績。

## 程式內 Loss 畫面補充

本機互動介面與逐回合更新已完成，入口 `loss`／`train --show-loss`／「查看Loss曲線.cmd」。實際核對真實run曲線點、最佳epoch、表格與切換；原CSV有hash驗證，synthetic明確區隔，資料缺失／讀取錯誤不繪造曲線。HTTP只提供指定run紀錄與CSV，路徑越界與外部Host拒絕。

全套76 passed／1 skipped（原Human資源缺少），最後缺省數量顯示修正後7項相關測試再通過。實際synthetic AE 3回合驗證前端自動更新與完成；此為功能證據。兩個真實run全部凍結hash保持不變，未新增正式訓練／Test成績。[驗證紀錄](validation/loss-viewer-checks.json)、[全套pytest](validation/pytest-loss-full.xml)、[最後相關測試](validation/pytest-loss-viewer-final.xml)、[操作](loss-viewer.md)。

## 接續訓練：旋轉骨架對照

真實run openpose18-rotation-001已完成100回合、正常Validation選checkpoint及Validation門檻校準；最佳Epoch95、Loss0.499633。本輪未執行Test，不能標為完整獨立Test驗收通過。相同Validation跌倒警示9/15、正常誤報6/8、unknown8/23，改善有限。

110幀真實Validation影片、7個共同有效端點分數與狀態/events一致，舊run全部凍結hash不變；11項相關測試通過，瀏覽器已核對新run曲線與摘要。[訓練結果](rotation-training.md)、[整合證據](validation/rotation-training-integration.json)、[pytest](validation/pytest-rotation-training.xml)。


## 低 FPS 改善與重新訓練

已修正24 FPS來源在30 FPS重採樣時的假缺幀，真缺幀與因果性限制保留。161段獨立cache完成，正常Train57/62段4180窗、正常Validation11/12段286窗；真實AE訓練88回合、最佳Epoch78。新政策顯式配置，舊fingerprint與run不變。Pose只沿用hash相同的真實觀測，ST-GCN完整重算；两段真實影片另做全流程重新推論。

規格F1校準與替代平衡校準分開保存，同一38段Validation的正常誤報分別10/12、1/12；跌倒警示分別20/26、12/26，unknown均8/38。替代方案以增加漏報換取較少誤報，並非整體成功；本輪無Test評估。來源AE/scaler/Loss逐檔hash相同，替代run不宣稱再次訓練。

全套92 passed／1 skipped／11 warnings（17.76秒），ruff/format/compileall通過。真實正常影片48與跌倒影片7個共同可見有效端點raw score差0；UR狀態/events一致。正常影片因標註區間與當下品質遮罩，僅44/48狀態相同、1個端點不顯示分數；两邊均無事件，不宣稱全狀態等價。Loss瀏覽器畫面與實際影片抽幀已檢查。舊三個run及新F1run凍結hash保持不變。

[操作與結果](timing-improvement.md)、[真實整合證據](validation/improvement-timing-integration.json)、[全套pytest](validation/pytest-improvement-full.xml)、[最後檢查](validation/improvement-final-checks.json)。原Human權重阻塞與完整CPU/泛化未驗證狀態不變。


## 接續 AE 訓練與 Windows 進度檔修正

三組真實AE共577回合完成；同一38段Validation整體正確判定率未超過既有55.3%。影片均衡組最低正常Validation Loss0.197624不當作準確率改善；所有F1基準、平衡門檻與失敗重跑均保留。[全部結果與操作](further-training.md)、[機讀比較](validation/further-training-results.json)。

新增按正常Train影片均衡抽樣，Scaler與Validation Loss計算不變；變更Test特徵不影響AE/scaler/threshold的回歸涵蓋兩種抽樣。Windows存檔短暫鎖定有界重試、持續錯誤拋出、失敗run標記與Loss畫面驗證通過。全套100 passed／1 skipped／11 warnings，19.01秒；ruff/format/compileall通過。原Human權重skip原因不變。

compact平衡版兩段真實Validation影像359/110幀完整重新推論與解碼；48＋7共同可見有效端點raw score差0，UR狀態/events一致；Le2i44/48狀態相同、1端點顯示遮罩，不宣稱全狀態一致。未新增Test，故不是完整real_end_to_end驗收；舊五個run全部凍結hash不變。證據[整合](validation/further-training-integration.json)、[最終pytest](validation/pytest-further-training-final.xml)、[最後檢查](validation/further-training-final-checks.json)。


## CAUCAFall 接入驗證（完成）

100 支官方完整 AVI 已下載，逐支 ZIP CRC 與实际 SHA256 已保存。合併 231 支 selected raw 全部解碼驗證通過。14 項資料協定風險測試與 20 FPS 時間比較回歸均通過；全套目前 117 passed/1 skipped/11 warnings（22.52秒），skip 仍為原 Human 權重缺件，替代 OpenPose18/ST-GCN 的實際 strict load/forward 已通過。新的 real_feature_smoke 使用三支 Train/Validation 真實影片，49/19/62 個特徵。已完成231支prepare、97回合AE、先凍結後Test與两支新來源streaming demo；本輪real_end_to_end流程驗證完成，辨識品質仍不足。[測試證據](validation/pytest-caucafall-final.xml)。

CAUCAFall Test75.0%、總Test59.3%，unknown分別1/20與10/81；測試表與舊61支同組比較見[實驗結果](caucafall-training.md)。新demo338/299幀通過解碼，62/18可見端點與快取分數完全相同，狀態/events一致；Fall另1低品質端點隱藏分數，未驗證真實起跌延遲。Loss頁面97回合/最佳87與曲線已視覺確認；新兩run和舊五run凍結hash不變。

## 95% 目標改善實驗（本輪完成，目標未達）

新增官方 YOLO11m Pose 的獨立 18 點映射 recipe、明確的 1 ms 到期容差及 50 ms 有界時間抖動政策。17 個原始 COCO 點完整映射，衍生 neck 不計入 12 個真實關節品質要求。691 個真實權重嚴格載入；與官方 checkpoint 的真實影像輸出差 0。原 ST-GCN 仍為固定 Kinetics 權重的分類頭前 256D，沒有使用跌倒分類頭或隨機特徵。

完成 231 支全新實際擷取、231 份相同骨架的時間對照、三個 AE run 共 293 回合。Train 僅 ADL，Validation 選 checkpoint／門檻及候選，只有選定且凍結的 `caucafall-yolo11-jitter-001` 執行本輪 Test。全部 81 支準確率 59.3%→70.4%，CAUCAFall 75%→90%，unknown10→1；未達95%，不能宣稱可靠部署或新独立泛化驗證。[完整方法與結果](target95-improvement.md)。

132 passed／1 skipped／12 warnings（29.99 秒），唯一 skip 為原 Human 資源缺失；ruff／format／compileall 通過。時間因果性、明確缺失不填補、recipe/cache 隔離、預設 fingerprint 相容、衍生點不灌水、CPU 不污染後續 CUDA 選擇均驗證。真實 CPU YOLO、CPU／CUDA 原特徵模型測試通過；全資料 YOLO 使用 CUDA，完整 CPU 實驗未執行。

兩支 Validation 原始 AVI 全新推論，338／299 幀均完整解碼；62＋45 個有效端點 raw score 差 0，狀態／events 一致。正常無警示、跌倒有警示，兩張預覽已查看，但未驗證起跌延遲。程式內 Loss 曲線100回合／最佳Epoch95實際確認。16 個 frozen run 均通過 artifact 核對，4 個候選的 Test 前 hash 不變，100 支 CAUCAFall 原始影片 hash 再次核對。

证據：[風險測試](validation/pytest-target95-final.xml)、[候選凍結](validation/target95-before-test-freeze.json)、[成績](validation/target95-results.json)、[真實展示整合](validation/target95-integration.json)、[最後檢查](validation/target95-final-checks.json)。本輪 `real_end_to_end` 流程驗證完成，不等於 95% 效能驗收通過。

## 2026-09-17 — 影片 sample 展示

- 使用 `runs/caucafall-yolo11-jitter-001` 重新執行兩次 `python -m fall_detect demo --no-display`，輸入為 CAUCAFall 驗證集 Subject 5 的 FallForwardS5.avi 與 WalkS5.avi；輸出 `outputs/sample-fall-001.mp4` 和 `outputs/sample-walk-001.mp4`，並保留 JSON、逐窗 CSV 與事件 CSV。
- 真實完整推論：跌倒影片 299 幀，第一個警示在影片 4466.67 ms；正常走路 338 幀、0 警示。此為觀看範例，不新增準確率或偵測延遲宣稱。
- 增加 `outputs/sample-viewer.html`，本機觀看網址 `http://127.0.0.1:8767/sample-viewer.html`。使用 `.tools` 中的 imageio-ffmpeg 0.6.0 將實際輸出轉為 H.264 MP4（`sample-*-001-browser.mp4`），保留原始模型輸出；轉檔完整解碼檢查通過。
- 內建瀏覽器 WebM 播放嘗試曾當機；切換 H.264 後，已觀察跌倒影片實際播放至 7.69 秒（paused=false、readyState=4）及紅色疑似跌倒警示。沒有修改模型、資料切分、門檻或推論規則。
