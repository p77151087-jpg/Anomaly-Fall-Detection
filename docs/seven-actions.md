# 七類動作與AE異常警示並列

2026-09-18，使用者明確要求實作七類動作展示。這次授權補充先前的簡短英文顯示契約：從完整、嚴格載入的Human七類分類頭取得真實輸出，作為獨立動作資訊；不以分類結果代替正常資料AE。

畫面左側ACTION顯示STANDING、WALKING、SITTING、LYING、STANDING UP、SITTING DOWN、FALLING，右側ANOMALY (AE)維持NORMAL／FALL ALERT／PENDING／UNKNOWN。PENDING與UNKNOWN是資料狀態，不是動作類別。兩者可能不同，必須如實呈現。分類頭未重新訓練，使用含跌倒標註的Le2i預訓練；不把分類分數當校準後機率或AE辨識成績。

僅原30幀、per_frame的Human recipe可啟用；其他骨架或實驗前處理沒有相符七類介接，不任意套用。CLI預設auto會為相符Human選actions，其他模型選simple，實際模式記錄於metadata；顯式actions遇不相符模型即拒絕。

每個有效特徵窗口的原始512D同時供原AE與原分類頭使用，不更動Scaler、AE、門檻、K3、原始骨架或事件。分類不須等AE連續三窗，但目前Pose低品質、track gap、無效窗口、過期輸出均清除動作標籤與分數，禁止沿用舊分類。沒有另加分類門檻或展示平滑。新-actions.csv另存各類原始sigmoid分數及argmax；原推論CSV與事件CSV保持相同格式。

開始前已保存37個凍結run、1165個來源cache檔案hash、設定與manifest、四個待修改程式的原版本；另外重新輸出四支Validation的舊版影片作精確比較。正式Notebook／五分鐘影片仍依辨識品質優先指示暫停，無新訓練、Test或公開上傳。

## 實作與驗收完成

已增加`HumanFeatureExtractor.action_scores()`，沿用同一完整checkpoint的`sigmoid(fcn(raw512D))`，不經Scaler，不修改特徵陣列或模型。其輸出與完整原模型forward相符；七類順序另與固定上游原始碼比對。`ActionDisplayState`獨立處理每個track的分類、品質與新鮮度；AE的EventState、閾值與原CSV完全不受分類控制。

CLI預設`--overlay auto`，原30幀Human選用新的actions畫面；顯式`--overlay simple`仍為原AE畫面，debug與presentation保留。Python `demo()`原API的debug預設保持不變，呼叫端可明確指定auto／actions。未知模型或短窗／per_window不會默默套用原七類分類。新模式保留完整原畫面，新增92px雙欄標頭；兩欄代表同一人，多人時標明track，人物框也各自顯示動作與AE狀態。動作FALLING為紅色、LYING為橘色、其餘有效動作為綠色，低品質時隱藏骨架。

32項相關測試通過（5.47秒）；完整191 passed／1 skipped（30.70秒）。唯一skip仍是歷史預設權重路徑，新的Human真實權重分類頭測試已實跑。之後只為真實權重測試加入跨機器缺資源時的skip條件，15項分類測試再次通過（本機0skip）；產品程式未再變更。初次沙箱測試26 passed／6 setup errors，原因是Windows既有pytest暫存目錄權限；取得執行權限後完整通過，沒有為通過測試放寬模型或數值條件。12則既有CuBLAS／上游相容性warning保留。

四支真實Validation影片均重新跑人物偵測、姿態、ST-GCN、分類頭及AE，總計503幀。原AE推論CSV與事件CSV相較本次修改前版本**逐位元相同**；313個可見分類窗口與保存的原始512D特徵重算七類分數，最大差0，顯示argmax與分數一致。所有UNKNOWN／PENDING列沒有保留分類分數；H264播放副本全片解碼，時間、影格數與尺寸均一致。37個舊run、1165個舊cache、設定／manifest／真實權重及修改前影片hash全部保留。

| 案例 | 影格數 | 有效分類窗口 | 實際顯示的動作類別（另有PENDING／UNKNOWN） | AE前後 |
| --- | ---: | ---: | --- | --- |
| adl-04 | 149 | 118 | STANDING、WALKING | CSV／事件完全相同 |
| fall-09 | 185 | 127 | STANDING、WALKING、FALLING | CSV／事件完全相同 |
| fall-29 | 99 | 48 | STANDING、WALKING、FALLING | CSV／事件完全相同 |
| fall-30 | 70 | 20 | SITTING、STANDING UP | CSV／事件完全相同 |

實際目視四支影片的接觸表與fall-09首次FALLING影格：字體與雙欄可讀、骨架配色清楚，fall-09在4.70秒已顯示FALLING但AE仍NORMAL；有7個影格為此組合。fall-29也有5個影格FALLING／AE NORMAL，後續可见FALL ALERT仍僅2幀。fall-30在1.6秒身體向下時分類為STANDING UP，後段身體出框時顯示UNKNOWN，並未辨識成FALLING。七類介接正確不代表七類辨識都正確，沒有七類逐影格GT，本次不宣稱其Accuracy；四片也未實際涵蓋LYING與SITTING DOWN，這兩類的映射由原模型類別契約及測試驗證。

原AE的23支Validation全部重算相符：TP12、TN8、FP0、FN3、unknown0，可判定及全體Accuracy均20/23=86.96%、F1=.8889，跌倒召回12/15=80%，窗口coverage84.39%。分類頭顯示FALLING不計入AE的TP，也未宣稱提高這些指標。歷史Test已查看與反覆Validation限制保留。

證據分級：手製骨架／影像的狀態與隔離測試為synthetic；真實權重上的完整forward parity只證明介接數值，不能當成辨識成績；503幀原影像重跑與凍結指標核對屬real_end_to_end。本次沒有新增訓練pilot或Test評估。

## 成果與操作

- [跌倒動作与AE不同步的例子 fall-09](../outputs/seven-actions-validation/fall-09-preview.mp4)
- [跌倒 fall-29](../outputs/seven-actions-validation/fall-29-preview.mp4)
- [正常活動 adl-04](../outputs/seven-actions-validation/adl-04-preview.mp4)
- [坐姿後跌倒仍有失敗的例子 fall-30](../outputs/seven-actions-validation/fall-30-preview.mp4)
- [整合核對與hash](validation/action-display-integration.json)、[完整測試](validation/pytest-action-display-full.xml)、[相關測試](validation/pytest-action-display-related-final.xml)

每支影片旁另有原始.mp4、原AE.csv、-events.csv、-actions.csv、metadata.json與接觸表。`head_score`及七欄分數是原分類頭sigmoid輸出，不保證相加為1，不呈現成「跌倒機率」。

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
# 核對既有成品與模型，不重訓、不改寫成果
.\.venv\Scripts\python.exe scripts/verify_action_display.py
# 播放已完成的新版
Start-Process 'outputs/seven-actions-validation/fall-09-preview.mp4'
# 用目前最佳Human模型產生另一份；輸出檔名須尚未存在
.\.venv\Scripts\python.exe -m fall_detect demo --run runs/ur-sample-human-s1-001 --input data/raw/urfall/fall-09 --output outputs/demo/fall-09-seven-actions-new.mp4 --overlay actions --no-display
```

舊實驗脚本的source lock原樣保留，因此會拒絕目前新增方法後的原始碼；請以本頁的新驗證指令核對當前版本，不能修改舊lock假裝原始碼沒有變。沒有外部阻塞；此展示修改已完成，不包含重訓、七類Accuracy驗收、修復全部骨架、達到課程90%或正式五分鐘影片。
