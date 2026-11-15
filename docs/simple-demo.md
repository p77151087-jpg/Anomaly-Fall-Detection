# 簡短英文展示狀態

2026-09-18後續：依使用者明確授權，已新增[七類動作＋獨立AE展示](seven-actions.md)。CLI預設改為auto，原30幀Human用actions，其餘用simple；明確`--overlay simple`仍維持本頁原AE畫面。下方「本次沒有新增姿態分類」是較早版本紀錄，新版已接真正的預訓練七類分類頭，仍不從AE分數推造動作。

## 骨架亂飛的顯示修正（2026-09-18）

使用者指出跌倒時骨架亂飛。核對原始Pose cache：fall-25第71影格（2.335秒）同一人被拆成兩個track，各只有6／7個可信真實關節，不滿足設定要求的12個。晚段影片另有Pose錯落到背景椅子附近。這是姿態估測／追蹤不穩，部分身體出畫面與遮擋也限制了觀測；不能視為正確骨架或正常辨識結果。

原renderer雖會過濾單一關節信心低於0.2的點，仍會把整副不合格骨架剩下的點連線。現在simple依實際run的整幀品質條件，隱藏不合格Pose的骨架、人物框與局部標籤，保留全局UNKNOWN與新鮮度提示。衍生neck不冒充額外真實關節，個別繪圖關節也沿用設定信心門檻。符合品質但位置錯誤的Pose仍可能出現，這不是完整的Pose修復。

只改顯示，不修改原始keypoints、追蹤ID、ST-GCN輸入、AE分數、事件或unknown判定。debug與舊presentation保留原始可視化供診斷。三支Validation以同一實驗run ur-stride1-decay-001重跑，共419幀；CSV與事件CSV與改版前逐位元相同，28個frozen run及舊影片hash保留。14項相关測試通過，Ruff與格式通過，H264副本全部解碼，抽幀目視确认不合格骨架隱藏、有效骨架保留。[整合證據](validation/skeleton-visibility-integration.json)、[pytest](validation/pytest-skeleton-visibility-final.xml)。

- [新版fall-25](../outputs/skeleton-quality-display/fall-25-preview.mp4)
- [新版fall-09](../outputs/skeleton-quality-display/fall-09-preview.mp4)
- [正常活動adl-04](../outputs/skeleton-quality-display/adl-04-preview.mp4)

這些為顯示診斷短片；stride1仍是未獲採用的實驗模型，不能把隱藏錯誤骨架當成Accuracy改善。正式Notebook／期末影片仍暫停。

2026-09-18，使用者要求讓展示接近同學影片的簡短英文標示。這次修改展示UI，另存4支短片；不更新正式Notebook／五分鐘影片、不訓練、不調門檻、不增加Test評分。

CLI `demo`預設使用`--overlay simple`。主畫面只有大字狀態、人物骨架／框、原影片時間及簡短新鮮度提示；不顯示追蹤編號、AE分數或連續三窗等技術說明。對應如下：

| 內部狀態 | 顯示文字 | 語意 |
| --- | --- | --- |
| warming_up | PENDING | 收集足夠的動作資訊 |
| normal | NORMAL | 未觸發異常警示 |
| suspected_fall | FALL ALERT | 已觸發疑似跌倒警示 |
| unknown | UNKNOWN | 目前資料不足，沒有有效判定 |

NORMAL不能冒稱STANDING，UNKNOWN也不能冒稱LYING或合併成正常。若要輸出站立／躺下等姿態，需額外實作並驗證姿態分類；本次沒有新增這項能力。unknown顯示「No valid result」，不沿用舊分數或舊更新時間；PENDING顯示「Collecting motion」，有效判定顯示「Updated ... s」。追蹤不穩定仍存在，隱藏編號沒有修復它。

原`--overlay presentation`中文模式與`--overlay debug`技術模式保留。Log仍保留track_id、raw_score、event_score、reason及score_age_ms，方便診斷。simple使用OpenCV英文字型，不需要額外中文字型檔。

## 已產出與驗證

使用已凍結`course-ur-balanced-001`，4支Validation共628影格：

- [正常活動 adl-04](../outputs/simple-status-20260918T095239/adl-04-preview.mp4)
- [跌倒警示 fall-02](../outputs/simple-status-20260918T095239/fall-02-preview.mp4)，事件判對不代表警示時機正確。
- [正常活動誤報 adl-20](../outputs/simple-status-20260918T095239/adl-20-preview.mp4)
- [跌倒漏報 fall-19](../outputs/simple-status-20260918T095239/fall-19-preview.mp4)

每支新影片的推論CSV與事件CSV都與舊中文影片逐位元相同；模型frozen hash不變。原始30fps，無加速或刪剪，H264播放副本640×544已全部解碼核對影格數；4種狀態已抽幀目視確認，舊影片hash不變。[整合證據](validation/simple-overlay-integration.json)。這是`real_end_to_end`呈現及推論一致性證據，不是新的Accuracy改善。

相關測試5 passed：[pytest](validation/pytest-simple-overlay.xml)，包含synthetic輸入不被renderer修改及既有unknown／報表契約。Ruff與CLI help通過。Graph連線回報Transport closed，search與coverage均已嘗試；改以原始碼閱讀／rg查核本次範圍，不宣稱graph coverage通過。

## 操作

於專案根目錄執行。使用新的輸出檔名；既有影片不允許覆蓋：

```powershell
.\.venv\Scripts\python.exe -m fall_detect demo --run runs/course-ur-balanced-001 --input data/raw/urfall/fall-02 --output outputs/demo/fall-02-simple-new.mp4 --device cuda --overlay simple
```

檢查內部資訊時改用`--overlay debug`，並另指定輸出名稱。此次變更沒有解決UR Fall Accuracy不足；正式交件的訓練改善仍按Accuracy優先計畫進行。
