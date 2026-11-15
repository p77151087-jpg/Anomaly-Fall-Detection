# 四段雙肩 demo 單頁人工覆核

已交付 `ur-shoulder-display-review-20261112-002`，本機頁為 **http://127.0.0.1:8851/index.html**。四段依序為 fall-19、fall-09、fall-18、fall-30，每段一筆展示確認，一次匯出 `shoulder-four-clips-human-review.json`。已在瀏覽器開啟並保留，交付時人工進度為 **0/4**，未代填判斷或勾選本人確認。

## 使用方式

1. 點片名切換，先看「播放重點」，必要時使用完整影片、首警示跳轉、半速／四分之一或逐幀。
2. 每段選三項：首警示時的動作、同一人與骨架合理性、當前警示與事件歷史的易讀性；可選「有問題」或「無法確認」。備註可省略，再勾本段親自查看。
3. 四段完成後，只填一次覆核者名稱／代稱及本人確認，再匯出一份正式 JSON。請交回對話驗證。自動草稿與下載／匯入備份可用；重載／匯入會取消全域本人確認，正式匯出前須重新勾選。

不需重標起訖、不需畫點、不需逐幀做紀錄。開始／落地既有時間範圍和骨架備註以唯讀參照顯示，也隨正式 JSON 帶回，不由本次展示確認取代。

| 影片 | 重點區間（秒） | 新模式首警示（秒） | 沿用人工資料 |
|---|---:|---:|---|
| fall-19 | 1.1–2.1 | 1.500 | 原第1–32幀正常、33–61轉換、62–100倒地；首警示原第45幀的整體wrong_or_drift保留。這些是姿態階段，不是時刻最早／最晚範圍。 |
| fall-09 | 4.2–5.1 | 4.566667 | 開始4.304–4.571、落地4.705–5.973秒的可能範圍，以及本人出框備註。 |
| fall-18 | 1.0–1.8 | 1.300 | 開始0.700–1.101、落地1.301–1.534秒的可能範圍；當前警示4幀，原出框／無骨架備註保留。 |
| fall-30 | 1.2–2.1 | 1.566667 | 開始1.167–1.534、落地1.635–1.868秒的可能範圍；當前警示4幀，原倒地後骨架異常備註保留。 |

## 實作與驗證

- 新 builder、HTML、bundle、來源 SHA 與四份既有 H.264 的逐位元複本另存。只讀已完成 Validation demo 及已驗證人工衍生資料，未載入模型／特徵、推論、擬合、重訓、校準或讀取／評估真實 Test。
- 既有人工檔先與原雙肩 lock 的 SHA 核對；每筆 JSON 綁定頁面 bundle、影片 SHA 及原參照內容 SHA。Python validator 拒絕少片、重複、未完成、草稿、無本人確認、錯來源、改寫原參照或冒充新增時間真值的輸入。負面／不確定判斷可正式匯出。
- **synthetic**：37 Python 風險／HTTP Range 測試通過；獨立假 DOM 測試覆蓋四筆一次匯出、草稿、未完成禁匯出、切換保留、來源不符與小數秒逐幀定位。這些使用 synthetic fixture，沒有產生真實人標。
- **真實頁面**：瀏覽器核四片切換、唯讀既有標註、四片首警示定位、半速播放／重點自停及下一格；fall-09 由sample137到138已確認。四片HTTP均206單一byte-range。交付時三選項空白、本人確認未勾、正式匯出disabled、0/4。
- 本輪 **real_feature_smoke 未執行，模型 real_end_to_end 未執行**；沿用前輪已驗收影片，不把UI測試稱為辨識改善。23/23 Validation 為前輪結果，沒有本輪新Accuracy或時序成績。
- 312份有界來源SHA核對通過；兩份更新前文件bytes另存。實際重建被拒、既有頁面／影片hash不變。這不是全工作區所有檔案的完整驗證。
- 初版`001`保留。瀏覽器發現WebKit小數秒截斷會讓fall-09「下一幀」停同格，`002`只加足夠的讀值浮點容差，未重編影片或改事件。首版來源／輸出與原設定不覆寫；新增回歸fixture曾因修改錯synthetic副本而失敗，初fixture保留、修fixture後通過。
- 本機服務首次沙箱背景程序未成功提供頁面、連線受限，日誌保留；經執行環境核准於沙箱外啟動僅綁127.0.0.1的服務後，實際UI與Range驗證通過。未關閉任何安全設定、未公開服務。
- codebase-memory Tier2：先graph／coverage再讀相關來源。相關新舊來源為no_recorded_issue／metadata_match，仍屬best effort；輸出目錄設計排除，直接核來源檔與SHA，不宣稱完整graph證明。

本頁是展示素材的本人覆核，不是教師新增的成績門檻、完整多人時間GT或模型再選擇。正式 Notebook／五分鐘影片仍暫緩，不公開上傳，舊預設／模型／品質／門檻不變。收到真人JSON後才驗證與記錄四筆結果。

## 位置及命令

- [覆核頁本機檔](../outputs/ur-shoulder-display-review-20261112-002/index.html)、[唯讀bundle](../outputs/ur-shoulder-display-review-20261112-002/review-bundle.json)、[來源SHA](../outputs/ur-shoulder-display-review-20261112-002/sources.json)。
- [實際頁面截圖](validation/ur-shoulder-display-review-20261112-002/review-page.png)、[Python測試](validation/ur-shoulder-display-review-20261112-002/risk-tests.xml)、[HTTP Range](validation/ur-shoulder-display-review-20261112-002/http-range.json)、[交付核對](validation/ur-shoulder-display-review-20261112-002/final-verification.json)。

專案根目錄 PowerShell：

```powershell
# 必要時重啟；若8851已在運作，直接開網址即可
.\.venv\Scripts\python.exe -B -X utf8 scripts/shoulder_display_review_v2.py serve --port 8851

# 只讀核對頁面、影片、來源和原參照
.\.venv\Scripts\python.exe -B -X utf8 scripts/shoulder_display_review_v2.py verify

# 收到真人匯出檔後，僅驗證JSON；不改模型或成績
.\.venv\Scripts\python.exe -B -X utf8 scripts/shoulder_display_review_v2.py validate-review --annotation 'C:/Users/Huayi/Downloads/shoulder-four-clips-human-review.json'
```
