# 去噪AE增強資料保存補充

去噪AE原訓練採Train-only即時Gaussian .1增強，原始30iso觀測及clean特徵未改。原訓練沒有逐batch擷取保存noise張量；已另外以相同固定程式、sequence-balanced sampler／DataLoader seed42、獨立CUDA noise generator seed42，在相同Torch／CUDA／GPU重播並保存40個epoch的3638筆Train來源索引及noisy標準化特徵。這是**訓練後可重現RNG重播，不是當時forward的現場擷取**，沒有模型forward、Loss、optimizer或新增訓練epochs。

clean標準化Train矩陣與逐epoch衍生NPZ分開存於`data/cache/ur-normal-methods-20261012-001/denoise-augmentation-replay`，原24影片與原cache完整保留，Val／Test不增強、不搬進Train。另一組同規格獨立generator重建每個來源索引與每筆noisy特徵並逐值驗證；全部NPZ及來源／scaler／recipe／程式hash見`provenance.json`與`docs/validation/ur-normal-methods-20261012-001/augmentation-preservation.json`。

新增保存145520個重播列，不增加原影片數或模型訓練數，不改成績。八候選、62真實AEepochs及正式CSV／門檻／權重保持相同；原final-qa不覆寫。補充檔另存，原final-qa所列388檔仍逐hash相同。重播亦有防覆寫，重跑命令會拒絕既有輸出。

```powershell
Set-Location 'C:\成大專案\fall-detection-anomaly'
Get-Content docs/validation/ur-normal-methods-20261012-001/augmentation-preservation.json -Encoding UTF8
```
