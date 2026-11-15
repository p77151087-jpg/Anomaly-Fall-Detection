# 下一輪擴充方向

2026-09-17再次查核。已完成[低FPS修正](timing-improvement.md)與[三組接續AE訓練](further-training.md)：Le2i四段正常Validation已有有效特徵，但後續三組訓練未提高整體正確判定率。下列候選新資料尚未加入本專案；上一輪577回合只使用既有UR Fall＋Le2i。

## 優先順序

1. **處理仍無法判定的八段Validation，並檢查新資料的骨架可用性。** 先前Le2i四段正常Validation缺窗已定位並修正為低FPS重採樣問題，不能繼續把它們列為零特徵。現有8段unknown均只有0～2個有效端點，無法形成K=3事件；需要在既有Train/Validation檢查追蹤、遮擋與骨架品質。保留原品質規則作基準，缺骨架仍標unknown。換模型或點位結構需要新recipe、可核對真實權重與相容性測試，不能補假點或拿分類分數代替特徵。
2. **補不同人物及場景的正常活動。** 優先坐下、起身、彎腰撿物、蹲下、跪下、正常躺下等可能觸發警示的活動。採用新資料時，先按人物與原始連續錄影分Train／Validation／新Test；同步雙鏡頭和衍生窗口不能跨組。保留正常Validation，不能把新增正常影片全放Train。
3. **做可歸因的比較實驗。** 分開比較只增加資料、只改骨架／追蹤、兩者一起改。AE訓練仍只使用正常Train，scaler不看Validation/Test，參數與門檻只用Validation。現有兩組Test已看過，後續比較必須標為探索性，並保留新資料作未查看的最終測試。
4. **補齊報告證據。** Loss之外，並列誤報／漏報、unknown、每個資料集coverage、事件時間與典型失敗案例。低Loss本身不等於可靠跌倒偵測。現已補[高解析Loss圖與圖說](loss-figures.md)。

## 可加入的資料來源

| 來源 | 核對到的內容 | 對此專案的用途與接入注意 |
| --- | --- | --- |
| [CAUCAFall v4](https://data.mendeley.com/datasets/7w7fccy7ky/4) | 10名受試者，各5種跌倒、5種日常活動；提供AVI、PNG及標籤，包含光線、遮擋與背景變化；CC BY 4.0 | 可優先小型驗證日常活動與新場景；既有影像標籤不等於OpenPose18骨架，仍須跑相同pose與時間序列流程 |
| [UP-Fall官方頁](https://sites.google.com/up.edu.mx/har-up/)／[原始論文](https://www.mdpi.com/1424-8220/19/9/1988) | 17名受試者、11種活動、每種3次；CameraX ZIP為攝影機影像序列。作者頁明列subject8/activity11的trial2、3缺失 | 適合增加人物與視角；必須取得RGB及對應時間資訊，不能把IMU或Optical Flow特徵當成本專案骨架輸入；同步兩視角歸同一來源分組 |
| [GMDCSA24作者專案](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos)／[作者v2.0公開存檔](https://zenodo.org/records/12921216) | 作者repo有Subject 1～4目錄；Zenodo公開頁列1.1GB ZIP及MD5，另提示有較新版本 | 可補獨立正常／跌倒影像來源；下載前固定實際版本，逐檔核對活動、人物、重複與完整序列，不能把版本或同源衍生片混入不同split |
| [OmniFall作者專案頁](https://dsch.ai/omnifall/) | 整合多個實拍資料集及人物／視角切分，也有另列的生成資料 | 可沿用標註及split規則，逐個核對原始來源；不代表所有模型／訓練配方可直接套用。生成資料與實拍證據仍分開 |

下一輪優先對CAUCAFall建立按人物／原始影片隔離的協定，少量Train/Validation做真實feature smoke；其10名人物各10種活動對應100段AVI，官方列23FPS及CC BY 4.0，正常活動含坐下、跪下、撿物等。這些活動適合檢查目前誤報，是本專案推論的接入優先順序，不是來源保證會提升準確率。再評估UP-Fall與GMDCSA24。以上目前僅核對官方來源、資料描述與公開下載入口，尚未完整下載或計算本地hash，也沒有加入正式訓練。

Multiple Cameras Fall的蒙特婁官方頁本次回傳Anubis Access Denied，不能列為已驗證可直接下載。UP-Fall官方頁仍描述公開CameraX下載，但本次沒有逐個驗證內嵌下載元件及所有影像ZIP。不能把有來源頁等同所有檔案已取得。

自錄資料也可先從平常的走路、坐下、起身、彎腰等活動補充；拍攝日期／人物／連續片段作分組，衍生窗口沿用原組，不把同一段影片切散到Train與Test。
