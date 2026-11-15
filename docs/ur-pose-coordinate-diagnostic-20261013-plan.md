# fall-19 可見左手肘座標控制診斷：事前鎖定

ID：`ur-pose-coordinate-diagnostic-20261013-001`。使用者於2026-10-13授權小規模骨架錯點診斷。本文件先於任何介入分數計算保存並鎖hash。

## 假設、範圍與證據

既有人工分類標註指出第44–51幀骨架不符合人物，但未提供正確關節座標。原RGB中可見左手肘彎曲處；凍結第44–51幀、同track1、human13 joint3（人物左手肘／畫面右侧），共8點。檢驗「這一個可見關節的定位偏差是否影響原wide AE分數」。不修復其他關節，不補出框或遮擋點，不改信心值。不能由此推定整副骨架正確或唯一漏報原因。

AI一次性暫定位置依原RGB指定，未經人工確認；固定後不依分數移動／搜尋座標。僅作敏感度結果，不稱人工錯點修正。另提供8點人工覆核頁，使用者一次匯出單一JSON；未收到真實人工確認前，人工作業狀態為pending。看不清的點記為not_visible_or_uncertain、xy=null，保留原觀測而非補造。

## 固定控制

- 參照原30幀isotropic `ur-isotropic-wide-20261012-001`，512D真實嚴格ST-GCN權重、Train-only Scaler、256/64 AE、threshold=0.5341113805770874、strict >、K3、原品質及因果重採樣全凍結。
- 完整fall-19原100觀測先做sham重算，23×512特徵及71窗口須吻合原cache（特徵atol=1e-6/rtol=1e-5；分數atol=1e-6）；再只覆寫鎖定XY。原始觀測另存，AI介入和日後human介入使用不同hash及資料夾。
- 不訓練、不fit、不校準、不挑模型、不跑完整Validation比較；Train adl-05、正常Validation adl-04各前3個合法窗口為不變控制，只推論。
- Test不載入cache、不解碼影像、不評估；只可在檔案完整性盤點中hash既有檔案，hash不構成Test模型評估。demo及舊正式Validation22/23不改。
- 衍生recipe為frozen-model coordinate sensitivity，不聲稱與舊model部署recipe相容；不能用於正式訓練／候選評分或部署。

## 驗收與停止

sham吻合、有限512D、信心／bbox／track／timestamp不變、71窗口品質掩碼一致、只有包含已改source同track的窗口允許改變、快取隔離與舊hash一致，才允許解讀。任何一项不成立即停，保存失敗，不追加其他joint／frames／模型／門檻。

比較每窗raw MSE及delta、max raw、連續K3 max、threshold餘量、合法覆蓋23/71、事件及人標姿態阶段。分數增加但仍無事件不能稱漏報改善；影片判對與警示時段分開。人工阶段33–61／62–100是整段姿態分類，非精確開始或落地時間真值。單一手動影片介入不改正式Accuracy/F1，不採用為模型。

synthetic＝輸入邊界／因果／凍結測試；real_feature_smoke＝真實權重及控制窗口；real_end_to_end＝fall-19 frozen inference敏感度。人工座標真值尚待覆核，這一層不得以AI實验取代。
