# 模型與第三方來源

**2026-09-18新增sample同源Human實驗：** 原Drive連結重試仍失敗，已由[上游Issue117](https://github.com/GajuuzZ/Human-Falling-Detect-Tracks/issues/117)所列社群鏡像取得完整四資源；三個網路嚴格載入及真實影像前測通過。另存`weights/sample-human-manifest.json`及`weights/pretrained/sample-human/`，原manifest保持原狀。鏡像沒有作者公布checksum，不能宣稱與同學checkpoint逐位元一致；也不能由取得權重推論Accuracy改善。[完整實作與結果](sample-human-reproduction.md)。

2026-09-17目前使用第二條路：Lightweight OpenPose18＋原始ST-GCN Kinetics。兩個權重已取得並完成真實端到端實驗，詳細來源在本文件後段「2026-09-17」紀錄及 `weights/openpose18-manifest.json`。以下Human缺權重描述保留為原方案歷史，並非新方案仍受阻。

固定 repository：[GajuuzZ/Human-Falling-Detect-Tracks](https://github.com/GajuuzZ/Human-Falling-Detect-Tracks)，實際查得與 checkout HEAD：`7ed2faa4d6147dfd576f58869b6c25545208af35`。上游 checkout 保留在 `third_party/human/`，不加入本專案 Git。

已逐段查閱 ActionsEstLoader.py、pose_utils.py、Actionsrecognition/Models.py、Utils.py、PoseEstimateLoader.py、DetectorLoader.py、SPPE 模型／crop／decode、Tracker 與 main.py。雙流各 pooled 256D，合併 512D，原分類頭 Linear(512,7)。測試對照 `sigmoid(fcn(features))` 與原 forward，並驗證凍結參數與 BatchNorm；目前是 synthetic 結構證據，不是預訓練權重證據。

## 下載與驗證狀態

四個資源的來源頁、Drive ID、角色、路徑、狀態在 `weights/manifest.json`。2026-09-11 透過 gdown 6.2.0 使用上游 README 指向的 URL 均回報 `Cannot retrieve the public link of the file`；網頁工具也未能開啟 Drive 權重頁。未繞過存取限制，未下載其他不明鏡像，未停用 TLS 驗證。所有 SHA256／檔案大小為 null、status=missing。原因不能確定是檔案分享設定、流量或環境存取，報告保留原始錯誤，不推測模型可用。

| 角色 | Drive ID | 本地路徑 |
| --- | --- | --- |
| detector | 1obEbWBSm9bXeg10FriJ7R2cGLRsg-AfP | weights/pretrained/detector.pth |
| detector_config | 19sPzBZjAjuJQ3emRteHybm2SG25w9Wn5 | weights/pretrained/yolov3-tiny-onecls.cfg |
| pose (ResNet50) | 1IPfCDRwCmQDnQy94nT1V-_NVtTEi4VmU | weights/pretrained/fast_res50_256x192.pth |
| stgcn | 1mQQ4JHe58ylKbBqTjuKzpwN2nwKOWJ9u | weights/pretrained/tsstg-model.pth |

固定來源：[上游 README](https://github.com/GajuuzZ/Human-Falling-Detect-Tracks/blob/7ed2faa4d6147dfd576f58869b6c25545208af35/README.md)。待取得後以 `doctor` 註冊 hash，再 `smoke-test --mode real` 驗證。strict_load 先查所有鍵與 shape；只允許全部 key 都帶 `module.` 時統一剝除並記錄。僅受限 tensor state dict 載入，沒有自動關閉 pickle 保護的選项。

## 本地相容性處理

- 直接建構上游 Darknet 與 FastPose('resnet50',17)，由 adapter 傳入 device／map_location／strict checkpoint。避開舊 InferenNet 建構子無條件 `.cuda()`；不修改網路運算與 state_dict 鍵名。
- 使用上游 RGB resize、crop_dets、getPrediction、13 點 heatmap 選擇與肩中點順序；固定 pose 輸入 256×192、detector 384×384，全部包含 recipe fingerprint。
- `pose_nms_compatible` 沿用上游 parametric distance、PCK matching、p_merge_fast 與門檻，將 tensor 刪除改成 boolean tensor selection，避免混合 NumPy/Torch 型別；有單人對照與多人物保留測試。
- 人物框從 detector resize/padding 座標還原至原圖，再 crop。每序列重設上游 Tracker(max_age=30,n_init=3)；只輸出當幀成功更新的 confirmed track，不把預測／缺失位置當作觀測。
- 骨架始終原圖 width/height；不沿用上游 main.py 把 frame.shape[:2] 傳入 width/height 的混淆。非正方形測試覆蓋此契約。
- 未變更上游 checkout；本地 adapter／前處理／型別與實驗流程由此專案新增。

上游原 Tiny-YOLO 用 COCO person，動作權重用 Le2i (Coffee room/Home) 人工七類標註，包含 Fall Down。AE 僅正常訓練不能把整套方法稱為「從未接觸跌倒」。AlphaPose 精確 checkpoint 訓練來源／權重內容仍需取得後核對，不能以一般 AlphaPose 名稱當驗證。

## 替代路徑（取得權重前的歷史紀錄）

已查閱 [原始 ST-GCN OLD_README](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md)：其 demo 需 OpenPose Python API，模型指向 Google Drive／BaiduYun。当前未取得完整、相容的 OpenPose18＋Kinetics checkpoint 組合，因此未啟用替代 recipe，也未宣稱已實作第二套推論。主路徑仍可補回四個檔案後接續；不能將 Human 13 點補成缺少眼耳的 OpenPose18、或把 256D 補零冒充 512D。

## 授權與識別

Human repository 根目錄未附整體 license，不能假定可任意再散布程式／權重。SPPE 子目錄另有 MIT License，Copyright (c) 2018 Jeff-sjtu，原文保存在 `docs/third-party/SPPE-LICENSE.txt`。本地 NMS 相容程式參考上游，未把上游網路冒作自行設計。資料授權另見 data-protocol。未取得上游權重預期 checksum；將來下載後自行計算的 hash 用於實驗重現，不是發布方驗證。

## 2026-09-17 已啟用的第二條路

新 recipe：`lightweight_openpose18_stgcn_kinetics_v1`；設定 `configs/openpose18.yaml`。Human 仍保留，原四個檔案不再是這個 recipe 的依賴。

- 姿態：Daniil-Osokin/lightweight-human-pose-estimation.pytorch，commit `d23c284b09acf27a163e1febd511e7482cac25ed`。作者 README 直接連到 Intel download.01.org 的 `checkpoint_iter_370000.pth`；已取得 87,959,810 bytes，SHA256 `517c86f769c6636583083f1467e3d212a0006c27109edb3aeffc19a79622d411`。原生完整 OpenPose18，不使用 Human13 補點。完整三個 refinement stage 的 426 個 state keys 嚴格載入，推論使用第一個 refinement，與作者的一階段 demo 語義對照；不使用作者會忽略缺失參數的 load_state。
- 動作：yysijie/st-gcn，commit `221c0e152054b8da593774c0d483e59befdb9061`。官方 OLD_README 的 Google Drive folder `1IYKoSrjeI3yYJ9bO0_z_eDo92i7ob_aF` 目前可下載；其中 `st_gcn.kinetics.pt` ID `1MDIUlJ_X-IpNgLZYXgdMuclCsnYTepU1`，12,763,702 bytes，SHA256 `12c56ed5535dd28a075e7d04dec96f572f8b1dc87548004761157e15b7d9e12e`。172 個 state keys 完全一致，包含400類分類頭。官方 shell script 指向的 S3 URL 回傳403，使用的是作者另列出的公開 Drive 來源，未繞過限制。
- 特徵：固定模型 eval/無梯度，輸入 `[1,3,30,18,1]`；取 extract_feature 第二項 `[1,256,T',18,1]` 並在時間／關節／唯一真實人物維度平均。重接原400類 head 與 model.forward 數值對照。400類分數不輸入 AE。
- 前處理：BGR，作者 mean128/scale1/256、height256、stride8 padding、上採樣4；原尺寸還原保留float座標與heatmap confidence。缺失點 confidence=0。Kinetics 座標為 x/width−0.5、y/height−0.5；confidence=0 的 x/y 清零，不使用 Human min/max。
- 關節：nose, neck, right shoulder/elbow/wrist, left shoulder/elbow/wrist, right hip/knee/ankle, left hip/knee/ankle, right eye, left eye, right ear, left ear。
- 追蹤：作者的相鄰幀關節相似度 greedy matching，至少3個相似關節，停用 smoothing。每序列 reset ID；來源時間缺口或整幀沒人不連接前後軌跡。遮擋、躺倒、多人交錯可能導致 ID 中斷，保留 unknown。
- 品質：18點中至少12點 confidence>=0.2，30幀窗至少80%影格合格；缺整幀／track gap 的整窗不可判定。30fps/30幀/stride5 為此 recipe 明訂短窗，與原始 Kinetics feeder 的300位置不同，並非原論文重現。
- 分布差異：Lightweight OpenPose 是另一個 COCO 姿態估計器；即使點位相同，也不能保證與原 ST-GCN 預訓練使用的 CMU OpenPose 分布相同。以新的256D快取、正常train-only scaler/AE、validation門檻重新建立基準，不沿用Human的任何特徵或門檻。

來源URL、實際hash、strict/forward狀態另存 `weights/openpose18-manifest.json`。本地hash是重現識別，不是假稱作者提供的checksum。原始ST-GCN及Lightweight OpenPose授權原文保存在 `docs/third-party/`。下載資料夾中其他NTU/Caffe權重未用於此recipe。


## Le2i擴充的模型設定

擴充重用上述兩個相同、嚴格載入的checkpoint及固定upstream，不新增未知權重。仍採height256／第一refinement／256D，但在原向未達固定骨架品質時額外以±90度影像推論，再還原原圖座標並抑制重複候選；不補點、不混合不同姿態的關節。`rotation_angles`與`rotation_policy`納入recipe hash，與歷史不旋轉run隔離。

384／第三refinement及無條件旋轉僅為Train/Validation候選試驗；最終選擇及六張真實骨架抽幀的限制見[Le2i紀錄](le2i-extension.md)。既有greedy追蹤未被宣稱已解決遮擋與ID分裂。新增資料只重新擬合Scaler／AE及Validation門檻，Pose與ST-GCN都保持凍結。
