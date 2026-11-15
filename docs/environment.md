# 已測環境

2026-09-11 在 `C:\成大專案\fall-detection-anomaly` 檢查與執行。系統為 Windows 11 家用版（Windows build 26200；Python platform 顯示 Windows-10-10.0.26200）。Intel Core i5-14450HX，10 cores／16 logical processors，RAM 約 16 GB；初始 C 槽可用約 732 GB。

GPU：NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB，driver 591.44，compute capability (12,0)。nvidia-smi 顯示 driver 最高 CUDA 13.1 支援；實際安裝的 PyTorch runtime 是 CUDA 12.8，二者不混為版本。

| 元件 | 實際版本 |
| --- | --- |
| 系統 Python（未改動） | 3.13.14 |
| 專案 Python | 3.11.16，python-build-standalone，`.python/` |
| venv | `.venv/`，不使用系統 site-packages |
| 環境管理工具 | uv 0.12.13，安裝在 `.tools/` |
| torch／torchvision | 2.8.0+cu128／0.23.0+cu128 |
| NumPy／SciPy | 2.4.6／1.17.1 |
| OpenCV headless | 5.0.0.93 |
| scikit-learn／Matplotlib | 1.9.1／3.11.1 |
| PyYAML／Pillow | 6.0.3／12.3.0 |
| pytest／ruff | 9.1.1／0.16.7 |
| gdown／requests／truststore | 6.2.0／2.34.2／0.10.4 |

完整 freeze 见 `requirements/windows-cu128.freeze.txt`；含盤點時裝入但核心未使用的 pandas/h5py 等，直接需求另列 core-tested 與 pose-download-tested。

實際安裝命令已列在 README。最初 uv 使用自身 trust anchors 下載 Python 失敗，錯誤 UnknownIssuer；改用 `--system-certs` 後安裝成功。模型／資料下載用 truststore 載入系統憑證，沒有 verify=False 或跳過 TLS 的設定。PyTorch wheel 從 [官方 CUDA 12.8 index](https://download.pytorch.org/whl/cu128) 取得，版本組合依 [官方安裝表](https://pytorch.org/get-started/previous-versions/#v280)。

## 驗證範圍

- CPU 與 CUDA tensor matmul 都實際執行；CUDA 裝置辨識為上述 RTX 5060。
- CPU 完整 synthetic AE train→calibrate→evaluate 流程通過，保存／重載分數一致。
- 原上游雙流 ST-GCN 隨機初始化的結構／分類頭對照在 CPU、CUDA 通過；只作 synthetic 結構測試，未使用作正式特徵。
- FastPose ResNet50 網路結構在 CPU/CUDA 以 synthetic tensor forward 測試；crop／heatmap decode／NMS 的合成 fixture 對照已通過。
- 真實 Tiny-YOLO／AlphaPose／ST-GCN checkpoint 與真實影像 forward **blocked**，所以整條 full CPU／CUDA 均不標 supported。
- mp4v(.mp4)、MJPG(.avi) 產生六幀再完整解碼，尺寸／fps／幀數吻合；synthetic 展示 MP4 85 frames、640×480、30 fps 可解碼。輸出不依賴 GUI。
- Windows DataLoader num_workers=0；torch_threads=4。seed 同時設定 Python/NumPy/Torch；PyTorch deterministic algorithms 採 warn_only，cudnn benchmark=False。未承諾跨 GPU／wheel 的 bitwise 重現。

上游 SciPy misc 與 NumPy/Torch array protocol 有 DeprecationWarning，測試未隱藏；不是本次測試失敗。未驗證移除這些 API 的未來版本，應沿用已記錄版本或在新環境重跑驗收。

## 2026-09-17 替代方案環境

沿用原 `.venv` Python3.11.16、Torch2.8.0+cu128、RTX5060 Laptop；沒有修改系統Python，也沒有新增OpenPose/Caffe編譯依賴。新模型採Lightweight OpenPose的純PyTorch程式與原始ST-GCN，使用既有NumPy/OpenCV。

GPU real feature smoke、CPU/CUDA真實checkpoint與真實影像/骨架窗測試通過。Windows CPU oneDNN的depthwise convolution經bounded profile發現耗時很長；LightweightPoseBackend在CPU推論範圍內暫停mkldnn並使用native CPU kernels，離開後恢復，不更換裝置、參數或網路運算。原始上游stage1與介接輸出仍做數值parity。單幀CPU profile約8–12秒（當時並行工作負載下），不承諾即時；完整CPU120幀smoke中止，不列為通過。GPU用於完整正式實驗。

CPU上的torch.backends.mkldnn.flags可能發出Intel GPU TF32支援警告；CPU/GPU測試均完成，這不是自動切換到Intel GPU。
