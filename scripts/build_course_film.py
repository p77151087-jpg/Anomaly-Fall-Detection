"""Local five-minute course film: frozen metrics, original-speed demos, Chinese narration."""

import argparse
import math
import subprocess
import wave
from datetime import UTC, datetime

import cv2
import imageio_ffmpeg
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fall_detect.config import project_path
from fall_detect.evaluation.course_report import load_course_results
from fall_detect.io_utils import ContractError, read_json, sha256, write_json

OUT = project_path("outputs/course-film")
FONT = "C:/Windows/Fonts/msjh.ttc"
BOLD = "C:/Windows/Fonts/msjhbd.ttc"
W, H, FPS = 1280, 720, 30
BG, FG, MUTED, CYAN, YELLOW = "#101d30", "#eef4fc", "#afc1d8", "#6bd4d0", "#ffd078"


def plan():
    result = load_course_results()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "storyboard.json"
    if path.exists():
        raise ContractError("Storyboard exists; preserve earlier film")
    m = result["test"]["metrics"]
    # Narrative is tied to these verified artifacts, not arbitrary handwritten scores.
    if (m["TP"], m["TN"], m["FP"], m["FN"], m["n_unknown"]) != (7, 4, 4, 7, 1):
        raise ContractError("Narrative belongs to the frozen 2026-09-18 result")
    scenes = [
        {
            "seconds": 18,
            "kind": "intro",
            "title": "從正常動作學習\n跌倒異常偵測",
            "lead": "UR Fall · 期末專案",
            "bullets": ["ST-GCN 特徵 ＋ 正常資料 Autoencoder", "包含成功案例、失敗案例與正式成績"],
            "narration": "這個專案用正常活動學習動作特徵，再以重建誤差偵測異常。本次以 UR Fall 作為主要成果，接下來展示方法、正常與跌倒影片，以及誤報、漏報和測試限制。現有辨識品質仍有改善空間。",
        },
        {
            "seconds": 24,
            "kind": "method",
            "title": "方法：骨架到異常分數",
            "lead": "使用真實預訓練權重，保留特徵與 AE 方法",
            "bullets": [
                "YOLO11m Pose",
                "OpenPose18 骨架",
                "ST-GCN · 256D",
                "Train ADL Scaler",
                "正常資料 AE",
                "MSE ＋ 連續三窗",
            ],
            "narration": "影像先經過 YOLO 姿態模型，映射為十八個骨架節點。凍結的 ST GCN 提取二百五十六維特徵，經正常訓練資料擬合的標準化器與自編碼器，得到重建誤差。這個流程沒有改成七類動作分類，也不把誤差稱為跌倒機率。",
        },
        {
            "seconds": 24,
            "kind": "split",
            "title": "原始序列先切分",
            "lead": "共 70 支 UR Fall；沒有重新切分",
            "bullets": [
                "Train：24 ADL／0 Fall",
                "Validation：8 ADL／15 Fall",
                "Test：8 ADL／15 Fall",
            ],
            "narration": "資料保持原來的七十支影片切分。訓練只用二十四支正常活動，共六百五十九個有效窗口。驗證集有八支正常與十五支跌倒，用來選檢查點、門檻與模型。測試集同樣有二十三支。歷史測試資料已查看，因此這是固定切分的工程對照，不能稱為全新獨立測試。",
        },
        {
            "seconds": 24,
            "kind": "rules",
            "title": "分數、連續窗口與 unknown",
            "lead": "30 fps · 30 幀窗口 · stride 5 · K = 3",
            "bullets": [
                "AE 分數：重建 MSE，不是機率",
                "三個連續有效窗口都超門檻才警示",
                "骨架不足或分數過期 → 暫時無法判定",
                "正常 / 分析中 / 無法判定是不同狀態",
            ],
            "narration": "每個窗口約一秒，每五格更新一次。三個連續有效窗口都超過門檻，才顯示疑似跌倒。骨架不足或分數過期時，畫面顯示暫時無法判定，並清除舊分數。分析中表示資料尚未收集完整，不能當成正常。右側保留分數更新時間，方便判斷資訊是否新鮮。",
        },
        {
            "seconds": 20,
            "kind": "video",
            "clip": "adl-04",
            "title": "正常活动案例",
            "lead": "Validation · adl-04 · 正常影片，沒有確認警示",
            "bullets": [
                "完整片段，原速播放",
                "綠色骨架：正常活動",
                "剛開始仍需收集窗口",
                "此片段是展示案例",
            ],
            "narration": "這段是驗證集的正常活動，完整片段維持原速播放。開始時先收集骨架窗口，取得合法分數後顯示綠色的正常活動。這支影片沒有確認警示。單一展示案例的成功不代表所有正常活動都不會誤報。",
        },
        {
            "seconds": 22,
            "kind": "video",
            "clip": "fall-02",
            "title": "跌倒案例與骨架失效",
            "lead": "Validation · fall-02 · 影片層級陽性",
            "bullets": [
                "完整片段，原速播放",
                "紅色骨架：疑似跌倒",
                "骨架失效時仍切換 unknown",
                "警示時間另需 onset 驗證",
            ],
            "narration": "這段驗證影片包含跌倒，模型曾發出疑似跌倒警示。紅色文字與骨架表示警示狀態；後續若追蹤中斷，仍會回到無法判定。整支影片判對，並不能證明警示剛好發生在起跌瞬間。這次沒有把影片層級命中，宣稱為精確事件時間辨識。",
        },
        {
            "seconds": 22,
            "kind": "video",
            "clip": "adl-20",
            "title": "誤報：正常活動也會超門檻",
            "lead": "Validation · adl-20 · 真實標籤 ADL",
            "bullets": [
                "完整片段，原速播放",
                "紅色警示在此是誤報",
                "正常／跌倒分數分布重疊",
                "保留失敗畫面，不隱藏警示",
            ],
            "narration": "這支影片的真實標籤是正常活動，卻出現了疑似跌倒，所以屬於誤報。自編碼器偵測的是偏離正常訓練分布的程度，並不直接理解跌倒語意。正常活動也可能有高誤差。我們保留完整片段，避免只展示成功案例造成誤解。",
        },
        {
            "seconds": 22,
            "kind": "video",
            "clip": "fall-19",
            "title": "漏報：可判定不等於全程可用",
            "lead": "Validation · fall-19 · 真實標籤 Fall",
            "bullets": [
                "完整片段，原速播放",
                "只有 5 個有效特徵窗口",
                "100 張中 42 張未偵測到人",
                "後半段沒有有效窗口",
            ],
            "narration": "這支跌倒影片沒有確認警示，屬於漏報。逐窗檢查顯示只有五個有效特徵窗口，一百張原始影格中有四十二張未偵測到人物，而且有效窗口只在前段。這是骨架與時間覆蓋不足的實測證據，但還不能解釋全部漏報，也沒有足夠標註量測起跌延遲。",
        },
        {
            "seconds": 20,
            "kind": "diagnosis",
            "title": "先診斷，再鎖定有限候選",
            "lead": "候選形成只使用 Train／Validation",
            "bullets": [
                "Train 每序列有效窗口：9–55",
                "嘗試序列均衡取樣",
                "比較 latent 32 與 latent 16",
                "三個候選後停止，不依 Test 加試",
            ],
            "narration": "診斷发现，訓練影片每支有效窗口數從九個到五十五個不等。因此事前鎖定三個候選，比較窗口等權、序列均衡取樣，以及較小的潛在維度。所有候選、選擇規則與停止條件都在訓練前保存，沒有看過本輪測試結果後再追加候選。",
        },
        {
            "seconds": 26,
            "kind": "loss",
            "title": "三候選訓練與 Validation 選擇",
            "lead": "合計 297 epochs；選定 UR-only 序列均衡版本",
            "bullets": [
                "Validation 正確：16 / 23",
                "Validation F1：75.9%",
                "以正常 Validation MSE 選 checkpoint",
                "門檻：0.335563；選定後凍結",
            ],
            "narration": "三個候選實際訓練共二百九十七個回合。左圖是選定模型的訓練與正常驗證損失。按照事前規則，選出潛在維度三十二的序列均衡版本，驗證集二十三支判對十六支，F 一約百分之七十五點九。模型與門檻凍結後才執行測試。損失下降本身不代表跌倒辨識已可靠。",
        },
        {
            "seconds": 28,
            "kind": "metrics",
            "title": "正式 UR Test：所有分母一起呈現",
            "lead": "固定 23 支影片；unknown 保留在全體分母",
            "bullets": [
                "可判定 Accuracy：50.0%（11/22）",
                "可判定 F1：56.0%",
                "全體 Accuracy：47.8%（11/23）",
                "全體跌倒召回：46.7%（7/15）",
            ],
            "narration": "正式測試共有二十三支，其中二十二支可判定，一支無法判定。只看可判定影片，準確率為百分之五十，F 一為百分之五十六。包含無法判定影片的全體正確率，是十一除以二十三，約百分之四十七點八。十五支跌倒中警示七支，全體跌倒召回約百分之四十六點七。表格同時列出誤報與漏報。",
        },
        {
            "seconds": 26,
            "kind": "comparison",
            "title": "與既有混合模型的取捨",
            "lead": "相同 UR Test；兩個模型分開保存",
            "bullets": [
                "跌倒判對：3 → 7",
                "正常誤報：1 → 4",
                "全體正確：10/23 → 11/23",
                "F1：33.3% → 56.0%（可判定）",
            ],
            "narration": "和既有混合資料模型比較，新版本跌倒判對從三支增加到七支，可判定 F 一也提高。然而正常誤報從一支增加到四支，全體只多判對一支。這是召回與誤報的取捨，不能包裝成大幅提升整體可靠度。原模型、設定、資料與雜湊都保留，新實驗使用新的運行目錄。",
        },
        {
            "seconds": 24,
            "kind": "limits",
            "title": "成果、限制與下一步",
            "lead": "有可重現的成果，也保留未解決的問題",
            "bullets": [
                "未達 UR 課程參考：Accuracy 90% / F1 0.87",
                "歷史 Test 已看過；未保證人物隔離",
                "尚未驗證警示 onset / 延遲",
                "官方協定待確認；本機完成，尚未公開上傳",
            ],
            "narration": "目前成果包括可執行的分析筆記本、中文展示，以及這支五分鐘影片。但成績未達課程所列的參考基準，歷史測試已查看，也未保證人物隔離。警示時機與官方評分協定仍待確認。我們不自行換算課程分數，也不把功能測試通過當成辨識品質改善。影片目前僅產出本機成品。",
        },
    ]
    assert sum(s["seconds"] for s in scenes) == 300
    elapsed = 0
    for i, scene in enumerate(scenes):
        scene.update(index=i, start=elapsed, audio=f"voice-{i:02d}.wav")
        elapsed += scene["seconds"]
    inputs = {
        str(p.relative_to(project_path("."))).replace("\\", "/"): sha256(p)
        for p in [
            project_path("docs/validation/course-report-inputs.json"),
            project_path("notebooks/course-results.ipynb"),
        ]
    }
    for sid in ("adl-04", "fall-02", "adl-20", "fall-19"):
        for suffix in (".mp4", ".csv", ".json", "-events.csv"):
            p = project_path(f"outputs/course-demo/{sid}-presentation{suffix}")
            inputs[str(p.relative_to(project_path("."))).replace("\\", "/")] = sha256(p)
    write_json(
        path,
        {
            "run": result["run"].name,
            "fps": FPS,
            "duration_seconds": 300,
            "source_kind": "real_end_to_end",
            "scenes": scenes,
            "input_hashes": inputs,
            "voice": "Microsoft Hanhan Desktop - Chinese (Taiwan)",
            "voice_type": "local synthesized narration",
            "public_upload": False,
        },
    )
    lines = [
        "# 五分鐘期末展示腳本",
        "",
        "2026-09-18；中文合成旁白；原始案例片段以1倍速度完整播放，之後明示定格分析。選片為同一模型的Validation案例，不用展示片段計算Accuracy。",
        "",
        f"Run：{result['run'].name}。歷史Test已查看；正式數字從Notebook及機讀輸出核對。",
        "",
    ]
    for scene in scenes:
        start, end = scene["start"], scene["start"] + scene["seconds"]
        lines += [
            f"## {start // 60:02d}:{start % 60:02d}–{end // 60:02d}:{end % 60:02d}　{scene['title'].replace(chr(10), ' ')}",
            "",
            scene["lead"],
            "",
            scene["narration"],
            "",
        ]
    lines += [
        "## 來源与授權",
        "",
        "UR Fall、YOLO11m、Kinetics ST-GCN 的來源／授權／實際hash見run的sources.json及資料manifest。字型使用Windows內建微軟正黑體，僅渲染文字、不散布字型檔。旁白使用本機Microsoft Hanhan合成，不模仿或冒充真人聲音。無背景音樂。",
        "",
        "本機產出不等於YouTube已上傳；公開上傳等使用者指定。",
    ]
    project_path("docs/course-demo-script.md").write_text("\n".join(lines), encoding="utf-8")
    print(path)


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def wrap(text, width, size):
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    result, line = [], ""
    for char in text:
        if char == "\n" or (line and draw.textlength(line + char, font=font(size)) > width):
            result.append(line)
            line = "" if char == "\n" else char
        else:
            line += char
    if line:
        result.append(line)
    return result


def text_block(draw, x, y, text, width, size=27, fill=FG, bold=False, gap=10):
    for line in wrap(text, width, size):
        draw.text((x, y), line, font=font(size, bold), fill=fill)
        y += size + gap
    return y


def bullet_list(draw, items, x, y, width, size=27, gap=23):
    for item in items:
        draw.ellipse((x, y + 13, x + 7, y + 20), fill=CYAN)
        y = text_block(draw, x + 22, y, item, width - 22, size) + gap


def base(scene, result, loss_image):
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    d.text((34, 15), f"UR FALL  /  {scene['index'] + 1:02d} OF 13", font=font(16, True), fill=CYAN)
    d.text((765, 16), "course-ur-balanced-001 · 中文結果說明", font=font(17), fill=MUTED)
    if scene["kind"] == "intro":
        text_block(d, 65, 127, scene["title"], 1130, 69, FG, True, 20)
        d.text((69, 330), scene["lead"], font=font(30), fill=CYAN)
        bullet_list(d, scene["bullets"], 73, 411, 1080, 30)
        return canvas
    d.text((33, 52), scene["title"], font=font(37, True), fill=FG)
    d.text((35, 103), scene["lead"], font=font(21), fill=MUTED)
    kind = scene["kind"]
    if kind == "method":
        for i, item in enumerate(scene["bullets"]):
            col, row = i % 3, i // 3
            x, y = 45 + col * 410, 180 + row * 165
            d.rounded_rectangle(
                (x, y, x + 365, y + 120), 14, fill="#1c314b", outline="#39627c", width=2
            )
            d.text((x + 18, y + 14), f"0{i + 1}", font=font(19, True), fill=CYAN)
            text_block(d, x + 18, y + 48, item, 340, 29, FG, True)
        d.text(
            (48, 527),
            "使用分類頭前特徵；Scaler / AE 僅用正常 Train 擬合",
            font=font(25),
            fill=YELLOW,
        )
    elif kind == "split":
        labels = [
            ("TRAIN", "24", "ADL only", "659 個有效窗口"),
            ("VALIDATION", "23", "8 ADL + 15 Fall", "checkpoint / 門檻 / 選模"),
            ("TEST", "23", "8 ADL + 15 Fall", "模型凍結後工程對照"),
        ]
        for i, (name, count, classes, use) in enumerate(labels):
            x = 45 + i * 410
            d.rounded_rectangle((x, 174, x + 365, 493), 14, fill="#1c314b")
            d.text((x + 25, 199), name, font=font(24, True), fill=CYAN)
            d.text((x + 25, 244), count, font=font(90, True), fill=FG)
            d.text((x + 25, 372), classes, font=font(28), fill=FG)
            text_block(d, x + 25, 433, use, 325, 22, fill=MUTED)
        d.text(
            (51, 534),
            "歷史 Test 已查看；沒有重新切分，不宣稱恢復獨立性。",
            font=font(25),
            fill=YELLOW,
        )
    elif kind == "video":
        bullet_list(d, scene["bullets"], 850, 171, 390, 24, 18)
        d.text((852, 504), "案例選自 Validation", font=font(21), fill=YELLOW)
    elif kind == "loss":
        canvas.paste(loss_image, (30, 160))
        bullet_list(d, scene["bullets"], 886, 172, 354, 24, 19)
    elif kind == "metrics":
        outcomes = result["test"]["metrics"]["outcomes_with_unknown"]
        d.text((50, 175), "真實 / 預測", font=font(23), fill=MUTED)
        for c, label in enumerate(["正常", "跌倒", "unknown"]):
            d.text((235 + c * 171, 175), label, font=font(25, True), fill=FG)
        for r, label in enumerate(["ADL", "Fall"]):
            y = 237 + r * 125
            d.text((53, y + 33), label, font=font(32, True), fill=FG)
            for c, value in enumerate(outcomes[r]):
                x = 213 + c * 171
                d.rounded_rectangle(
                    (x, y, x + 150, y + 105), 10, fill="#294660" if c < 2 else "#584e3c"
                )
                d.text((x + 52, y + 17), str(value), font=font(53, True), fill=FG)
        d.text((55, 519), "n_total 23  /  evaluable 22  /  unknown 1", font=font(24), fill=YELLOW)
        bullet_list(d, scene["bullets"], 811, 185, 430, 25, 19)
    elif kind == "comparison":
        mixed = result["mixed_ur"]
        current = result["test"]["metrics"]
        for i, (name, v) in enumerate([("混合資料模型", mixed), ("UR-only 選定模型", current)]):
            x = 45 + i * 610
            d.rounded_rectangle((x, 172, x + 580, 504), 14, fill="#1c314b")
            d.text((x + 28, 193), name, font=font(32, True), fill=CYAN)
            d.text(
                (x + 28, 261), f"{v['TP'] + v['TN']} / {v['n_total']}", font=font(71, True), fill=FG
            )
            d.text(
                (x + 28, 356), f"全體正確率 {v['operational_accuracy']:.1%}", font=font(30), fill=FG
            )
            d.text(
                (x + 28, 423),
                f"跌倒判對 {v['TP']}  ·  正常誤報 {v['FP']}",
                font=font(29),
                fill=YELLOW,
            )
        d.text((52, 536), "多抓到跌倒，也增加誤報；全體只多判對 1 支。", font=font(27), fill=YELLOW)
    else:
        bullet_list(d, scene["bullets"], 65, 182, 1140, 32, 25)
    return canvas


def captions(text):
    pieces, part = [], ""
    for char in text:
        part += char
        if len(part) >= 35 and char in "，。；、" or len(part) >= 47:
            pieces.append(part)
            part = ""
    if part:
        pieces.append(part)
    return pieces


def verify_inputs(story):
    load_course_results()
    for path, expected in story["input_hashes"].items():
        if sha256(project_path(path)) != expected:
            raise ContractError(f"Film input changed: {path}")


def soundtrack(story):
    target = OUT / "narration.wav"
    if target.exists():
        raise ContractError("Soundtrack exists")
    rate, channels, width = None, None, None
    chunks = []
    audio_meta = []
    for scene in story["scenes"]:
        original = OUT / scene["audio"]
        path = original.with_stem(original.stem + "-timed")
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-n",
                "-i",
                str(original),
                "-af",
                "atempo=1.25",
                "-c:a",
                "pcm_s16le",
                str(path),
            ],
            check=True,
        )
        with wave.open(str(path), "rb") as wav:
            params = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
            if rate is None:
                rate, channels, width = params
            if params != (rate, channels, width):
                raise ContractError("Narration format changed")
            frames = wav.getnframes()
            raw = wav.readframes(frames)
        seconds = frames / rate
        if seconds > scene["seconds"] - 1.3:
            raise ContractError(f"Narration too long, scene {scene['index']}: {seconds:.2f}s")
        start_pad = round(rate * 0.6)
        pad = round(rate * scene["seconds"]) - frames - start_pad
        chunks.append(bytes(start_pad * channels * width) + raw + bytes(pad * channels * width))
        audio_meta.append(
            {
                "scene": scene["index"],
                "duration": seconds,
                "sha256": sha256(path),
                "original_sha256": sha256(original),
                "narration_tempo": 1.25,
            }
        )
    with wave.open(str(target), "wb") as wav:
        wav.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        for chunk in chunks:
            wav.writeframes(chunk)
    return target, audio_meta


def render():
    story = read_json(OUT / "storyboard.json")
    verify_inputs(story)
    target = OUT / "UR_Fall_Final_Project_5min.mp4"
    intermediate = OUT / "UR_Fall_Final_Project_5min_silent.mp4"
    if target.exists() or intermediate.exists():
        raise ContractError("Film output exists; choose a new version")
    audio, audio_meta = soundtrack(story)
    result = load_course_results()
    history = result["history"][story["run"]]
    fig, ax = plt.subplots(figsize=(8.3, 4.25), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(
        [int(r["epoch"]) for r in history],
        [float(r["train_loss"]) for r in history],
        color=CYAN,
        label="Train ADL",
    )
    ax.plot(
        [int(r["epoch"]) for r in history],
        [float(r["val_normal_loss"]) for r in history],
        color=YELLOW,
        label="Validation ADL",
    )
    ax.set(xlabel="Epoch", ylabel="Reconstruction MSE")
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    legend = ax.legend(facecolor=BG, edgecolor=MUTED)
    for label in legend.get_texts():
        label.set_color(FG)
    fig.tight_layout()
    fig.savefig(OUT / "selected-loss.png", dpi=100)
    plt.close(fig)
    loss_image = Image.open(OUT / "selected-loss.png").convert("RGB")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-threads",
        "4",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(intermediate),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    srt = []
    subtitle_count = 0
    frame_count = 0
    try:
        for scene, voice in zip(story["scenes"], audio_meta):
            background = base(scene, result, loss_image)
            video_frames = []
            source_fps = None
            if scene["kind"] == "video":
                cap = cv2.VideoCapture(
                    str(project_path(f"outputs/course-demo/{scene['clip']}-presentation.mp4"))
                )
                source_fps = cap.get(cv2.CAP_PROP_FPS)
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    video_frames.append(
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(
                            (780, 462), Image.Resampling.LANCZOS
                        )
                    )
                cap.release()
                if not video_frames:
                    raise ContractError("No demo frames")
            pieces = captions(scene["narration"])
            total = sum(map(len, pieces))
            boundaries = []
            cursor = 0.6
            for piece in pieces:
                duration = voice["duration"] * len(piece) / total
                boundaries.append((cursor, cursor + duration, piece))
                cursor += duration
                subtitle_count += 1

                def stamp(t):
                    ms = round(t * 1000)
                    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

                srt.append(
                    f"{subtitle_count}\n{stamp(scene['start'] + boundaries[-1][0])} --> {stamp(scene['start'] + boundaries[-1][1])}\n{piece}\n"
                )
            preview_saved = False
            for local_frame in range(scene["seconds"] * FPS):
                elapsed = local_frame / FPS
                canvas = background.copy()
                d = ImageDraw.Draw(canvas)
                if video_frames:
                    source_time = max(0.0, elapsed - 0.8)
                    index = min(len(video_frames) - 1, math.floor(source_time * source_fps + 1e-7))
                    canvas.paste(video_frames[index], (28, 143))
                    # Full source plays once at original grid FPS; any tail is visibly frozen.
                    note = (
                        "原速播放 · 完整片段"
                        if source_time < len(video_frames) / source_fps
                        else "定格分析 · 原片段已完整播放"
                    )
                    text_block(d, 852, 543, note, 380, 18, YELLOW)
                d.rectangle((0, 608, W, 695), fill="#08121f")
                active = next((t for start, end, t in boundaries if start <= elapsed < end), "")
                text_block(d, 42, 618, active, 1196, 24, FG, gap=7)
                progress = (scene["start"] + elapsed) / 300
                d.rectangle((0, 704, int(W * progress), 709), fill=CYAN)
                d.text(
                    (1105, 681),
                    f"{int(scene['start'] + elapsed) // 60:02d}:{int(scene['start'] + elapsed) % 60:02d} / 05:00",
                    font=font(15),
                    fill=MUTED,
                )
                if not preview_saved and local_frame == int(scene["seconds"] * 0.5 * FPS):
                    canvas.save(OUT / f"scene-{scene['index']:02d}.png")
                    preview_saved = True
                process.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
                frame_count += 1
            print(f"Rendered scene {scene['index'] + 1}/13", flush=True)
        process.stdin.close()
        error = process.stderr.read().decode("utf8", errors="replace")
        code = process.wait()
        if code:
            raise ContractError(error)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    (OUT / "subtitles.zh-TW.srt").write_text("\n".join(srt), encoding="utf-8")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-i",
            str(intermediate),
            "-i",
            str(audio),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-t",
            "300",
            "-movflags",
            "+faststart",
            str(target),
        ],
        check=True,
    )
    # Decode all frames with ffmpeg and OpenCV to catch truncated files, not just headers.
    subprocess.run([ffmpeg, "-v", "error", "-i", str(target), "-f", "null", "-"], check=True)
    cap = cv2.VideoCapture(str(target))
    decoded = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (H, W):
            raise ContractError("Film dimensions changed")
        decoded += 1
    cap.release()
    if decoded != 9000 or frame_count != 9000 or abs(fps - FPS) > 0.01:
        raise ContractError("Film duration/frame verification failed")
    verify_inputs(story)
    write_json(
        project_path("docs/validation/course-film-verification.json"),
        {
            "created_at": datetime.now(UTC).isoformat(),
            "source_kind": "real_end_to_end",
            "run": story["run"],
            "file": str(target),
            "sha256": sha256(target),
            "frames": decoded,
            "fps": fps,
            "duration_seconds": decoded / fps,
            "size_wh": [W, H],
            "video_codec": "H264",
            "silent_file": str(intermediate),
            "silent_sha256": sha256(intermediate),
            "audio_codec": "AAC",
            "full_decode_verified": True,
            "original_speed_demos": True,
            "freeze_frames_labeled": True,
            "narration": audio_meta,
            "storyboard_sha256": sha256(OUT / "storyboard.json"),
            "public_upload": False,
        },
    )
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["plan", "render"])
    args = parser.parse_args()
    globals()[args.stage]()
