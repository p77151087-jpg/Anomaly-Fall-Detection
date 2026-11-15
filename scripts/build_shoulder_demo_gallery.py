"""Create new browser-compatible copies of accepted fresh RGB demos, without inference."""

import json
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from validate_shoulder_demo import OUT, base, current_text_templates, require

from fall_detect.io_utils import read_json, sha256, write_json


def red_history(image):
    x = image[30:56, 334:632].astype(int)
    return int(((x[:, :, 2] > 150) & (x[:, :, 2] > x[:, :, 1] * 1.5)).sum()) > 30


def main():
    report = read_json(OUT / "validation-acceptance.json")
    require(report["passed"] and len(report["sequences"]) == 23, "Full RGB acceptance required")
    destination = OUT / "gallery"
    require(not destination.exists(), "Gallery exists; choose a new version")
    destination.mkdir()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    manifest, cards = {}, []
    templates = current_text_templates()
    for row in sorted(
        report["sequences"], key=lambda r: (r["sequence_id"] != "fall-19", r["sequence_id"])
    ):
        sid = row["sequence_id"]
        source, target = base(sid).with_suffix(".mp4"), destination / f"{sid}.mp4"
        relative = str(source.relative_to(OUT))
        require(sha256(source) == row["output_sha256"][relative], "Accepted original video changed")
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ]
        subprocess.run(command, check=True)
        a, b = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(target))
        require(abs(b.get(cv2.CAP_PROP_FPS) - 30) < 1e-6, "H264 FPS differs")
        count = 0
        while True:
            ok, original = a.read()
            valid, encoded = b.read()
            require(ok == valid, "H264 length differs")
            if not ok:
                break
            require(original.shape == encoded.shape, "H264 size differs")
            roi_a, roi_b = (
                original[84:103, 334:632].astype(float),
                encoded[84:103, 334:632].astype(float),
            )
            require(
                int(np.argmin([np.abs(roi_a - t).mean() for t in templates]))
                == int(np.argmin([np.abs(roi_b - t).mean() for t in templates])),
                "H264 current text differs",
            )
            require(red_history(original) == red_history(encoded), "H264 history color differs")
            if sid == "fall-19" and count == 45:
                ok_png, png = cv2.imencode(".png", encoded)
                require(ok_png, "Screenshot encoding failed")
                png.tofile(destination / "fall-19-first-alert.png")
            count += 1
        a.release()
        b.release()
        require(count == row["frames"], "H264 frame count differs")
        manifest[sid] = {
            "source_sha256": sha256(source),
            "h264_sha256": sha256(target),
            "frames_verified": count,
            "current_glyph_and_history_color": "matched",
        }
        first = row["first_alarm_ms"]
        timing = f"{first / 1000:.3f} 秒" if first is not None else "無事件"
        cards.append(
            {
                "id": sid,
                "label": "跌倒" if row["label"] else "正常",
                "prediction": row["prediction"],
                "first": first,
                "summary": f"{timing}；當前警示 {row['current_alert_frames']} 幀；事件 {row['event_count']}",
            }
        )
        print(f"H264 verified {sid}: {count} frames", flush=True)
    data = json.dumps(cards, ensure_ascii=False).replace("<", "\\u003c")
    page = """<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>雙肩可選 demo：23 支 Validation</title><style>body{font:17px system-ui;max-width:1050px;margin:24px auto;padding:16px;background:#101923;color:#e8edf1}video{width:100%;max-height:73vh;background:#000}button,select{font:inherit;padding:8px;margin:5px}a{color:#8bd5ff}.note{color:#bbcad6;line-height:1.6}</style>
<h1>雙肩可選 demo｜原影片實跑驗收</h1><p class="note">原 wide AE 與雙肩分支各自 K3 後合併。右側區分事件歷史與當前警示；左側七類 ACTION 是獨立對照。分數不是跌倒機率。本批 Validation 已反覆使用，Test 未評估；人工精確時間真值未涵蓋全部影片。</p>
<select id="case"></select><button id="event">跳到首次事件</button><button id="back">前一格</button><button id="next">下一格</button><select id="speed"><option value="1">1 倍</option><option value="0.5">0.5 倍</option></select><p id="info"></p>
<video id="video" controls preload="metadata"></video><p><a id="download" download>下載 H.264 片段</a></p>
<script>const cases=DATA;const sel=document.getElementById('case'),v=document.getElementById('video');for(const c of cases){const o=document.createElement('option');o.value=c.id;o.textContent=c.id+' · '+c.label;sel.append(o)}function load(){const c=cases.find(x=>x.id===sel.value);v.src=c.id+'.mp4';v.playbackRate=Number(document.getElementById('speed').value);document.getElementById('info').textContent=c.summary;document.getElementById('download').href=v.src;document.getElementById('event').disabled=c.first===null}sel.onchange=load;document.getElementById('event').onclick=()=>{const c=cases.find(x=>x.id===sel.value);v.pause();v.currentTime=c.first/1000};document.getElementById('back').onclick=()=>{v.pause();v.currentTime=Math.max(0,v.currentTime-1/30)};document.getElementById('next').onclick=()=>{v.pause();v.currentTime=Math.min(v.duration,v.currentTime+1/30)};document.getElementById('speed').onchange=e=>v.playbackRate=Number(e.target.value);load();</script></html>""".replace(
        "DATA", data
    )
    (destination / "index.html").write_text(page, encoding="utf-8")
    write_json(
        destination / "verification.json",
        {
            "codec": "H.264/yuv420p",
            "clips": manifest,
            "frames": sum(r["frames_verified"] for r in manifest.values()),
            "index_sha256": sha256(destination / "index.html"),
            "screenshot_sha256": sha256(destination / "fall-19-first-alert.png"),
        },
    )


if __name__ == "__main__":
    main()
