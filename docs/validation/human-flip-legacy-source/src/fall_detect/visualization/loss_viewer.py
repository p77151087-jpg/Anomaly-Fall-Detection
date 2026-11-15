"""Local, read-only training history viewer. No inference or model loading."""

import csv
import io
import json
import math
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

from ..io_utils import ContractError, read_json


def validate_history(rows):
    result = []
    try:
        for expected, row in enumerate(rows, 1):
            point = {"epoch": int(row["epoch"])}
            if point["epoch"] != expected:
                raise ValueError("Epochs must start at 1 and be consecutive")
            for key in ("train_loss", "val_normal_loss"):
                point[key] = float(row[key])
                if not math.isfinite(point[key]) or point[key] < 0:
                    raise ValueError("Loss must be finite and non-negative")
            result.append(point)
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"無法讀取 Loss 紀錄：{exc}") from exc
    return result


def normal_counts(run):
    path = run / "dataset-index.json"
    if not path.exists():
        return {}, []
    index = read_json(path)
    counts, groups = {}, {}
    for source in index["split_manifest"]["sequences"]:
        split = source["split"]
        if split not in ("train", "validation") or source["label"] != 0:
            continue
        metadata = index["sequences"][source["sequence_id"]]
        if "valid_features" not in metadata:
            # Old/synthetic indexes can omit counts; missing metadata is not zero coverage.
            return {}, []
        windows = metadata["valid_features"]
        count = counts.setdefault(split, {"planned": 0, "usable": 0, "windows": 0})
        count["planned"] += 1
        count["usable"] += int(windows > 0)
        count["windows"] += windows
        if split == "validation":
            dataset = source.get("dataset", "urfall")
            group = groups.setdefault(dataset, {"planned": 0, "windows": 0})
            group["planned"] += 1
            group["windows"] += windows
    warnings = [
        f"{dataset} 的 {group['planned']} 段正常驗證影片沒有有效特徵，未納入驗證 Loss。"
        for dataset, group in groups.items()
        if group["windows"] == 0
    ]
    return counts, warnings


def read_loss_run(run):
    run = Path(run)
    status = read_json(run / "status.json")
    source_kind = status.get("source_kind", "unknown")
    training = status.get("training", "unknown")
    history_path = run / "training-history.csv"
    progress_path = run / "training-progress.json"
    history, updated_at, verified = [], None, False
    if training == "complete":
        # Verify the plotted file itself; do not load weights or test features.
        manifest = run / "frozen-artifacts.json"
        if not manifest.exists():
            manifest = run / "training-artifacts.json"
        expected = read_json(manifest).get("training-history.csv") if manifest.exists() else None
        content = history_path.read_bytes()
        if expected:
            import hashlib

            if hashlib.sha256(content).hexdigest() != expected:
                raise ContractError("訓練紀錄 hash 不符，停止顯示曲線。")
            verified = True
        history = validate_history(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
        if not history or len(history) != status.get("epochs"):
            raise ContractError("訓練回合數與已保存狀態不符。")
        if min(row["val_normal_loss"] for row in history) != status.get("best_val_normal_loss"):
            raise ContractError("最佳驗證 Loss 與已保存狀態不符。")
        updated_at = history_path.stat().st_mtime
    elif progress_path.exists():
        progress = read_json(progress_path)
        if progress.get("source_kind") != source_kind:
            raise ContractError("即時紀錄與 run 的資料來源不同。")
        history = validate_history(progress["history"])
        if training == "running":
            training = progress.get("training", training)
        updated_at = progress_path.stat().st_mtime
    counts, warnings = normal_counts(run)
    if training in {"failed", "interrupted"}:
        warnings.append("此訓練已失敗或中斷，曲線僅保留已完成的回合，不能當作可用模型。")
    if status.get("training_reused_from"):
        change = (
            "另以 Train 正常資料估計重建誤差權重，再校準門檻"
            if status.get("residual_fit") == "complete"
            else "只另行校準門檻"
        )
        warnings.append(
            f"此版本沿用 {status['training_reused_from']} 的模型與 Loss 紀錄，"
            f"{change}，沒有重複訓練。"
        )
    sources_path = run / "sources.json"
    if sources_path.is_file():
        sampling = read_json(sources_path).get("training_sampling", {})
        if sampling.get("strategy") == "sequences":
            warnings.append(
                "此訓練按正常影片均衡抽樣，可重複抽到同一窗口。Train Loss 是抽樣窗口的平均；"
                "Validation Loss 仍是全部有效正常驗證窗口的平均。"
            )
    return {
        "run": run.name,
        "source_kind": source_kind,
        "training": training,
        "history": history,
        "best": min(history, key=lambda row: row["val_normal_loss"]) if history else None,
        "last": history[-1] if history else None,
        "counts": counts,
        "warnings": warnings,
        "updated_at": updated_at,
        "history_verified": verified,
    }


def available_runs(root):
    root = Path(root).resolve()
    if not root.exists():
        return []
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.resolve().parent == root and (path / "status.json").is_file()
    ]
    return [p.name for p in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)]


def resolve_run(root, name):
    if name not in available_runs(root):
        raise ContractError("找不到指定的訓練紀錄。")
    return Path(root).resolve() / name


def make_server(root, port=8765, selected=None):
    root = Path(root).resolve()
    if not 0 <= port <= 65535:
        raise ContractError("Port 必須介於 0 和 65535。")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, content, content_type, code=200, filename=None):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            port = self.server.server_port
            if self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                self.respond(b"Local access only", "text/plain", 403)
                return
            request = urlsplit(self.path)
            try:
                if request.path == "/":
                    self.respond(
                        Path(__file__).with_name("loss_page.html").read_bytes(),
                        "text/html; charset=utf-8",
                    )
                    return
                if request.path == "/api/runs":
                    payload = {"runs": available_runs(root), "selected": selected}
                elif request.path in ("/api/loss", "/download/history.csv"):
                    name = parse_qs(request.query).get("run", [""])[0]
                    payload = read_loss_run(resolve_run(root, name))
                    if request.path == "/download/history.csv":
                        stream = io.StringIO(newline="")
                        writer = csv.DictWriter(
                            stream, fieldnames=["epoch", "train_loss", "val_normal_loss"]
                        )
                        writer.writeheader()
                        writer.writerows(payload["history"])
                        self.respond(
                            stream.getvalue().encode("utf-8-sig"),
                            "text/csv; charset=utf-8",
                            filename="training-loss.csv",
                        )
                        return
                else:
                    self.respond(b"Not found", "text/plain", 404)
                    return
                self.respond(
                    json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
            except (ContractError, OSError, ValueError, KeyError, TypeError) as exc:
                self.respond(
                    json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                    400,
                )

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


class LossViewer:
    def __init__(self, root, port=8765, selected=None):
        self.server = make_server(root, port, selected)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        query = "?" + urlencode({"run": selected}) if selected else ""
        self.url = f"http://127.0.0.1:{self.server.server_port}/{query}"

    def start(self, open_browser=True):
        self.thread.start()
        print(f"Loss 檢視畫面：{self.url}\n按 Ctrl+C 關閉檢視服務。", flush=True)
        if open_browser:
            try:
                webbrowser.open(self.url)
            except webbrowser.Error:
                print("請在瀏覽器開啟上方網址。", flush=True)
        return self

    def wait(self):
        try:
            while self.thread.is_alive():
                self.thread.join(timeout=0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        if self.thread.is_alive():
            self.server.shutdown()
            self.thread.join(timeout=5)
        self.server.server_close()
