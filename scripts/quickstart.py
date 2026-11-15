"""Acquire pinned demo resources and run the existing CLI without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import ssl
import stat
import subprocess
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "configs/quickstart-runtime.json"
DEMO_CONFIG = "configs/ur-sample-human-s1-001.yaml"
RUN = "runs/ur-sample-human-s1-001"
UPSTREAM = "third_party/human"
UPSTREAM_REPOSITORY = "https://github.com/GajuuzZ/Human-Falling-Detect-Tracks.git"
UPSTREAM_COMMIT = "7ed2faa4d6147dfd576f58869b6c25545208af35"
RUNTIME_PREFIXES = (
    "runs/ur-sample-human-s1-001/",
    "runs/ur-isotropic-wide-20261012-001/",
    "outputs/ur-fall19-shoulder-descent-20261107-001/",
)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
WEIGHT_PATHS = {
    "weights/pretrained/sample-human/detector.pth",
    "weights/pretrained/sample-human/yolov3-tiny-onecls.cfg",
    "weights/pretrained/sample-human/fast_res50_256x192.pth",
    "weights/pretrained/sample-human/tsstg-model.pth",
}


class SetupError(RuntimeError):
    """An actionable setup failure; existing experiment files stay untouched."""


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise SetupError("Empty or invalid resource path")
    parts = relative.split("/")
    if any(p in ("", ".", "..") or ":" in p or "\\" in p for p in parts):
        raise SetupError(f"Unsafe resource path: {relative}")
    if PurePosixPath(relative).is_absolute():
        raise SetupError(f"Absolute resource path: {relative}")
    path = root.joinpath(*parts)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise SetupError(f"Resource path escapes the project: {relative}")
    return path


def read_manifest(root: Path) -> dict:
    try:
        manifest = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SetupError(f"Cannot read {CONFIG_PATH}: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise SetupError("Unsupported quickstart runtime manifest")
    if not SHA256.fullmatch(str(manifest.get("sha256", ""))):
        raise SetupError("Runtime release checksum is not configured")
    if urlparse(manifest.get("url", "")).scheme != "https":
        raise SetupError("Runtime release must use HTTPS")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or len(files) > 128:
        raise SetupError("Runtime manifest has no valid bounded file list")
    seen = set()
    for relative, metadata in files.items():
        safe_path(root, relative)
        if not relative.startswith(RUNTIME_PREFIXES):
            raise SetupError(f"Unexpected runtime scope: {relative}")
        if relative.casefold() in seen:
            raise SetupError(f"Duplicate runtime path: {relative}")
        seen.add(relative.casefold())
        if not isinstance(metadata, dict) or not SHA256.fullmatch(str(metadata.get("sha256", ""))):
            raise SetupError(f"Invalid resource checksum: {relative}")
        if type(metadata.get("size_bytes")) is not int or metadata["size_bytes"] < 0:
            raise SetupError(f"Invalid resource size: {relative}")
    limit = manifest.get("max_extract_bytes", 64 * 1024 * 1024)
    if type(limit) is not int or not 0 < limit <= 128 * 1024 * 1024:
        raise SetupError("Invalid runtime extraction limit")
    if sum(item["size_bytes"] for item in files.values()) > limit:
        raise SetupError("Runtime exceeds extraction limit")
    download_limit = manifest.get("max_download_bytes", 16 * 1024 * 1024)
    if type(download_limit) is not int or not 0 < download_limit <= 128 * 1024 * 1024:
        raise SetupError("Invalid runtime download limit")
    weights = manifest.get("pretrained_weights")
    if not isinstance(weights, dict) or set(weights) != WEIGHT_PATHS:
        raise SetupError("Expected the four pinned Human pretrained resources")
    for relative, metadata in weights.items():
        safe_path(root, relative)
        validate_file_metadata(relative, metadata)
    samples = manifest.get("sample_sequences")
    if not isinstance(samples, dict) or set(samples) != {"adl-05", "fall-02"}:
        raise SetupError("Quickstart may only prepare Train adl-05 and Validation fall-02")
    for name, sample in samples.items():
        if not isinstance(sample, dict) or not all(
            SHA256.fullmatch(str(sample.get(key, ""))) for key in ("archive_sha256", "sync_sha256")
        ):
            raise SetupError(f"Invalid sample provenance: {name}")
        frames = sample.get("frames")
        if (
            not isinstance(frames, dict)
            or not frames
            or len(frames) > 2000
            or sample.get("png_count") != len(frames)
        ):
            raise SetupError(f"Invalid bounded frame inventory: {name}")
        for relative, metadata in frames.items():
            safe_path(root, relative)
            if not relative.startswith(f"{name}-cam0-rgb/") or not relative.endswith(".png"):
                raise SetupError(f"Unexpected sample frame path: {relative}")
            validate_file_metadata(relative, metadata)
            if metadata["size_bytes"] > 8 * 1024 * 1024:
                raise SetupError(f"Sample frame exceeds its size limit: {relative}")
    return manifest


def validate_file_metadata(relative: str, metadata: dict) -> None:
    if not isinstance(metadata, dict) or not SHA256.fullmatch(str(metadata.get("sha256", ""))):
        raise SetupError(f"Invalid resource checksum: {relative}")
    if type(metadata.get("size_bytes")) is not int or metadata["size_bytes"] < 0:
        raise SetupError(f"Invalid resource size: {relative}")


def file_state(root: Path, relative: str, metadata: dict) -> str:
    path = safe_path(root, relative)
    if not path.exists():
        return "missing"
    if not path.is_file():
        return "mismatch"
    if path.stat().st_size != metadata["size_bytes"]:
        return "mismatch"
    return "ready" if digest(path) == metadata["sha256"] else "mismatch"


def runtime_state(root: Path, manifest: dict) -> dict[str, str]:
    return {
        relative: file_state(root, relative, meta) for relative, meta in manifest["files"].items()
    }


def refuse_mismatches(states: dict[str, str]) -> None:
    changed = [name for name, state in states.items() if state == "mismatch"]
    if changed:
        raise SetupError(
            "Existing resources differ from the pinned release; they were preserved. "
            "Use a separate fresh clone or inspect these files:\n  " + "\n  ".join(changed)
        )


def download_runtime(root: Path, manifest: dict) -> Path:
    directory = root / ".artifacts/quickstart-downloads"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / f"{manifest['sha256']}.zip"
    if archive.is_file() and digest(archive) == manifest["sha256"]:
        return archive
    temporary = archive.with_suffix(".part")
    limit = manifest.get("max_download_bytes", 16 * 1024 * 1024)
    request = Request(manifest["url"], headers={"User-Agent": "fall-detection-quickstart/1"})
    print("Downloading the pinned trained-model bundle...", flush=True)
    try:
        import truststore

        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        # Normal TLS verification stays enabled. Incomplete bytes are never installed.
        with (
            urlopen(request, timeout=60, context=context) as response,
            temporary.open("wb") as output,
        ):
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > limit:
                raise SetupError("Runtime download exceeds its size limit")
            count = 0
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > limit:
                    raise SetupError("Runtime download exceeds its size limit")
                output.write(chunk)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, ImportError) as exc:
        raise SetupError(
            "Could not download the trained-model release. Check the network and repository "
            "access; a private or unpublished GitHub release cannot be downloaded anonymously. "
            f"Retry start.cmd after the release is accessible. Detail: {exc}"
        ) from exc
    if digest(temporary) != manifest["sha256"]:
        raise SetupError("Runtime download SHA256 mismatch; no experiment files were changed")
    temporary.replace(archive)
    return archive


def install_runtime(root: Path, manifest: dict, archive: Path) -> None:
    refuse_mismatches(runtime_state(root, manifest))
    if digest(archive) != manifest["sha256"]:
        raise SetupError("Runtime archive SHA256 mismatch")
    staging_parent = root / ".artifacts"
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="quickstart-stage-", dir=staging_parent) as temporary:
        stage = Path(temporary)
        try:
            with zipfile.ZipFile(archive) as bundle:
                names = set()
                extracted = set()
                for member in bundle.infolist():
                    name = member.filename.rstrip("/") if member.is_dir() else member.filename
                    safe_path(root, name)
                    if name.casefold() in names:
                        raise SetupError(f"Duplicate ZIP member: {name}")
                    names.add(name.casefold())
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode):
                        raise SetupError(f"ZIP links are not allowed: {name}")
                    if member.is_dir():
                        if not any(p.startswith(name + "/") for p in manifest["files"]):
                            raise SetupError(f"Unexpected ZIP directory: {name}")
                        continue
                    if name not in manifest["files"]:
                        raise SetupError(f"Unexpected ZIP member: {name}")
                    metadata = manifest["files"][name]
                    if member.file_size != metadata["size_bytes"]:
                        raise SetupError(f"ZIP member size mismatch: {name}")
                    target = safe_path(stage, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                    if digest(target) != metadata["sha256"]:
                        raise SetupError(f"ZIP member SHA256 mismatch: {name}")
                    extracted.add(name)
                if extracted != set(manifest["files"]):
                    raise SetupError("Runtime ZIP is incomplete")
        except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
            if isinstance(exc, SetupError):
                raise
            raise SetupError(f"Could not validate the runtime ZIP: {exc}") from exc
        # Validate the entire archive and all old files before publishing any new file.
        states = runtime_state(root, manifest)
        refuse_mismatches(states)
        for name, state in states.items():
            if state == "ready":
                continue
            target = safe_path(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Atomic create-only publication. A race never overwrites an existing run.
                if os.name == "nt":
                    # Windows rename fails if target exists; no hard-link privileges needed.
                    os.rename(stage / name, target)
                else:
                    os.link(stage / name, target)
            except FileExistsError:
                if file_state(root, name, manifest["files"][name]) != "ready":
                    raise SetupError(f"Resource appeared during setup; preserved: {name}") from None
            except OSError as exc:
                raise SetupError(
                    f"Cannot publish {name} atomically on this filesystem: {exc}"
                ) from exc


def ensure_runtime(root: Path, manifest: dict) -> None:
    states = runtime_state(root, manifest)
    refuse_mismatches(states)
    if all(value == "ready" for value in states.values()):
        print("Trained-model bundle: ready", flush=True)
        return
    install_runtime(root, manifest, download_runtime(root, manifest))
    if any(state != "ready" for state in runtime_state(root, manifest).values()):
        raise SetupError("Runtime installation did not finish; rerun start.cmd to continue")


def upstream_state(root: Path) -> str:
    directory = root / UPSTREAM
    if not directory.exists():
        return "missing"
    if not (directory / ".git").exists():
        return "mismatch"
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        changed = subprocess.check_output(
            ["git", "-C", str(directory), "diff", "--name-only", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "mismatch"
    return "ready" if commit == UPSTREAM_COMMIT and not changed else "mismatch"


def ensure_upstream_checkout(root: Path) -> None:
    """Publish only a complete pinned checkout; failed fetches never poison its path."""
    state = upstream_state(root)
    if state == "ready":
        return
    if state != "missing":
        raise SetupError("Existing upstream checkout differs from the pin; it was preserved")
    destination = safe_path(root, UPSTREAM)
    staging_parent = safe_path(root, ".artifacts")
    staging_parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")
    print("Fetching the pinned Human upstream checkout...", flush=True)
    with tempfile.TemporaryDirectory(
        prefix="quickstart-upstream-", dir=staging_parent
    ) as temporary:
        staging_root = Path(temporary).resolve()
        if not staging_root.is_relative_to(staging_parent.resolve()):
            raise SetupError("Upstream staging path escaped the project")
        staged = staging_root / UPSTREAM
        staged.mkdir(parents=True)
        base = [
            "git",
            "-c",
            "http.lowSpeedLimit=100",
            "-c",
            "http.lowSpeedTime=30",
            "-C",
            str(staged),
        ]
        commands = [
            ["init", "--quiet"],
            ["remote", "add", "origin", UPSTREAM_REPOSITORY],
            ["fetch", "--no-tags", "--depth=1", "origin", UPSTREAM_COMMIT],
            ["checkout", "--quiet", "--detach", UPSTREAM_COMMIT],
        ]
        try:
            for arguments in commands:
                subprocess.run(base + arguments, check=True, env=environment, timeout=180)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise SetupError(
                "Could not fetch the pinned upstream source. Check the network and rerun "
                f"start.cmd; no existing upstream checkout was changed. Detail: {exc}"
            ) from exc
        if upstream_state(staging_root) != "ready":
            raise SetupError("Staged upstream checkout did not match the clean pinned commit")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if upstream_state(root) == "ready":
                return
            raise SetupError("An upstream checkout appeared during setup; it was preserved")
        try:
            # Windows rename is atomic and refuses an existing destination.
            os.rename(staged, destination)
        except FileExistsError:
            if upstream_state(root) != "ready":
                raise SetupError(
                    "An upstream checkout appeared during setup; it was preserved"
                ) from None
        except OSError as exc:
            raise SetupError(f"Could not publish the pinned upstream checkout: {exc}") from exc


def sample_state(root: Path, name: str, metadata: dict) -> str:
    directory = safe_path(root, f"data/raw/urfall/{name}")
    if not directory.exists():
        return "missing"
    frames = metadata["frames"]
    frame_states = {
        relative: file_state(directory, relative, item) for relative, item in frames.items()
    }
    if any(value == "mismatch" for value in frame_states.values()):
        return "mismatch"
    images = directory / f"{name}-cam0-rgb"
    actual = {p.relative_to(directory).as_posix() for p in images.glob("*.png") if p.is_file()}
    if actual - frames.keys():
        return "mismatch"
    marker = directory / "source.json"
    sync = directory / f"{name}-data.csv"
    if not marker.is_file() or not sync.is_file():
        return "missing"
    try:
        source = json.loads(marker.read_text(encoding="utf-8"))
        if source.get("archive_sha256") != metadata["archive_sha256"]:
            return "mismatch"
        if digest(sync) != metadata["sync_sha256"]:
            return "mismatch"
    except (OSError, ValueError):
        return "mismatch"
    return "ready" if all(state == "ready" for state in frame_states.values()) else "missing"


def repair_samples(root: Path, manifest: dict) -> None:
    """Complete interrupted extraction from a verified archive, preserving old bytes."""
    for name, metadata in manifest["sample_sequences"].items():
        directory = root / "data/raw/urfall" / name
        frames = metadata["frames"]
        if all(file_state(directory, rel, item) == "ready" for rel, item in frames.items()):
            continue
        archive = root / "data/raw/urfall/archives" / f"{name}-cam0-rgb.zip"
        if not archive.is_file():
            continue
        print(f"Completing the verified RGB extraction for {name}...", flush=True)
        raw_manifest = {"sha256": metadata["archive_sha256"], "files": frames}
        install_runtime(directory, raw_manifest, archive)


def source_state(root: Path, manifest: dict) -> dict[str, str]:
    result = {UPSTREAM: upstream_state(root)}
    for name, metadata in manifest["pretrained_weights"].items():
        result[name] = file_state(root, name, metadata)
    for name, metadata in manifest["sample_sequences"].items():
        if name not in ("adl-05", "fall-02"):
            raise SetupError("Quickstart sample scope may only contain adl-05 and fall-02")
        result[f"data/raw/urfall/{name}"] = sample_state(root, name, metadata)
    return result


def run_cli(root: Path, arguments: list[str]) -> None:
    subprocess.run(
        [sys.executable, "-B", "-X", "utf8", "-m", "fall_detect", *arguments], cwd=root, check=True
    )


def ensure_sources(root: Path, manifest: dict) -> None:
    states = source_state(root, manifest)
    refuse_mismatches(states)
    if states[UPSTREAM] != "ready":
        ensure_upstream_checkout(root)
    repair_samples(root, manifest)
    states = source_state(root, manifest)
    for attempt in range(1, 4):
        missing_models = states[UPSTREAM] != "ready" or any(
            states[name] != "ready" for name in manifest["pretrained_weights"]
        )
        missing_samples = any(
            states[f"data/raw/urfall/{name}"] != "ready" for name in manifest["sample_sequences"]
        )
        if not missing_models and not missing_samples:
            break
        arguments = ["fetch-assets", "--config", DEMO_CONFIG]
        if missing_models:
            arguments.append("--models")
        if missing_samples:
            arguments += ["--dataset", "urfall-cam0-rgb", "--limit-per-class", "1"]
        print(
            f"Preparing pretrained models and the two demo sequences (attempt {attempt}/3)...",
            flush=True,
        )
        try:
            run_cli(root, arguments)
        except subprocess.CalledProcessError as exc:
            states = source_state(root, manifest)
            refuse_mismatches(states)
            repair_samples(root, manifest)
            states = source_state(root, manifest)
            refuse_mismatches(states)
            if all(state == "ready" for state in states.values()):
                break
            if attempt == 3:
                raise SetupError(
                    "Source download failed after 3 attempts. Check the network and "
                    "rerun start.cmd; completed resources were preserved."
                ) from exc
            print("Source download was interrupted; retrying incomplete resources...", flush=True)
        else:
            repair_samples(root, manifest)
            break
    states = source_state(root, manifest)
    refuse_mismatches(states)
    missing = [name for name, state in states.items() if state != "ready"]
    if missing:
        raise SetupError(
            "Resource download is incomplete; rerun start.cmd:\n  " + "\n  ".join(missing)
        )
    print("Pretrained models, pinned upstream and demo sequences: ready", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--input", help="Optional MP4/AVI/image directory")
    result.add_argument("--output", help="New output MP4 path; existing results are preserved")
    result.add_argument("--no-open", action="store_true", help="Do not open the finished video")
    mode = result.add_mutually_exclusive_group()
    mode.add_argument(
        "--setup-only", action="store_true", help="Prepare resources without inference"
    )
    mode.add_argument(
        "--check-only", action="store_true", help="Offline, read-only readiness check"
    )
    return result


def main(argv: list[str] | None = None, *, root: Path = ROOT) -> int:
    args = parser().parse_args(argv)
    try:
        manifest = read_manifest(root)
        if args.check_only:
            states = {**runtime_state(root, manifest), **source_state(root, manifest)}
            ready = all(value == "ready" for value in states.values())
            print(json.dumps({"ready": ready, "resources": states}, indent=2))
            return 0 if ready else 2
        ensure_runtime(root, manifest)
        ensure_sources(root, manifest)
        if args.setup_only:
            print("Setup finished. Run start.cmd to generate the demo.", flush=True)
            return 0
        output = args.output or (
            "outputs/demo/quickstart-"
            + datetime.now(UTC).astimezone().strftime("%Y%m%d-%H%M%S-%f")
            + ".mp4"
        )
        arguments = [
            "demo",
            "--run",
            RUN,
            "--input",
            args.input or "data/raw/urfall/fall-02",
            "--output",
            output,
            "--device",
            args.device,
            "--overlay",
            "sample",
            "--no-display",
        ]
        print("Rendering the demo with the existing trained model...", flush=True)
        run_cli(root, arguments)
        target = Path(output)
        target = target if target.is_absolute() else root / target
        print(f"Demo saved: {target}", flush=True)
        if not args.no_open and os.name == "nt":
            try:
                os.startfile(str(target))
            except OSError as exc:
                print(f"Open the saved video manually: {exc}", flush=True)
        return 0
    except (SetupError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
