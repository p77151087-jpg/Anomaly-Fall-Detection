"""Download only links discovered on the fixed upstream/official source pages."""

import re
import shutil
import subprocess
import urllib.parse
import zipfile
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

from .config import project_path
from .data.splits import ensure_split, smoke_ids
from .io_utils import ContractError, contained_path, read_json, sha256, write_json
from .recipes import (
    OPENPOSE_RECIPE,
    POSE_COMMIT,
    POSE_REPO,
    STGCN_COMMIT,
    STGCN_FOLDER,
    STGCN_REPO,
    YOLO_RECIPE,
    fallback_manifest,
    yolo_manifest,
)

UPSTREAM = "https://github.com/GajuuzZ/Human-Falling-Detect-Tracks"
COMMIT = "7ed2faa4d6147dfd576f58869b6c25545208af35"
DATA_PAGE = "https://fenix.ur.edu.pl/~mkepski/ds/uf.html"
MODEL_FILES = {
    "detector": ("1obEbWBSm9bXeg10FriJ7R2cGLRsg-AfP", "detector.pth"),
    "detector_config": ("19sPzBZjAjuJQ3emRteHybm2SG25w9Wn5", "yolov3-tiny-onecls.cfg"),
    "pose": ("1IPfCDRwCmQDnQy94nT1V-_NVtTEi4VmU", "fast_res50_256x192.pth"),
    "stgcn": ("1mQQ4JHe58ylKbBqTjuKzpwN2nwKOWJ9u", "tsstg-model.pth"),
}


def session():
    # Use OS trust anchors, including enterprise certificates; never disable verification.
    import truststore

    truststore.inject_into_ssl()
    import requests

    return requests.Session()


def initial_manifest():
    return {
        "schema_version": 1,
        "recipe_id": "human_tsstg_coco_cut_v1",
        "resources": [
            {
                "role": role,
                "recipe_id": "human_tsstg_coco_cut_v1",
                "repository_url": UPSTREAM,
                "commit": COMMIT,
                "source_page": UPSTREAM + "/blob/" + COMMIT + "/README.md",
                "download_url": f"https://drive.google.com/uc?id={identifier}",
                "local_path": "weights/pretrained/" + filename,
                "size_bytes": None,
                "sha256": None,
                "upstream_expected_sha256": None,
                "license": "Upstream root has no license; SPPE has separate LICENSE. Research use; redistribution rights unverified.",
                "status": "missing",
            }
            for role, (identifier, filename) in MODEL_FILES.items()
        ],
    }


def ensure_weight_manifest(config):
    path = project_path(config["pose"]["weights_manifest"])
    if config["skeleton"]["recipe_id"] == YOLO_RECIPE:
        expected = yolo_manifest()
    else:
        expected = (
            fallback_manifest()
            if config["skeleton"]["recipe_id"] == OPENPOSE_RECIPE
            else initial_manifest()
        )
    if not path.exists():
        write_json(path, expected)
    manifest = read_json(path)
    if manifest.get("recipe_id") != expected["recipe_id"] or sorted(
        e["role"] for e in manifest["resources"]
    ) != sorted(e["role"] for e in expected["resources"]):
        raise ContractError(
            "Weight manifest recipe/roles mismatch; use a separate manifest per recipe"
        )
    if any(e.get("recipe_id") != expected["recipe_id"] for e in manifest["resources"]):
        raise ContractError("Resource recipe mismatch")
    return manifest


def register_weights(config):
    manifest = ensure_weight_manifest(config)
    for entry in manifest["resources"]:
        path = project_path(entry["local_path"])
        if not path.is_file():
            entry.update(status="missing", sha256=None, size_bytes=None)
            continue
        with path.open("rb") as handle:
            prefix = handle.read(256).lstrip().lower()
        if prefix.startswith((b"<!doctype", b"<html", b"<?xml")) or path.stat().st_size < 100:
            raise ContractError(
                f"{entry['role']}: expected model/config file, got invalid content: {path}"
            )
        current = sha256(path)
        if entry.get("sha256") and entry["sha256"] != current:
            raise ContractError(f"{entry['role']} SHA256 changed; register a new recipe manifest")
        entry.update(
            sha256=current,
            size_bytes=path.stat().st_size,
            status=entry["status"] if entry["status"] == "usable" else "downloaded_unverified",
        )
    write_json(project_path(config["pose"]["weights_manifest"]), manifest)
    return manifest


def require_weights(config):
    manifest = register_weights(config)
    missing = [e for e in manifest["resources"] if e["status"] == "missing"]
    if missing:
        detail = "\n".join(
            f"{e['role']}: {e['local_path']}\n  source: {e['source_page']}" for e in missing
        )
        raise ContractError(
            "Missing pretrained resources:\n"
            + detail
            + "\nRun: python -m fall_detect fetch-assets --models"
        )
    return {e["role"]: e for e in manifest["resources"]}


def ensure_upstream(config):
    if config["skeleton"]["recipe_id"] in (OPENPOSE_RECIPE, YOLO_RECIPE):
        ensure_checkout(project_path(config["features"]["upstream_dir"]), STGCN_REPO, STGCN_COMMIT)
        return ensure_checkout(project_path(config["pose"]["upstream_dir"]), POSE_REPO, POSE_COMMIT)
    path = project_path(config["pose"]["upstream_dir"])
    return ensure_checkout(path, UPSTREAM, COMMIT)


def ensure_checkout(path, repository, commit):
    if not path.exists():
        subprocess.run(
            ["git", "clone", "--no-checkout", repository + ".git", str(path)], check=True
        )
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", commit], check=True)
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != commit:
        raise ContractError(f"Upstream commit mismatch: need {commit}, found {actual}")
    return path


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def discover_dataset(http):
    response = http.get(DATA_PAGE, timeout=60)
    response.raise_for_status()
    parser = Links()
    parser.feed(response.text)
    links = {}
    for href in parser.links:
        url = urllib.parse.urljoin(DATA_PAGE, href)
        name = Path(urllib.parse.urlparse(url).path).name
        if re.fullmatch(r"(?:adl|fall)-\d{2}-(?:cam0-rgb\.zip|data\.csv)", name):
            links[name] = url
    if sum(name.endswith("-cam0-rgb.zip") for name in links) != 70:
        raise ContractError(
            "Official catalog does not expose exactly 70 cam0 RGB archives; inspect source"
        )
    return links


def download(http, url, path):
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with http.get(url, timeout=(30, 120), stream=True) as response:
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            raise ContractError(f"Server returned HTML instead of an asset: {url}")
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
    temporary.replace(path)


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        names = set()
        for member in members:
            relative = member.filename.replace("\\", "/")
            target = contained_path(destination, relative)
            if target in names:
                raise ContractError(f"Duplicate ZIP member: {relative}")
            names.add(target)
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ContractError(f"ZIP symlink rejected: {relative}")
        for member in members:
            target = contained_path(destination, member.filename.replace("\\", "/"))
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def fetch_sequence(root, sid, links):
    """Each worker owns one sequence and one HTTP session; publication stays atomic."""
    import requests

    try:
        print(f"Fetching {sid} camera 0 RGB and synchronization", flush=True)
        with requests.Session() as http:
            archive = root / "archives" / f"{sid}-cam0-rgb.zip"
            download(http, links[archive.name], archive)
            sync = root / sid / f"{sid}-data.csv"
            download(http, links[sync.name], sync)
        marker = root / sid / "source.json"
        digest = sha256(archive)
        if not marker.exists() or read_json(marker).get("archive_sha256") != digest:
            safe_extract(archive, root / sid)
            write_json(
                marker,
                {
                    "sequence_id": sid,
                    "archive_url": links[archive.name],
                    "archive_sha256": digest,
                    "sync_sha256": sha256(sync),
                    "license": "CC BY-NC-SA 4.0",
                    "source_page": DATA_PAGE,
                },
            )
        return {"sequence_id": sid, "status": "downloaded"}
    except (requests.RequestException, ContractError, OSError, zipfile.BadZipFile) as exc:
        print(f"Blocked {sid}: {exc}", flush=True)
        return {"sequence_id": sid, "error": str(exc)}


def fetch_assets(config, models=False, dataset=False, limit=None):
    import requests

    http = session()
    report = {"models": [], "dataset": [], "errors": []}
    if models and config["skeleton"]["recipe_id"] == YOLO_RECIPE:
        from .pose.yolo import fetch_yolo_weights

        ensure_upstream(config)
        fetch_yolo_weights(config, http)
        report["models"] = [
            {"role": e["role"], "exists": True} for e in register_weights(config)["resources"]
        ]
        models = False
    if models:
        upstream = ensure_upstream(config)
        readme = (upstream / "README.md").read_text(encoding="utf-8")
        manifest = ensure_weight_manifest(config)
        import gdown
        from gdown.exceptions import DownloadError

        for entry in manifest["resources"]:
            fallback = config["skeleton"]["recipe_id"] == OPENPOSE_RECIPE
            if fallback:
                if entry["role"] == "pose" and entry["download_url"] not in readme:
                    raise ContractError("Pose URL is not in pinned README")
                if entry["role"] == "stgcn":
                    source = (
                        project_path(config["features"]["upstream_dir"]) / "OLD_README.md"
                    ).read_text(encoding="utf-8")
                    if STGCN_FOLDER.rsplit("/", 1)[-1] not in source:
                        raise ContractError("ST-GCN source folder is not in pinned README")
            elif MODEL_FILES[entry["role"]][0] not in readme:
                raise ContractError("Weight URL is not in pinned upstream README")
            path = project_path(entry["local_path"])
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                print(f"Downloading {entry['role']} -> {path}", flush=True)
                try:
                    temporary = str(path) + ".part"
                    if fallback and entry["role"] == "pose":
                        download(http, entry["download_url"], path)
                    else:
                        result = gdown.download(
                            url=entry["download_url"], output=temporary, quiet=False
                        )
                        if result is None:
                            raise ContractError("Google Drive did not return a file")
                        Path(temporary).replace(path)
                except (DownloadError, requests.RequestException, ContractError, OSError) as exc:
                    report["errors"].append({"role": entry["role"], "error": str(exc)})
                    print(f"Blocked {entry['role']}: {exc}", flush=True)
            report["models"].append({"role": entry["role"], "exists": path.exists()})
        register_weights(config)
    if dataset:
        manifest_dir = project_path(config["data"]["manifest_dir"])
        split = ensure_split(manifest_dir)
        links = discover_dataset(http)
        write_json(manifest_dir / "source-catalog.json", {"source_page": DATA_PAGE, "links": links})
        ids = smoke_ids(split, limit) if limit else [s["sequence_id"] for s in split["sequences"]]
        root = project_path(config["data"]["raw_dir"])
        with ThreadPoolExecutor(max_workers=min(4, len(ids))) as pool:
            for result in pool.map(lambda sid: fetch_sequence(root, sid, links), ids):
                report["errors" if "error" in result else "dataset"].append(result)
    write_json(project_path("outputs/asset-fetch-report.json"), report)
    return report
