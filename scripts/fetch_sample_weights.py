"""Fetch the public Human mirror cited in upstream issue 117; preserve old assets."""

from datetime import UTC, datetime
from pathlib import Path

from fall_detect.assets import initial_manifest, session
from fall_detect.config import project_path
from fall_detect.io_utils import ContractError, read_json, sha256, write_json

SOURCE = "https://github.com/GajuuzZ/Human-Falling-Detect-Tracks/issues/117"
FOLDER = "https://drive.google.com/drive/folders/1lrTI56k9QiIfMJhG9kzNjBzJh98KCIIO"
IDS = {
    "detector": "1GbH6UQpBvOwLO_gozaH9QFnL23j8plPb",
    "detector_config": "1f-jEh05PBaHXLRwBvrYgSSPtYLuBeiPm",
    "pose": "1m5tIKH8INJ82GI7q6tWtPPDJ55jgpKfg",
    "stgcn": "1MAk2v0peCa4Fs9zw9-N3ssuG4Y1MIItO",
}


def main():
    import gdown

    session()  # OS certificate store, TLS verification remains enabled.
    target = project_path("weights/sample-human-manifest.json")
    manifest = read_json(target) if target.exists() else initial_manifest()
    manifest.update(
        retrieval_source=SOURCE,
        mirror_folder=FOLDER,
        provenance_limit="Community mirror; author-published checksum unavailable. Strict architecture and local hash checks do not establish byte identity to the original release.",
    )
    for entry in manifest["resources"]:
        destination = (
            project_path("weights/pretrained/sample-human") / Path(entry["local_path"]).name
        )
        entry.update(
            original_download_url=entry.get("original_download_url", entry["download_url"]),
            download_url=f"https://drive.google.com/uc?id={IDS[entry['role']]}",
            retrieval_source=SOURCE,
            local_path=str(destination.relative_to(project_path("."))).replace("\\", "/"),
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".part")
            if temporary.exists():
                raise ContractError(
                    f"Partial download preserved; inspect before retry: {temporary}"
                )
            print(f"Downloading {entry['role']}", flush=True)
            result = gdown.download(id=IDS[entry["role"]], output=str(temporary), quiet=False)
            if not result:
                raise ContractError(f"No public file returned for {entry['role']}")
            temporary.replace(destination)
        with destination.open("rb") as handle:
            prefix = handle.read(256).lstrip().lower()
        if (
            prefix.startswith((b"<html", b"<!doctype", b"<?xml"))
            or destination.stat().st_size < 100
        ):
            raise ContractError(f"Invalid asset content: {destination}")
        digest = sha256(destination)
        if entry.get("sha256") and entry["sha256"] != digest:
            raise ContractError(f"Preserved asset changed: {destination}")
        entry.update(
            sha256=digest,
            size_bytes=destination.stat().st_size,
            status="usable" if entry.get("status") == "usable" else "downloaded_unverified",
        )
        entry.setdefault("retrieved_at", datetime.now(UTC).isoformat())
        write_json(target, manifest)
        print(f"{entry['role']}: {entry['size_bytes']} bytes {digest}", flush=True)


if __name__ == "__main__":
    main()
