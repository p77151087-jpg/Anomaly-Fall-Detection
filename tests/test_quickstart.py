"""Pinned runtime installation must preserve experiments and stay bounded."""

import hashlib
import importlib.util
import io
import json
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import ClassVar

import pytest

spec = importlib.util.spec_from_file_location(
    "quickstart_under_test", Path(__file__).resolve().parents[1] / "scripts/quickstart.py"
)
quickstart = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quickstart)

PREFIX = "runs/ur-sample-human-s1-001/"


def metadata(content):
    return {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}


def bundle(tmp_path, entries=None, expected=None):
    entries = entries or {PREFIX + "autoencoder.pt": b"model", PREFIX + "scaler.npz": b"scaler"}
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for name, content in entries.items():
            output.writestr(name, content)
    manifest = {
        "schema_version": 1,
        "runtime_id": "unit-runtime",
        "url": "https://example.invalid/runtime.zip",
        "sha256": quickstart.digest(archive),
        "files": {name: metadata(content) for name, content in (expected or entries).items()},
        "pretrained_weights": {},
        "sample_sequences": {},
    }
    return archive, manifest


def write_file(root, name, content):
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def test_runtime_reuses_existing_files_and_repeated_setup_is_offline(tmp_path, monkeypatch):
    archive, manifest = bundle(tmp_path)
    root = tmp_path / "clone"
    root.mkdir()
    original = write_file(root, PREFIX + "autoencoder.pt", b"model")
    original_mtime = original.stat().st_mtime_ns
    downloads = []
    monkeypatch.setattr(
        quickstart, "download_runtime", lambda *_: downloads.append(True) or archive
    )
    quickstart.ensure_runtime(root, manifest)
    quickstart.ensure_runtime(root, manifest)
    assert downloads == [True]
    assert original.read_bytes() == b"model"
    assert original.stat().st_mtime_ns == original_mtime
    assert (root / (PREFIX + "scaler.npz")).read_bytes() == b"scaler"


def test_modified_existing_run_prevents_any_download_or_publication(tmp_path, monkeypatch):
    _, manifest = bundle(tmp_path)
    root = tmp_path / "clone"
    root.mkdir()
    old = write_file(root, PREFIX + "scaler.npz", b"different experiment")
    monkeypatch.setattr(quickstart, "download_runtime", lambda *_: pytest.fail("network"))
    with pytest.raises(quickstart.SetupError, match="preserved"):
        quickstart.ensure_runtime(root, manifest)
    assert old.read_bytes() == b"different experiment"
    assert not (root / (PREFIX + "autoencoder.pt")).exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/drive", "a\\b", "a/../b"])
def test_zip_traversal_is_rejected_before_publication(tmp_path, name):
    archive, manifest = bundle(
        tmp_path,
        entries={PREFIX + "autoencoder.pt": b"model", name: b"bad"},
        expected={PREFIX + "autoencoder.pt": b"model"},
    )
    root = tmp_path / "clone"
    root.mkdir()
    with pytest.raises(quickstart.SetupError):
        quickstart.install_runtime(root, manifest, archive)
    assert not (root / "runs").exists()


def test_archive_hash_mismatch_does_not_touch_run(tmp_path):
    archive, manifest = bundle(tmp_path)
    manifest["sha256"] = "0" * 64
    root = tmp_path / "clone"
    root.mkdir()
    with pytest.raises(quickstart.SetupError, match="archive SHA256 mismatch"):
        quickstart.install_runtime(root, manifest, archive)
    assert list(root.iterdir()) == []


def test_member_hash_mismatch_is_checked_before_any_file_is_installed(tmp_path):
    archive, manifest = bundle(tmp_path)
    manifest["files"][PREFIX + "scaler.npz"]["sha256"] = "0" * 64
    root = tmp_path / "clone"
    root.mkdir()
    with pytest.raises(quickstart.SetupError, match="member SHA256 mismatch"):
        quickstart.install_runtime(root, manifest, archive)
    assert not (root / "runs").exists()


def test_zip_links_and_duplicate_paths_are_rejected(tmp_path):
    archive = tmp_path / "links.zip"
    link = zipfile.ZipInfo(PREFIX + "autoencoder.pt")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(link, b"elsewhere")
    manifest = {
        "sha256": quickstart.digest(archive),
        "files": {link.filename: metadata(b"elsewhere")},
    }
    root = tmp_path / "clone"
    root.mkdir()
    with pytest.raises(quickstart.SetupError, match="ZIP links"):
        quickstart.install_runtime(root, manifest, archive)
    archive, manifest = bundle(
        tmp_path,
        entries={PREFIX + "autoencoder.pt": b"a", PREFIX + "AUTOENCODER.PT": b"b"},
        expected={PREFIX + "autoencoder.pt": b"a"},
    )
    with pytest.raises(quickstart.SetupError, match="Duplicate ZIP member"):
        quickstart.install_runtime(root, manifest, archive)


def test_oversized_member_and_incomplete_bundle_are_rejected(tmp_path):
    archive, manifest = bundle(tmp_path)
    manifest["files"][PREFIX + "scaler.npz"]["size_bytes"] = 1
    root = tmp_path / "clone"
    root.mkdir()
    with pytest.raises(quickstart.SetupError, match="member size mismatch"):
        quickstart.install_runtime(root, manifest, archive)
    manifest["files"][PREFIX + "scaler.npz"] = metadata(b"scaler")
    manifest["files"][PREFIX + "missing.json"] = metadata(b"missing")
    with pytest.raises(quickstart.SetupError, match="incomplete"):
        quickstart.install_runtime(root, manifest, archive)
    assert not (root / "runs").exists()


def test_download_hash_failure_never_becomes_installable_archive(tmp_path, monkeypatch):
    _, manifest = bundle(tmp_path)
    root = tmp_path / "clone"
    root.mkdir()

    class Response(io.BytesIO):
        headers: ClassVar[dict[str, str]] = {"Content-Length": "9"}

    monkeypatch.setattr(quickstart, "urlopen", lambda *_args, **_kwargs: Response(b"bad bytes"))
    with pytest.raises(quickstart.SetupError, match="download SHA256 mismatch"):
        quickstart.download_runtime(root, manifest)
    assert not list((root / ".artifacts/quickstart-downloads").glob("*.zip"))
    assert not (root / "runs").exists()


def test_download_size_limit_is_enforced(tmp_path, monkeypatch):
    _, manifest = bundle(tmp_path)
    manifest["max_download_bytes"] = 4
    root = tmp_path / "clone"
    root.mkdir()

    class Response(io.BytesIO):
        headers: ClassVar[dict[str, str]] = {"Content-Length": "10"}

    monkeypatch.setattr(quickstart, "urlopen", lambda *_args, **_kwargs: Response(b"0123456789"))
    with pytest.raises(quickstart.SetupError, match="size limit"):
        quickstart.download_runtime(root, manifest)


def test_default_flow_only_runs_existing_demo_and_never_train_or_test(tmp_path, monkeypatch):
    _, manifest = bundle(tmp_path)
    monkeypatch.setattr(quickstart, "read_manifest", lambda *_: manifest)
    prepared = []
    monkeypatch.setattr(quickstart, "ensure_runtime", lambda *_: prepared.append("runtime"))
    monkeypatch.setattr(quickstart, "ensure_sources", lambda *_: prepared.append("sources"))
    commands = []
    monkeypatch.setattr(quickstart, "run_cli", lambda _root, args: commands.append(args))
    assert quickstart.main(["--no-open"], root=tmp_path) == 0
    assert prepared == ["runtime", "sources"]
    assert len(commands) == 1 and commands[0][0] == "demo"
    assert commands[0][commands[0].index("--input") + 1] == "data/raw/urfall/fall-02"
    assert not {"train", "calibrate", "evaluate", "test"}.intersection(commands[0])


def test_setup_only_never_runs_inference(tmp_path, monkeypatch):
    _, manifest = bundle(tmp_path)
    monkeypatch.setattr(quickstart, "read_manifest", lambda *_: manifest)
    monkeypatch.setattr(quickstart, "ensure_runtime", lambda *_: None)
    monkeypatch.setattr(quickstart, "ensure_sources", lambda *_: None)
    monkeypatch.setattr(quickstart, "run_cli", lambda *_: pytest.fail("inference"))
    assert quickstart.main(["--setup-only"], root=tmp_path) == 0


def test_check_only_is_offline_and_read_only(tmp_path, monkeypatch):
    _, manifest = bundle(tmp_path)
    monkeypatch.setattr(quickstart, "read_manifest", lambda *_: manifest)
    monkeypatch.setattr(quickstart, "runtime_state", lambda *_: {"runtime": "missing"})
    monkeypatch.setattr(quickstart, "source_state", lambda *_: {"sample": "ready"})
    monkeypatch.setattr(quickstart, "ensure_runtime", lambda *_: pytest.fail("mutation"))
    monkeypatch.setattr(quickstart, "ensure_sources", lambda *_: pytest.fail("mutation"))
    monkeypatch.setattr(quickstart, "urlopen", lambda *_: pytest.fail("network"))
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    assert quickstart.main(["--check-only"], root=tmp_path) == 2
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_source_fetch_is_bounded_to_two_train_validation_samples(tmp_path, monkeypatch):
    manifest = {
        "pretrained_weights": {"weights/pretrained/model": {}},
        "sample_sequences": {"adl-05": {}, "fall-02": {}},
    }
    states = iter(
        [
            {
                quickstart.UPSTREAM: "missing",
                "weights/pretrained/model": "missing",
                "data/raw/urfall/adl-05": "missing",
                "data/raw/urfall/fall-02": "missing",
            },
            {
                quickstart.UPSTREAM: "missing",
                "weights/pretrained/model": "missing",
                "data/raw/urfall/adl-05": "missing",
                "data/raw/urfall/fall-02": "missing",
            },
            {
                quickstart.UPSTREAM: "ready",
                "weights/pretrained/model": "ready",
                "data/raw/urfall/adl-05": "ready",
                "data/raw/urfall/fall-02": "ready",
            },
        ]
    )
    monkeypatch.setattr(quickstart, "source_state", lambda *_: next(states))
    monkeypatch.setattr(quickstart, "ensure_upstream_checkout", lambda *_: None)
    monkeypatch.setattr(quickstart, "repair_samples", lambda *_: None)
    calls = []
    monkeypatch.setattr(quickstart, "run_cli", lambda _root, args: calls.append(args))
    quickstart.ensure_sources(tmp_path, manifest)
    assert calls == [
        [
            "fetch-assets",
            "--config",
            quickstart.DEMO_CONFIG,
            "--models",
            "--dataset",
            "urfall-cam0-rgb",
            "--limit-per-class",
            "1",
        ]
    ]


def test_incomplete_sample_is_repaired_without_overwriting_existing_rgb(tmp_path):
    sid = "fall-02"
    frames = {f"{sid}-cam0-rgb/001.png": b"frame one", f"{sid}-cam0-rgb/002.png": b"frame two"}
    archive_root = tmp_path / "data/raw/urfall/archives"
    archive_root.mkdir(parents=True)
    archive = archive_root / f"{sid}-cam0-rgb.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for name, content in frames.items():
            output.writestr(name, content)
    metadata_sample = {
        "archive_sha256": quickstart.digest(archive),
        "sync_sha256": hashlib.sha256(b"sync").hexdigest(),
        "png_count": 2,
        "frames": {name: metadata(content) for name, content in frames.items()},
    }
    directory = tmp_path / "data/raw/urfall" / sid
    old = write_file(directory, f"{sid}-cam0-rgb/001.png", b"frame one")
    mtime = old.stat().st_mtime_ns
    write_file(directory, f"{sid}-data.csv", b"sync")
    write_file(
        directory,
        "source.json",
        json.dumps({"archive_sha256": metadata_sample["archive_sha256"]}).encode(),
    )
    assert quickstart.sample_state(tmp_path, sid, metadata_sample) == "missing"
    quickstart.repair_samples(tmp_path, {"sample_sequences": {sid: metadata_sample}})
    assert quickstart.sample_state(tmp_path, sid, metadata_sample) == "ready"
    assert old.stat().st_mtime_ns == mtime
    old.write_bytes(b"changed")
    with pytest.raises(quickstart.SetupError, match="preserved"):
        quickstart.repair_samples(tmp_path, {"sample_sequences": {sid: metadata_sample}})
    assert old.read_bytes() == b"changed"


def test_source_download_retries_only_incomplete_resources(tmp_path, monkeypatch):
    model = "weights/pretrained/model"
    manifest = {
        "pretrained_weights": {model: {}},
        "sample_sequences": {"adl-05": {}, "fall-02": {}},
    }
    states = {
        quickstart.UPSTREAM: "ready",
        model: "missing",
        "data/raw/urfall/adl-05": "missing",
        "data/raw/urfall/fall-02": "missing",
    }
    monkeypatch.setattr(quickstart, "source_state", lambda *_: states.copy())
    monkeypatch.setattr(quickstart, "repair_samples", lambda *_: None)
    calls = []
    completed_mtime = None

    def fetch(_root, arguments):
        nonlocal completed_mtime
        calls.append(arguments)
        if len(calls) == 1:
            completed = write_file(tmp_path, model, b"completed model")
            completed_mtime = completed.stat().st_mtime_ns
            states[model] = "ready"
            states["data/raw/urfall/fall-02"] = "ready"
            raise subprocess.CalledProcessError(2, arguments)
        assert "--models" not in arguments
        states["data/raw/urfall/adl-05"] = "ready"

    monkeypatch.setattr(quickstart, "run_cli", fetch)
    quickstart.ensure_sources(tmp_path, manifest)
    assert len(calls) == 2 and "--models" in calls[0]
    assert calls[1] == [
        "fetch-assets",
        "--config",
        quickstart.DEMO_CONFIG,
        "--dataset",
        "urfall-cam0-rgb",
        "--limit-per-class",
        "1",
    ]
    assert (tmp_path / model).read_bytes() == b"completed model"
    assert (tmp_path / model).stat().st_mtime_ns == completed_mtime


def test_source_download_stops_after_three_failed_attempts(tmp_path, monkeypatch):
    manifest = {"pretrained_weights": {}, "sample_sequences": {"adl-05": {}}}
    states = {quickstart.UPSTREAM: "ready", "data/raw/urfall/adl-05": "missing"}
    monkeypatch.setattr(quickstart, "source_state", lambda *_: states.copy())
    monkeypatch.setattr(quickstart, "repair_samples", lambda *_: None)
    calls = []

    def fail(_root, arguments):
        calls.append(arguments)
        raise subprocess.CalledProcessError(2, arguments)

    monkeypatch.setattr(quickstart, "run_cli", fail)
    with pytest.raises(quickstart.SetupError, match="3 attempts.*rerun start.cmd"):
        quickstart.ensure_sources(tmp_path, manifest)
    assert len(calls) == 3
    assert all("--models" not in arguments for arguments in calls)


def test_changed_resource_aborts_download_retry_without_overwrite(tmp_path, monkeypatch):
    model = "weights/pretrained/model"
    existing = write_file(tmp_path, model, b"old model")
    manifest = {"pretrained_weights": {model: {}}, "sample_sequences": {"adl-05": {}}}
    states = {quickstart.UPSTREAM: "ready", model: "ready", "data/raw/urfall/adl-05": "missing"}
    monkeypatch.setattr(quickstart, "source_state", lambda *_: states.copy())
    monkeypatch.setattr(quickstart, "repair_samples", lambda *_: None)
    calls = []

    def fail_after_external_edit(_root, arguments):
        calls.append(arguments)
        existing.write_bytes(b"external edit")
        states[model] = "mismatch"
        raise subprocess.CalledProcessError(2, arguments)

    monkeypatch.setattr(quickstart, "run_cli", fail_after_external_edit)
    with pytest.raises(quickstart.SetupError, match="preserved"):
        quickstart.ensure_sources(tmp_path, manifest)
    assert len(calls) == 1
    assert existing.read_bytes() == b"external edit"


def test_upstream_uses_a_staged_shallow_pinned_checkout(tmp_path, monkeypatch):
    root = tmp_path / "clone"
    root.mkdir()
    commands = []

    def fake_run(command, **options):
        commands.append((command, options))
        staged = Path(command[command.index("-C") + 1])
        if "init" in command:
            (staged / ".git").mkdir()
        assert not (root / quickstart.UPSTREAM).exists()
        assert staged.is_relative_to(root / ".artifacts")

    def fake_check_output(command, **_options):
        return quickstart.UPSTREAM_COMMIT + "\n" if "rev-parse" in command else ""

    monkeypatch.setattr(quickstart.subprocess, "run", fake_run)
    monkeypatch.setattr(quickstart.subprocess, "check_output", fake_check_output)
    quickstart.ensure_upstream_checkout(root)
    assert len(commands) == 4
    fetch = commands[2][0]
    assert fetch[-5:] == ["fetch", "--no-tags", "--depth=1", "origin", quickstart.UPSTREAM_COMMIT]
    for command, options in commands:
        assert options["env"]["GIT_TERMINAL_PROMPT"] == "0"
        assert options["env"]["GCM_INTERACTIVE"] == "Never"
        assert options["timeout"] == 180
        assert "http.lowSpeedLimit=100" in command and "http.lowSpeedTime=30" in command
    assert (root / quickstart.UPSTREAM / ".git").is_dir()
    assert not list((root / ".artifacts").glob("quickstart-upstream-*"))
    assert quickstart.upstream_state(root) == "ready"


def test_failed_upstream_fetch_is_retryable_without_partial_destination(tmp_path, monkeypatch):
    root = tmp_path / "clone"
    root.mkdir()
    staging_paths = []
    fail_fetch = True

    def fake_run(command, **_options):
        staged = Path(command[command.index("-C") + 1])
        if "init" in command:
            staging_paths.append(staged)
            (staged / ".git").mkdir()
        if "fetch" in command and fail_fetch:
            raise subprocess.TimeoutExpired(command, 180)

    monkeypatch.setattr(quickstart.subprocess, "run", fake_run)
    monkeypatch.setattr(
        quickstart.subprocess,
        "check_output",
        lambda command, **_options: (
            quickstart.UPSTREAM_COMMIT + "\n" if "rev-parse" in command else ""
        ),
    )
    with pytest.raises(quickstart.SetupError, match="rerun start.cmd"):
        quickstart.ensure_upstream_checkout(root)
    assert not (root / quickstart.UPSTREAM).exists()
    fail_fetch = False
    quickstart.ensure_upstream_checkout(root)
    assert len(staging_paths) == 2 and staging_paths[0] != staging_paths[1]
    assert quickstart.upstream_state(root) == "ready"


def test_existing_modified_upstream_is_preserved_without_network(tmp_path, monkeypatch):
    root = tmp_path / "clone"
    directory = root / quickstart.UPSTREAM
    (directory / ".git").mkdir(parents=True)
    original = write_file(directory, "user-work.py", b"keep my changes")
    monkeypatch.setattr(
        quickstart.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("network")
    )
    monkeypatch.setattr(
        quickstart.subprocess,
        "check_output",
        lambda command, **_options: (
            quickstart.UPSTREAM_COMMIT + "\n" if "rev-parse" in command else "user-work.py\n"
        ),
    )
    with pytest.raises(quickstart.SetupError, match="preserved"):
        quickstart.ensure_upstream_checkout(root)
    assert original.read_bytes() == b"keep my changes"
    assert not (root / ".artifacts").exists()
