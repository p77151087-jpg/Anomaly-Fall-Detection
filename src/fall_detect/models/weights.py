import sys

import torch

from ..assets import COMMIT
from ..config import project_path
from ..io_utils import ContractError, sha256


def select_device(requested="auto", threads=4):
    torch.set_num_threads(threads)
    if requested == "cuda" and not torch.cuda.is_available():
        raise ContractError(
            "runtime.device=cuda requested but CUDA is unavailable; use cpu or install a compatible wheel"
        )
    return torch.device(
        "cuda"
        if requested == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested == "auto"
        else requested
    )


def import_upstream(config):
    import subprocess

    path = project_path(config["pose"]["upstream_dir"])
    if not (path / ".git").exists():
        raise ContractError("Missing pinned upstream checkout; run fetch-assets --models")
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != COMMIT:
        raise ContractError(f"Expected upstream {COMMIT}, got {actual}")
    # Run upstream unmodified; compatibility fixes are in our adapters.
    modified = subprocess.check_output(
        ["git", "-C", str(path), "diff", "--name-only", "HEAD"], text=True
    ).strip()
    if modified:
        raise ContractError(
            f"Pinned upstream sources were modified: {modified}; record a new backend recipe"
        )
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def strict_load(model, path, device, expected_sha=None):
    if expected_sha and sha256(path) != expected_sha:
        raise ContractError(f"Weight SHA256 mismatch: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise ContractError(f"Restricted checkpoint load failed: {path}: {exc}") from exc
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items())
    ):
        raise ContractError("Expected a tensor-only state dictionary")
    conversion = None
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: value for k, value in state.items()}
        conversion = "strip uniform module. prefix"
    expected = model.state_dict()
    if state.keys() != expected.keys():
        raise ContractError(
            f"Checkpoint key mismatch; missing={sorted(expected.keys() - state.keys())[:5]}, unexpected={sorted(state.keys() - expected.keys())[:5]}"
        )
    for key, value in state.items():
        if value.shape != expected[key].shape:
            raise ContractError(
                f"Checkpoint shape mismatch for {key}: {tuple(value.shape)} != {tuple(expected[key].shape)}"
            )
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ContractError(f"Checkpoint non-finite tensor: {key}")
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    return {"strict": True, "key_conversion": conversion, "sha256": sha256(path)}


def import_pinned(path, commit, module):
    """Import unchanged pinned sources, rejecting conflicting generic upstream packages."""
    import importlib
    import subprocess

    path = project_path(path)
    if not (path / ".git").exists():
        raise ContractError(
            f"Missing pinned checkout: {path}; run fetch-assets --models with the recipe config"
        )
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    modified = subprocess.check_output(
        ["git", "-C", str(path), "diff", "--name-only", "HEAD"], text=True
    ).strip()
    if actual != commit or modified:
        raise ContractError(
            f"Pinned source mismatch: {path}; expected {commit}; modified={modified}"
        )
    root_name = module.split(".")[0]
    existing = sys.modules.get(root_name)
    if existing is not None:
        locations = list(getattr(existing, "__path__", []))
        if getattr(existing, "__file__", None):
            locations.append(existing.__file__)
        if not locations or any(not project_path(p).is_relative_to(path) for p in locations):
            raise ContractError(f"Upstream import collision: {root_name}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    result = importlib.import_module(module)
    if not project_path(result.__file__).is_relative_to(path):
        raise ContractError(f"Imported {module} from an unexpected checkout")
    return result
