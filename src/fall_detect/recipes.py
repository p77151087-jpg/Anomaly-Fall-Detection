"""Explicit, incompatible pretrained recipes; no implicit model substitution."""

HUMAN_RECIPE = "human_tsstg_coco_cut_v1"
OPENPOSE_RECIPE = "lightweight_openpose18_stgcn_kinetics_v1"
YOLO_RECIPE = "yolo11m_coco17_openpose18_stgcn_kinetics_v1"
YOLO_VERSION = "8.3.228"
YOLO_SOURCE_SHA = "29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt"
STGCN_REPO = "https://github.com/yysijie/st-gcn"
STGCN_COMMIT = "221c0e152054b8da593774c0d483e59befdb9061"
POSE_REPO = "https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch"
POSE_COMMIT = "d23c284b09acf27a163e1febd511e7482cac25ed"
STGCN_FOLDER = "https://drive.google.com/drive/folders/1IYKoSrjeI3yYJ9bO0_z_eDo92i7ob_aF"
POSE_URL = "https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth"

RECIPES = {
    HUMAN_RECIPE: {"layout": "human13", "joints": 13, "dim": 512},
    OPENPOSE_RECIPE: {"layout": "openpose18", "joints": 18, "dim": 256},
    YOLO_RECIPE: {"layout": "openpose18", "joints": 18, "dim": 256, "derived_joints": [1]},
}


def recipe_spec(config):
    return RECIPES[config["skeleton"]["recipe_id"]]


def upstream_commits(config):
    if config["skeleton"]["recipe_id"] == HUMAN_RECIPE:
        return {"human": "7ed2faa4d6147dfd576f58869b6c25545208af35"}
    result = {"stgcn": STGCN_COMMIT, "lightweight_openpose": POSE_COMMIT}
    if config["skeleton"]["recipe_id"] == YOLO_RECIPE:
        result["ultralytics_package"] = YOLO_VERSION
    return result


def yolo_manifest():
    manifest = fallback_manifest()
    manifest["recipe_id"] = YOLO_RECIPE
    for entry in manifest["resources"]:
        entry["recipe_id"] = YOLO_RECIPE
        if entry["role"] == "pose":
            entry.update(
                repository_url="https://github.com/ultralytics/ultralytics",
                commit=None,
                package_version=YOLO_VERSION,
                source_page="https://github.com/ultralytics/assets/releases/tag/v8.3.0",
                download_url=YOLO_URL,
                source_sha256=YOLO_SOURCE_SHA,
                local_path="weights/pretrained/yolo11/yolo11m-pose-state.pt",
                license="AGPL-3.0 (Ultralytics code and pretrained weights)",
                conversion="Official trusted checkpoint to float32 tensor-only state_dict",
            )
    return manifest


def fallback_manifest():
    return {
        "schema_version": 1,
        "recipe_id": OPENPOSE_RECIPE,
        "resources": [
            {
                "role": role,
                "recipe_id": OPENPOSE_RECIPE,
                "repository_url": repo,
                "commit": commit,
                "source_page": f"{repo}/blob/{commit}/{readme}",
                "source_folder": STGCN_FOLDER if role == "stgcn" else None,
                "download_url": url,
                "local_path": "weights/pretrained/openpose18/" + filename,
                "size_bytes": None,
                "sha256": None,
                "upstream_expected_sha256": None,
                "license": license_name,
                "status": "missing",
            }
            for role, repo, commit, readme, url, filename, license_name in [
                (
                    "pose",
                    POSE_REPO,
                    POSE_COMMIT,
                    "README.md",
                    POSE_URL,
                    "checkpoint_iter_370000.pth",
                    "Apache-2.0 (upstream repository)",
                ),
                (
                    "stgcn",
                    STGCN_REPO,
                    STGCN_COMMIT,
                    "OLD_README.md",
                    "https://drive.google.com/uc?id=1MDIUlJ_X-IpNgLZYXgdMuclCsnYTepU1",
                    "st_gcn.kinetics.pt",
                    "BSD-2-Clause (upstream repository)",
                ),
            ]
        ],
    }
