import numpy as np
import pytest
import torch

from fall_detect.io_utils import ContractError, sha256
from fall_detect.models.autoencoder import Autoencoder
from fall_detect.models.stgcn import concatenate_streams
from fall_detect.models.weights import import_upstream, select_device, strict_load
from fall_detect.skeleton.preprocessing import human_preprocess


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_synthetic_stgcn_shapes_head_parity_frozen_buffers(config, skeleton, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    import_upstream(config)
    from Actionsrecognition.Models import TwoStreamSpatialTemporalGraph

    select_device(device)
    torch.manual_seed(42)
    # Random init ONLY in explicitly synthetic structure tests; no formal cache/scorer.
    model = (
        TwoStreamSpatialTemporalGraph({"strategy": "spatial"}, 7)
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    points, motion = (
        torch.from_numpy(x).to(device) for x in human_preprocess(skeleton, (640, 480))
    )
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    with torch.inference_mode():
        features = concatenate_streams(model, points, motion)
        assert features.shape == (1, 512) and torch.isfinite(features).all()
        torch.testing.assert_close(torch.sigmoid(model.fcn(features)), model((points, motion)))
    ae = Autoencoder(512).to(device)
    optimizer = torch.optim.Adam(ae.parameters())
    # Clone exits inference tensor semantics; only AE receives gradients.
    training_input = features.clone()
    loss = (ae(training_input) - training_input).square().mean()
    loss.backward()
    optimizer.step()
    assert all(torch.equal(before[k], v.cpu()) for k, v in model.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


def test_strict_checkpoint_rejects_shapes_keys_and_html(tmp_path):
    model = Autoencoder(512)
    path = tmp_path / "synthetic-state.pt"
    torch.save({"module." + k: v for k, v in model.state_dict().items()}, path)
    report = strict_load(Autoencoder(512), path, "cpu", sha256(path))
    assert report["strict"] and report["key_conversion"] == "strip uniform module. prefix"
    with pytest.raises(ContractError, match="shape mismatch"):
        strict_load(Autoencoder(256), path, "cpu")
    path.write_text("<html>error</html>")
    with pytest.raises(ContractError, match="Restricted"):
        strict_load(model, path, "cpu")


def test_pose_crop_decoder_and_nms_against_upstream(config):
    import_upstream(config)
    from pPose_nms import pose_nms
    from SPPE.src.utils.eval import getPrediction
    from SPPE.src.utils.img import crop_dets

    from fall_detect.pose.human import pose_nms_compatible

    rgb = np.full((240, 400, 3), 150, np.uint8)
    boxes = torch.tensor([[80.0, 40.0, 230.0, 220.0]])
    crops, pt1, pt2 = crop_dets(rgb, boxes, 256, 192)
    assert crops.shape == (1, 3, 256, 192) and torch.isfinite(crops).all()
    heatmaps = torch.zeros(1, 13, 64, 48)
    for joint in range(13):
        heatmaps[0, joint, 15 + joint * 2, 12 + joint] = 0.9
    _, points, scores = getPrediction(heatmaps, pt1, pt2, 256, 192, 64, 48)
    original = pose_nms(boxes.clone(), torch.tensor([0.8]), points.clone(), scores.clone())
    compatible = pose_nms_compatible(boxes, torch.tensor([0.8]), points, scores)
    assert len(original) == len(compatible) == 1
    torch.testing.assert_close(original[0]["keypoints"], compatible[0]["keypoints"])
    torch.testing.assert_close(original[0]["kp_score"], compatible[0]["kp_score"])
    multiple = pose_nms_compatible(
        torch.cat((boxes, boxes + 100)),
        torch.tensor([0.8, 0.7]),
        torch.cat((points, points + 100)),
        torch.cat((scores, scores)),
    )
    assert len(multiple) == 2


def test_real_checkpoint_loading_only_when_assets_present(config):
    from fall_detect.assets import register_weights
    from fall_detect.models.stgcn import HumanFeatureExtractor

    resources = {e["role"]: e for e in register_weights(config)["resources"]}
    if any(e["status"] == "missing" for e in resources.values()):
        pytest.skip("blocked_external_asset: four real pretrained resources are missing")
    extractor = HumanFeatureExtractor(config, resources, select_device("cpu"))
    assert extractor.load_report["strict"]


def test_explicit_unavailable_cuda_is_not_silently_changed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ContractError, match="CUDA is unavailable"):
        select_device("cuda")
    assert str(select_device("auto")) == "cpu"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_synthetic_fastpose_structure_is_device_portable(config, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    import_upstream(config)
    from SPPE.src.models.FastPose import FastPose

    select_device(device)
    model = FastPose("resnet50", 17).to(device).eval().requires_grad_(False)
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, 256, 192), device=device))
    assert output.shape == (1, 17, 64, 48) and torch.isfinite(output).all()
