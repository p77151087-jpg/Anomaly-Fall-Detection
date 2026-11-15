import numpy as np
import torch

from ..config import project_path
from ..contracts import RECIPE_ID
from ..io_utils import ContractError
from ..skeleton.preprocessing import human_preprocess
from .weights import import_upstream, strict_load


def concatenate_streams(model, points, motion):
    return torch.cat((model.pts_stream(points), model.mot_stream(motion)), dim=-1)


class HumanFeatureExtractor:
    feature_dim = 512
    recipe_id = RECIPE_ID

    def __init__(self, config, resources, device):
        import_upstream(config)
        from Actionsrecognition.Models import TwoStreamSpatialTemporalGraph

        self.model = TwoStreamSpatialTemporalGraph({"strategy": "spatial"}, 7)
        entry = resources["stgcn"]
        self.load_report = strict_load(
            self.model, project_path(entry["local_path"]), device, entry["sha256"]
        )
        self.device = device
        self.tolerance = config["skeleton"]["span_tolerance"]
        self.window_size = config["skeleton"]["window_size"]
        self.normalization = config["skeleton"].get("coordinate_normalization", "per_frame")
        self.forward_verified = False

    @torch.inference_mode()
    def extract_features(self, skeleton, frame_size_wh):
        points, motion = human_preprocess(
            skeleton, frame_size_wh, self.tolerance, self.window_size, self.normalization
        )
        self.model.eval()
        points = torch.from_numpy(points).to(self.device)
        motion = torch.from_numpy(motion).to(self.device)
        features = concatenate_streams(self.model, points, motion)
        if features.shape != (1, self.feature_dim) or not torch.isfinite(features).all():
            raise ContractError("ST-GCN failed to produce finite [1,512] features")
        if not self.forward_verified:
            original = self.model((points, motion))
            reconstructed = torch.sigmoid(self.model.fcn(features))
            torch.testing.assert_close(reconstructed, original, atol=1e-6, rtol=1e-5)
            self.forward_verified = True
        return features.cpu().numpy().astype(np.float32)
