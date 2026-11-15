import numpy as np
import torch

from ..config import project_path
from ..io_utils import ContractError
from ..recipes import OPENPOSE_RECIPE, STGCN_COMMIT
from ..skeleton.preprocessing import openpose_preprocess
from .weights import import_pinned, strict_load


class OpenPoseFeatureExtractor:
    feature_dim = 256
    recipe_id = OPENPOSE_RECIPE

    def __init__(self, config, resources, device):
        self.recipe_id = config["skeleton"]["recipe_id"]
        module = import_pinned(config["features"]["upstream_dir"], STGCN_COMMIT, "net.st_gcn")
        self.model = module.Model(3, 400, {"layout": "openpose", "strategy": "spatial"}, True)
        entry = resources["stgcn"]
        self.load_report = strict_load(
            self.model, project_path(entry["local_path"]), device, entry["sha256"]
        )
        self.device = device
        self.forward_verified = False

    @torch.inference_mode()
    def extract_features(self, skeleton, frame_size_wh):
        data = torch.from_numpy(openpose_preprocess(skeleton, frame_size_wh)).to(self.device)
        self.model.eval()
        _, feature_map = self.model.extract_feature(data)
        if feature_map.shape[1] != 256 or feature_map.shape[3:] != (18, 1):
            raise ContractError("Unexpected ST-GCN feature map; expected [1,256,T',18,1]")
        features = feature_map.mean(dim=(2, 3, 4))
        if features.shape != (1, 256) or not torch.isfinite(features).all():
            raise ContractError("ST-GCN failed to produce finite [1,256] features")
        if not self.forward_verified:
            reconstructed = self.model.fcn(features[:, :, None, None]).flatten(1)
            torch.testing.assert_close(reconstructed, self.model(data), atol=2e-5, rtol=1e-4)
            self.forward_verified = True
        return features.cpu().numpy().astype(np.float32)
