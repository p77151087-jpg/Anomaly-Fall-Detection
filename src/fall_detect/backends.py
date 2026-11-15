from .recipes import OPENPOSE_RECIPE, YOLO_RECIPE


def create_backends(config, resources, device):
    if config["skeleton"]["recipe_id"] == YOLO_RECIPE:
        from .models.openpose_stgcn import OpenPoseFeatureExtractor
        from .pose.yolo import YoloPoseBackend

        return YoloPoseBackend(config, resources, device), OpenPoseFeatureExtractor(
            config, resources, device
        )
    if config["skeleton"]["recipe_id"] == OPENPOSE_RECIPE:
        from .models.openpose_stgcn import OpenPoseFeatureExtractor
        from .pose.lightweight import LightweightPoseBackend

        return LightweightPoseBackend(config, resources, device), OpenPoseFeatureExtractor(
            config, resources, device
        )
    from .models.stgcn import HumanFeatureExtractor
    from .pose.human import HumanPoseBackend

    return HumanPoseBackend(config, resources, device), HumanFeatureExtractor(
        config, resources, device
    )
