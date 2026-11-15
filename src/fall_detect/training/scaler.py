from dataclasses import dataclass

import numpy as np

from ..io_utils import ContractError, write_npz


@dataclass
class FeatureScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features):
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
            raise ContractError("Scaler needs nonempty finite train-normal features")
        mean, scale = values.mean(axis=0), values.std(axis=0)
        scale[scale == 0] = 1.0
        return cls(mean, scale)

    def transform(self, features):
        features = np.asarray(features)
        if (
            features.ndim != 2
            or features.shape[1] != len(self.mean)
            or not np.isfinite(features).all()
        ):
            raise ContractError("Feature/scaler dimension mismatch or non-finite input")
        result = ((features - self.mean) / self.scale).astype(np.float32)
        if not np.isfinite(result).all():
            raise ContractError("Feature scaling overflow")
        return result

    def save(self, path):
        write_npz(path, mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as arrays:
            mean, scale = arrays["mean"], arrays["scale"]
        if (
            mean.ndim != 1
            or mean.shape != scale.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ContractError("Invalid saved scaler")
        return cls(mean, scale)
