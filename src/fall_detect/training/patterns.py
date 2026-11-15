"""Balance normal Train feature patterns within each original sequence."""

import numpy as np
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from ..io_utils import ContractError


def pattern_sampler(normalized_train, counts, generator, seed=42):
    values = np.asarray(normalized_train, dtype=np.float32)
    if (
        not counts
        or any(type(n) is not int or n < 0 for n in counts.values())
        or values.ndim != 2
        or len(values) != sum(counts.values())
        or len(values) == 0
        or not np.isfinite(values).all()
    ):
        raise ContractError(
            "Pattern sampling requires finite Train features matching sequence counts"
        )
    k = min(8, len(values))
    # Bound BLAS threads without changing process-wide environment or global RNGs.
    with threadpool_limits(limits=1):
        model = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300, algorithm="lloyd")
        labels = model.fit_predict(values)
    weights = np.zeros(len(values), dtype=np.float64)
    rows, start = {}, 0
    for sid, count in counts.items():
        assigned = labels[start : start + count]
        groups, size = np.unique(assigned, return_counts=True)
        for group, n in zip(groups, size):
            weights[start : start + count][assigned == group] = 1.0 / (len(groups) * n)
        rows[sid] = {str(int(g)): int(n) for g, n in zip(groups, size)}
        start += count
    if not np.isfinite(weights).all() or not (weights > 0).all():
        raise ContractError("Invalid Train pattern weights")
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=len(values), replacement=True, generator=generator
    )
    profile = {
        "fit_split": "train",
        "fit_label": 0,
        "test_used": False,
        "validation_used": False,
        "kind": "KMeans feature clusters, not action classes",
        "n_clusters": k,
        "seed": seed,
        "n_init": 10,
        "max_iter": 300,
        "centers": model.cluster_centers_.tolist(),
        "assignments": labels.tolist(),
        "sequence_cluster_counts": rows,
        "sample_weights": weights.tolist(),
        "rule": "Equal mass per nonempty sequence and per occupied cluster within each sequence",
    }
    return sampler, profile
