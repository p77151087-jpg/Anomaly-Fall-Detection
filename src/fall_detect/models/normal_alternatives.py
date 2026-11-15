"""Normal-data alternatives, separate from the historical frozen AE contract."""

from collections import deque
from dataclasses import replace

import numpy as np
import torch
from sklearn.svm import OneClassSVM
from torch import nn

from ..contracts import FeatureCache
from ..io_utils import ContractError


def svm_scores(model, values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ContractError("SVM requires finite 2D features")
    scores = -model.decision_function(values).reshape(-1)
    if not np.isfinite(scores).all():
        raise ContractError("Non-finite SVM anomaly score")
    return scores


def fit_svm(train, nu, gamma):
    train = np.asarray(train, dtype=np.float64)
    if not len(train) or train.ndim != 2 or not np.isfinite(train).all():
        raise ContractError("SVM needs nonempty finite Train normal features")
    return OneClassSVM(
        kernel="rbf", nu=nu, gamma=gamma, tol=0.001, shrinking=True, cache_size=512, max_iter=-1
    ).fit(train)


def temporal_inputs(cache, steps=8):
    """Causal histories: reset on any invalid endpoint, gap or track change.

    Returns unscaled original feature sequences and a cache that preserves ALL
    scheduled endpoints. Warm-up endpoints become unavailable, never vanish.
    """
    if steps < 2:
        raise ContractError("Temporal histories need at least two observations")
    histories, last, records, tensors, provenance = {}, {}, [], [], []
    identities = set()
    for r in sorted(cache.records, key=lambda row: (row.sample_index, row.track_id or -1)):
        if r.sequence_id != cache.metadata["sequence_id"]:
            raise ContractError("Cannot concatenate different videos")
        key = (r.sequence_id, r.track_id)
        ident = (r.sample_index, r.track_id)
        if ident in identities or r.available_at_ms < r.end_ms:
            raise ContractError("Duplicate or noncausal temporal endpoint")
        identities.add(ident)
        history = histories.setdefault(key, deque(maxlen=steps))
        previous = last.get(key)
        if previous is not None:
            if (
                r.sample_index <= previous.sample_index
                or r.available_at_ms <= previous.available_at_ms
            ):
                raise ContractError("Nonmonotonic temporal input")
            if r.sample_index != previous.sample_index + cache.metadata["stride"]:
                history.clear()
        last[key] = r
        if r.row_index is None or r.track_id is None:
            history.clear()
            records.append(replace(r, row_index=None))
            continue
        history.append(r)
        if len(history) < steps:
            records.append(replace(r, row_index=None, quality="temporal_history_warmup"))
            continue
        rows = [h.row_index for h in history]
        if any(h.track_id != r.track_id or h.available_at_ms > r.available_at_ms for h in history):
            raise ContractError("Temporal input crosses track or uses the future")
        tensors.append(cache.features[rows])
        provenance.append(
            {
                "sequence_id": r.sequence_id,
                "track_id": r.track_id,
                "target_sample_index": r.sample_index,
                "input_sample_indices": [h.sample_index for h in history],
                "input_row_indices": rows,
            }
        )
        records.append(replace(r, row_index=len(tensors) - 1))
    dim = cache.features.shape[1]
    inputs = np.asarray(tensors, dtype=np.float32).reshape(-1, steps, dim)
    if not np.isfinite(inputs).all():
        raise ContractError("Non-finite temporal inputs")
    # The event engine uses row count, records and scores, not these placeholder
    # current features. Do not serialize this as an ordinary ST-GCN cache.
    out = FeatureCache(
        np.asarray([t[-1] for t in inputs], dtype=np.float32).reshape(-1, dim),
        records,
        {**cache.metadata, "temporal_steps": steps},
    )
    return inputs, out, provenance


class TemporalAutoencoder(nn.Module):
    """GRU sequence reconstruction; all inputs precede score availability."""

    def __init__(self, dim=512, hidden=128, latent=64, steps=8):
        super().__init__()
        self.steps = steps
        self.embedding = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU())
        self.encoder = nn.GRU(hidden, latent, batch_first=True)
        self.decoder = nn.GRU(latent + 1, hidden, batch_first=True)
        self.output = nn.Linear(hidden, dim)

    def forward(self, values):
        if values.ndim != 3 or values.shape[1] != self.steps:
            raise ContractError("Temporal AE input shape mismatch")
        _, state = self.encoder(self.embedding(values))
        context = state[-1].unsqueeze(1).expand(-1, self.steps, -1)
        position = torch.linspace(0, 1, self.steps, device=values.device, dtype=values.dtype).view(
            1, self.steps, 1
        )
        decoded, _ = self.decoder(torch.cat((context, position.expand(len(values), -1, -1)), -1))
        return self.output(decoded)


def candidate_rank(metrics, order):
    """Unknown remains in total Accuracy; fixed order is the final tie breaker."""
    return (
        metrics["operational_accuracy"],
        metrics["sequence_fall_f1"],
        metrics["end_to_end_fall_recall"],
        -metrics["FP"],
        -order,
    )


def calibrate_normal_scores(rows):
    """Complete Validation threshold sweep, strict >, same candidate ranking."""
    from ..evaluation.metrics import sequence_metrics

    values = np.unique([r["sequence_score"] for r in rows if r["sequence_score"] is not None])
    if not len(values) or not np.isfinite(values).all():
        raise ContractError("No finite Validation event scores")
    if any(
        not any(r["label"] == label and r["sequence_score"] is not None for r in rows)
        for label in (0, 1)
    ):
        raise ContractError("Calibration requires both evaluable Validation classes")
    thresholds = [np.nextafter(float(values[0]), -np.inf)] + values.tolist()
    if not np.isfinite(thresholds).all():
        raise ContractError("Non-finite threshold")
    candidates = []
    for threshold in thresholds:
        predictions = [
            {
                **r,
                "prediction": int(r["sequence_score"] > threshold)
                if r["sequence_score"] is not None
                else None,
            }
            for r in rows
        ]
        candidates.append({"threshold": float(threshold), "metrics": sequence_metrics(predictions)})
    chosen = max(candidates, key=lambda c: (*candidate_rank(c["metrics"], 0), c["threshold"]))
    return chosen, candidates
