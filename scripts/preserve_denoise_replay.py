"""Persist online Train augmentation by deterministic RNG replay, without training.

These tensors are a post-training replay, not tensors captured during the
original forward. Original data, frozen models and final QA remain unchanged.
"""

import numpy as np
import torch
from ur_normal_methods import ROOT, C, E, save, validated_index, verify_lock

from fall_detect.io_utils import ContractError, read_json, sha256, write_npz
from fall_detect.training.experiment import normal_matrix, training_sampler
from fall_detect.training.scaler import FeatureScaler


def main():
    verify_lock()
    run = ROOT / "runs/ur-denoise-wide-methods-20261012-001"
    training = read_json(run / "training.json")
    folder = C / "denoise-augmentation-replay"
    if folder.exists():
        raise ContractError("Augmentation replay exists; preserve evidence")
    frozen = read_json(run / "frozen-artifacts.json")
    assert all(sha256(run / n) == d for n, d in frozen.items())
    index = validated_index()
    train, hashes, counts = normal_matrix(index, "train", True)
    assert list(counts) == training["train_sequences"] and len(counts) == 24
    assert hashes == read_json(run / "sources.json")["train_feature_hashes"]
    values = FeatureScaler.load(run / "scaler.npz").transform(train)
    generator = torch.Generator().manual_seed(42)
    sampler = training_sampler(counts, "sequences", generator)
    loader = torch.utils.data.DataLoader(
        torch.arange(len(values)), batch_size=32, sampler=sampler, generator=generator
    )
    noise_generator = torch.Generator(device="cuda").manual_seed(42)
    folder.mkdir()
    write_npz(folder / "clean-train-standardized.npz", clean=values)
    files = {}
    for epoch in range(1, training["epochs"] + 1):
        selected, noisy = [], []
        for rows in loader:
            clean = torch.from_numpy(values[rows.numpy()]).cuda()
            augmented = (
                clean + torch.randn(clean.shape, generator=noise_generator, device="cuda") * 0.1
            )
            selected.extend(rows.tolist())
            noisy.append(augmented.cpu().numpy())
        path = folder / f"epoch-{epoch:03d}.npz"
        write_npz(
            path,
            source_train_rows=np.array(selected, dtype=np.int64),
            noisy_inputs=np.concatenate(noisy),
        )
        files[path.name] = sha256(path)
        if epoch % 10 == 0:
            print(
                f"Persisted Train-only augmentation replay {epoch}/{training['epochs']}; no model forward/optimizer",
                flush=True,
            )
    # Independent repeat of the known two generator streams validates saved row
    # associations and every noisy tensor, without invoking an anomaly model.
    generator2 = torch.Generator().manual_seed(42)
    sampler2 = training_sampler(counts, "sequences", generator2)
    loader2 = torch.utils.data.DataLoader(
        torch.arange(len(values)), batch_size=32, sampler=sampler2, generator=generator2
    )
    noise2 = torch.Generator(device="cuda").manual_seed(42)
    for epoch in range(1, training["epochs"] + 1):
        with np.load(folder / f"epoch-{epoch:03d}.npz", allow_pickle=False) as saved:
            offset = 0
            for rows in loader2:
                clean = torch.from_numpy(values[rows.numpy()]).cuda()
                replay = clean + torch.randn(clean.shape, generator=noise2, device="cuda") * 0.1
                np.testing.assert_array_equal(
                    saved["source_train_rows"][offset : offset + len(rows)], rows.numpy()
                )
                np.testing.assert_array_equal(
                    saved["noisy_inputs"][offset : offset + len(rows)], replay.cpu().numpy()
                )
                offset += len(rows)
            assert offset == 3638
    assert all(sha256(run / n) == d for n, d in frozen.items())
    note = {
        "kind": "post_training_deterministic_online_augmentation_replay",
        "original_forward_capture": False,
        "model_forward_executed": False,
        "optimizer_updates": 0,
        "additional_training_epochs": 0,
        "train_sequences": training["train_sequences"],
        "train_feature_hashes": hashes,
        "train_counts": counts,
        "seed": 42,
        "noise_std": 0.1,
        "epochs_replayed": training["epochs"],
        "rows_per_epoch": len(values),
        "sampling": "same sequence-balanced WeightedRandomSampler and DataLoader generator42; independent CUDA noise generator42",
        "saved_arrays_separate_from_original": True,
        "exact_rng_repeat_verified": True,
        "validation_augmented": False,
        "test_read": False,
        "scaler_sha256": sha256(run / "scaler.npz"),
        "source_sha256": sha256(__file__),
        "feature_recipe_hash": index["recipe_hash"],
        "method_recipe_hash": read_json(run / "method-config.json")["method_recipe_hash"],
        "clean_array_sha256": sha256(folder / "clean-train-standardized.npz"),
        "epoch_array_sha256": files,
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "limitation": "Reproduced from the fixed original code/generator seeds on the same software/device; original minibatch tensors were not logged during training. Do not describe replay as original-forward capture.",
        "frozen_run_unchanged": True,
        "prior_final_qa_sha256": sha256(E / "final-qa.json"),
    }
    save(folder / "provenance.json", note)
    save(E / "augmentation-preservation.json", note)
    print(
        {
            "saved_epochs": 40,
            "saved_noisy_rows": 40 * 3638,
            "additional_training_epochs": 0,
            "all_replay_arrays_verified": True,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
