from copy import deepcopy

import pytest
import torch

from fall_detect.config import load_config, recipe_fingerprint, validate
from fall_detect.io_utils import ContractError
from fall_detect.training.experiment import training_reconstruction_loss


def test_noise_preserves_clean_target_and_rng_and_zero_is_legacy():
    clean = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    original = clean.clone()
    identity = torch.nn.Identity()
    rng = torch.Generator().manual_seed(123)
    expected_noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(123))
    global_state = torch.random.get_rng_state().clone()
    loss = training_reconstruction_loss(identity, clean, 0.1, rng)
    # Adding a small perturbation to float32 values up to 23 incurs cancellation error.
    assert loss.item() == pytest.approx((expected_noise * 0.1).square().mean().item(), rel=1e-5)
    assert torch.equal(clean, original)
    assert torch.equal(global_state, torch.random.get_rng_state())
    assert torch.equal(
        loss,
        training_reconstruction_loss(identity, clean, 0.1, torch.Generator().manual_seed(123)),
    )
    model = torch.nn.Linear(8, 8)
    state = torch.random.get_rng_state().clone()
    assert torch.equal(
        training_reconstruction_loss(model, clean),
        torch.nn.functional.mse_loss(model(clean), clean),
    )
    assert torch.equal(state, torch.random.get_rng_state())


def test_regularization_config_is_finite_and_does_not_change_pose_recipe():
    config = load_config()
    original = deepcopy(config)
    for key in ("weight_decay", "input_noise_std"):
        assert config["autoencoder"][key] == 0
        for value in (-1, float("nan"), float("inf"), True):
            invalid = deepcopy(config)
            invalid["autoencoder"][key] = value
            with pytest.raises(ContractError):
                validate(invalid)
        config["autoencoder"][key] = 0.1
    validate(config)
    assert recipe_fingerprint(config, {}) == recipe_fingerprint(original, {})
