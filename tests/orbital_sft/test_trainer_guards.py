import torch
import pytest

from swarmshield.posttraining.trainer import validate_gradients


def test_trainable_adapter_gradient_must_be_finite():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([float("nan")])
    with pytest.raises(ValueError, match="nonfinite gradient"):
        validate_gradients([("layer.lora_B", parameter)])


def test_finite_trainable_adapter_gradient_is_allowed():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([0.5])
    validate_gradients([("layer.lora_B", parameter)])


def test_frozen_parameters_must_not_receive_gradients():
    parameter = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
    parameter.grad = torch.tensor([0.5])
    with pytest.raises(ValueError, match="Frozen parameter"):
        validate_gradients([("layer.base", parameter)])
