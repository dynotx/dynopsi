from unittest.mock import MagicMock

import pytest
import torch

from dynopsi.data.featurization import Features
from dynopsi.flow_matching.interpolant import EuclideanInterpolant, Interpolant, LinearInterpolantSchedule
from dynopsi.flow_matching.priors import IsotropicGaussian, Sampleable
from dynopsi.model.model_config import DynoPsiModelConfig
from dynopsi.simulation.store import Store
from dynopsi.simulation.time_sampler import LinearTimeSampler


@pytest.fixture
def priors() -> dict[str, Sampleable]:
    return {"coords": IsotropicGaussian(std=1.0, sample_shape=None)}


@pytest.fixture
def interpolants() -> dict[str, Interpolant]:
    return {"coords": EuclideanInterpolant(interpolant_schedule=LinearInterpolantSchedule())}


@pytest.fixture
def inference_model_config(priors: dict[str, Sampleable], interpolants: dict[str, Interpolant]) -> DynoPsiModelConfig:
    mock_model = MagicMock()
    mock_model.priors = priors
    mock_model.interpolants = interpolants

    def model_side_effect(x_in, feature_dict, times):
        """Mock the model's forward pass. Returns x_in + times["coords"]"""
        required_keys = {"coords"}
        feature_dict_keys = set(feature_dict.keys())
        assert required_keys.issubset(feature_dict_keys), (
            f"Missing keys in feature_dict: {required_keys - feature_dict_keys}"
        )
        # just return x_in + times["coords"]
        output = {k: v.clone() for k, v in x_in.items()}
        output["coords"] = output["coords"] + times["coords"]
        return output

    mock_model.side_effect = model_side_effect

    def infer_cvf(x_in, feature_dict, times, self_conditioning_inputs=None):
        output = model_side_effect(x_in, feature_dict, times)
        return output, {k: None for k in output}

    mock_model.infer_cvf.side_effect = infer_cvf

    # construct a DynoPsiModelConfig without any validation
    # because we only care about the priors and interpolants
    config = DynoPsiModelConfig.model_construct(dynopsi_model=mock_model)
    config._dynopsi_model = mock_model
    return config


@pytest.fixture
def num_timesteps() -> int:
    return 3


@pytest.fixture
def seq_len() -> int:
    return 10


@pytest.fixture
def batch_size() -> int:
    return 5


@pytest.fixture
def x_tensor(batch_size: int, seq_len: int) -> torch.Tensor:
    return torch.ones(batch_size, seq_len, 9)


@pytest.fixture
def cvf_tensor(batch_size: int, seq_len: int) -> torch.Tensor:
    return torch.full((batch_size, seq_len, 9), 2.0)


@pytest.fixture
def t_tensor(batch_size: int) -> torch.Tensor:
    return torch.full((batch_size, 1, 1), 0.5)


def make_store(features: dict[str, Features], num_timesteps: int, batch_size: int | None = None) -> Store:
    """Make a simple store with the input features and a linear time schedule that has 3 steps."""
    time_sampler = LinearTimeSampler(num_steps=num_timesteps)
    time_schedule = time_sampler.get_full_schedule(torch.device("cpu"))
    if batch_size is None:
        return Store(features=features, time_schedule=time_schedule)
    else:
        return Store(features=features, time_schedule=time_schedule, batch_slice=slice(0, batch_size))
