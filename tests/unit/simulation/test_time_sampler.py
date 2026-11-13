from collections.abc import Callable

import pytest
import torch
from pydantic import ValidationError

from dynopsi.simulation.time_sampler import LinearTimeSampler, LogTimeSampler, QuadraticTimeSampler, TimeSamplerType


class TimeSamplerContractTests:
    """Contract tests that must pass for every TimeSampler subclass.

    Subclasses must override the `make_sampler` fixture to return a factory
    callable that accepts at least (num_steps, modalities) and returns a
    concrete TimeSampler instance.
    """

    @pytest.fixture
    def make_sampler(self) -> Callable:
        raise NotImplementedError

    @pytest.mark.parametrize("num_steps", [1, 5, 10, 100])
    def test_get_steps(self, num_steps: int, make_sampler: Callable) -> None:
        sampler = make_sampler(num_steps=num_steps)
        steps = sampler.get_steps()
        assert isinstance(steps, range)
        steplist = list(steps)
        assert len(steplist) == num_steps
        assert steplist == list(range(num_steps))

    def test_get_full_schedule_shapes(self, make_sampler: Callable) -> None:
        num_steps = 10
        modalities = ("coords",)
        sampler = make_sampler(num_steps=num_steps, modalities=modalities)
        ts, dts = sampler.get_full_schedule(torch.device("cpu"))

        expected_shape = (1, num_steps, 1, 1)
        for modality in modalities:
            assert modality in ts, f"Missing modality {modality} in ts"
            assert modality in dts, f"Missing modality {modality} in dts"
            assert ts[modality].shape == expected_shape, (
                f"ts[{modality}] has shape {ts[modality].shape}, expected {expected_shape}"
            )
            assert dts[modality].shape == expected_shape, (
                f"dts[{modality}] has shape {dts[modality].shape}, expected {expected_shape}"
            )

    def test_get_full_schedule_modality_keys(self, make_sampler: Callable) -> None:
        modalities = ("coords",)
        sampler = make_sampler(num_steps=5, modalities=modalities)
        ts, dts = sampler.get_full_schedule(torch.device("cpu"))
        assert set(ts.keys()) == set(modalities)
        assert set(dts.keys()) == set(modalities)

    def test_get_full_schedule_modalities_are_independent_tensors(self, make_sampler: Callable) -> None:
        modalities = ("coords", "sequence")
        sampler = make_sampler(num_steps=5, modalities=modalities)
        ts, dts = sampler.get_full_schedule(torch.device("cpu"))

        ts["coords"][0, 0, 0, 0] = 999.0
        assert ts["sequence"][0, 0, 0, 0] != 999.0, "Modality tensors should be independent (cloned)"


class TestLinearTimeSampler(TimeSamplerContractTests):
    @pytest.fixture
    def make_sampler(self) -> Callable:
        def factory(num_steps: int = 5, modalities: tuple[str, ...] = ("coords",), **kwargs) -> LinearTimeSampler:
            return LinearTimeSampler(num_steps=num_steps, modalities=modalities, **kwargs)

        return factory

    @pytest.mark.parametrize(
        ("num_steps", "start_time", "end_time", "modalities"),
        [
            (10, 0.0, 1.0, ("coords",)),
            (1, 0.0, 1.0, ("coords",)),
            (5, 0.5, 1.0, ("coords", "sequence")),
        ],
        ids=["standard", "single_step", "multi_modality"],
    )
    def test_instantiation_valid(
        self, num_steps: int, start_time: float, end_time: float, modalities: tuple[str, ...]
    ) -> None:
        sampler = LinearTimeSampler(
            num_steps=num_steps, start_time=start_time, end_time=end_time, modalities=modalities
        )
        assert sampler.num_steps == num_steps
        assert sampler.start_time == start_time
        assert sampler.end_time == end_time
        assert sampler.modalities == modalities

    @pytest.mark.parametrize(
        "invalid_kwargs",
        [
            {"num_steps": 0, "start_time": 0.0, "end_time": 1.0, "modalities": ("coords",)},
            {"num_steps": -5, "start_time": 0.0, "end_time": 1.0, "modalities": ("coords",)},
            {"num_steps": 10, "start_time": 1.0, "end_time": 0.0, "modalities": ("coords",)},
            {"num_steps": 10, "start_time": 0.5, "end_time": 0.5, "modalities": ("coords",)},
            {"num_steps": 10, "start_time": 0.0, "end_time": 1.0, "modalities": ()},
        ],
        ids=[
            "num_steps_zero",
            "num_steps_negative",
            "start_after_end",
            "start_equals_end",
            "empty_modalities",
        ],
    )
    def test_instantiation_invalid(self, invalid_kwargs: dict) -> None:
        with pytest.raises(ValidationError):
            LinearTimeSampler(**invalid_kwargs)

    def test_get_full_schedule_timestep_values(self) -> None:
        sampler = LinearTimeSampler(num_steps=4, start_time=0.0, end_time=1.0, modalities=("coords",))
        ts, dts = sampler.get_full_schedule(torch.device("cpu"))

        timesteps = ts["coords"].squeeze()
        expected = torch.tensor([0.0, 0.25, 0.5, 0.75])
        assert torch.allclose(timesteps, expected, atol=1e-6), f"Timesteps {timesteps} do not match expected {expected}"

    def test_get_full_schedule_constant_step_sizes(self) -> None:
        sampler = LinearTimeSampler(num_steps=5, start_time=0.0, end_time=1.0, modalities=("coords",))
        ts, dts = sampler.get_full_schedule(torch.device("cpu"))

        step_sizes = dts["coords"].squeeze()
        expected_dt = 0.2
        assert torch.allclose(step_sizes, torch.full_like(step_sizes, expected_dt), atol=1e-6)

    @pytest.mark.parametrize("start_time, end_time", [(-1.0, 1.0), (1.0, 4.0)])
    def test_get_full_schedule_non_unit_range(self, start_time: float, end_time: float) -> None:
        with pytest.raises(ValidationError):
            LinearTimeSampler(num_steps=3, start_time=start_time, end_time=end_time, modalities=("coords",))

    def test_type_discriminator(self) -> None:
        sampler = LinearTimeSampler(num_steps=5, modalities=("coords",))
        assert sampler.type == "linear_time_sampler"


class TestLogTimeSampler(TimeSamplerContractTests):
    @pytest.fixture
    def make_sampler(self) -> Callable:
        def factory(
            num_steps: int = 5, modalities: tuple[str, ...] = ("coords",), p: float = 1.0, **kwargs
        ) -> LogTimeSampler:
            return LogTimeSampler(num_steps=num_steps, modalities=modalities, p=p, **kwargs)

        return factory

    @pytest.mark.parametrize(
        ("num_steps", "p", "modalities"),
        [
            (10, 1.0, ("coords",)),
            (1, 0.1, ("coords",)),
            (5, 10.0, ("coords", "sequence")),
            (50, 0.001, ("coords",)),
        ],
        ids=["standard", "single_step", "multi_modality", "small_p"],
    )
    def test_instantiation_valid(self, num_steps: int, p: float, modalities: tuple[str, ...]) -> None:
        sampler = LogTimeSampler(num_steps=num_steps, p=p, modalities=modalities)
        assert sampler.num_steps == num_steps
        assert sampler.p == p
        assert sampler.modalities == modalities

    @pytest.mark.parametrize(
        "invalid_kwargs",
        [
            {"num_steps": 10, "p": 0.0, "modalities": ("coords",)},
            {"num_steps": 10, "p": -1.0, "modalities": ("coords",)},
            {"num_steps": 0, "p": 1.0, "modalities": ("coords",)},
        ],
        ids=["p_zero", "p_negative", "num_steps_zero"],
    )
    def test_instantiation_invalid(self, invalid_kwargs: dict) -> None:
        with pytest.raises(ValidationError):
            LogTimeSampler(**invalid_kwargs)

    def test_get_full_schedule_normalized_range(self) -> None:
        sampler = LogTimeSampler(num_steps=10, p=1.0, modalities=("coords",))
        ts, dts = sampler.get_full_schedule(torch.device("cpu"))

        timesteps = ts["coords"].squeeze()
        assert timesteps[0].item() == pytest.approx(0.0, abs=1e-6), "First timestep should be 0"
        assert timesteps[-1].item() < 1.0, "Last timestep should be less than 1 (end is excluded)"

    def test_get_full_schedule_step_sizes_positive(self) -> None:
        sampler = LogTimeSampler(num_steps=10, p=2.0, modalities=("coords",))
        ts, dts = sampler.get_full_schedule(torch.device("cpu"))

        step_sizes = dts["coords"].squeeze()
        assert (step_sizes > 0).all(), "All step sizes should be positive"

    def test_get_full_schedule_step_sizes_sum_to_one(self) -> None:
        sampler = LogTimeSampler(num_steps=20, p=1.0, modalities=("coords",))
        ts, dts = sampler.get_full_schedule(torch.device("cpu"))

        total = dts["coords"].sum().item()
        assert total == pytest.approx(1.0, abs=1e-5), f"Step sizes should sum to 1, got {total}"

    @pytest.mark.parametrize("p", [0.5, 1.0, 5.0, 10.0])
    def test_get_full_schedule_non_uniform_steps(self, p: float) -> None:
        sampler = LogTimeSampler(num_steps=20, p=p, modalities=("coords",))
        ts, dts = sampler.get_full_schedule(torch.device("cpu"))

        step_sizes = dts["coords"].squeeze()
        assert not torch.allclose(step_sizes, step_sizes[0].expand_as(step_sizes), atol=1e-4), (
            f"LogTimeSampler with p={p} should produce non-uniform step sizes"
        )

    def test_type_discriminator(self) -> None:
        sampler = LogTimeSampler(num_steps=5, p=1.0, modalities=("coords",))
        assert sampler.type == "log_time_sampler"


class TestQuadraticTimeSampler(TimeSamplerContractTests):
    @pytest.fixture
    def make_sampler(self) -> Callable:
        def factory(num_steps: int = 5, modalities: tuple[str, ...] = ("coords",), **kwargs) -> QuadraticTimeSampler:
            return QuadraticTimeSampler(num_steps=num_steps, modalities=modalities, **kwargs)

        return factory

    @pytest.mark.parametrize(
        ("num_steps", "modalities"),
        [
            (10, ("coords",)),
            (1, ("coords",)),
            (5, ("coords", "sequence")),
        ],
        ids=["standard", "single_step", "multi_modality"],
    )
    def test_instantiation_valid(self, num_steps: int, modalities: tuple[str, ...]) -> None:
        sampler = QuadraticTimeSampler(num_steps=num_steps, modalities=modalities)
        assert sampler.num_steps == num_steps
        assert sampler.modalities == modalities

    @pytest.mark.parametrize(
        "invalid_kwargs",
        [
            {"num_steps": 0, "modalities": ("coords",)},
            {"num_steps": -3, "modalities": ("coords",)},
        ],
        ids=["num_steps_zero", "num_steps_negative"],
    )
    def test_instantiation_invalid(self, invalid_kwargs: dict) -> None:
        with pytest.raises(ValidationError):
            QuadraticTimeSampler(**invalid_kwargs)

    def test_get_full_schedule_timestep_values(self) -> None:
        sampler = QuadraticTimeSampler(num_steps=4, modalities=("coords",))
        ts, _ = sampler.get_full_schedule(torch.device("cpu"))

        timesteps = ts["coords"].squeeze()
        # f(x) = (x/4)^2 for x in {0,1,2,3} → {0, 0.0625, 0.25, 0.5625}
        expected = torch.tensor([0.0, 0.0625, 0.25, 0.5625])
        assert torch.allclose(timesteps, expected, atol=1e-6), f"Timesteps {timesteps} do not match expected {expected}"

    def test_get_full_schedule_step_sizes_sum_to_one(self) -> None:
        sampler = QuadraticTimeSampler(num_steps=20, modalities=("coords",))
        _, dts = sampler.get_full_schedule(torch.device("cpu"))

        total = dts["coords"].sum().item()
        assert total == pytest.approx(1.0, abs=1e-5), f"Step sizes should sum to 1, got {total}"

    def test_get_full_schedule_step_sizes_monotonically_increasing(self) -> None:
        sampler = QuadraticTimeSampler(num_steps=10, modalities=("coords",))
        _, dts = sampler.get_full_schedule(torch.device("cpu"))

        step_sizes = dts["coords"].squeeze()
        assert (step_sizes[1:] >= step_sizes[:-1]).all(), "Quadratic step sizes should be monotonically non-decreasing"

    def test_get_full_schedule_denser_near_zero(self) -> None:
        sampler = QuadraticTimeSampler(num_steps=10, modalities=("coords",))
        _, dts = sampler.get_full_schedule(torch.device("cpu"))

        step_sizes = dts["coords"].squeeze()
        first_half = step_sizes[: len(step_sizes) // 2].sum().item()
        second_half = step_sizes[len(step_sizes) // 2 :].sum().item()
        assert first_half < second_half, "Quadratic sampler should concentrate smaller steps near t=0"

    def test_type_discriminator(self) -> None:
        sampler = QuadraticTimeSampler(num_steps=5, modalities=("coords",))
        assert sampler.type == "quadratic_time_sampler"


class TestTimeSamplerTypeDiscriminator:
    @pytest.mark.parametrize(
        ("config", "expected_type"),
        [
            (
                {"type": "linear_time_sampler", "num_steps": 10, "modalities": ["coords"]},
                LinearTimeSampler,
            ),
            (
                {"type": "log_time_sampler", "num_steps": 10, "p": 1.0, "modalities": ["coords"]},
                LogTimeSampler,
            ),
            (
                {"type": "quadratic_time_sampler", "num_steps": 10, "modalities": ["coords"]},
                QuadraticTimeSampler,
            ),
        ],
        ids=["linear", "log", "quadratic"],
    )
    def test_discriminated_union_deserializes_correct_type(self, config: dict, expected_type: type) -> None:
        from pydantic import TypeAdapter

        adapter = TypeAdapter(TimeSamplerType)
        sampler = adapter.validate_python(config)
        assert isinstance(sampler, expected_type)
