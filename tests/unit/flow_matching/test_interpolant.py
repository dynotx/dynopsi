import numpy as np
import pytest
import torch

from dynopsi.flow_matching.interpolant import (
    EuclideanInterpolant,
    FisherInterpolant,
    LinearInterpolantSchedule,
)

test_params_dict = {
    "batch_size": 1,
    "L": 10,
    "d": 3,
}


def random_rotation_matrix(has_positive_determinant: bool = True) -> torch.Tensor:
    A = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(A)
    if has_positive_determinant and torch.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    elif not has_positive_determinant and torch.linalg.det(Q) > 0:
        Q[:, -1] *= -1
    return Q


@pytest.fixture
def test_params() -> dict:
    """Common test parameters used across test cases."""
    return test_params_dict


class TestLinearEuclideanInterpolant:
    def _sample_x(self, test_params: dict) -> torch.Tensor:
        return torch.randn(test_params["batch_size"], test_params["L"], test_params["d"])

    def _sample_t(self, test_params: dict) -> torch.Tensor:
        return torch.rand(test_params["batch_size"], 1, 1)

    def test_interpolate_shape(self, test_params: dict) -> None:
        x0 = self._sample_x(test_params)
        x1 = self._sample_x(test_params)
        t = self._sample_t(test_params)
        interpolant = EuclideanInterpolant(LinearInterpolantSchedule())
        xt = interpolant.interpolate(x0, x1, t)
        assert xt.shape == (test_params["batch_size"], test_params["L"], test_params["d"])

    def test_interpolate_value(self, test_params: dict) -> None:
        x0 = self._sample_x(test_params)
        x1 = self._sample_x(test_params)
        t = self._sample_t(test_params)
        interpolant = EuclideanInterpolant(LinearInterpolantSchedule())
        xt = interpolant.interpolate(x0, x1, t)
        xt_expected = x0 * (1 - t) + x1 * t
        assert torch.allclose(xt, xt_expected, atol=1e-5)

    def test_interpolate_boundary_x0(self, test_params: dict) -> None:
        x0 = self._sample_x(test_params)
        x1 = self._sample_x(test_params)
        t = torch.zeros(test_params["batch_size"], 1, 1)
        interpolant = EuclideanInterpolant(LinearInterpolantSchedule())
        xt = interpolant.interpolate(x0, x1, t)
        assert torch.allclose(xt, x0, atol=1e-5)

    def test_interpolate_boundary_x1(self, test_params: dict) -> None:
        x0 = self._sample_x(test_params)
        x1 = self._sample_x(test_params)
        t = torch.ones(test_params["batch_size"], 1, 1)
        interpolant = EuclideanInterpolant(LinearInterpolantSchedule())
        xt = interpolant.interpolate(x0, x1, t)
        assert torch.allclose(xt, x1, atol=1e-5)

    def test_conditional_vector_field_shape(self, test_params: dict) -> None:
        xt = self._sample_x(test_params)
        x1 = self._sample_x(test_params)
        t = self._sample_t(test_params)
        interpolant = EuclideanInterpolant(LinearInterpolantSchedule())
        ut = interpolant.conditional_vector_field(xt, x1, t)
        assert ut.shape == (test_params["batch_size"], test_params["L"], test_params["d"])

    def test_conditional_vector_field_value(self, test_params: dict) -> None:
        xt = self._sample_x(test_params)
        x1 = self._sample_x(test_params)
        # Ensure max time of 0.9 to avoid impact of torch.clamp on final value for u_t with xt
        t = torch.clamp(self._sample_t(test_params), min=0, max=0.9)
        interpolant = EuclideanInterpolant(LinearInterpolantSchedule())
        ut = interpolant.conditional_vector_field(xt, x1, t)
        ut_expected = (x1 - xt) / (1 - t)
        assert torch.allclose(ut, ut_expected, atol=1e-5)

    def test_recovering_x1_via_vector_field(self, test_params: dict) -> None:
        x0 = self._sample_x(test_params)
        x1 = self._sample_x(test_params)
        # Ensure max time of 0.9 to avoid impact of torch.clamp on final value for u_t with xt
        t = torch.clamp(self._sample_t(test_params), min=0, max=0.9)
        interpolant = EuclideanInterpolant(LinearInterpolantSchedule())
        xt = interpolant.interpolate(x0, x1, t)
        ut = interpolant.conditional_vector_field(xt, x1, t)
        x1_again = EuclideanInterpolant.apply_conditional_vector_field(xt, ut, 1 - t)
        assert torch.allclose(x1, x1_again, atol=1e-5)

    def test_conditional_vector_field_value_with_x0(self, test_params: dict) -> None:
        x0 = self._sample_x(test_params)
        x1 = self._sample_x(test_params)
        # Ensure max time of 0.9 to avoid impact of torch.clamp on final value for u_t with xt
        t = torch.clamp(self._sample_t(test_params), min=0, max=0.9)
        xt = t * x1 + (1 - t) * x0
        interpolant = EuclideanInterpolant(LinearInterpolantSchedule())
        ut_xt = interpolant.conditional_vector_field(xt, x1, t)
        ut_x0 = interpolant.conditional_vector_field(xt, x1, t, x0)
        ut_expected = x1 - x0
        assert torch.allclose(ut_x0, ut_expected, atol=1e-5)
        assert torch.allclose(ut_xt, ut_x0, atol=1e-5)


class TestFisherInterpolant:
    @pytest.fixture(autouse=True)
    def _need_geoopt(self) -> None:
        pytest.importorskip("geoopt")

    def _sphere(self):
        import geoopt

        return geoopt.Sphere()

    def _sample_x(self, test_params: dict) -> torch.Tensor:
        torch.manual_seed(0)
        sphere = self._sphere()
        return sphere.random_uniform(test_params["batch_size"], test_params["L"], test_params["d"]).abs()

    def _sample_t(self, test_params: dict) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.rand(test_params["batch_size"], 1, 1)

    def test_interpolate_shape(self, test_params: dict) -> None:
        x0 = self._sample_x(test_params)
        x1 = self._sample_x(test_params)
        t = self._sample_t(test_params)
        interpolant = FisherInterpolant()
        xt = interpolant.interpolate(x0, x1, t)
        assert xt.shape == (test_params["batch_size"], test_params["L"], test_params["d"])

    def test_interpolate_value(self, test_params: dict) -> None:
        x0 = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(0)
        x1 = torch.tensor([0.0, 0.0, 1.0]).unsqueeze(0)
        t = torch.tensor([0.5]).unsqueeze(0).unsqueeze(0)
        interpolant = FisherInterpolant()
        xt = interpolant.interpolate(x0, x1, t)
        expected_xt = torch.tensor([1 / np.sqrt(2.0), 0.0, 1 / np.sqrt(2.0)]).unsqueeze(0).float()
        assert torch.allclose(xt, expected_xt, atol=1e-5)

    def test_conditional_vector_field_shape(self, test_params: dict) -> None:
        xt = self._sample_x(test_params)
        x1 = self._sample_x(test_params)
        t = self._sample_t(test_params)
        interpolant = FisherInterpolant()
        ut = interpolant.conditional_vector_field(xt, x1, t)
        assert ut.shape == (test_params["batch_size"], test_params["L"], test_params["d"])

    def test_recovering_x1_via_vector_field(self, test_params: dict) -> None:
        x0 = self._sample_x(test_params).to(torch.float64)
        x1 = self._sample_x(test_params).to(torch.float64)
        # Ensure max time of 0.9 to avoid impact of torch.clamp on final value for u_t with xt
        t = torch.clamp(self._sample_t(test_params), min=0, max=0.9)
        interpolant = FisherInterpolant()
        xt = interpolant.interpolate(x0, x1, t)
        ut = interpolant.conditional_vector_field(xt, x1, t)
        x1_again = interpolant.apply_conditional_vector_field(xt, ut, 1 - t)
        assert torch.allclose(x1, x1_again, atol=1e-5), (
            f"torch.max(torch.abs(x1 - x1_again)) = {torch.max(torch.abs(x1 - x1_again))}"
        )
