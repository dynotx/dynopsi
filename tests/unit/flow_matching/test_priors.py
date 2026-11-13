from __future__ import annotations

import pytest
import torch

from dynopsi.flow_matching.priors import (
    DirichletPrior,
    HarmonicPrior,
    IsotropicGaussian,
    ProjectDirichletToSphere,
    UniformSpherePrior,
)


class TestHarmonicPrior:
    def test_sample_shape_no_mask(self):
        prior = HarmonicPrior(num_atoms=1, d=3.8)
        samples = prior.sample(num_samples=4, seq_len=10)
        assert samples.shape == (4, 10, 3)

    def test_sample_shape_backbone(self):
        prior = HarmonicPrior(num_atoms=3, d=1.52)
        samples = prior.sample(num_samples=2, seq_len=6)
        assert samples.shape == (2, 6, 9)

    def test_masked_sample_respects_unmasked(self):
        prior = HarmonicPrior(num_atoms=1, d=3.8)
        unmasked = torch.zeros(3, 5, 3)
        mask = torch.zeros(3, 5, 3)
        mask[:, :3, :] = 1
        out = prior.sample(num_samples=3, seq_len=5, unmasked=unmasked, mask=mask)
        # Positions where mask=0 should equal unmasked (zeros)
        assert torch.allclose(out[:, 3:, :], torch.zeros(3, 2, 3), atol=1e-5)

    def test_samples_are_finite(self):
        prior = HarmonicPrior(num_atoms=3, d=3.8)
        samples = prior.sample(num_samples=8, seq_len=20)
        assert torch.isfinite(samples).all()


class TestDirichletPrior:
    def test_sample_shape(self):
        prior = DirichletPrior(alphabet_size=20, alpha=1.0)
        samples = prior.sample(num_samples=4, seq_len=8)
        assert samples.shape == (4, 8, 20)

    def test_samples_sum_to_one(self):
        prior = DirichletPrior(alphabet_size=20, alpha=1.0)
        samples = prior.sample(num_samples=4, seq_len=8)
        sums = samples.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4, 8), atol=1e-5)

    def test_samples_non_negative(self):
        prior = DirichletPrior(alphabet_size=20, alpha=1.0)
        samples = prior.sample(num_samples=4, seq_len=8)
        assert (samples >= 0).all()

    def test_custom_alpha_list(self):
        alpha = [0.5] * 10 + [2.0] * 10
        prior = DirichletPrior(alphabet_size=20, alpha=alpha)
        samples = prior.sample(num_samples=4, seq_len=5)
        assert samples.shape == (4, 5, 20)

    def test_requires_seq_len(self):
        prior = DirichletPrior(alphabet_size=5, alpha=1.0)
        with pytest.raises(ValueError, match="seq_len"):
            prior.sample(num_samples=2, seq_len=None)

    def test_alpha_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            DirichletPrior(alphabet_size=5, alpha=[1.0, 2.0])

    def test_masked_sample(self):
        prior = DirichletPrior(alphabet_size=4, alpha=1.0)
        unmasked = torch.zeros(2, 6, 4)
        mask = torch.zeros(2, 6, 4)
        mask[:, :3, :] = 1
        out = prior.sample(num_samples=2, seq_len=6, unmasked=unmasked, mask=mask)
        assert torch.allclose(out[:, 3:, :], torch.zeros(2, 3, 4), atol=1e-5)


@pytest.fixture
def test_params_isotropic() -> dict:
    """Common test parameters used across test cases."""
    return {"sample_shape": [6, 7], "num_samples": 10}


@pytest.fixture
def test_params_dirichlet() -> dict:
    """Common test parameters used across test cases."""
    return {"seq_len": 10, "alphabet_size": 33, "alpha": 0.5, "num_samples": 10}


@pytest.fixture
def test_params_uniform_sphere() -> dict:
    """Common test parameters used across test cases."""
    return {"d": 33, "seq_len": 10, "num_samples": 10}


class TestIsotropicGaussian:
    def test_shape_no_mask(self, test_params_isotropic: dict) -> None:
        sample_shape = test_params_isotropic["sample_shape"]
        num_samples = test_params_isotropic["num_samples"]
        prior = IsotropicGaussian(std=1.0, sample_shape=sample_shape[1:])
        samples = prior.sample(num_samples, seq_len=sample_shape[0])
        assert samples.shape == (num_samples, *sample_shape)

    def test_shape_yes_mask(self, test_params_isotropic: dict) -> None:
        sample_shape = test_params_isotropic["sample_shape"]
        num_samples = test_params_isotropic["num_samples"]
        prior = IsotropicGaussian(std=1.0, sample_shape=sample_shape[1:])
        unmasked = torch.randn(num_samples, *sample_shape)
        mask = (torch.rand(num_samples, *sample_shape) < 0.5).to(torch.long)
        samples = prior.sample(num_samples=num_samples, seq_len=sample_shape[0], unmasked=unmasked, mask=mask)
        assert samples.shape == (num_samples, *sample_shape)

    def test_shape_yes_mask_num_samples_of_mask(self, test_params_isotropic: dict) -> None:
        sample_shape = test_params_isotropic["sample_shape"]
        num_samples = test_params_isotropic["num_samples"]
        prior = IsotropicGaussian(std=1.0, sample_shape=sample_shape[1:])
        unmasked = torch.randn((num_samples, *sample_shape))
        mask = (torch.rand((num_samples, *sample_shape)) < 0.5).to(torch.long)
        samples = prior.sample(num_samples=num_samples, seq_len=sample_shape[0], unmasked=unmasked, mask=mask)
        assert samples.shape == (num_samples, *sample_shape)

    def test_masking_unmasked_stays_fixed(self, test_params_isotropic: dict) -> None:
        sample_shape = test_params_isotropic["sample_shape"]
        num_samples = test_params_isotropic["num_samples"]
        prior = IsotropicGaussian(std=1.0, sample_shape=sample_shape[1:])
        unmasked = torch.randn(num_samples, *sample_shape)
        mask = (torch.rand(num_samples, *sample_shape) < 0.5).to(torch.long)
        samples = prior.sample(num_samples=num_samples, seq_len=sample_shape[0], unmasked=unmasked, mask=mask)
        assert torch.allclose(samples * (1 - mask), unmasked * (1 - mask))

    def test_masking_masked_changes(self, test_params_isotropic: dict) -> None:
        sample_shape = test_params_isotropic["sample_shape"]
        num_samples = test_params_isotropic["num_samples"]
        prior = IsotropicGaussian(std=1.0, sample_shape=sample_shape[1:])
        unmasked = torch.randn(num_samples, *sample_shape)
        mask = torch.ones(num_samples, *sample_shape)
        samples = prior.sample(num_samples=num_samples, seq_len=sample_shape[0], unmasked=unmasked, mask=mask)
        assert not torch.allclose(samples * mask, unmasked * mask)


class TestUniformSpherePrior:
    @pytest.fixture(autouse=True)
    def _need_geoopt(self) -> None:
        pytest.importorskip("geoopt")

    def test_shape_no_mask(self, test_params_uniform_sphere: dict) -> None:
        d = test_params_uniform_sphere["d"]
        seq_len = test_params_uniform_sphere["seq_len"]
        num_samples = test_params_uniform_sphere["num_samples"]
        prior = UniformSpherePrior(d=d)
        samples = prior.sample(num_samples, seq_len)
        assert samples.shape == (num_samples, seq_len, d)

    def test_shape_yes_mask(self, test_params_uniform_sphere: dict) -> None:
        d = test_params_uniform_sphere["d"]
        seq_len = test_params_uniform_sphere["seq_len"]
        num_samples = test_params_uniform_sphere["num_samples"]
        prior = UniformSpherePrior(d=d)
        unmasked = torch.randn(num_samples, seq_len, d)
        mask = (torch.rand(num_samples, seq_len, d) < 0.5).to(torch.long)
        samples = prior.sample(num_samples, seq_len, unmasked, mask)
        assert samples.shape == (num_samples, seq_len, d)

    def test_shape_yes_mask_num_samples_of_mask(self, test_params_uniform_sphere: dict) -> None:
        d = test_params_uniform_sphere["d"]
        seq_len = test_params_uniform_sphere["seq_len"]
        num_samples = test_params_uniform_sphere["num_samples"]
        prior = UniformSpherePrior(d=d)
        unmasked = torch.randn((num_samples, seq_len, d))
        mask = (torch.rand((num_samples, seq_len, d)) < 0.5).to(torch.long)
        samples = prior.sample(num_samples, seq_len, unmasked, mask)
        assert samples.shape == (num_samples, seq_len, d)

    def test_masking_unmasked_stays_fixed(self, test_params_uniform_sphere: dict) -> None:
        d = test_params_uniform_sphere["d"]
        seq_len = test_params_uniform_sphere["seq_len"]
        num_samples = test_params_uniform_sphere["num_samples"]
        prior = UniformSpherePrior(d=d)
        unmasked = torch.randn(num_samples, seq_len, d)
        mask = (torch.rand(num_samples, seq_len, d) < 0.5).to(torch.long)
        samples = prior.sample(num_samples, seq_len, unmasked, mask)
        assert torch.allclose(samples * (1 - mask), unmasked * (1 - mask))

    def test_masking_masked_changes(self, test_params_uniform_sphere: dict) -> None:
        d = test_params_uniform_sphere["d"]
        seq_len = test_params_uniform_sphere["seq_len"]
        num_samples = test_params_uniform_sphere["num_samples"]
        prior = UniformSpherePrior(d=d)
        unmasked = torch.randn(num_samples, seq_len, d)
        mask = torch.ones(num_samples, seq_len, d)
        samples = prior.sample(num_samples, seq_len, unmasked, mask)
        assert not torch.allclose(samples * mask, unmasked * mask)

    def test_sample_is_on_sphere(self, test_params_uniform_sphere: dict) -> None:
        d = test_params_uniform_sphere["d"]
        seq_len = test_params_uniform_sphere["seq_len"]
        num_samples = test_params_uniform_sphere["num_samples"]
        prior = UniformSpherePrior(d=d)
        samples = prior.sample(num_samples, seq_len)
        assert torch.allclose(torch.norm(samples, dim=-1), torch.ones(num_samples, seq_len))


class TestProjectDirichletToSphere:
    def test_sample_is_on_sphere(self, test_params_dirichlet: dict) -> None:
        seq_len = test_params_dirichlet["seq_len"]
        alphabet_size = test_params_dirichlet["alphabet_size"]
        alpha = test_params_dirichlet["alpha"]
        num_samples = test_params_dirichlet["num_samples"]
        prior = ProjectDirichletToSphere(alphabet_size, alpha)
        samples = prior.sample(num_samples, seq_len)
        assert torch.allclose(torch.norm(samples, dim=-1), torch.ones(num_samples, seq_len))
