import numpy as np
import pytest
import torch

import dynopsi.data.featurization.binder_center as binder_center
from dynopsi.data.constants import PocketFeature
from dynopsi.data.featurization.base import Features
from dynopsi.data.featurization.binder_center import (
    HotspotGeometryUnavailableError,
    HotspotOutwardGeometry,
    compute_binder_center_sphere_exclusion,
    extract_hotspot_outward_geometry,
    sphere_exclusion_center,
    sphere_exclusion_radii,
)
from dynopsi.data.featurization.primitives import new_chains


def _geometry(
    ca: torch.Tensor, hotspot_mask: torch.Tensor, c_H: torch.Tensor, n0: torch.Tensor
) -> HotspotOutwardGeometry:
    return HotspotOutwardGeometry(clash_ca=ca, hotspot_mask=hotspot_mask, c_H=c_H, n0=n0)


def test_sphere_exclusion_radii_rejects_nonpositive_length() -> None:
    with pytest.raises(ValueError, match="binder_length must be positive"):
        sphere_exclusion_radii(0, exclusion_offset=0.0, exclusion_radius=None)


def test_sphere_exclusion_radii_l60_unclipped() -> None:
    d_star, r_excl = sphere_exclusion_radii(60, exclusion_offset=0.0, exclusion_radius=None)
    expected = 2.2 * 60**0.38 + 1.7
    assert d_star == pytest.approx(expected, rel=1e-6)
    assert d_star == pytest.approx(12.13, abs=0.02)
    assert r_excl == d_star


def test_sphere_exclusion_radii_offset_moves_both() -> None:
    d_star, r_excl = sphere_exclusion_radii(60, exclusion_offset=-3.0, exclusion_radius=None)
    coupled, _ = sphere_exclusion_radii(60, exclusion_offset=0.0, exclusion_radius=None)
    assert d_star == pytest.approx(coupled - 3.0)
    assert r_excl == d_star


def test_sphere_exclusion_radii_constant_wall_ignores_length() -> None:
    d_star_60, r_60 = sphere_exclusion_radii(60, exclusion_offset=0.0, exclusion_radius=5.0)
    d_star_80, r_80 = sphere_exclusion_radii(80, exclusion_offset=1.0, exclusion_radius=5.0)
    assert r_60 == 5.0
    assert r_80 == 5.0
    assert d_star_80 != d_star_60


def test_solver_places_center_beyond_exclusion() -> None:
    hotspot = torch.tensor([[0.0, 0.0, 0.0]])
    unselected = torch.tensor([[0.0, 0.0, -6.0]])
    ca = torch.cat([hotspot, unselected], dim=0)
    hotspot_mask = torch.tensor([True, False])
    c_H = hotspot[0]
    n0 = torch.tensor([0.0, 0.0, 1.0])
    center = compute_binder_center_sphere_exclusion(
        _geometry(ca, hotspot_mask, c_H, n0), target_dist=8.0, exclusion_radius=8.0
    )
    min_clearance = torch.linalg.vector_norm(ca - center, dim=-1).min()
    assert min_clearance >= 8.0 - 1e-3
    assert center[2] > 0.0
    assert not torch.allclose(center, torch.tensor([0.0, 0.0, 1.0]))


def test_solver_constant_wall_allows_closer_than_d_star() -> None:
    hotspot = torch.tensor([[0.0, 0.0, 0.0]])
    unselected = torch.tensor([[0.0, 0.0, -6.0], [0.0, 20.0, 0.0]])
    ca = torch.cat([hotspot, unselected], dim=0)
    hotspot_mask = torch.tensor([True, False, False])
    c_H = hotspot[0]
    n0 = torch.tensor([0.0, 0.0, 1.0])
    geometry = _geometry(ca, hotspot_mask, c_H, n0)
    coupled = compute_binder_center_sphere_exclusion(geometry, target_dist=12.13, exclusion_radius=12.13)
    thin = compute_binder_center_sphere_exclusion(geometry, target_dist=12.13, exclusion_radius=5.0)
    far_atom = unselected[1]
    coupled_clearance = float(torch.linalg.vector_norm(coupled - far_atom))
    thin_clearance = float(torch.linalg.vector_norm(thin - far_atom))
    assert coupled_clearance >= 12.13 - 1e-3
    assert thin_clearance >= 5.0 - 1e-3
    assert thin_clearance < coupled_clearance


def test_solver_slsqp_moves_seed_when_wall_thinner_than_standoff(monkeypatch: pytest.MonkeyPatch) -> None:
    hotspot = torch.tensor([[0.0, 0.0, 0.0]])
    unselected = torch.tensor([[0.0, 0.0, -6.0]])
    ca = torch.cat([hotspot, unselected], dim=0)
    hotspot_mask = torch.tensor([True, False])
    captured: dict[str, np.ndarray] = {}
    original = binder_center._slsqp_refine

    def _capture_refine(seed: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        refined = original(seed, *args, **kwargs)
        captured["seed"] = np.asarray(seed, dtype=np.float64)
        captured["refined"] = np.asarray(refined, dtype=np.float64)
        return refined

    monkeypatch.setattr(binder_center, "_slsqp_refine", _capture_refine)
    center = compute_binder_center_sphere_exclusion(
        _geometry(ca, hotspot_mask, hotspot[0], torch.tensor([0.0, 0.0, 1.0])),
        target_dist=12.13,
        exclusion_radius=5.0,
    )
    seed = captured["seed"]
    refined = captured["refined"]
    assert float(np.linalg.norm(refined - seed)) > 1.0
    assert float(np.linalg.norm(refined)) > float(np.linalg.norm(seed))
    torch.testing.assert_close(center, torch.tensor(refined, dtype=center.dtype))


def test_solver_fail_loud_when_refine_clashes(monkeypatch: pytest.MonkeyPatch) -> None:
    hotspot = torch.tensor([[0.0, 0.0, 0.0]])
    unselected = torch.tensor([[0.0, 0.0, -6.0]])
    ca = torch.cat([hotspot, unselected], dim=0)
    hotspot_mask = torch.tensor([True, False])
    monkeypatch.setattr(
        "dynopsi.data.featurization.binder_center._slsqp_refine",
        lambda *args, **kwargs: np.zeros(3),
    )
    with pytest.raises(RuntimeError, match="clash-feasible"):
        compute_binder_center_sphere_exclusion(
            _geometry(ca, hotspot_mask, hotspot[0], torch.tensor([0.0, 0.0, 1.0])),
            target_dist=8.0,
            exclusion_radius=8.0,
        )


def test_extract_empty_unselected_message_differs_from_far_unselected() -> None:
    hotspot = torch.tensor([[0.0, 0.0, 0.0]])
    resolved = torch.ones(1, dtype=torch.bool)
    with pytest.raises(HotspotGeometryUnavailableError, match="UNSELECTED"):
        extract_hotspot_outward_geometry(hotspot, torch.tensor([PocketFeature.POCKET]), resolved)

    coords = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    pocket = torch.tensor([PocketFeature.POCKET, PocketFeature.UNSELECTED])
    with pytest.raises(HotspotGeometryUnavailableError, match="within 12A"):
        extract_hotspot_outward_geometry(coords, pocket, torch.ones(2, dtype=torch.bool))


def test_sphere_exclusion_center_is_clash_feasible_near_standoff() -> None:
    coords = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -6.0]])
    pocket = torch.tensor([PocketFeature.POCKET, PocketFeature.UNSELECTED])
    resolved = torch.ones(2, dtype=torch.bool)
    d_star, r_excl = sphere_exclusion_radii(60, 0.0, None)
    center = sphere_exclusion_center(coords, pocket, resolved, 60, 0.0, None)
    min_clearance = float(torch.linalg.vector_norm(coords - center, dim=-1).min())
    hotspot_dist = float(torch.linalg.vector_norm(center - coords[0]))
    assert min_clearance >= r_excl - 1e-3
    assert hotspot_dist == pytest.approx(d_star, abs=0.5)


def _fixed_hotspot_offset_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two Cα: hotspot at the origin and a proximal UNSELECTED at ``-6 z``.

    ``c_H`` is the origin and ``n0`` is ``+z``, so the 10 Å offset lands at
    ``[0, 0, 10]``.
    """
    coords_ca = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -6.0]])
    pocket = torch.tensor([PocketFeature.POCKET, PocketFeature.UNSELECTED])
    resolved = torch.ones(2, dtype=torch.bool)
    return coords_ca, pocket, resolved


def _fixed_hotspot_offset_features(coords_ca: torch.Tensor, pocket: torch.Tensor, resolved: torch.Tensor) -> Features:
    """Wrap Cα coordinates so ``_coords_ca`` reads ``coords[0, 1::3]``."""
    n_tokens = coords_ca.shape[0]
    coords = torch.zeros(1, n_tokens * 3, 3)
    coords[0, 1::3, :] = coords_ca
    return Features(
        feature_dict={
            "coords": coords,
            "frames_idx": torch.arange(n_tokens * 3).reshape(n_tokens, 3),
            "pocket_feature": pocket,
            "frame_resolved_mask": resolved.to(dtype=torch.float32),
        }
    )


def test_fixed_hotspot_offset_estimator_places_center_ten_angstrom() -> None:
    features = _fixed_hotspot_offset_features(*_fixed_hotspot_offset_fixture())
    center = new_chains.FixedHotspotOffsetEstimator()(features, (60,))[0]
    torch.testing.assert_close(center, torch.tensor([0.0, 0.0, 10.0]))


def test_fixed_hotspot_offset_estimator_offset_is_length_independent() -> None:
    features = _fixed_hotspot_offset_features(*_fixed_hotspot_offset_fixture())
    center_60 = new_chains.FixedHotspotOffsetEstimator()(features, (60,))[0]
    center_80 = new_chains.FixedHotspotOffsetEstimator()(features, (80,))[0]
    torch.testing.assert_close(center_60, center_80)


def test_fixed_hotspot_offset_estimator_differs_from_radius_of_gyration() -> None:
    features = _fixed_hotspot_offset_features(*_fixed_hotspot_offset_fixture())
    fixed = new_chains.FixedHotspotOffsetEstimator()(features, (60,))[0]
    rg = new_chains.BinderRgCenterOfMassEstimator()(features, (60,))[0]
    assert not torch.allclose(fixed, rg)
    assert float(torch.linalg.vector_norm(fixed)) == pytest.approx(10.0)


def test_fixed_hotspot_offset_estimator_rejects_multiple_new_chain_lengths() -> None:
    features = _fixed_hotspot_offset_features(*_fixed_hotspot_offset_fixture())
    with pytest.raises(ValueError, match="length 1"):
        new_chains.FixedHotspotOffsetEstimator()(features, (60, 80))


def test_fixed_hotspot_offset_estimator_wraps_unavailable_as_new_chains_error() -> None:
    coords_ca = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    pocket = torch.tensor([PocketFeature.POCKET, PocketFeature.UNSELECTED])
    features = _fixed_hotspot_offset_features(coords_ca, pocket, torch.ones(2, dtype=torch.bool))
    with pytest.raises(new_chains.BinderCenterOfMassUnavailableError, match="within 12A") as exc_info:
        new_chains.FixedHotspotOffsetEstimator()(features, (60,))
    assert type(exc_info.value) is new_chains.BinderCenterOfMassUnavailableError


def test_fixed_hotspot_offset_check_num_chains_compatible() -> None:
    estimator = new_chains.FixedHotspotOffsetEstimator()
    assert estimator.check_num_chains_compatible(1)
    assert not estimator.check_num_chains_compatible(2)
