from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.optimize
import torch

from dynopsi.data.constants import PocketFeature

_FEASIBILITY_TOL = 1e-4
_SEED_SAFETY = 1e-4
_POWER_LAW_SCALE = 2.2
_POWER_LAW_EXPONENT = 0.38
_CA_CLEARANCE = 1.7
_HUBER_DELTA = 1.0
_VECTOR_NORM_EPS = 1e-6
_ZERO_DIST_EPS = 1e-12
_CA_CA_INTERACTION_DIST = 12.0


class HotspotGeometryUnavailableError(ValueError):
    pass


class HotspotOutwardGeometry(NamedTuple):
    clash_ca: torch.Tensor
    hotspot_mask: torch.Tensor
    c_H: torch.Tensor
    n0: torch.Tensor


def extract_hotspot_outward_geometry(
    coords_ca: torch.Tensor,
    pocket_feature: torch.Tensor,
    resolved_mask: torch.Tensor,
) -> HotspotOutwardGeometry:
    """Return clash-set Cα, hotspot mask on that set, hotspot mean ``c_H``, and unit ``n0``.

    Parameters
    ----------
    coords_ca : Tensor, shape [N, 3]
        Cα coordinates.
    pocket_feature : Tensor, shape [N]
        Pocket labels aligned with ``coords_ca``.
    resolved_mask : Tensor, shape [N]
        True where the residue is resolved.

    Clash set is resolved ``POCKET ∪ UNSELECTED``. ``n0`` uses hotspots union
    proximal UNSELECTED (Cα–Cα ``<`` 12 Å).

    Returns
    -------
    HotspotOutwardGeometry
        ``clash_ca`` shape ``[C, 3]``, ``hotspot_mask`` shape ``[C]``,
        ``c_H`` and ``n0`` shape ``[3]``.
    """
    if (pocket_feature == PocketFeature.UNSPECIFIED).all():
        raise ValueError("All of features['pocket_feature'] are UNSPECIFIED. This is not supported.")
    if not (pocket_feature == PocketFeature.POCKET).any():
        raise ValueError(
            "No residues designated as `POCKET` in `features['pocket_feature']`. Cannot estimate the binder center of mass."
        )

    resolved = resolved_mask.bool()
    hotspot_token = pocket_feature == PocketFeature.POCKET
    unselected_token = pocket_feature == PocketFeature.UNSELECTED

    coords_hotspot = coords_ca[hotspot_token & resolved]
    if not coords_hotspot.numel():
        raise HotspotGeometryUnavailableError(
            "No resolved hotspot residues found, cannot estimate the binder center of mass."
        )
    coords_unselected = coords_ca[unselected_token & resolved]
    if not coords_unselected.numel():
        raise HotspotGeometryUnavailableError(
            "No resolved UNSELECTED residues found, cannot estimate the binder center of mass."
        )
    dist = torch.cdist(coords_hotspot, coords_unselected)
    proximal = coords_unselected[(dist < _CA_CA_INTERACTION_DIST).any(dim=0)]
    if not proximal.numel():
        raise HotspotGeometryUnavailableError(
            f"No resolved target residues within {_CA_CA_INTERACTION_DIST:.0f}A of the hotspot found, cannot estimate the binder center of mass."
        )
    c_H = coords_hotspot.mean(dim=0)
    c_ref = torch.cat([proximal, coords_hotspot]).mean(dim=0)
    vector = c_H - c_ref
    vector_norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(vector_norm) or vector_norm <= _VECTOR_NORM_EPS:
        raise HotspotGeometryUnavailableError(
            "The binder center of mass is unavailable because the vector from the target center of mass to the hotspot center of mass is not finite or too small."
        )
    n0 = vector / vector_norm
    clash_token = (hotspot_token | unselected_token) & resolved
    return HotspotOutwardGeometry(
        clash_ca=coords_ca[clash_token],
        hotspot_mask=hotspot_token[clash_token],
        c_H=c_H,
        n0=n0,
    )


def center_on_hotspot_axis(
    coords_ca: torch.Tensor,
    pocket_feature: torch.Tensor,
    resolved_mask: torch.Tensor,
    distance: float,
) -> torch.Tensor:
    """Return ``c_H + distance * n0`` for the hotspot outward axis.

    Raises
    ------
    HotspotGeometryUnavailableError
        Extract cannot form an outward hotspot axis.
    ValueError
        Missing hotspot residues or all-UNSPECIFIED pocket.
    """
    geometry = extract_hotspot_outward_geometry(coords_ca, pocket_feature, resolved_mask)
    return geometry.c_H + distance * geometry.n0


def sphere_exclusion_radii(
    binder_length: int,
    exclusion_offset: float,
    exclusion_radius: float | None,
) -> tuple[float, float]:
    """Return ``(D*, R_excl)`` for a binder of ``binder_length`` residues.

    ``D* = 2.2 · L^0.38 + 1.7 + exclusion_offset``. ``R_excl`` equals ``D*`` unless
    ``exclusion_radius`` is set.
    """
    if binder_length <= 0:
        raise ValueError("binder_length must be positive")
    d_star = _POWER_LAW_SCALE * binder_length**_POWER_LAW_EXPONENT + _CA_CLEARANCE + exclusion_offset
    r_excl = d_star if exclusion_radius is None else exclusion_radius
    return float(d_star), float(r_excl)


def sphere_exclusion_center(
    coords_ca: torch.Tensor,
    pocket_feature: torch.Tensor,
    resolved_mask: torch.Tensor,
    binder_length: int,
    exclusion_offset: float,
    exclusion_radius: float | None,
) -> torch.Tensor:
    """Place a clash-aware binder COM from hotspot geometry and binder length.

    Raises
    ------
    HotspotGeometryUnavailableError
        Extract cannot form an outward hotspot axis.
    RuntimeError
        Refine is not clash-feasible.
    ValueError
        Missing hotspot residues, all-UNSPECIFIED pocket, or non-finite coordinates.
    """
    geometry = extract_hotspot_outward_geometry(coords_ca, pocket_feature, resolved_mask)
    target_dist, r_excl = sphere_exclusion_radii(binder_length, exclusion_offset, exclusion_radius)
    return compute_binder_center_sphere_exclusion(geometry, target_dist, r_excl)


def _nearest_feasible_seed(
    c_H: np.ndarray,
    d: np.ndarray,
    ca_np: np.ndarray,
    exclusion_radius: float,
) -> np.ndarray:
    """Return the first clash-feasible point just outside the exclusion spheres along ``d``.

    The last overlapping interval along the ray ends at ``t_star``. The returned
    seed is ``c_H + (t_star + safety) * d``, one epsilon outside that wall.
    """
    q = c_H - ca_np
    b = q @ d
    q2 = np.einsum("ij,ij->i", q, q)
    disc = b * b - q2 + exclusion_radius * exclusion_radius

    pos = disc > 0.0
    sqrt_disc = np.sqrt(disc[pos])
    lo_all = -b[pos] - sqrt_disc
    hi_all = -b[pos] + sqrt_disc
    intervals = [(max(0.0, float(lo)), float(hi)) for lo, hi in zip(lo_all, hi_all, strict=True) if hi > 0.0]

    if not intervals:
        t_star = 0.0
    else:
        intervals.sort(key=lambda iv: iv[0])
        if intervals[0][0] > 0.0:
            t_star = 0.0
        else:
            end = intervals[0][1]
            for lo, hi in intervals[1:]:
                if lo <= end:
                    end = max(end, hi)
                else:
                    break
            t_star = end

    return c_H + (t_star + _SEED_SAFETY) * d


def _huber(u: np.ndarray, delta: float) -> np.ndarray:
    abs_u = np.abs(u)
    return np.where(abs_u <= delta, 0.5 * u * u, delta * (abs_u - 0.5 * delta))


def _mean_huber(p: np.ndarray, hotspot_np: np.ndarray, target_dist: float, delta: float) -> float:
    gamma = np.linalg.norm(hotspot_np - p, axis=-1) - target_dist
    return float(np.mean(_huber(gamma, delta)))


def _mean_huber_grad(p: np.ndarray, hotspot_np: np.ndarray, target_dist: float, delta: float) -> np.ndarray:
    diff = p - hotspot_np
    dists = np.linalg.norm(diff, axis=-1)
    gamma = dists - target_dist
    huber_grad = np.where(np.abs(gamma) <= delta, gamma, delta * np.sign(gamma))
    safe = np.where(dists > _ZERO_DIST_EPS, dists, 1.0)
    unit = diff / safe[:, None]
    return (huber_grad[:, None] * unit).mean(axis=0)


def _slsqp_refine(
    seed: np.ndarray,
    ca_np: np.ndarray,
    hotspot_np: np.ndarray,
    exclusion_radius: float,
    target_dist: float,
    delta: float,
    maxiter: int = 100,
    ftol: float = 1e-6,
) -> np.ndarray:
    def objective(p: np.ndarray) -> float:
        return _mean_huber(p, hotspot_np, target_dist, delta)

    def objective_grad(p: np.ndarray) -> np.ndarray:
        return _mean_huber_grad(p, hotspot_np, target_dist, delta)

    def constraint_fun(p: np.ndarray) -> np.ndarray:
        return np.linalg.norm(ca_np - p, axis=-1) - exclusion_radius

    def constraint_jac(p: np.ndarray) -> np.ndarray:
        diff = p - ca_np
        dists = np.linalg.norm(diff, axis=-1)
        safe = np.where(dists > _ZERO_DIST_EPS, dists, 1.0)
        return diff / safe[:, None]

    result = scipy.optimize.minimize(  # type: ignore[call-overload]  # scipy stubs reject the constraints dict
        objective,
        seed,
        method="SLSQP",
        jac=objective_grad,
        constraints={"type": "ineq", "fun": constraint_fun, "jac": constraint_jac},
        options={"maxiter": maxiter, "ftol": ftol},
    )
    return np.asarray(result.x, dtype=np.float64)


def _clash_feasible(p: np.ndarray, ca_np: np.ndarray, exclusion_radius: float) -> bool:
    if not np.isfinite(p).all():
        return False
    min_clearance = float((np.linalg.norm(ca_np - p, axis=-1) - exclusion_radius).min())
    return min_clearance >= -_FEASIBILITY_TOL


def compute_binder_center_sphere_exclusion(
    geometry: HotspotOutwardGeometry,
    target_dist: float,
    exclusion_radius: float,
) -> torch.Tensor:
    """Place a binder COM by ray-march along ``geometry.n0`` then one SLSQP refine.

    Parameters
    ----------
    geometry :
        Clash-set Cα, hotspot mask on that set, hotspot mean, and unit outward seed.
    target_dist :
        Huber standoff ``D*``.
    exclusion_radius :
        Hard wall ``R_excl``. Must already be resolved; this function does not add 1.7.

    Returns
    -------
    center : Tensor, shape (3,)

    Raises
    ------
    ValueError
        Empty hotspot mask or non-finite coordinates.
    RuntimeError
        No clash-feasible refine.
    """
    if not geometry.hotspot_mask.any():
        raise ValueError("hotspot_mask selects no residues")
    hotspot_mask_np = geometry.hotspot_mask.detach().cpu().numpy().astype(bool)
    ca_np = geometry.clash_ca.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(ca_np).all():
        raise ValueError("ca contains non-finite values")
    hotspot_np = ca_np[hotspot_mask_np]
    c_H_np = geometry.c_H.detach().cpu().numpy().astype(np.float64)
    n0_np = geometry.n0.detach().cpu().numpy().astype(np.float64)

    seed = _nearest_feasible_seed(c_H_np, n0_np, ca_np, exclusion_radius)
    refined = _slsqp_refine(seed, ca_np, hotspot_np, exclusion_radius, target_dist, _HUBER_DELTA)
    if not _clash_feasible(refined, ca_np, exclusion_radius):
        raise RuntimeError("sphere-exclusion centering did not find a clash-feasible center")
    return torch.tensor(refined, dtype=geometry.clash_ca.dtype, device=geometry.clash_ca.device)
