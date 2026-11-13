import random

import numpy as np
import pytest
import torch

from dynopsi.data.utils.alignment_utils import (
    align_by_mask,
    kabsch_align_batch,
    kabsch_align_batch_multi_atom,
    kabsch_align_single,
    kabsch_torch,
)


@pytest.fixture(autouse=True)
def seed_everything() -> None:
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_rotation_matrix(has_positive_determinant: bool = True) -> torch.Tensor:
    A = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(A)
    # enforce right-handed rotation (det=+1)
    if has_positive_determinant and torch.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    elif not has_positive_determinant and torch.linalg.det(Q) > 0:
        Q[:, -1] *= -1
    return Q


@pytest.fixture
def test_params() -> dict:
    return {
        "batch_size": 2,
        "L": 10,
        "d": 3,
    }


@pytest.mark.parametrize("has_positive_determinant", [True, False])
def test_kabsch_torch(has_positive_determinant: bool, test_params: dict) -> None:
    P = torch.randn(test_params["L"], test_params["d"])
    R = random_rotation_matrix(has_positive_determinant)
    if not has_positive_determinant:
        R[:, -1] *= -1
    Q = P @ R
    R_hat = kabsch_torch(P, Q)
    assert torch.allclose(R_hat, R, atol=1e-5)


@pytest.mark.parametrize("has_positive_determinant", [True, False])
def test_align_by_mask(has_positive_determinant: bool, test_params: dict) -> None:
    P = torch.randn(test_params["L"], test_params["d"])
    R = random_rotation_matrix(has_positive_determinant)
    random_translation = torch.randn(test_params["d"])
    if not has_positive_determinant:
        R[:, -1] *= -1
    Q = P @ R + random_translation

    # add more random coordinates to P and Q
    P_extra = torch.cat([P, torch.randn(5, test_params["d"])], dim=0)
    Q_extra = torch.cat([Q, torch.randn(8, test_params["d"])], dim=0)

    mask_P = torch.cat([torch.ones(test_params["L"]), torch.zeros(5)]).bool()
    mask_Q = torch.cat([torch.ones(test_params["L"]), torch.zeros(8)]).bool()
    reconstructed_P = align_by_mask(P_extra, Q_extra, mask_P, mask_Q)

    assert torch.allclose(reconstructed_P[mask_P], Q, atol=1e-5)


@pytest.mark.parametrize("has_positive_determinant", [True, False])
def test_kabsch_align_single_with_mask(has_positive_determinant: bool, test_params: dict) -> None:
    P = torch.randn(test_params["L"], test_params["d"])
    R = random_rotation_matrix(has_positive_determinant)
    if not has_positive_determinant:
        R[:, -1] *= -1
    t = torch.randn(test_params["d"])
    Q = P @ R + t

    # concat additional points
    additional_points = 5
    P_extra = torch.randn(additional_points, test_params["d"])
    Q_extra = torch.randn(additional_points, test_params["d"])

    P = torch.cat([P, P_extra], dim=0)
    Q = torch.cat([Q, Q_extra], dim=0)

    mask = torch.cat([torch.ones(test_params["L"]), torch.zeros(additional_points)]).bool()
    # align
    P_hat = kabsch_align_single(Q, P, mask, reapply_mean=False)

    # should equal P for unmasked points and Q_extra for masked points
    assert torch.allclose(P_hat[: test_params["L"]], P[: test_params["L"]], atol=1e-5)
    assert torch.allclose(P_hat[test_params["L"] :], Q_extra, atol=1e-5)


def test_kabsch_align_single_with_mask_reapply_mean(test_params: dict) -> None:
    P = torch.randn(test_params["L"], test_params["d"])
    Q = torch.randn(test_params["L"], test_params["d"])

    mask = torch.ones(test_params["L"]).bool()
    P_hat = kabsch_align_single(P, Q, mask, reapply_mean=True)
    assert torch.allclose(torch.mean(P_hat, dim=0), torch.mean(P, dim=0), atol=1e-5)


@pytest.mark.parametrize("has_positive_determinant", [True, False])
def test_kabsch_align_batch(has_positive_determinant: bool, test_params: dict) -> None:
    P = torch.randn(test_params["batch_size"], test_params["L"], test_params["d"])
    Q = torch.zeros_like(P)
    for i in range(test_params["batch_size"]):
        R = random_rotation_matrix(has_positive_determinant)
        if not has_positive_determinant:
            R[:, -1] *= -1
        t = torch.randn(test_params["d"])
        Q[i] = P[i] @ R + t

    mask = torch.ones(test_params["batch_size"], test_params["L"]).bool()
    P_hat = kabsch_align_batch(Q, P, mask, reapply_mean=False)
    assert torch.allclose(P_hat, P, atol=1e-5)


@pytest.mark.parametrize("has_positive_determinant", [True, False])
def test_kabsch_align_batch_multi_atom(has_positive_determinant: bool, test_params: dict) -> None:
    num_atoms = 3
    P = torch.randn(test_params["batch_size"], test_params["L"] * num_atoms, test_params["d"])
    Q = torch.zeros_like(P)
    for i in range(test_params["batch_size"]):
        R = random_rotation_matrix(has_positive_determinant)
        if not has_positive_determinant:
            R[:, -1] *= -1
        t = torch.randn(test_params["d"])
        Q[i] = P[i] @ R + t
    mask = torch.ones(test_params["batch_size"], test_params["L"]).bool()

    mask_flat = torch.repeat_interleave(mask, num_atoms, dim=1)
    P_hat_base = kabsch_align_batch(Q, P, mask_flat, reapply_mean=False)

    P_multi_atom = P.reshape(test_params["batch_size"], test_params["L"], num_atoms, test_params["d"])
    Q_multi_atom = Q.reshape(test_params["batch_size"], test_params["L"], num_atoms, test_params["d"])
    P_hat_multi_atom = kabsch_align_batch_multi_atom(Q_multi_atom, P_multi_atom, mask, reapply_mean=False)
    P_hat_multi_atom = P_hat_multi_atom.reshape(
        test_params["batch_size"], test_params["L"] * num_atoms, test_params["d"]
    )
    assert torch.allclose(P_hat_multi_atom, P_hat_base, atol=1e-5)


@pytest.mark.parametrize("has_positive_determinant", [True, False])
def test_kabsch_align_shape_dependency(has_positive_determinant: bool, test_params: dict) -> None:
    num_atoms = 3
    P = torch.randn(test_params["batch_size"], test_params["L"] * num_atoms, test_params["d"])
    Q = torch.zeros_like(P)
    for i in range(test_params["batch_size"]):
        R = random_rotation_matrix(has_positive_determinant)
        if not has_positive_determinant:
            R[:, -1] *= -1
        t = torch.randn(test_params["d"])
        Q[i] = P[i] @ R + t
        # Make Q not a perfect rotation of P by randomly adding a bit of coordinate noise to some positions
        # WHen ! is a perfect rotation of P, the 9D alignment is also able to find the correct 3D rotation
        noise_mask = torch.randint(0, 2, Q[i].shape, device=Q[i].device).bool()
        Q[i] += torch.randn_like(Q[i]) * 0.01 * noise_mask
    mask = torch.ones(test_params["batch_size"], test_params["L"]).bool()

    mask_3d = torch.repeat_interleave(mask, num_atoms, dim=1)
    P_hat_3d = kabsch_align_batch(Q, P, mask_3d, reapply_mean=False)

    P_9d = P.reshape(test_params["batch_size"], test_params["L"], num_atoms * test_params["d"])
    Q_9d = Q.reshape(test_params["batch_size"], test_params["L"], num_atoms * test_params["d"])
    P_hat_9d = kabsch_align_batch(Q_9d, P_9d, mask, reapply_mean=False)
    P_hat_9d_reshaped = P_hat_9d.reshape(test_params["batch_size"], test_params["L"] * num_atoms, test_params["d"])
    assert not torch.allclose(P_hat_9d_reshaped, P_hat_3d, atol=1e-5)

    P_multi_atom = P.reshape(test_params["batch_size"], test_params["L"], num_atoms, test_params["d"])
    Q_multi_atom = Q.reshape(test_params["batch_size"], test_params["L"], num_atoms, test_params["d"])
    P_hat_multi_atom = kabsch_align_batch_multi_atom(Q_multi_atom, P_multi_atom, mask, reapply_mean=False)
    P_hat_multi_atom = P_hat_multi_atom.reshape(
        test_params["batch_size"], test_params["L"] * num_atoms, test_params["d"]
    )
    assert torch.allclose(P_hat_multi_atom, P_hat_3d, atol=1e-5)
