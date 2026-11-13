import einops
import torch


def slice_coords_by_frames_idx(coords: torch.Tensor, frames_idx: torch.Tensor, num_atoms: int) -> torch.Tensor:
    """Gather backbone atom coordinates using frames_idx.

    Parameters
    ----------
    coords : torch.Tensor
        Full atom coordinates, shape (batch_size, 1, all_atoms, 3).
    frames_idx : torch.Tensor
        Indices of N, CA, C backbone atoms for each residue, shape (batch_size, seq_len, 3).
    num_atoms : int
        Atoms to return: 3 → N/CA/C backbone, 1 → CA only.

    Returns
    -------
    torch.Tensor
        Shape (batch_size, seq_len, num_atoms * 3).
    """
    batch_size = frames_idx.shape[0]
    seq_len = frames_idx.shape[1]
    frames_idx_expand = frames_idx.reshape(batch_size, 1, -1, 1).expand(-1, -1, -1, 3)
    backbone_coords = torch.gather(coords, dim=2, index=frames_idx_expand).reshape(batch_size, seq_len, -1)
    if num_atoms == 3:
        return backbone_coords
    elif num_atoms == 1:
        return backbone_coords.view(batch_size, seq_len, 3, 3)[:, :, 1, :]
    else:
        raise ValueError(f"Unsupported num_atoms={num_atoms}; supported values are 1 and 3")


def kabsch_torch(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Return the optimal rotation R that minimizes ||PR - Q||_F.

    Assumes coordinates are centered.

    Parameters
    ----------
    P : torch.Tensor
        [N, 3] tensor of coordinates
    Q : torch.Tensor
        [N, 3] tensor of coordinates

    Returns
    -------
    R : torch.Tensor
        [3, 3] tensor of rotation matrix
    """
    H = torch.mm(P.T, Q)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.linalg.det(torch.mm(Vt.T, U.T))

    factor = torch.where(d < 0, torch.tensor(-1.0, device=Vt.device), torch.tensor(1.0, device=Vt.device))
    Vt[-1, :] *= factor

    R = torch.mm(U, Vt)
    return R.to(device=P.device)


def align_by_mask(
    coords1: torch.Tensor, coords2: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor
) -> torch.Tensor:
    """
    Align coords1 to coords2 (which can be different lengths) by the equal-length masked regions of both.

    Parameters
    ----------
    coords1 : torch.Tensor
        Coordinates of shape [N, 3]
    coords2 : torch.Tensor
        Coordinates of shape [M, 3]
    mask1 : torch.Tensor
        Mask of shape [N]
    mask2 : torch.Tensor
        Mask of shape [M]. mask1.sum() must equal mask2.sum().

    Returns
    -------
    aligned_coords1 : torch.Tensor
        Aligned coordinates of shape [N, 3]
    """
    assert mask1.sum() == mask2.sum()
    assert mask1.shape[0] == coords1.shape[0]
    assert mask2.shape[0] == coords2.shape[0]
    coords1_masked = coords1[mask1]
    coords2_masked = coords2[mask2]

    # center
    mean1 = coords1_masked.mean(dim=0, keepdim=True)
    mean2 = coords2_masked.mean(dim=0, keepdim=True)

    coords1_centered = coords1_masked - mean1
    coords2_centered = coords2_masked - mean2

    # align
    R = kabsch_torch(coords2_centered, coords1_centered)

    # rotate (full) coords1 by R and translate by mean2
    aligned_coords1 = (coords1 - mean1) @ R.T + mean2

    return aligned_coords1


def kabsch_align_single(
    coords1_i: torch.Tensor,
    coords2_i: torch.Tensor,
    mask_i: torch.Tensor,
    reapply_mean: bool = False,
    replace_masked_with_original_coords1_i: bool = True,
) -> torch.Tensor:
    """Align a single pair of coordinates using the Kabsch algorithm.

    Rotation is applied to the first input tensor.

    If reapply_mean is True, then the pre-alignment mean of the first input tensor
    (`coords1_i`) is added back to the aligned coordinates. Otherwise, the mean
    of the returned input coordinates will be the mean of the second input tensor
    (`coords2_i`).

    Note that reapply_mean=False has the desirable behavior that if coords2_i = a and
    coords1_i = aR+t for a random rotation matrix R and translation t,
    then the output will be a.

    Parameters
    ----------
    coords1_i : torch.Tensor
        Coordinates of shape [N, 3]
    coords2_i : torch.Tensor
        Coordinates of shape [N, 3]
    mask_i : torch.Tensor
        Mask of shape [N]
    reapply_mean : bool
        If True, the mean of the first input tensor is added back to the aligned coordinates
        replace_masked_with_original_coords1_i: bool
            If True, replace output aligned_coords_i at positions mask_i with the values form coords1_i

    Returns
    -------
    aligned_coords1_i : torch.Tensor
        Aligned coordinates of shape [N, 3]
    """
    mask_i = mask_i.unsqueeze(-1)
    coords1_i_masked = coords1_i * mask_i
    coords2_i_masked = coords2_i * mask_i

    mean1 = coords1_i_masked.sum(dim=0, keepdim=True) / mask_i.sum(dim=0, keepdim=True)
    mean2 = coords2_i_masked.sum(dim=0, keepdim=True) / mask_i.sum(dim=0, keepdim=True)

    # center
    coords1_i_centered = coords1_i_masked - mean1
    coords2_i_centered = coords2_i_masked - mean2

    # reapply mask so that masked points don't affect the alignment
    coords1_i_centered *= mask_i
    coords2_i_centered *= mask_i

    # align
    R = kabsch_torch(coords2_i_centered, coords1_i_centered)

    # rotate coords1 into coords2's frame
    aligned_coords1_i = (coords1_i - mean1) @ R.T

    # apply correct mean
    if reapply_mean:
        aligned_coords1_i = aligned_coords1_i + mean1
    else:
        aligned_coords1_i = aligned_coords1_i + mean2

    # replace masked positions with original coords
    if replace_masked_with_original_coords1_i:
        aligned_coords1_i = torch.where(mask_i.bool(), aligned_coords1_i, coords1_i)

    return aligned_coords1_i


def kabsch_align_batch(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    mask: torch.Tensor,
    reapply_mean: bool = False,
    replace_masked_with_original: bool = True,
) -> torch.Tensor:
    """Align two sets of coordinates using the Kabsch algorithm.

    The i-th element of the first input tensor is aligned to the i-th element of the second input tensor,
    and rotation is applied to the first input tensor.

    Parameters
    ----------
    coords1 : torch.Tensor
        Coordinates of shape [batch_size, N, 3]
    coords2 : torch.Tensor
        Coordinates of shape [batch_size, N, 3]
    mask : torch.Tensor
        Mask of shape [batch_size, N]
    reapply_mean : bool
        If True, the mean of the first input tensor is added back to the aligned coordinates for each batch element
    replace_masked_with_original : bool
        If True, at positions where mask is False, replace aligned output with values from coords1 (per batch element)

    Returns
    -------
    torch.Tensor
        Aligned coordinates of shape [batch_size, N, 3]

    See Also
    --------
    kabsch_align_single : Aligns a single pair of coordinates; see for more details on
        reapply_mean and replace_masked_with_original.
    """
    return torch.vmap(lambda c1, c2, m: kabsch_align_single(c1, c2, m, reapply_mean, replace_masked_with_original))(
        coords1, coords2, mask
    )


def kabsch_align_batch_multi_atom(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    mask: torch.Tensor,
    reapply_mean: bool = False,
    replace_masked_with_original: bool = True,
) -> torch.Tensor:
    """Align two sets of multi-atom coordinates using the Kabsch algorithm.

    The i-th element of the first input tensor is aligned to the i-th element of the second input tensor,
    and rotation is applied to the first input tensor. Within each batch element, coordinates are
    flattened over the atom dimension for alignment, then reshaped back to [batch_size, N, num_atoms, 3].

    Parameters
    ----------
    coords1 : torch.Tensor
        Coordinates of shape [batch_size, N, num_atoms, 3]
    coords2 : torch.Tensor
        Coordinates of shape [batch_size, N, num_atoms, 3]
    mask : torch.Tensor
        Mask of shape [batch_size, N].
        Applied per residue/position; the same mask value is used for
        all atoms at that position when flattening for alignment.
    replace_masked_with_original : bool
            If True, at positions where mask is False, replace aligned output with values from coords1.


    Returns
    -------
    torch.Tensor
        Aligned coordinates of shape [batch_size, N, num_atoms, 3]

    See Also
    --------
    kabsch_align_batch : Batch alignment of single-atom coordinates.
    kabsch_align_single : Aligns a single pair of coordinates; see for more details on
            reapply_mean and replace_masked_with_original.
    """
    if coords1.shape[-1] != 3 or coords2.shape[-1] != 3:
        raise ValueError(
            f"coordinates must have shape [batch_size, N, num_atoms, 3], got {coords1.shape} and {coords2.shape}. If the final dimension is not 3, the Kabsch algorithm will not return a valid 3D rotation matrix"
        )

    num_atoms = coords1.shape[2]

    # Reshape coordinates to [batch_size, N * num_atoms, 3] for Kabsch algorithm
    coords1_flat = einops.rearrange(coords1, "b n n_atoms d -> b (n n_atoms) d", n_atoms=num_atoms, d=3)
    coords2_flat = einops.rearrange(coords2, "b n n_atoms d -> b (n n_atoms) d", n_atoms=num_atoms, d=3)

    # Expand mask to match reshaped coordinates: [batch_size, N] -> [batch_size, N * num_atoms]
    mask_expanded = mask.unsqueeze(2).expand(-1, -1, num_atoms)
    mask_flat = einops.rearrange(mask_expanded, "b n n_atoms -> b (n n_atoms)", n_atoms=num_atoms)

    aligned_flat = kabsch_align_batch(coords1_flat, coords2_flat, mask_flat, reapply_mean, replace_masked_with_original)

    # Reshape aligned coordinates back to (batch_size, N, num_atoms, 3)
    return einops.rearrange(aligned_flat, "b (n n_atoms) d -> b n n_atoms d", n_atoms=num_atoms, d=3)
