import torch

from dynopsi.data.constants import ATOM37_ORDER, AtomFilter


def reshape_coords_for_model(coords: torch.Tensor, frames_idx: torch.Tensor, atoms: AtomFilter) -> torch.Tensor:
    """Extract and return frame coordinates in model-compatible shape.

    Uses `frames_idx` tensor to extract per-residue frame atoms from `coords`.
    For BB, each frame is [N, CA, C]. For CA, each frame is [CA].
    Returns flattened atom 3D coordinates per residue.

    Parameters
    ----------
    coords : torch.Tensor
        Atom 3D coordinates with shape `(batch_size, 1, num_atoms, 3)`.
    frames_idx : torch.Tensor
        Index into `coords` to provide per-residue atom coordinates,
        shape `(batch_size, num_residues, n_frame_atoms)`.
    atoms : AtomFilter
        Which atoms to include; supported values are `AtomFilter.BB` and
        `AtomFilter.CA`.

    Returns
    -------
    torch.Tensor
        Frame coordinates of shape `(batch_size, num_residues, 3 * n_frame_atoms)`.

    Raises
    ------
    ValueError
        If `atoms` is not one of the supported atom filters.
    """
    batch_size = frames_idx.shape[0]
    seq_len = frames_idx.shape[1]
    frames_idx_expand = frames_idx.reshape(batch_size, 1, -1, 1).expand(
        -1, -1, -1, 3
    )  # shape: (batch_size, 1, seq_len, 3)
    backbone_coords = torch.gather(coords, dim=2, index=frames_idx_expand).reshape(
        batch_size, seq_len, -1
    )  # shape: (batch_size, seq_len, 3 * 3)
    if atoms in (AtomFilter.BB, AtomFilter.CA):
        return backbone_coords
    else:
        raise ValueError(f"Unsupported atoms == {atoms}")


def center_coordinates(atom_coords: torch.Tensor, atom_center_mask: torch.Tensor) -> torch.Tensor:
    """Center coordinates on the center of mass of selected atoms.

    Computes the center of mass using only the atoms selected in the `atom_center_mask`,
    and translates all coordinates so that this center of mass is at the origin.

    Parameters
    ----------
    atom_coords : torch.Tensor
        The coordinates of the atoms, shape `(1, num_atoms, 3)`
    atom_center_mask : torch.Tensor
        The mask of the atoms that will define the center of mass, shape `(num_atoms)`

    Returns
    -------
    torch.Tensor:
        The centered coordinates, shape `(1, num_atoms, 3)`.
    """
    if not atom_center_mask.any():
        raise ValueError("No atoms to center on")
    atom_center_mask = atom_center_mask.unsqueeze(0).unsqueeze(2)  # (1, num_atoms, 1)

    atom_center_mean = torch.sum(atom_coords * atom_center_mask, dim=1, keepdim=True) / torch.sum(
        atom_center_mask, dim=1, keepdim=True
    )
    centered_coords = atom_coords - atom_center_mean

    return centered_coords


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.
    Snippet from Boltz

    Parameters
    ----------
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns
    -------
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def random_quaternions(n: int, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.
    Snippet from Boltz

    Parameters
    ----------
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns
    -------
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Snippet from Boltz

    Parameters
    ----------
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns
    -------
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def randomly_rotate(coords: torch.Tensor) -> torch.Tensor:
    """Randomly rotate a batch of coordinates.
    Adapted from Boltz

    Parameters
    ----------
        coords: torch.Tensor of shape (batch_size, num_atoms, 3)
            Contains the coordinates to rotate.

    Returns
    -------
        torch.Tensor of shape (batch_size, num_atoms, 3)
            Contains the rotated coordinates.
    """
    quaternions = random_quaternions(len(coords), coords.dtype, coords.device)
    R = quaternion_to_matrix(quaternions)
    return torch.bmm(coords, R)


def add_oxygen_coords(backbone_coords: torch.Tensor) -> torch.Tensor:
    """Add imputed oxygen coordinates to backbone N, CA, C coordinates.

    Parameters
    ----------
    backbone_coords : torch.Tensor
        Backbone atom 3D coordinates of shape `(N, 3, 3)`.

    Returns
    -------
    torch.Tensor
        3D atomic coordinates for [N, CA, C, O], shape `(N, 4, 3)`.

    See Also
    --------
    adjust_oxygen_pos : Used to impute O from N, CA, C positions.
    """
    backbone_indices = torch.tensor([ATOM37_ORDER.index(atom_name) for atom_name in ["N", "CA", "C"]])
    backbone_indices_w_o = torch.tensor([ATOM37_ORDER.index(atom_name) for atom_name in ["N", "CA", "C", "O"]])

    atom_37 = torch.zeros(backbone_coords.shape[0], 37, 3)
    atom_37[:, backbone_indices, :] = backbone_coords

    atom_37_w_o = adjust_oxygen_pos(atom_37)

    backbone_coords_w_o = atom_37_w_o[:, backbone_indices_w_o, :]

    return backbone_coords_w_o


def adjust_oxygen_pos(atom_37: torch.Tensor, pos_is_known: torch.Tensor | None = None) -> torch.Tensor:
    """Impute backbone oxygen position using adjacent frame positions.

    Positions the oxygen atom in the plane created by the C-alpha and C
    of the current frame and the N of the next frame, at the appropriate
    C-O bond length from C in the direction away from the Ca-C-N triangle.

    When the next frame is not available (e.g. C-terminus), oxygen is placed
    in the N-Ca-C plane of the current frame, pointing in the direction of the
    average of the Ca-C and Ca-N vectors.

    Implementation from FrameFlow: https://github.com/microsoft/protein-frame-flow/blob/f50d8dbbdae827be291e9f73d732b61b195f8816/data/utils.py

    Parameters
    ----------
    atom_37 : torch.Tensor
        3D atom coordinates in atom_37 ordering (i.e. [N, CA, C, CB, O, ...]), shape `(N, 37, 3)`.
    pos_is_known : torch.Tensor | None
        Boolean mask indicating residues with known positions, shape `(N,)`.

    Returns
    -------
    torch.Tensor
        `atom_37` with O positions overwritten with imputed oxygen
        coordinates, shape `(N, 37, 3)`.
    """
    N = atom_37.shape[0]
    assert atom_37.shape == (N, 37, 3)

    # Get vectors to Carbonyl from Carbon alpha and N of next residue. (N-1, 3)
    # Note that the (N,) ordering is from N-terminal to C-terminal.

    # Calpha to carbonyl both in the current frame.
    calpha_to_carbonyl: torch.Tensor = (atom_37[:-1, 2, :] - atom_37[:-1, 1, :]) / (
        torch.norm(atom_37[:-1, 2, :] - atom_37[:-1, 1, :], keepdim=True, dim=1) + 1e-7
    )
    # For masked positions, they are all 0 and so we add 1e-7 to avoid division by 0.
    # The positions are in Angstroms and so are on the order ~1 so 1e-7 is an insignificant change.

    # Nitrogen of the next frame to carbonyl of the current frame.
    nitrogen_to_carbonyl: torch.Tensor = (atom_37[:-1, 2, :] - atom_37[1:, 0, :]) / (
        torch.norm(atom_37[:-1, 2, :] - atom_37[1:, 0, :], keepdim=True, dim=1) + 1e-7
    )

    carbonyl_to_oxygen: torch.Tensor = calpha_to_carbonyl + nitrogen_to_carbonyl  # (N-1, 3)
    carbonyl_to_oxygen = carbonyl_to_oxygen / (torch.norm(carbonyl_to_oxygen, dim=1, keepdim=True) + 1e-7)

    atom_37[:-1, 4, :] = atom_37[:-1, 2, :] + carbonyl_to_oxygen * 1.23

    # Now we deal with frames for which there is no next frame available.

    # Calpha to carbonyl both in the current frame. (N, 3)
    calpha_to_carbonyl_term: torch.Tensor = (atom_37[:, 2, :] - atom_37[:, 1, :]) / (
        torch.norm(atom_37[:, 2, :] - atom_37[:, 1, :], keepdim=True, dim=1) + 1e-7
    )
    # Calpha to nitrogen both in the current frame. (N, 3)
    calpha_to_nitrogen_term: torch.Tensor = (atom_37[:, 0, :] - atom_37[:, 1, :]) / (
        torch.norm(atom_37[:, 0, :] - atom_37[:, 1, :], keepdim=True, dim=1) + 1e-7
    )
    carbonyl_to_oxygen_term: torch.Tensor = calpha_to_carbonyl_term + calpha_to_nitrogen_term  # (N, 3)
    carbonyl_to_oxygen_term = carbonyl_to_oxygen_term / (
        torch.norm(carbonyl_to_oxygen_term, dim=1, keepdim=True) + 1e-7
    )

    # Create a mask that is 1 when the next residue is not available either
    # due to this frame being the C-terminus or the next residue is not
    # known due to pos_is_known being false.

    if pos_is_known is None:
        pos_is_known = torch.ones((atom_37.shape[0],), dtype=torch.int64, device=atom_37.device)

    next_res_gone: torch.Tensor = ~pos_is_known.bool()  # (N,)
    next_res_gone = torch.cat([next_res_gone, torch.ones((1,), device=pos_is_known.device).bool()], dim=0)  # (N+1, )
    next_res_gone = next_res_gone[1:]  # (N,)

    atom_37[next_res_gone, 4, :] = atom_37[next_res_gone, 2, :] + carbonyl_to_oxygen_term[next_res_gone, :] * 1.23

    return atom_37
