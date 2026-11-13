from pathlib import Path

import numpy as np


def write_ca_pdb(
    path: Path,
    chain_id: str,
    ca_coords: np.ndarray,
    *,
    res_name: str = "ALA",
) -> Path:
    lines: list[str] = []
    for index, xyz in enumerate(ca_coords):
        x, y, z = (float(value) for value in xyz)
        serial = index + 1
        resseq = index + 1
        lines.append(
            f"ATOM  {serial:5d}  CA  {res_name} {chain_id}{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def linear_ca(n: int) -> np.ndarray:
    return np.stack([np.arange(n, dtype=float) * 3.8, np.zeros(n), np.zeros(n)], axis=1)


def folded_ca(n: int) -> np.ndarray:
    """Compact 3-wide zigzag, a different fold from a straight CA line."""
    coords = np.zeros((n, 3), dtype=float)
    for i in range(n):
        coords[i] = [(i % 3) * 3.8, (i // 3) * 3.8, (i % 2) * 2.0]
    return coords
