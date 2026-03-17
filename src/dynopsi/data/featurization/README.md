# Definition of Indices

- **`token_index`** `(bs, seq_length)` — Index that starts at 0, increases monotonically, and does not restart for new chains.

- **`residue_index`** `(bs, seq_length)` — Index that starts at 0 and restarts for new chains; this is relative to the current structure and will change if regions are inpainted, therefore potentially increasing or decreasing the full sequence length.

- **`original_residue_index`** `(bs, seq_length)` — Index that describes the `residue_index` for the original structure. This will **not** change if regions are inpainted. If a new region is inpainted, each residue in the new region will receive the `original_residue_index` of the first residue in the original inpainted region that was removed.

- **`auth_residue_index`** `(bs, seq_length)` — Index that describes the `residue_index` for the original structure. This will **not** change if regions are inpainted. If a new region is inpainted, each residue in the new region will receive the `auth_residue_index` of the first residue in the original inpainted region that was removed.

### Example: Inpaint chain A2–4 with 5 residues (chain A has 6 residues total)

**Before inpainting:**

- `residue_index`:          [0, 1, 2, 3, 4, 5]
- `original_residue_index`: [0, 1, 2, 3, 4, 5]
- `auth_residue_index`:     [1, 2, 3, 4, 5, 6]

**After inpainting:**

- `residue_index`:          [0, 1, 2, 3, 4, 5, 6, 7]
- `original_residue_index`: [0, 1, 1, 1, 1, 1, 4, 5]
- `auth_residue_index`:     [1, 2, 2, 2, 2, 2, 5, 6]