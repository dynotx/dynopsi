# Examples Table of Contents

- `configurations` - run these using the `dynopsi check` and `dynopsi sample` CLI commands
    - `il7ra_3di3_config.yaml` : The IL-7RA example from the main README.
    - `multiple_designs_multiple_targets_pipeline.yaml` : A design pipeline for each of multiple targets in a single file. 
    - `vegf_1bj1_multichain_target.yaml` : Cropping syntax in the featurization configuration for designing against a multi-chain target.
    - `multiple_residue_specifications.yaml` : Cropping and hotspots using both the `original_residue_index` and `auth_residue_index` formats. The two featurization configurations in this file differ only by the residue input format and produce exactly the same output features for simulation.
    - `alternative_time_sampler.yaml` : Log time sampler for the IL-7RA example. It also uses a different diffusion coefficient, noise scale, and ODE warmup and cooldown than `il7ra_3di3_config.yaml`.
    - `il7ra_3di3_loop_conditioning_config.yaml` : IL-7RA example. `design_conditioning` sets a loop-or-other binder and no homooligomer partner.
- notebooks
    - `binder_design_pipeline_il7ra_3di3.ipynb` : IL-7RA example from the main README in a notebook environment.

The six configuration examples above use Cicada (Psi-1.5 Cicada), the default
checkpoint with short name `v1.5-cicada`.
Five use the non-loop, no-partner values. The loop example sets `LOOP_OR_OTHER` with `NO_PARTNER`.

# Customizing Your Design
## Customizing Featurization
### Chain IDs and Residue Indices
Chain ids and residue positions can be specified in the target crop and hotspot inputs using either the original or author residue index options:
- `auth_residue_index`: Use the Author-defined indexing: residue indices and chain IDs are defined by the author of the deposited structure file.
- `original_residue_index`: chains are labeled A, B, C, … and each chain's residues start at 1.

We recommend loading your structure in [Mol*Viewer](https://molstar.org/viewer/) to select chains and indices. Mol*Viewer explicitly annotates with "AUTH" in scenarios where the author-defined chain ID and residue indexing diverges from standard PDB indexing.

For example, these are two equivalent featurizations for [TrkA](https://www.rcsb.org/structure/1WWW) (see also `example/configurations/multiple_residue_specifications.yaml` for a full configuration file example):

    - Author residue index: 
        - Crop X282-382
        - Hotspot X294,X296,X333
    - `original_residue_index`:
        - Crop C1-101
        - Hotspot C13,C15,C52


### Choosing a binder center of mass
Our model requires the user to specify an estimate for the center of mass of the binder chain relative to the target. We have achieved the best results by defining custom centers of mass. Below is an example of how to select a center of mass for [IL-7RA](https://www.rcsb.org/structure/3DI3) in [PyMOL](https://www.pymol.org/).

```pymol
# load the target structure into PyMOL
fetch 3di3
hide everything
show cartoon, (chain A and 3di3) or (chain B and 3di3)
color gray, chain B and 3di3

# create a pseudo atom with a Van-der Waals radius of 10A
# empirically this works for binders of length 60-120
pseudoatom atom, pos=(0,0,0), vdw=10
set sphere_transparency, 0.5, selection=atom
color yellow, atom
```
Drag the pseudoatom to the desired position, then retrieve the center of mass with the following command:
```pymol
print(cmd.get_atom_coords("atom"))
```

## Customizing Simulation
### Time samplers
Three time samplers are currently supported for controlling the denoising trajectory schedule:

`YAML` / `Python`
- `linear_time_sampler` / `LinearTimeSampler` : evenly spaces `num_steps` timesteps between `start_time` and `end_time`. We recommend this as a starting point with `start_time = 0.0` and `end_time = 1.0`.
- `log_time_sampler` / `LogTimeSampler` : spaces steps on a logarithmic scale controlled by parameter `p > 0`. Higher `p` concentrates more steps near the end of the trajectory.
- `quadratic_time_sampler` / `QuadraticTimeSampler` : steps follow `(x / num_steps)^2`, with more steps near `t=0`.

Higher `num_steps` generally produces better results, with diminishing returns above 100 steps. The `sde_simulation` defaults are `num_steps: 60` and `self_conditioning_mode: previous_step`. Psi-1 settings are in the [README checkpoints section](https://github.com/dynotx/dynopsi/blob/main/README.md#checkpoints).

### Time-dependent weighting of SDE/ODE component terms
The ODE and SDE component terms receive time-dependent weights according to the diffusion coefficient, noise scale, and score scale inputs. These inputs are parameterized as functions of time `t ∈ [0, 1]`, and four functional forms are currently supported:

`YAML` / `Python`
- `constant` / `ConstantFunction`
- `power_law` / `PowerLawFunction`:  `a * t^b + c`, clamped to `[min_val, max_val]`.
- `inverse_parameter` / `InverseParameterFunction`: `1 / (eps + t)`, optionally clamped with `clamp_min` and `clamp_max`.
- `piecewise_power` / `PiecewisePowerFunction`: ramps from 0 to `y_max_value` and back to 0.
