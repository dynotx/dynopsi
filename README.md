# Dyno Psi-1

[Link to white paper](link-to-paper)

This repository enables inference and sampling for Dyno Psi-1, a *de novo* miniprotein binder design model. The binder design pipeline is configurable and modular, supporting binder generation against both single and multi-chain targets. Backbone atom coordinates are output for each designed binder and can be exported in multiple formats for downstream analysis. [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) is recommended to generate accompanying sequences, and recommended settings are provided below [in the sequence design section](#sequence-design-with-proteinmpnn).
Dyno Psi model checkpoints and supplementary datasets are hosted on Hugging Face
[![🤗 Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-DynoTx-F1E0CC)](https://huggingface.co/dynotx).

## Installation

### Requirements

- Python 3.10+
- CUDA-enabled GPU

### Setup

```bash
pip install dynopsi
```

## Overview

A binder design run is defined by a configuration, specified as a YAML file or directly in Python. The example below demonstrates both options, and a more detailed explanation of possible configurations can be found in `example/README.md`. The configuration contains sections for defining featurization parameters (e.g. binder lengths, target crops, hotspot residues), simulation parameters (e.g. ODE/SDE solvers, time schedules), and output formats and file locations. The input features to the model defined by the configuration can be viewed and validated before starting sampling. Generated structure samples are the inputs to sequence design and filtering steps.
Each step is decribed in more detail in the sections below:

1. [Configuration](#configuration)
    1. [Featurization](#featurization)
    2. [Simulation](#simulation)
    3. [Output](#output-configuration)
    4. [Putting it all together](#putting-it-all-together)
2. [Input Validation](#check-design-pipeline)
3. [Backbone Generation!](#run-design-pipeline)
4. [Sequence Design with ProteinMPNN](#sequence-design-with-proteinmpnn)
5. [Filter Designs](#filter-designs)

## Configuration

The design pipeline consists of three separately configurable components, which you can choose to configure as a YAML file or directly in Python (which provides more flexibility). These are
(1) [Featurization](#featurization), (2) [Simulation](#simulation), and (3) [Output Configuration](#output-configuration).

### Featurization

This is where you provide a target, optionally crop your target, specify hotspots, etc. For the target, you can provide either an RCSB record ID, an AFDB ID, or a local path to a structure file. For the first two options, the structure will be downloaded. For binder chains to be designed, you must specify the desired length (or multiple lengths) as well as an estimated center of mass of the final design. Our recommendation is to load your structure in a visualization tool like PyMOL and manually position the binder relative to the target.

### Required inputs to featurization

- `name` : A user-readable name that defines this featurization setup.
- Define your target structure with exactly one of the following:
  - `rcsb_record_id` : The RCSB ID of your target; it will be downloaded from the RCSB PDB. This requires an internet connection.
  - `afdb_uniprot_id` : The Uniprot ID of the target; it will be downloaded from AFDB (v6). All isoforms are allowed (e.g. P16871, P16871-2). This requires an internet connection.
  - `structure_path` : Local path to the target structure, in .pdb or .mmcif format.
- Define the chain ID / indexing strategy used for the featurization specification
  - `index_type` : Which index strategy is used. Options are `original_residue_index`, `auth_residue_index`, described below:
    - `auth_residue_index`: Use the Author-defined indexing: residue indices and chain IDs are defined by the author of the deposited structure file.
    - `original_residue_index`: Use standard PDB indexing: all residue indices start at 1 within each chain and the chain ID is the standard PDB chain ID.

    We recommend loading your structure in [Mol*Viewer](https://molstar.org/viewer/) to select chains and indices. Mol*Viewer explicitly annotates with "AUTH" in scenarios where the author-defined chain ID and residue indexing diverges from standard PDB indexing.

    It is strongly recommended that you [check](#check-design-pipeline) the output featurization before generating designs to make sure that you have defined the correct chain/index parameters, especially if you use `auth_residue_index`.

Four additional featurization arguments are required: `crops`, `hotspots`, `new_chains_lengths`, and `new_chains_centers_of_mass`. Each accepts multiple configurations, which are combined in a matrix, producing one featurization for every combination across all four arguments. Separate simulations will be run sequentially for each combination specified.

- `crops` : A dictionary of {crop_name : crop_range}.
- `hotspots` : A dictionary of {hotspot_name : hotspot_positions}.
- `new_chains_lengths` : A list of different chain lengths to generate.
- `new_chains_centers_of_mass` : A dictionary of {center_name : center_of_mass_function}.

### An example

The same binder design task can be specified using either YAML or Python.

YAML version:

```yaml
featurization_pipelines:
    il7ra_3di3_binder_design:  # name this something unique
        type: dynopsi_binder_design_featurization_pipeline  # this is the only option currently supported
        index_type: auth_residue_index  # `auth_residue_index` or `original_residue_index`
        rcsb_record_id: 3di3
        # afdb_uniprot_id: P16871
        # structure_path: /path/to/pdb_or_cif/file
        crops:
            crop_17_209: B17-209  # name crop_17_209 can be whatever you want
        hotspots:
            hotspot_58V_80L_139Y: B58,B80,B139  # name hotspot_58V_80L_139Y can be whatever you want
        new_chains_lengths: [60, 80, 100, 120] # produce binders of length 60, 80, 100, 120
        new_chains_centers_of_mass:
            il7center:  # name il7center can be whatever you want
                type: predefined_center_of_mass
                centers_of_mass: [[28.1, 41.4, 47.9]]
```

Python version:

```python
from dynopsi.data.featurization import DynoPsiBinderDesignFeaturizationPipeline, primitives

featurization_pipeline = DynoPsiBinderDesignFeaturizationPipeline(
    name="il7ra_3di3_binder_design",
    index_type="auth_residue_index",
    rcsb_record_id="3di3",
    crops={"crop_17_209": "B17-209"},
    hotspots={"hotspot_58V_80L_139Y": "B58,B80,B139"},
    new_chains_lengths=[60, 80, 100, 120],
    new_chains_centers_of_mass={
        "il7center": primitives.PredefinedCenterOfMassEstimator(centers_of_mass=[[28.1, 41.4, 47.9]])
    },
)
```

### Simulation

The simulation configuration controls the sampler, including which solver to use (ODE/SDE) and its parameters. Beyond the default solvers, sampling behavior can be customized by composing `Ops` — instructions that define the denoising trajectory — into a `Simulation` procedure. This makes it straightforward to introduce guidance (coming soon!), impose symmetry, or otherwise modify the sampling dynamics. For vanilla miniprotein binder design, we recommend the following defaults:

YAML version:

```yaml
simulation_pipelines:
    sde_100_steps_default:  # name this whatever you want
        type: sde_simulation  # supported types include `ode_simulation` and `sde_simulation`
        time_sampler:
            type: linear_time_sampler  # supported types include `linear_time_sampler` and `log_time_sampler`
            num_steps: 100  # sample quality generally increases with more steps, but we see saturation ~100
        diffusion_coefficient_fn:
            type: inverse_parameter
            eps: 0.02
            clamp_max: 10.0
        noise_scale_fn:
            type: constant
            constant: 0.1
        score_scale_fn:
            type: constant
            constant: 1.5
        t_thresh_score_weighting_only: 0.9
```

Python version:

```python
from dynopsi.simulation import LinearTimeSampler, SDESimulation
from dynopsi.utils import ConstantFunction, InverseParameterFunction

simulation = SDESimulation(
    name="SDE_linear_100_steps",
    inference_model_config=DynoPsiModelConfig(
        repo_id="dynotx/dynopsi", filename="dynopsi-1.ckpt",
    ),
    diffusion_coefficient_fn=InverseParameterFunction(eps=0.1),
    noise_scale_fn=ConstantFunction(constant=0.1),
    score_scale_fn=ConstantFunction(constant=1.5),
    t_thresh_score_weighting_only=0.9,
    time_sampler=LinearTimeSampler(num_steps=100),
)
```

### Output Configuration

Specify the output format (`.pdb` or `.cif`), output directory, and whether to save the full sampling trajectory.

> **Note:** Saving trajectories requires significantly more time and disk space and is not recommended for large-scale runs.

YAML version:

```yaml
output_specifications:
    default_output_specification:
        output_dir: /path/to/outputs
        formats: [pdb, npz]
        save_trajectory: False
```

Python version:

```python
from dynopsi import OutputSpecification

output_specification = OutputSpecification(
    output_dir="./example/output",
    formats=["pdb", "npz"],
    save_trajectory=False,
)
```

### Putting it all together

Combine the pieces you specified above into a design pipeline.

YAML version:

```yaml
design_pipelines:
    il7ra_3di3_binder_design:
        type: dynopsi_binder_design_pipeline
        featurization_pipeline: il7ra_3di3_binder_design
        simulation_pipeline: sde_100_steps_default
        output_specification: default_output_specification
```

Python version:

```python
from dynopsi import DynoPsiBinderDesignPipeline

design_pipeline = DynoPsiBinderDesignPipeline(
    name="dynopsi_binder_design_pipeline",
    inference_featurization_pipeline=featurization_pipeline,
    simulation_pipeline=simulation,
    output_specification=output_specification
)
```

## Validate & Run a Design Pipeline

The `dynopsi` CLI command only supports calls to `.yaml` configurations, with examples shown below. If you prefer to run design pipelines via Python scripts, refer to the `.ipynb` examples in `example/notebooks/` for a starting point to define your own script.

### Check design pipeline

Dry run the design specification and verify that the solver configuration is valid. It is strongly reccommended to run this validation and inspect your featurization, especially before long sampling runs.

Optional Arguments:

- `output_format`: File format for the output structure. Accepted values: `"pdb"`, `"mmcif"`.
- `overwrite` : Boolean of whether to overwrite samples. If False, checks for the presence of /path/to/outputs/<design_pipeline_name>/verify_configuration. If True, clears the entire directory prior to verifying the configuration.

CLI + YAML version:

```bash
dynopsi check example/configurations/il7ra_3di3_config.yaml
```

Python version:

```python
design_pipeline.verify_configuration()
```

View the featurization configuration (path below) in a structure viewer. Color by b-factor; this will show

| B Factor | Hotspot Feature                |
|----------|--------------------------------|
| 100      | hotspot                        |
| 50       | binder (center of mass)        |
| 0        | other target residues          |

```text
/path/to/outputs/
└── <design_pipeline_name>/
    └── verify_configuration/
        └── <reference_structure_info>___<crop_info>___<hotspot_info>___length<length>_<center_of_mass_name>.pdb
```

### Run design pipeline

Required arguments

- `num_samples` : Each matrix-product of featurization that you defined will generate `num_samples` designs.

Optional arguments:

- `overwrite` : Boolean of whether to overwrite samples. If False, checks for the presence of /path/to/outputs/<design_pipeline_name>/<simulation_name>. If True, clears the entire directory prior to sampling.
- `batch_size` : Size of batch for inference; if not provided, batch size is inferred from the available CUDA-enabled GPUs.

CLI + YAML version:

```bash
dynopsi sample example/configurations/il7ra_3di3_config.yaml --num_samples 10
```

Python version:

```python
design_pipeline.sample(num_samples=10)
```

### Inspect outputs

Outputs will be saved with the following pattern:

```text
/path/to/outputs/
└── <design_pipeline_name>/
    └── <simulation_name>/
        └── <featurization_pipeline_name>/
            └── <reference_structure_info>___<crop_info>___<hotspot_info>___length<length>_<center_of_mass_name>/
                └── X_T_final_state_<sample_idx>.<output_format>
                └── X_T_trajectory_<sample_idx>.<output_format>
```

## Sequence design with ProteinMPNN

Dyno Psi-1 generates backbone (N, CA, C, O) coordinates for the binder, and preserves the original target sidechains. We recommend designing the binder sequence with target-aware [ProteinMPNN](https://github.com/dauparas/ProteinMPNN).  

```bash
python protein_mpnn_run.py \
--model_name v_48_020 \
--sampling_temp 0.0001 \ # low sampling temp leads to higher designability, but lower diversity
--backbone_noise 0.0 \
--omit_AAs C \
--num_seqs_per_target 4 \
--use_soluble_model \
--pdb_path_chains <binder_chain_id> \ 
--fixed_positions_jsonl \ # path to JSON representing dict with {chain_id: list[fixed_residues]}. We recommend including the entire target.
--pdb_path <dyno_psi_design.pdb> \
--out_folder <...>
```

## Filter designs

We strongly recommend that you refold designs with AlphaFold2 (AF2) and filter on binder and interface quality metrics. The white paper shows filtering results from refolding with AF2 monomer in initial guess mode with a target template provided and 3 recycles. The designed structures are compared to the refolded structures, and designs that satify the following constraints are retained.

1. Binder RMSD (designed complex vs. refolded complex) < 1 Angstrom
2. Binder pLDDT > 0.8
3. Inter-chain pAE (ipAE) < 10

The filter thresholds match the binder design in silico benchmarking thresholds from [Watson, Juergens, Bennett (2023)](https://doi.org/10.1038/s41586-023-06415-8).

We show the specific code for refolding using [ColabDesign](https://github.com/sokrypton/ColabDesign/tree/main) below. We use the Kabsch alignment algorithm to align the designed and refolded binder to compute the Binder RMSD.

```python
complex_prediction_model = mk_afdesign_model(
    protocol="binder",
    num_recycles=3,
    data_dir="/path/to/weights”,
    use_multimer=False,
    use_initial_guess=True,
    use_initial_atom_pos=False,
)

complex_prediction_model.prep_inputs(
    pdb_filename=<dyno_psi_design.pdb>,
    chain=<target_chain_id>,
    binder_chain=<binder_chain_id>,
    binder_len=<length_binder_sequence>,
    use_binder_template=False,
    rm_target_seq=False,
    rm_target_sc=False,
    rm_template_ic=True,
)

models = ["model_1_ptm"]

complex_prediction_model.predict(
        seq=<binder_sequence>,
        models=models,
        num_models=len(models),
        sample_models=False,
        num_recycles=3,
        verbose=False,
    )
```
