# Migrating from 1.0.0 to 2.0.0

Package 1.0.0 → 2.0.0. Default model: Psi-1 (`dynopsi-1.ckpt`, short name `v1`) → Psi-1.5 Cicada (`dynopsi-1.5-cicada.ckpt`, short name `v1.5-cicada`).

`dynopsi` 2.0.0 is the package; the default model is Psi-1.5 Cicada. Psi-1 remains `checkpoint: v1`.

`dynopsi==1.0.0` supports Python `>=3.10,<3.11`. Version 2.0.0 supports Python `>=3.12,<3.13`. An unpinned install gets 2.0.0 in a Python 3.12 environment. On Python 3.10, create a Python 3.12 environment before installing `dynopsi==2.0.0`. To keep running 1.0.0, see [Checkpoint](#checkpoint).

## Representative YAML upgrade

A 1.0.0 example that moves to Cicada and the default SDE schedule changes like this:

```diff
 featurization_pipelines:
     il7ra_3di3_binder_design:
         type: dynopsi_binder_design_featurization_pipeline
+        binder_atoms_to_include: bb
+        design_conditioning:
+            chain_is_loopy:
+                non_loopy: NON_LOOP
+            has_homooligomer_partner:
+                no_partner: NO_PARTNER

 simulation_pipelines:
     sde_default:
         type: sde_simulation
-        score_scale_fn:
-            type: constant
-            constant: 1.5
-        t_thresh_score_weighting_only: 0.9
+        ode_warmup_score_scale_fn:
+            type: constant
+            constant: 1.0
+        ode_warmup_max_t: 0.30
+        ode_cooldown_score_scale_fn:
+            type: constant
+            constant: 1.3
+        ode_cooldown_min_t: 0.87
```

`non_loopy` and `no_partner` are labels you choose. To keep the 1.0.0 weights, set `checkpoint: v1` on the simulation pipeline and omit `design_conditioning`. `v1` does not declare those keys. The diff uses the 2.0.0 schedule. The 1.0.0 threshold is under [Change 2](#change-2).

## YAML

A configuration shaped like the 1.0.0 examples fails in this order:

1. [Change 9](#change-9).
2. [Change 2](#change-2).
3. The conditioning check below.

A simulation pipeline entry that contains `inference_model_config` fails at [Change 8](#change-8) after change 9 and before change 2. A 1.0.0 example has no such block and skips change 8.

If you only write YAML, skip [the Python failure order](#python).

### Conditioning check

After the configuration parses, `dynopsi check` compares its conditioning keys with the checkpoint. Cicada declares `chain_is_loopy` and `has_homooligomer_partner`. A 1.0.0 configuration sets neither. Sampling that checkpoint uses `NON_LOOP` and `NO_PARTNER`. Set both on the featurization pipeline:

```yaml
design_conditioning:
  chain_is_loopy:
    <name>: NON_LOOP
  has_homooligomer_partner:
    <name>: NO_PARTNER
```

`<name>` is a label you choose. A declared key that is omitted, or set to an empty map, raises `ConditioningCheckpointError` and names `design_conditioning.<key>` as missing.

## Python

An SDE configuration fails in this order:

1. [Change 9](#change-9).
2. [Change 8](#change-8).
3. [Change 2](#change-2).
4. [Change 4](#change-4).

An ODE configuration skips change 2: changes 9, 8, and 4.

[Change 3](#change-3) keeps `DynoPsiBinderDesignPipeline` as a deprecated alias of `DesignPipeline`. It is not one of these failures.

## The nine changes

### Change 1

**Old form.** The 1.0.0 package has no `design_is_loopy` key, so a configuration taken from the 1.0.0 examples has nothing to rename. A featurization pipeline that set it used:

```yaml
design_is_loopy:
  <name>: <value>
```

**New form.**

```yaml
design_conditioning:
  chain_is_loopy:
    <name>: <value>
```

**Required change.** Rename that pipeline field. `ChainIsLoopyConditioningFeaturizer` still takes `design_is_loopy`. A `design_is_loopy` key left on the pipeline is rejected. An unrecognized key inside `design_conditioning` is rejected.

### Change 2

**Old form.** On an SDE simulation, as YAML or as Python constructor fields. The 1.0.0 example uses these values:

```yaml
score_scale_fn:
  type: constant
  constant: 1.5
t_thresh_score_weighting_only: 0.9
```

**New form.** These fields replace them and already have defaults:

| Field | Default |
| --- | --- |
| `ode_warmup_score_scale_fn` | constant `1.0` |
| `ode_warmup_max_t` | `0.30` |
| `ode_cooldown_score_scale_fn` | constant `1.3` |
| `ode_cooldown_min_t` | `0.87` |

**Required change.** Delete `score_scale_fn` and `t_thresh_score_weighting_only`. Deleting them passes validation. Keeping the old names fails validation on field `score_scale_fn` and field `t_thresh_score_weighting_only`, type `extra_forbidden`. Omitting the four new fields does not fail. The defaults in the table are the 2.0.0 schedule, not a rename of the old values.

The warmup band is `t <= ode_warmup_max_t`. A negative `ode_warmup_max_t` leaves that band empty. Require `ode_warmup_max_t` < `ode_cooldown_min_t`. The two warmup fields have no 1.0.0 equivalent. To keep the old numbers, set `ode_cooldown_min_t` to the old threshold and `ode_cooldown_score_scale_fn` to the old function. The 1.0.0 schedule, one threshold and no warmup band, needs a negative `ode_warmup_max_t` plus both cooldown fields set to the old values:

```yaml
ode_warmup_max_t: -1
ode_cooldown_score_scale_fn:
  type: constant
  constant: 1.5
ode_cooldown_min_t: 0.9
```

Leave `diffusion_coefficient_fn` and `noise_scale_fn` as they are. An SDE configuration hits this failure. An ODE configuration does not. This `-1` block reproduces 1.0.0. The README Psi-1 starting schedule uses `ode_warmup_max_t: 0.10`.

### Change 3

**Old form.** `from dynopsi import DynoPsiBinderDesignPipeline`.

**New form.** `from dynopsi import DesignPipeline`.

**Required change.** The old name remains a deprecated alias of `DesignPipeline`. The import still succeeds and warns. The next major release removes the alias. The alias does not make a 1.0.0 configuration run.

### Change 4

**Old form.** `DesignPipeline(inference_featurization_pipeline=...)`.

**New form.** `DesignPipeline(featurization_pipeline=...)`.

**Required change.** Pass `featurization_pipeline`. The YAML key was already `featurization_pipeline` in 1.0.0. This change is Python only. Passing the old keyword fails validation on field `featurization_pipeline`, type `missing`, and on field `inference_featurization_pipeline`, type `extra_forbidden`.

### Change 5

**Old form.** A `type` key on a `design_pipelines` entry, such as `type: dynopsi_binder_design_pipeline`.

**New form.** The same entry naming `featurization_pipeline`, `simulation_pipeline`, and `output_specification`, without `type`.

**Required change.** None for a configuration that already runs. 2.0.0 reads those three names and drops every other key on the entry, including `type`, as 1.0.0 did.

### Change 6

**Old form.** A `plan_init` step with `type: initialize_unconditional_pocket_features`.

**New form.** The step is absent.

**Required change.** Remove the step. That `type` is rejected. No 1.0.0 example sets it.

### Change 7

**Old form.** The 1.0.0 examples use `formats: [pdb, npz]`.

**New form.** The 2.0.0 examples use `formats: [mmcif, npz]`.

**Required change.** None if you keep `formats: [pdb, npz]`. That list still works. 1.0.0 already accepts `mmcif` in `formats`. The recommendation moved. The library does not default to `mmcif`.

Leaving `formats` out of an `output_dir` / `formats` / `save_trajectory` block raises. Leaving it out is valid when `observable_output_configs` supplies the formats. In the `formats` list, and in the Python `output_format` argument, the spelling is `mmcif`. `dynopsi check --format` accepts `pdb` or `cif` and converts `cif` to `mmcif`. Sampling writes `mmcif` as `.cif`. `dynopsi check --format cif` writes the preview as `.mmcif`.

### Change 8

**Old form.** YAML may carry an `inference_model_config` block on a simulation pipeline entry. In Python:

```python
DynoPsiModelConfig(repo_id="dynotx/dynopsi", filename="dynopsi-1.ckpt")
```

**New form.** On that same simulation pipeline entry, name a published model by short name:

```yaml
simulation_pipelines:
  <name>:
    checkpoint: v1.5-cicada
```

Python passes one loader:

```python
from dynopsi.model.model_config import DynoPsiModelConfig, HuggingFaceLoaderConfig

DynoPsiModelConfig(
    loader=HuggingFaceLoaderConfig(
        repo_id="dynotx/dynopsi",
        filename="dynopsi-1.5-cicada.ckpt",
        revision="f55d9e684659aaf627b5bfa4ab05f735a3398951",
    ),
)
```

**Required change.** Delete the YAML `inference_model_config` key. 2.0.0 rejects it. Set `checkpoint:` to the short name of a published model. In Python, those three failures arrive together: field `loader` has type `missing`, and fields `repo_id` and `filename` have type `extra_forbidden`.

### Change 9

**Old form.** `binder_atoms_to_include` is absent. 1.0.0 used backbone atoms.

**New form.** On the featurization pipeline, in YAML or on `DynoPsiBinderDesignFeaturizationPipeline`. Accepted values are backbone (`bb`) and Cα (`ca`). The field is required and has no default.

```yaml
binder_atoms_to_include: bb
```

**Required change.** Add `binder_atoms_to_include: bb`. That restores the 1.0.0 behavior. Omitting the field fails validation on field `binder_atoms_to_include`, type `missing`. Every 1.0.0 configuration omits it.

## Checkpoint

Pin `dynopsi==1.0.0` to keep running 1.0.0. An unpinned upgrade changes the default weights to Cicada. The file is `dynopsi-1.5-cicada.ckpt` in `dynotx/dynopsi`. The short name for `checkpoint:` and `--checkpoint` is `v1.5-cicada`.

## Other behavior on a 1.0.0 example

An SDE configuration that omits `validate_before_run` stops validating before the run; set `validate_before_run: true` to keep the 1.0.0 behavior. `predefined_center_of_mass` and `estimate_binder_center_of_mass_radius_of_gyration` still mean what they meant in 1.0.0, and `sphere_exclusion` is optional and is not the default.
