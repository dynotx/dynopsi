# Changelog

## [2.0.0] - 2026-11-12

Upgrading from 1.0.0: [migration guide](MIGRATION.md).

### Added

- `checkpoint:` on a simulation pipeline, and `--checkpoint` on `sample` and `check`. If both are set they must name the same weights. Either one alone selects the weights. Omitting both uses the default short name `v1.5-cicada`.
- An unpinned upgrade changes the default weights to Psi-1.5 Cicada, file [`dynopsi-1.5-cicada.ckpt`](https://huggingface.co/dynotx/dynopsi/blob/f55d9e684659aaf627b5bfa4ab05f735a3398951/dynopsi-1.5-cicada.ckpt), short name `v1.5-cicada`. The release pins Hugging Face revision `f55d9e684659aaf627b5bfa4ab05f735a3398951`; the checkpoint SHA-256 is `7a5bbc43efde45bbf04489ef280d2067751ba6f8a652c3aab8b510b91d1b3d39`.
- The default checkpoint declares new `chain_is_loopy` and `has_homooligomer_partner` conditioning inputs.
- `sphere_exclusion` is an optional `type` under `new_chains_centers_of_mass` on the featurization pipeline. `predefined_center_of_mass` and `estimate_binder_center_of_mass_radius_of_gyration` still mean what they meant in 1.0.0, and `sphere_exclusion` is not the default.

### Changed — YAML

- `design_is_loopy` on the featurization pipeline is now `design_conditioning.chain_is_loopy`. A `design_is_loopy` key left on the pipeline is rejected. The 1.0.0 examples do not set it. `ChainIsLoopyConditioningFeaturizer` still takes `design_is_loopy`.
- `score_scale_fn` and `t_thresh_score_weighting_only` are replaced by `ode_warmup_score_scale_fn`, `ode_warmup_max_t`, `ode_cooldown_score_scale_fn`, and `ode_cooldown_min_t`.
- A `type` key on a `design_pipelines` entry, such as `type: dynopsi_binder_design_pipeline`, is ignored. The entry names `featurization_pipeline`, `simulation_pipeline`, and `output_specification`.
- A `plan_init` step with `type: initialize_unconditional_pocket_features` is rejected. Remove the step.
- Examples and the documented recommendation now use `formats: [mmcif, npz]` instead of `formats: [pdb, npz]`. `formats: [pdb, npz]` still works. 1.0.0 already accepts `mmcif`. The library does not default to `mmcif`.
- YAML `inference_model_config` is rejected. Name the weights with `checkpoint:`.
- `binder_atoms_to_include` is required. Use `bb` for backbone atoms, which is what 1.0.0 used.

### Changed — Python

- `ode_warmup_score_scale_fn`, `ode_warmup_max_t`, `ode_cooldown_score_scale_fn`, and `ode_cooldown_min_t` replace `score_scale_fn` and `t_thresh_score_weighting_only` as constructor fields. An ODE configuration does not include these fields.
- `DynoPsiBinderDesignPipeline` is a deprecated alias of `DesignPipeline`. The import still succeeds and warns. The next major release removes it.
- `featurization_pipeline=` replaces `inference_featurization_pipeline=`. The YAML key was already `featurization_pipeline`.
- `DynoPsiModelConfig(repo_id=..., filename=...)` is rejected. Pass `loader=HuggingFaceLoaderConfig(...)`.
- `binder_atoms_to_include` is a required constructor field. Use `bb` for backbone atoms, which is what 1.0.0 used.

## [1.0.0] - 2026-03-18

v1 release.

## [0.0.2rc1] - 2026-03-18

Release candidate, a prerelease, published the same day twelve minutes earlier.
