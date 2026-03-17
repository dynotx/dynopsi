"""Verify that the simulation pipelines do not error."""

import argparse
import logging
import os

import torch

from dynopsi.data.featurization import Features, primitives
from dynopsi.data.featurization.inference_featurization_pipeline import DynoPsiBinderDesignFeaturizationPipeline
from dynopsi.model.model_config import DynoPsiModelConfig
from dynopsi.simulation import LinearTimeSampler, LogTimeSampler
from dynopsi.simulation.observable import ObservableOutputConfig
from dynopsi.simulation.simulation_pipeline.ode_simulation import ODESimulation
from dynopsi.simulation.simulation_pipeline.sde_simulation import SDESimulation
from dynopsi.simulation.time_sampler import TimeSamplerType
from dynopsi.simulation.utils import infer_batch_size
from dynopsi.utils import ConstantFunction, InverseParameterFunction, SingleParameterFunctionType

logger = logging.getLogger(__name__)

FEATURIZATION_PIPELINES: dict[str, DynoPsiBinderDesignFeaturizationPipeline] = {
    "vegfa_1bj1": DynoPsiBinderDesignFeaturizationPipeline(
        name="vegfa_1bj1",
        index_type="auth_residue_index",
        rcsb_record_id="1bj1",
        crops={
            "crop_14_107": "V14-107,W14-107",  # NOTE: chain C is author chain V, chain D is author chain W
        },
        hotspots={"hotspot_81M_83I_91I": "W81,W83,W91"},
        new_chains_lengths=[60],
        new_chains_centers_of_mass={
            "testvibecenter": primitives.PredefinedCenterOfMassEstimator(
                centers_of_mass=[[32.0603, -6.7852173, 29.20792]]
            ),
        },
    ),
    "il7ra_3di3": DynoPsiBinderDesignFeaturizationPipeline(
        name="il7ra_3di3_binder_design",
        index_type="auth_residue_index",
        rcsb_record_id="3di3",
        crops={"crop_17_209": "B17-209"},
        hotspots={"hotspot_58V_80L_139Y": "B58,B80,B139"},
        new_chains_lengths=[60],
        new_chains_centers_of_mass={
            "il7center": primitives.PredefinedCenterOfMassEstimator(centers_of_mass=[[28.1, 41.4, 47.9]]),
        },
    ),
}

OBSERVABLE_OUTPUT_CONFIGS: dict[str, ObservableOutputConfig] = {
    "X_T": ObservableOutputConfig(state_type="clean_data_coords", outputs={"final": ("pdb", "npz")}),
    "CVF_T": ObservableOutputConfig(state_type="conditional_vector_fields", outputs={"any": ("npz",)}),
}


INFERENCE_MODEL_CONFIG: DynoPsiModelConfig = DynoPsiModelConfig(repo_id="dynotx/dynopsi", filename="dynopsi-1.ckpt")

DIFFUSION_COEFFICIENT_FUNCTIONS: dict[str, SingleParameterFunctionType] = {
    "eps=0.0": InverseParameterFunction(eps=0.0),
    "eps=0.1": InverseParameterFunction(eps=0.1),
    "eps=1.0": InverseParameterFunction(eps=1.0),
}

NOISE_SCALE_FUNCTIONS: dict[str, SingleParameterFunctionType] = {
    "noise_scale=0.0": ConstantFunction(constant=0.0),
    "noise_scale=0.1": ConstantFunction(constant=0.1),
}

SCORE_SCALE_FUNCTIONS: dict[str, SingleParameterFunctionType] = {
    "score_scale=0.0": ConstantFunction(constant=0.0),
    "score_scale=1.5": ConstantFunction(constant=1.5),
}

TIME_SAMPLERS: dict[str, TimeSamplerType] = {
    "linear": LinearTimeSampler(num_steps=1),
    "log_p1": LogTimeSampler(num_steps=1, p=1.0),
}

T_THRESH_SCORE_ONLY_VALUES: dict[str, float] = {
    "t_thresh_score_only=0.0": 0.0,
    "t_thresh_score_only=1.0": 1.0,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_ode_simulation", action="store_true")
    parser.add_argument("--run_sde_simulation", action="store_true")
    args = parser.parse_args()

    # USER SETUP
    base_output_dir = f"{os.getenv('HOME')}/temp/sampling_tests"
    num_samples = 1

    # SYSTEM SETUP
    run_ode_simulation = args.run_ode_simulation
    run_sde_simulation = args.run_sde_simulation
    device_names = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    devices = [torch.device(device_name) for device_name in device_names]
    batch_size = infer_batch_size(devices)
    observable_output_configs = OBSERVABLE_OUTPUT_CONFIGS
    inference_model_config = INFERENCE_MODEL_CONFIG

    for featurization_pipeline_name, featurization_pipeline in FEATURIZATION_PIPELINES.items():
        print(f"FEATURIZATION: {featurization_pipeline_name}")
        features = next(featurization_pipeline([Features(atoms_to_include=featurization_pipeline.atoms_to_include)]))
        featurization_name = featurization_pipeline.get_featurization_name(features)
        for time_sampler_name, time_sampler in TIME_SAMPLERS.items():
            shared_simulation_kwargs = dict(
                inference_model_config=inference_model_config,
                observable_output_configs=observable_output_configs,
                time_sampler=time_sampler,
            )

            # Create ODE simulation
            if run_ode_simulation:
                ode_simulation_name = f"ode_{time_sampler_name}"
                print(f"ODE SIMULATION: {ode_simulation_name}")
                ode_simulation = ODESimulation(name=ode_simulation_name, **shared_simulation_kwargs)
                features_save_dir = os.path.join(base_output_dir, ode_simulation_name, featurization_name)
                os.makedirs(features_save_dir, exist_ok=True)
                ode_simulation.run(
                    {"features": features},
                    features_save_dir,
                    num_samples=num_samples,
                    batch_size=batch_size,
                    devices=devices,
                )

            # Create SDE simulations
            if run_sde_simulation:
                for (
                    diffusion_coefficient_function_name,
                    diffusion_coefficient_function,
                ) in DIFFUSION_COEFFICIENT_FUNCTIONS.items():
                    for noise_scale_function_name, noise_scale_function in NOISE_SCALE_FUNCTIONS.items():
                        for score_scale_function_name, score_scale_function in SCORE_SCALE_FUNCTIONS.items():
                            for (
                                t_thresh_score_only_name,
                                t_thresh_score_only_value,
                            ) in T_THRESH_SCORE_ONLY_VALUES.items():
                                additional_sde_kwargs = dict(
                                    diffusion_coefficient_fn=diffusion_coefficient_function,
                                    noise_scale_fn=noise_scale_function,
                                    score_scale_fn=score_scale_function,
                                    t_thresh_score_weighting_only=t_thresh_score_only_value,
                                )
                                sde_simulation_name = (
                                    f"sde_{time_sampler_name}"
                                    f"_{diffusion_coefficient_function_name}"
                                    f"_{noise_scale_function_name}"
                                    f"_{score_scale_function_name}"
                                    f"_{t_thresh_score_only_name}"
                                )
                                sde_simulation = SDESimulation(
                                    name=sde_simulation_name, **shared_simulation_kwargs, **additional_sde_kwargs
                                )
                                features_save_dir = os.path.join(
                                    base_output_dir, sde_simulation_name, featurization_name
                                )
                                os.makedirs(features_save_dir, exist_ok=True)
                                print(f"SDE SIMULATION: {sde_simulation_name}")
                                sde_simulation.run(
                                    {"features": features},
                                    features_save_dir,
                                    num_samples=num_samples,
                                    batch_size=batch_size,
                                    devices=devices,
                                )
