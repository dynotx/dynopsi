import argparse
import logging
import os
import shutil
import sys
from typing import Literal

from dynopsi.design_pipeline import DesignPipeline
from dynopsi.design_pipeline_parser import DesignLanguageParser

logger = logging.getLogger(__name__)


def _load_design_pipelines(config_path: str) -> list[DesignPipeline]:
    """Load design pipelines from a YAML config file."""
    if not (config_path.endswith(".yaml") or config_path.endswith(".yml")):
        raise ValueError(f"Unsupported config format: {config_path}. Use a .yaml or .yml file.")
    parser = DesignLanguageParser()
    return parser.parse(config_path)


def _cmd_sample(
    config_path: str,
    num_samples: int,
    batch_size: int | None,
    overwrite: bool,
) -> None:
    """Load a config and generate samples.

    Parameters
    ----------
    config_path : str, path to the configuration file specifying the design pipeline.
    num_samples : int, the number of samples to generate per featurization specification.
    batch_size : int, batch size for inference. If None, it will be inferred by the available CUDA-enabled GPUs.
    overwrite : bool, whether to check if the output directory for a `design_pipeline` already exists. If True, entire folder for the design pipeline will be removed and rewritten.

    Raises
    ------
    FileExistsError if the output directory for a `design_pipeline` already exists and `overwrite = False`.
    """
    design_pipelines = _load_design_pipelines(config_path)

    for design_pipeline in design_pipelines:
        save_dir = os.path.join(
            design_pipeline.output_specification.output_dir,
            design_pipeline.name,
            design_pipeline.simulation_pipeline.name,
        )
        if os.path.exists(save_dir):
            if not overwrite:
                raise FileExistsError(
                    f"Output directory {save_dir} already exists. Use --overwrite to allow overwriting."
                )
            shutil.rmtree(save_dir)
        design_pipeline.sample(num_samples, batch_size=batch_size)


def _cmd_check(
    config_path: str,
    output_format: Literal["pdb", "mmcif"],
    overwrite: bool,
) -> None:
    """Load config and run design_pipeline.verify_configuration(). Output dir is from the config YAML.

    Parameters
    ----------
    config_path : str, path to the configuration file specifying the design pipelines.
    output_format : Literal["pdb", "mmcif"], output format for the featurization preview (CLI "cif" is converted to mmcif before calling).
    overwrite : bool, whether to check if the output directory for a `design_pipeline` already exists. If True, entire folder for the `design_pipeline` featurization preview will be removed and rewritten.

    Raises
    ------
    FileExistsError if the output directory for a `design_pipeline` already exists and `overwrite = False`.
    """
    design_pipelines = _load_design_pipelines(config_path)

    for design_pipeline in design_pipelines:
        verify_outdir = os.path.join(
            design_pipeline.output_specification.output_dir,
            design_pipeline.name,
            "verify_configuration",
        )
        if os.path.exists(verify_outdir):
            if not overwrite:
                raise FileExistsError(
                    f"Output directory {verify_outdir} already exists. Use --overwrite to allow overwriting."
                )
            shutil.rmtree(verify_outdir)

    for design_pipeline in design_pipelines:
        design_pipeline.verify_configuration(output_format=output_format)

    logger.info("Check complete: config loaded, verification outputs and validation done (output dir from config).")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="dynopsi",
        description="Ψ Dyno Psi-1: Sampling pipelines for de novo miniprotein binder design.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # Sample command parsing
    sample_parser = subparsers.add_parser("sample", help="Run a binder design sampling pipeline from YAML config")
    sample_parser.add_argument(
        "config",
        type=str,
        help="Path to YAML config file (.yaml or .yml). Output directory is defined in the config.",
    )
    sample_parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to generate (default: 10)",
    )
    sample_parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for sampling (default: inferred from GPU memory)",
    )
    sample_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output directory",
    )

    # Check command parsing
    check_parser = subparsers.add_parser(
        "check",
        help="Dry run the design specification to verify that the solver configuration is valid and write featurization as PDB/CIF for inspection",
    )
    check_parser.add_argument(
        "config",
        type=str,
        help="Path to YAML config file (.yaml or .yml). Output directory is defined in the config.",
    )
    check_parser.add_argument(
        "--format",
        type=str,
        choices=["pdb", "cif"],
        default="pdb",
        help="Output structure format for featurization (default: pdb)",
    )
    check_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output directory",
    )

    args = parser.parse_args()

    if args.command == "sample":
        _cmd_sample(
            config_path=args.config,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
    elif args.command == "check":
        # CLI exposes pdb/cif; convert cif -> mmcif so internally we only see pdb or mmcif
        output_format: Literal["pdb", "mmcif"] = "mmcif" if args.format == "cif" else "pdb"
        _cmd_check(
            config_path=args.config,
            output_format=output_format,
            overwrite=args.overwrite,
        )
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
