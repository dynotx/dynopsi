import argparse
import os

from dynopsi.design_pipeline_parser import DesignLanguageParser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--overwrite", type=bool, required=False, default=False)
    parser.add_argument("--batch_size", type=int, required=False, default=None)
    args = parser.parse_args()

    design_language_parser = DesignLanguageParser()
    design_pipelines = design_language_parser.parse(args.config_path)
    # check if the output directory exists
    for design_pipeline in design_pipelines:
        if os.path.exists(design_pipeline.output_specification.output_dir) and not args.overwrite:
            raise ValueError(f"Output directory {design_pipeline.output_specification.output_dir} already exists.")
        design_pipeline.sample(args.num_samples, batch_size=args.batch_size)
