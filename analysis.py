#!/usr/bin/env python3

"""
ZprimeAnalysis framework for applying object and event-level systematic corrections
on NanoAOD ROOT files and producing histograms of observables like mtt. Supports both
correctionlib-based and function-based corrections.
"""
import logging
import sys

from analysis.nondiff import NonDiffAnalysis
from analysis.diff import DifferentiableAnalysis

from utils.configuration import config as ZprimeConfig
from utils.input_files import construct_fileset
from utils.schema import Config, load_config_with_restricted_cli


# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(level=logging.INFO, format="[%(name)s::%(levelname)s] %(message)s")
logger = logging.getLogger("AnalysisDriver")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

# -----------------------------
# Main Driver
# -----------------------------
def main():
    """
    Main driver function for running the Zprime analysis framework.
    Loads configuration, runs preprocessing, and dispatches analysis over datasets.
    """

    cli_args = sys.argv[1:]
    full_config = load_config_with_restricted_cli(ZprimeConfig, cli_args)
    config = Config(**full_config)  # Pydantic validation
    # ✅ You now have a fully validated config object
    logger.info(f"Luminosity: {config.general.lumi}")



    fileset = construct_fileset(
        n_files_max_per_sample=config.general.max_files
    )

    analysis_mode = config.general.analysis
    if analysis_mode == "nondiff":
        logger.info("Running Non-Differentiable Analysis")
        nondiff_analysis = NonDiffAnalysis(config)
        nondiff_analysis.run_analysis_chain(fileset)

    elif analysis_mode == "diff":
        logger.info("Running Differentiable Analysis")
        diff_analysis = DifferentiableAnalysis(config)
        diff_analysis.optimize_analysis_cuts(fileset)
    else:
        logger.info("Running both Non-Differentiable and Differentiable Analysis")
        # Non-differentiable analysis
        nondiff_analysis = NonDiffAnalysis(config)
        nondiff_analysis.run_analysis_chain(fileset)
        # Differentiable analysis
        diff_analysis = DifferentiableAnalysis(config)
        diff_analysis.optimize_analysis_cuts(fileset)


    # plot_nominal_histograms("output/histograms/histograms.root")
    # plot_cms_style(histograms_file="output/histograms/histograms.pkl")




if __name__ == "__main__":
    main()
