#!/usr/bin/env python3

"""
ZprimeAnalysis framework for applying object and event-level systematic corrections
on NanoAOD ROOT files and producing histograms of observables like mtt. Supports both
correctionlib-based and function-based corrections.
"""
import logging
import sys

from analysis.diff import DifferentiableAnalysis
from analysis.nondiff import NonDiffAnalysis
from utils.configuration import config as ZprimeConfig
from utils.input_files import construct_fileset
from utils.logging import ColoredFormatter
from utils.schema import Config, load_config_with_restricted_cli

# -----------------------------
# Logging Configuration
# -----------------------------
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter())
if root_logger.hasHandlers():
    root_logger.handlers.clear()
root_logger.addHandler(handler)

logger = logging.getLogger("AnalysisDriver")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

# ANSI color codes
MAGENTA = "\033[95m"
RESET = "\033[0m"

def _banner(text: str) -> str:
    """Creates a magenta-colored banner for logging."""
    return (
        f"\n{MAGENTA}\n{'=' * 80}\n"
        f"{' ' * ((80 - len(text)) // 2)}{text.upper()}\n"
        f"{'=' * 80}{RESET}"
    )
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
    logger.info(f"Luminosity: {config.general.lumi}")

    fileset = construct_fileset(
        max_files_per_sample=config.general.max_files
    )

    analysis_mode = config.general.analysis
    if analysis_mode == "nondiff":
        logger.info(_banner("Running Non-Differentiable Analysis"))
        nondiff_analysis = NonDiffAnalysis(config)
        nondiff_analysis.run_analysis_chain(fileset)

    elif analysis_mode == "diff":
        logger.info(_banner("Running Differentiable Analysis"))
        diff_analysis = DifferentiableAnalysis(config)
        diff_analysis.run_analysis_optimisation(fileset)
    else:
        logger.info(_banner("Running both Non-Differentiable and Differentiable Analysis"))
        # Non-differentiable analysis
        logger.info("Running Non-Differentiable Analysis")
        nondiff_analysis = NonDiffAnalysis(config)
        nondiff_analysis.run_analysis_chain(fileset)
        # Differentiable analysis
        logger.info("Running Differentiable Analysis")
        diff_analysis = DifferentiableAnalysis(config)
        diff_analysis.run_analysis_optimisation(fileset)


if __name__ == "__main__":
    main()
