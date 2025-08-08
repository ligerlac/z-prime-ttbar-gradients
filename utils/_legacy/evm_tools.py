import logging

import jax
import jax.numpy as jnp
import evermore as evm
from typing import NamedTuple, Dict
from jaxtyping import Array, PyTree
from tabulate import tabulate

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s: %(name)s: %(lineno)s - %(funcName)20s()] %(message)s",
)
logger = logging.getLogger("EvmTools")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

# =============================================================================
# Data Structures
# =============================================================================


# =============================================================================
# FitResult
# =============================================================================
class FitResult(NamedTuple):
    """
    Container for fit results with associated uncertainties.

    Attributes
    ----------
    params : PyTree
        Fitted parameter values as a PyTree structure
    loss : float
        Final negative log-likelihood value
    uncertainties : Dict[str, float]
        Parameter uncertainties keyed by parameter name
    covariance : Array
        Covariance matrix of unfrozen parameters
    param_values : Dict[str, float]
        Parameter values for summary printing (name: value)
    """

    params: PyTree
    loss: float
    uncertainties: Dict[str, float]
    covariance: Array
    param_values: Dict[str, float]


# =============================================================================
# ChannelData
# =============================================================================
class ChannelData(NamedTuple):
    """
    Container for per-channel data and process information.

    Attributes
    ----------
    region : str
        Analysis region name (e.g., 'signal_region')
    observable : str
        Observable name (e.g., 'm_ll')
    data : Array
        Observed data histogram as JAX array
    processes : Dict[str, Dict]
        Dictionary of process information where:
        key: Process name (e.g., 'signal', 'background')
        value: Dictionary with keys:
            'nominal': Nominal histogram (JAX array)
            'systematics': Dictionary of systematic variations:
                key: Systematic name
                value: Varied histogram (JAX array)
    """

    region: str
    observable: str
    data: Array
    processes: Dict[str, Dict]


# =============================================================================
# Parameters
# =============================================================================
class Parameters(NamedTuple):
    """
    Parameter structure for the statistical model.

    Attributes
    ----------
    mu : evm.Parameter
        Signal strength parameter
    norm : Dict[str, evm.Parameter]
        Normalization parameters keyed by process name
    nuis : Dict[str, evm.NormalParameter]
        Nuisance parameters keyed by systematic name
    """

    mu: evm.Parameter
    norm: Dict[str, evm.Parameter]
    nuis: Dict[str, evm.NormalParameter]


# =============================================================================
# Summaries
# =============================================================================
# =============================================================================
# summarize_fit_result
# =============================================================================
def summarize_fit_result(result: FitResult) -> None:
    """
    Print fit results using professional table formatting.

    Parameters
    ----------
    result : FitResult
        Object containing fit information
    """
    # Prepare parameter table data
    param_table = []
    for name in sorted(result.param_values.keys()):
        value = result.param_values[name]
        unc = result.uncertainties.get(name, None)
        # Format value ± uncertainty
        val_str = (
            f"{jax.device_get(value).item():.4f} ±"
            f"{jax.device_get(unc).item():.4f}"
        )
        param_table.append([name, val_str])

    # Prepare fit quality metrics
    quality_table = [
        [
            "Negative Log-Likelihood",
            f"{jax.device_get(result.loss).item():.4f}",
        ],
        ["Number of Parameters", len(result.param_values)],
        [
            "Covariance Condition",
            f"{jax.device_get(jnp.linalg.cond(result.covariance)).item():.2e}",
        ],
    ]

    if param_table:
        # Format table with newline for proper separation
        table_str = tabulate(
            param_table,
            headers=["Parameter", "Value ± Uncertainty"],
            tablefmt="grid",
            floatfmt=".4f",
        )
        logger.info("Fitted parameter values:\n%s", table_str)

    # Format quality table with newline
    table_str = tabulate(
        quality_table, headers=["Metric", "Value"], tablefmt="grid"
    )
    logger.info("Fit quality metrics:\n%s", table_str)


# =============================================================================
# print_significance_summary
# =============================================================================
def print_significance_summary(significance: float, q0: float) -> None:
    """
    Print significance results with professional table formatting.

    Parameters
    ----------
    significance : float
        Computed significance value (Z)
    q0 : float
        Profile likelihood ratio test statistic
    """
    # Results table
    results_table = [
        ["Test Statistic (q0)", f"{jax.device_get(q0).item():.4f}"],
        ["Significance (Z)", f"{jax.device_get(significance).item():.4f}σ"],
    ]

    # Format table with newline for proper separation
    table_str = tabulate(
        results_table, headers=["Metric", "Value"], tablefmt="grid"
    )

    # Using logger for formatted output with proper newlines
    logger.info("Significance results :\n%s" + table_str)
