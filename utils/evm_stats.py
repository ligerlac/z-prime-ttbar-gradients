import logging

import jax
import jax.numpy as jnp
import equinox as eqx
import evermore as evm
import optax
from typing import NamedTuple, Optional, List, Dict, Tuple, Callable
from jaxtyping import Array, PyTree
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format="[%(levelname)s: %(name)s] %(message)s")
logger = logging.getLogger("evermore")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

# Enable double precision for numerical stability
jax.config.update("jax_enable_x64", True)

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
# Core Significance Calculation
# =============================================================================

# =============================================================================
# calculate_significance
# =============================================================================
def calculate_significance(histograms: Dict, channels: List) -> Array:
    """
    Compute asymptotic significance using profile likelihood ratio.

    Parameters
    ----------
    histograms : Dict
        Nested dictionary of histogram data with structure:
        {
            "data": {
                "nominal": {
                    "region1": {"obs1": array, ...},
                    ...
                }
            },
            "process1": {
                "nominal": {...},
                "syst1": {...},
                ...
            },
            ...
        }
    channels : List
        List of channel configuration objects with attributes:
        name: Region name
        fit_observable: Observable name
        use_in_diff: Flag indicating if channel should be used

    Returns
    -------
    Array
        Computed significance value (Z)
    """
    # Validate and filter input channels
    valid_channels = _get_valid_channels(channels)
    if not valid_channels:
        logger.warning("No valid channels found for significance calculation")
        return jnp.array(0.0)

    # Prepare structured data for fitting
    channel_data, all_systematics, all_processes = _prepare_channel_data(
        histograms, valid_channels
    )
    if not channel_data:
        logger.warning("No channel data available after preparation")
        return jnp.array(0.0)

    # Define model structure and fitting function
    ParamStruct = _create_parameter_structure(all_processes, all_systematics)
    model_fn = _create_model_function()

    # Fit alternative hypothesis (μ unconstrained)
    logger.info("Fitting alternative hypothesis (μ free)...")
    result_alt = _fit_hypothesis(
        ParamStruct, model_fn, channel_data, frozen_mu=None
    )
    #summarize_fit_result(result_alt)

    # Fit null hypothesis (μ = 0)
    logger.info("Fitting null hypothesis (μ = 0)...")
    result_null = _fit_hypothesis(
        ParamStruct, model_fn, channel_data, frozen_mu=0.0
    )
    #summarize_fit_result(result_null)

    # Compute test statistic and significance
    q0 = 2 * (result_null.loss - result_alt.loss)
    significance = jnp.sqrt(jnp.clip(q0, 0.0))

    #print_significance_summary(significance, q0)

    return significance


# =============================================================================
# Enhanced Table-Based Result Summarization
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
        val_str = f"{jax.device_get(value).item():.4f} ± {jax.device_get(unc).item():.4f}"
        param_table.append([name, val_str])

    # Prepare fit quality metrics
    quality_table = [
        ["Negative Log-Likelihood", f"{result.loss:.4f}"],
        ["Number of Parameters", len(result.param_values)],
        ["Covariance Condition",
         f"{jax.device_get(jnp.linalg.cond(result.covariance)).item():.2e}"]
    ]

    if param_table:
        # Format table with newline for proper separation
        table_str = tabulate(
            param_table,
            headers=["Parameter", "Value ± Uncertainty"],
            tablefmt="grid",
            floatfmt=".4f"
        )
        logger.info("Fitted parameter values:\n%s", table_str)

    # Format quality table with newline
    table_str = tabulate(
        quality_table,
        headers=["Metric", "Value"],
        tablefmt="grid"
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
        ["Test Statistic (q0)", f"{q0:.4f}"],
        ["Significance (Z)", f"{significance:.4f}σ"],
    ]

    # Format table with newline for proper separation
    table_str = tabulate(
        results_table,
        headers=["Metric", "Value"],
        tablefmt="grid"
    )

    # Using logger for formatted output with proper newlines
    logger.info("Significance results :\n%s" + table_str)

# =============================================================================
# Helper Functions
# =============================================================================

# =============================================================================
# _get_valid_channels
# =============================================================================
def _get_valid_channels(channels: List) -> List[Tuple[str, str]]:
    """
    Filter valid (region, observable) pairs for fitting.

    Parameters
    ----------
    channels : List
        List of channel configuration objects

    Returns
    -------
    List[Tuple[str, str]]
        List of valid (region_name, observable_name) tuples
    """
    return [
        (ch.name, ch.fit_observable)
        for ch in channels
        if ch.use_in_diff and hasattr(ch, "fit_observable")
    ]

# =============================================================================
# _prepare_channel_data
# =============================================================================
def _prepare_channel_data(
    histograms: Dict,
    channels: List[Tuple[str, str]]
) -> Tuple[List[ChannelData], List[str], List[str]]:
    """
    Prepare structured channel data from raw histograms.

    Parameters
    ----------
    histograms : Dict
        Nested dictionary of histogram data
    channels : List[Tuple[str, str]]
        List of valid (region, observable) tuples

    Returns
    -------
    Tuple[List[ChannelData], List[str], List[str]]
        channel_data: List of ChannelData objects
        all_systematics: Sorted list of systematic names
        all_processes: Sorted list of process names
    """
    channel_objects = []
    systematics = set()
    processes = set()

    for region, obs in channels:
        # Extract data histogram using nested dict lookups
        data_hist = (histograms.get("data", {})
                     .get("nominal", {})
                     .get(region, {})
                     .get(obs))
        if data_hist is None:
            continue

        process_info = {}
        for pname, variations in histograms.items():
            if pname == "data":
                continue

            # Get nominal histogram for this process/region/observable
            nominal_hist = (variations.get("nominal", {})
                            .get(region, {})
                            .get(obs))
            if nominal_hist is None:
                continue

            # Collect systematic variations
            syst_vars = {}
            for syst_name, syst_data in variations.items():
                if syst_name == "nominal":
                    continue
                hist = syst_data.get(region, {}).get(obs)
                if hist is not None:
                    syst_vars[syst_name] = hist
                    systematics.add(syst_name)

            process_info[pname] = {
                "nominal": nominal_hist,
                "systematics": syst_vars
            }
            processes.add(pname)

        if process_info:
            channel_objects.append(ChannelData(
                region=region,
                observable=obs,
                data=data_hist,
                processes=process_info
            ))

    return (
        channel_objects,
        sorted(systematics),
        sorted(processes)
    )

# =============================================================================
# _create_parameter_structure
# =============================================================================
def _create_parameter_structure(
    processes: List[str],
    systematics: List[str]
) -> NamedTuple:
    """
    Define parameter structure for the fit.

    Parameters
    ----------
    processes : List[str]
        List of process names
    systematics : List[str]
        List of systematic names

    Returns
    -------
    NamedTuple
        Parameters NamedTuple with:
        mu: Signal strength parameter
        norm: Normalization parameters (only for 'ttbar_semilep')
        nuis: Nuisance parameters for systematics
    """
    return Parameters(
        mu=evm.Parameter(1.0, lower=0, upper=1000),
        # Only ttbar_semilep gets free normalization
        norm={p: evm.Parameter(1.0) for p in processes if p == "ttbar_semilep"},
        nuis={s: evm.NormalParameter(0.0) for s in systematics},
    )

# =============================================================================
# _create_model_function
# =============================================================================
def _create_model_function() -> Callable:
    """
    Create model function for expectation calculation.

    Returns
    -------
    Callable
        Function that computes expected histograms given parameters and channel data
    """
    # =============================================================================
    # model
    # =============================================================================
    def model(params: NamedTuple, channel_data: List[ChannelData]) -> List[Array]:
        expectations = []
        for ch in channel_data:
            total = jnp.zeros_like(ch.data)
            for pname, pinfo in ch.processes.items():
                base = pinfo["nominal"]
                # Apply appropriate scaling:
                # - Signal scaled by μ
                # - ttbar_semilep has free normalization
                # - Other backgrounds fixed to nominal
                if pname == "signal":
                    scaled = params.mu.scale()(base)
                elif pname == "ttbar_semilep":
                    scaled = params.norm[pname].scale()(base)
                else:
                    scaled = base
                total += scaled
            expectations.append(total)
        return expectations
    return model

# =============================================================================
# _fit_hypothesis
# =============================================================================
def _fit_hypothesis(
    param_struct: NamedTuple,
    model_fn: Callable,
    channel_data: List[ChannelData],
    frozen_mu: Optional[float] = None,
    steps: int = 100
) -> FitResult:
    """
    Fit a single hypothesis (either null or alternative).

    Parameters
    ----------
    param_struct : NamedTuple
        Initial parameter structure
    model_fn : Callable
        Model function for expectation calculation
    channel_data : List[ChannelData]
        List of ChannelData objects
    frozen_mu : Optional[float]
        Value to freeze μ at (for null hypothesis)
    steps : int
        Number of optimization steps

    Returns
    -------
    FitResult
        Object with fit results
    """
    # Handle frozen μ case for null hypothesis
    if frozen_mu is not None:
        params = param_struct._replace(
            mu=evm.Parameter(frozen_mu, frozen=True)
        )
    else:
        params = param_struct

    # Split parameters into differentiable and static parts
    diffable, static = evm.parameter.partition(params)
    optimizer = optax.adam(0.05)
    opt_state = optimizer.init(diffable)

    # =============================================================================
    # train_step
    # =============================================================================
    @jax.jit
    def train_step(diffable, static, opt_state):
        loss_val, grads = eqx.filter_value_and_grad(
            _make_loss_fn(model_fn, channel_data)
        )(diffable, static)
        updates, opt_state = optimizer.update(grads, opt_state)
        diffable = optax.apply_updates(diffable, updates)
        return diffable, opt_state, loss_val

    # Optimization loop
    for step in range(steps):
        diffable, opt_state, loss_val = train_step(diffable, static, opt_state)
        #if step % 10 == 0:
            #logger.info(f"Step {step}: loss = {jax.device_get(loss_val).item():.4f}")

    # Post-processing after optimization
    final_params = eqx.combine(diffable, static)
    clean_diffable = _remove_frozen_parameters(diffable)
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(clean_diffable)

    # =============================================================================
    # hess_fn
    # =============================================================================
    def hess_fn(flat_params):
        params_unraveled = unravel_fn(flat_params)
        loss_val = _make_loss_fn(model_fn, channel_data)(params_unraveled, static)
        return jnp.sum(loss_val)  # Ensure scalar output for hessian

    hess = jax.hessian(hess_fn)(flat_params)
    cov_matrix = jnp.linalg.pinv(hess)
    std_devs = jnp.sqrt(jnp.clip(jnp.diag(cov_matrix), 0.0))

    # Prepare parameter names and values for results
    param_names = _get_unfrozen_parameter_names(clean_diffable)
    param_values_dict = {
        name: val.astype(float)
        for name, val in zip(param_names, flat_params)
    }
    uncertainties = {
        name: std.astype(float)
        for name, std in zip(param_names, std_devs)
    }

    return FitResult(
        params=final_params,
        loss=loss_val, #float(jax.device_get(loss_val).item()),
        uncertainties=uncertainties,
        covariance=cov_matrix,
        param_values=param_values_dict
    )

# =============================================================================
# _make_loss_fn
# =============================================================================
def _make_loss_fn(
    model_fn: Callable,
    channel_data: List[ChannelData]
) -> Callable:
    """
    Create negative log-likelihood loss function.

    Parameters
    ----------
    model_fn : Callable
        Model function for expectation calculation
    channel_data : List[ChannelData]
        List of ChannelData objects

    Returns
    -------
    Callable
        Loss function that computes negative log-likelihood
    """
    # =============================================================================
    # loss
    # =============================================================================
    def loss(diffable: PyTree, static: PyTree) -> Array:
        params = eqx.combine(diffable, static)
        expectations = model_fn(params, channel_data)
        total_nll = 0.0
        # Poisson NLL for each channel
        for i, ch in enumerate(channel_data):
            total_nll -= evm.pdf.Poisson(expectations[i]).log_prob(ch.data).sum()
        # Constraint terms for parameters
        total_nll -= evm.util.sum_over_leaves(evm.loss.get_log_probs(params))
        return jnp.sum(total_nll)
    return loss

# =============================================================================
# _remove_frozen_parameters
# =============================================================================
def _remove_frozen_parameters(tree: PyTree) -> PyTree:
    """
    Filter out frozen parameters from PyTree structure.

    Parameters
    ----------
    tree : PyTree
        Parameter PyTree structure

    Returns
    -------
    PyTree
        New PyTree with frozen parameters removed
    """
    if isinstance(tree, dict):
        return {
            k: _remove_frozen_parameters(v)
            for k, v in tree.items()
            if _remove_frozen_parameters(v) is not None
        }
    elif hasattr(tree, "_fields"):  # NamedTuple
        return type(tree)(*(
            _remove_frozen_parameters(getattr(tree, f))
            for f in tree._fields
        ))
    else:
        return tree if not getattr(tree, "frozen", False) else None

# =============================================================================
# _get_unfrozen_parameter_names
# =============================================================================
def _get_unfrozen_parameter_names(
    tree: PyTree,
    prefix: str = ""
) -> List[str]:
    """
    Get full paths for unfrozen parameters in PyTree.

    Parameters
    ----------
    tree : PyTree
        Parameter PyTree structure
    prefix : str
        Current path prefix (for recursion)

    Returns
    -------
    List[str]
        List of parameter paths (e.g., 'mu.value', 'norm.ttbar_semilep.value')
    """
    if tree is None:
        return []

    names = []
    if isinstance(tree, dict):
        for k, v in tree.items():
            new_prefix = f"{prefix}{k}."
            names.extend(_get_unfrozen_parameter_names(v, new_prefix))
    elif hasattr(tree, "_fields"):  # NamedTuple
        for field in tree._fields:
            new_prefix = f"{prefix}{field}."
            names.extend(_get_unfrozen_parameter_names(
                getattr(tree, field), new_prefix
            ))
    else:
        # Remove trailing dot from final parameter name
        names.append(prefix.rstrip('.'))
    return names