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
class FitResult(NamedTuple):
    params: PyTree
    loss: float
    uncertainties: Dict[str, float]
    covariance: Array
    param_values: Dict[str, float]  # Added for summary printing

class ChannelData(NamedTuple):
    region: str
    observable: str
    data: Array
    processes: Dict[str, Dict]  # Process info dicts

class Parameters(NamedTuple):
    mu: evm.Parameter
    norm: Dict[str, evm.Parameter]
    nuis: Dict[str, evm.NormalParameter]

# =============================================================================
# Core Significance Calculation
# =============================================================================
def calculate_significance(histograms: Dict, channels: List) -> Array:
    """
    Compute asymptotic significance using profile likelihood ratio.

    Args:
        histograms: Nested dictionary of histogram data
        channels: List of channel configurations

    Returns:
        sqrt(q0) significance value
    """
    # Step 1: Collect and validate input channels
    valid_channels = _get_valid_channels(channels)
    if not valid_channels:
        return jnp.array(0.0)

    # Step 2: Prepare channel data for modeling
    channel_data, all_systematics, all_processes = _prepare_channel_data(
        histograms, valid_channels
    )
    if not channel_data:
        return jnp.array(0.0)

    # Step 3: Define model and parameters
    ParamStruct = _create_parameter_structure(all_processes, all_systematics)
    model_fn = _create_model_function()

    # Step 4: Fit hypotheses and compute significance
    logger.info("Fitting alternative hypothesis (μ free)...")
    result_alt = _fit_hypothesis(
        ParamStruct, model_fn, channel_data, frozen_mu=None
    )
    summarize_fit_result(result_alt)

    logger.info("Fitting null hypothesis (μ = 0)...")
    result_null = _fit_hypothesis(
        ParamStruct, model_fn, channel_data, frozen_mu=0.0
    )
    summarize_fit_result(result_null)

    # Step 5: Compute final significance
    q0 = 2 * (result_null.loss - result_alt.loss)
    significance = jnp.sqrt(jnp.clip(q0, 0.0))

    print_significance_summary(significance, q0)

    return significance


# =============================================================================
# Enhanced Table-Based Result Summarization
# =============================================================================
def summarize_fit_result(result: FitResult) -> None:
    """
    Print fit results using professional table formatting.

    Args:
        result: FitResult object containing fit information
        title: Descriptive title for this fit result
    """
    # Prepare parameter table data
    param_table = []
    for name in sorted(result.param_values.keys()):
        value = result.param_values[name]
        unc = result.uncertainties.get(name, None)

        param_table.append([
            name,
            f"{jax.device_get(value).item():.4f} ± {jax.device_get(unc).item():.4f}",
        ])

    # Prepare fit quality table
    quality_table = [
        ["Negative Log-Likelihood", f"{result.loss:.4f}"],
        ["Number of Parameters", len(result.param_values)],
        ["Covariance Condition", f"{jax.device_get(jnp.linalg.cond(result.covariance)).item():.2e}"]
    ]

    if param_table:
        print("\nFITTED PARAMETERS:")
        print(tabulate(
            param_table,
            headers=["Parameter", "Value ± Uncertainty", "Significance"],
            tablefmt="grid",
            floatfmt=".4f"
        ))

    print("\nFIT QUALITY:")
    print(tabulate(quality_table, headers=["Metric", "Value"], tablefmt="grid"))

def print_significance_summary(significance: float, q0: float) -> None:
    """
    Print significance results with professional table formatting.

    Args:
        significance: Computed significance value (Z)
        q0: Profile likelihood ratio test statistic
    """
    z_value = significance
    # Results table
    results_table = [
        ["Test Statistic (q0)", f"{q0:.4f}"],
        ["Significance (Z)", f"{z_value:.4f}σ"],
    ]

    # Print tables
    print("\n" + "=" * 60)
    print(" SIGNIFICANCE RESULTS ".center(60))
    print("=" * 60)

    print("\nSTATISTICAL RESULTS:")
    print(tabulate(results_table, headers=["Metric", "Value"], tablefmt="grid"))
    print("\n" + "=" * 60)

# =============================================================================
# Helper Functions
# =============================================================================
def _get_valid_channels(channels: List) -> List[Tuple[str, str]]:
    """Filter valid (region, observable) pairs for fitting"""
    return [
        (ch.name, ch.fit_observable)
        for ch in channels
        if ch.use_in_diff and hasattr(ch, "fit_observable")
    ]

def _prepare_channel_data(histograms: Dict, channels: List[Tuple[str, str]]
    ) -> Tuple[List[ChannelData], List[str], List[str]]:
    """
    Prepare structured channel data from raw histograms

    Returns:
        channel_data: Processed channel information
        all_systematics: Sorted list of systematic names
        all_processes: Sorted list of process names
    """
    channel_objects = []
    systematics = set()
    processes = set()

    for region, obs in channels:
        # Get data histogram
        data_hist = histograms.get("data", {}).get("nominal", {}).get(region, {}).get(obs)
        if data_hist is None:
            continue

        # Collect process information
        process_info = {}
        for pname, variations in histograms.items():
            if pname == "data":
                continue

            nominal_hist = variations.get("nominal", {}).get(region, {}).get(obs)
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

            process_info[pname] = {"nominal": nominal_hist, "systematics": syst_vars}
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

def _create_parameter_structure(processes: List[str], systematics: List[str]) -> NamedTuple:
    """Define parameter structure for the fit"""
    return Parameters(
        mu=evm.Parameter(1.0, lower=0, upper=1000),
        norm={p: evm.Parameter(1.0) for p in processes if p == "ttbar_semilep"},
        nuis={s: evm.NormalParameter(0.0) for s in systematics},
    )

def _create_model_function() -> Callable:
    """Create model function for expectation calculation"""
    def model(params: NamedTuple, channel_data: List[ChannelData]) -> List[Array]:
        expectations = []
        for ch in channel_data:
            total = jnp.zeros_like(ch.data)
            for pname, pinfo in ch.processes.items():
                base = pinfo["nominal"]
                # Apply appropriate scaling
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

def _fit_hypothesis(
    param_struct: NamedTuple,
    model_fn: Callable,
    channel_data: List[ChannelData],
    frozen_mu: Optional[float] = None,
    steps: int = 100
) -> FitResult:
    """Fit a single hypothesis (either null or alternative)"""
    # Prepare parameters
    params = param_struct if frozen_mu is None else \
             param_struct._replace(mu=evm.Parameter(frozen_mu, frozen=True))

    # Initialize optimizer
    diffable, static = evm.parameter.partition(params)
    optimizer = optax.adam(0.05)
    opt_state = optimizer.init(diffable)

    # JIT-compiled training step
    @jax.jit
    def train_step(diffable, static, opt_state):
        loss_val, grads = eqx.filter_value_and_grad(_make_loss_fn(model_fn, channel_data))(diffable, static)
        updates, opt_state = optimizer.update(grads, opt_state)
        diffable = optax.apply_updates(diffable, updates)
        return diffable, opt_state, loss_val

    # Run optimization
    for step in range(steps):
        diffable, opt_state, loss_val = train_step(diffable, static, opt_state)
        if step % 10 == 0:
            logger.info(f"Step {step}: loss = {jax.device_get(loss_val).item():.4f}")

    # Post-processing
    final_params = eqx.combine(diffable, static)
    clean_diffable = _remove_frozen_parameters(diffable)
    flat_params, unravel_fn = jax.flatten_util.ravel_pytree(clean_diffable)

    # Hessian and covariance calculation
    def hess_fn(flat_params):
        params_unraveled = unravel_fn(flat_params)
        loss_val = _make_loss_fn(model_fn, channel_data)(params_unraveled, static)
        return jnp.sum(loss_val)  # Ensure scalar output for hessian

    hess = jax.hessian(hess_fn)(flat_params)
    cov_matrix = jnp.linalg.pinv(hess)
    std_devs = jnp.sqrt(jnp.clip(jnp.diag(cov_matrix), 0.0))

    # Parameter names
    param_names = _get_unfrozen_parameter_names(clean_diffable)
    param_values_dict = {name: val.astype(float) for name, val in zip(param_names, flat_params)}
    uncertainties = {name: std.astype(float) for name, std in zip(param_names, std_devs)}

    return FitResult(
        params=final_params,
        loss=float(jax.device_get(loss_val).item()),
        uncertainties=uncertainties,
        covariance=cov_matrix,
        param_values=param_values_dict
    )

def _make_loss_fn(model_fn: Callable, channel_data: List[ChannelData]) -> Callable:
    """Create loss function for given model and data"""
    def loss(diffable: PyTree, static: PyTree) -> Array:
        params = eqx.combine(diffable, static)
        expectations = model_fn(params, channel_data)
        total_nll = 0.0
        for i, ch in enumerate(channel_data):
            total_nll -= evm.pdf.Poisson(expectations[i]).log_prob(ch.data).sum()
        total_nll -= evm.util.sum_over_leaves(evm.loss.get_log_probs(params))
        return  jnp.sum(total_nll)
    return loss

def _remove_frozen_parameters(tree: PyTree) -> PyTree:
    """Filter out frozen parameters from PyTree"""
    if isinstance(tree, dict):
        return {k: _remove_frozen_parameters(v) for k, v in tree.items()
                if _remove_frozen_parameters(v) is not None}
    elif hasattr(tree, "_fields"):  # NamedTuple
        return type(tree)(*(_remove_frozen_parameters(getattr(tree, f))
                          for f in tree._fields))
    else:
        return tree if not getattr(tree, "frozen", False) else None

def _get_unfrozen_parameter_names(tree: PyTree, prefix: str = "") -> List[str]:
    """Get full paths for unfrozen parameters in PyTree"""
    if tree is None:
        return []

    names = []
    if isinstance(tree, dict):
        for k, v in tree.items():
            names.extend(_get_unfrozen_parameter_names(v, f"{prefix}{k}."))
    elif hasattr(tree, "_fields"):  # NamedTuple
        for field in tree._fields:
            names.extend(_get_unfrozen_parameter_names(
                getattr(tree, field), f"{prefix}{field}."
            ))
    else:
        names.append(prefix.rstrip('.'))
    return names