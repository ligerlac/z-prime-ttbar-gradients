from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import evermore as evm

from jaxtyping import Array, Float, PyTree
from typing import NamedTuple, TypeAlias, Any, Callable

import logging
# Configure module-level logger for debugging and monitoring
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

jax.config.update("jax_enable_x64", True)  # Use 64-bit precision

FScalar: TypeAlias = Float[Array, ""]  # Scalar float array type
Hist1D: TypeAlias = Float[Array, "bins"]  # 1D histogram type (sumw)
Params: TypeAlias = dict[str, evm.AbstractParameter[FScalar]]  # Parameter dictionary


def model_per_channel(params: Params, hists: dict[str, Hist1D]) -> dict[str, Hist1D]:
    hists["signal"] = params["mu"].scale()(hists["signal"])
    hists["ttbar_semilep"] = params["scale_ttbar"].scale()(hists["ttbar_semilep"])
    # Other backgrounds are fixed, no scaling applied
    return hists


def loss_per_channel(dynamic: Params, static: Params, hists: dict[str, Hist1D], observation: Hist1D) -> FScalar:
    params = evm.tree.combine(dynamic, static)
    expected = evm.util.sum_over_leaves(model_per_channel(params, hists))
    # Poisson NLL of the expectation and observation
    log_likelihood = (
        evm.pdf.PoissonContinuous(lamb=expected).log_prob(observation).sum()
    )
    # Add parameter constraints from logpdfs
    constraints = evm.loss.get_log_probs(params)
    # Sum over all constraints (i.e., priors)
    constraints = jax.tree.map(jnp.sum, constraints)
    log_likelihood += evm.util.sum_over_leaves(constraints)
    return -jnp.sum(log_likelihood)


class ChannelData(NamedTuple):
    name: str
    observed_counts: Hist1D
    templates: dict[str, Hist1D]
    bin_edges: jax.Array


def total_loss(dynamic: Params, static: Params, channels: list[ChannelData]) -> FScalar:
    loss = 0.0
    for channel in channels:
        # Compute loss for each channel
        loss += loss_per_channel(
            dynamic=dynamic,
            static=static,
            hists=channel.templates,
            observation=channel.observed_counts,
        )
    return loss


def fit(params: Params, channels: list[ChannelData]) -> Params:
    solver = optx.BFGS(rtol=1e-5, atol=1e-7)

    dynamic, static = evm.tree.partition(params)

    # wrap the loss function to match optimistix's expectations
    def optx_loss(dynamic, args):
        return total_loss(dynamic, *args)

    fitresult = optx.minimise(
        optx_loss,
        solver,
        dynamic,
        has_aux=False,
        args=(static, channels),
        options={},
        max_steps=10_000,
        throw=True,
    )
    # NLL
    nll = total_loss(fitresult.value, static, channels)
    # bestfit parameters
    bestfit_params = evm.tree.combine(fitresult.value, static)
    return (nll, bestfit_params)


@eqx.filter_jit
def q0_test(
    params: Params,
    channels: list[ChannelData],
    test_poi: float,
    poi_where: Callable,
) -> tuple[FScalar, Params]:
    """Calculate expected p-values via q0 test."""
    # global fit
    two_nll, bestfit_params = fit(params, channels)

    # conditional fit at test_poi
    # Fix `mu` and freeze the parameter
    params = eqx.tree_at(lambda t: poi_where(t).frozen, params, True)
    params = eqx.tree_at(lambda t: poi_where(t).raw_value, params, evm.parameter.to_value(test_poi))
    two_nll_conditional, _ = fit(params, channels)

    # Calculate the likelihood ratio
    # q0 = -2 ln [L(μ=0, θ̂̂) / L(μ̂, θ̂)]
    likelihood_ratio = 2.0 * (two_nll_conditional - two_nll)

    poi_hat = poi_where(bestfit_params).value
    q0 = jnp.where(poi_hat >= test_poi, likelihood_ratio, 0.0)
    # p = 1 - Φ(√q₀)
    p0 = 1.0 - jax.scipy.stats.norm.cdf(jnp.sqrt(q0))
    return (p0, evm.tree.pure(bestfit_params))



def build_channel_data_scalar(
    histogram_dictionary: PyTree[Hist1D],
    channel_configurations: list[Any],
) -> tuple[list[ChannelData], None]:
    """
    Construct ChannelData objects from nested histogram dictionary.

    Parameters
    ----------
    histogram_dictionary : PyTree[Hist1D]
        Nested histogram structure:
            Level 1: Process names (e.g., "signal", "ttbar")
            Level 2: Systematic variations (e.g., "nominal", "scale_up")
            Level 3: Channel names
            Level 4: Observable names → (counts, bin_edges) or counts
    channel_configurations : list[Any]
        Channel configuration objects with attributes:
            - name: Channel identifier
            - fit_observable: Key for target observable
            - use_in_discovery: Flag to include channel

    Returns
    -------
    channel_data_list : list[ChannelData]
        Constructed channel data containers


    Notes
    -----
    - Only uses "nominal" systematic variation
    - Automatically creates zero templates for missing required processes
    - Skips channels not marked for discovery use
    - Converts all inputs to JAX arrays for compatibility
    """
    channel_data_list = []

    # Process each channel configuration
    for config in channel_configurations:
        # Skip channels excluded from discovery fit
        if not getattr(config, "use_in_discovery", True):
            continue
        if not getattr(config, "use_in_diff", True):
            continue

        channel_name = config.name
        observable_key = config.fit_observable

        # =====================================================================
        # Step 1: Extract observed data
        # =====================================================================
        try:
            # Navigate nested dictionary: data → nominal → channel → observable
            data_container = (
                histogram_dictionary.get("data", {})
                .get("nominal", {})
                .get(channel_name, {})
                .get(observable_key, None)
            )

            if data_container is None:
                logger.warning(f"Missing data for {channel_name}/{observable_key}")
                continue
        except KeyError:
            logger.exception(f"Data access error for {channel_name}")
            continue

        # Handle different histogram storage formats
        if isinstance(data_container, tuple):
            # Tuple format: (counts, bin_edges)
            observed_counts, bin_edges = data_container
            observed_counts = jnp.asarray(observed_counts)
            bin_edges = jnp.asarray(bin_edges)
        else:
            # Assume array is counts only
            observed_counts = jnp.asarray(data_container)
            bin_edges = jnp.array([])  # Empty bin edges

        # =====================================================================
        # Step 2: Build process templates
        # =====================================================================
        process_templates = {}
        for process_name, variations in histogram_dictionary.items():
            # Skip data entry (already handled)
            if process_name == "data":
                continue

            try:
                # Extract nominal histogram for this process/channel/observable
                nominal_hist = (
                    variations
                    .get("nominal", {})
                    .get(channel_name, {})
                    .get(observable_key, None)
                )

                if nominal_hist is None:
                    continue
            except KeyError:
                logger.warning(
                    f"Missing nominal histogram for {process_name} in {channel_name}"
                )
                continue

            # Handle different storage formats
            if isinstance(nominal_hist, tuple):
                # Tuple format: (counts, edges) - extract counts only
                counts = jnp.asarray(nominal_hist[0])
            else:
                # Assume it's counts array
                counts = jnp.asarray(nominal_hist)

            process_templates[process_name] = counts

        # =====================================================================
        # Step 3: Ensure required processes exist
        # =====================================================================
        # Create zero templates for required processes if missing
        zero_template = jnp.zeros_like(observed_counts)
        if "signal" not in process_templates:
            logger.info(f"Adding zero signal template for {channel_name}")
            process_templates["signal"] = zero_template
        if "ttbar_semilep" not in process_templates:
            logger.info(f"Adding zero ttbar template for {channel_name}")
            process_templates["ttbar_semilep"] = zero_template

        # =====================================================================
        # Step 4: Create channel container
        # =====================================================================
        channel_data = ChannelData(
            name=channel_name,
            observed_counts=observed_counts,
            templates=process_templates,
            bin_edges=bin_edges,
        )
        channel_data_list.append(channel_data)
    return channel_data_list, None


def compute_discovery_pvalue(
    histogram_dictionary: PyTree[Hist1D],
    channel_configurations: list[Any],
    parameters: Params,
    signal_strength_test_value: float = 0.0,
) -> tuple[FScalar, Params]:
    """
    Calculate discovery p-value using profile likelihood ratio.

    Implements the test statistic:
        q₀ = -2 ln [ L(μ=0, θ̂̂) / L(μ̂, θ̂) ]
    where:
        μ = signal strength
        θ = nuisance parameters (here κ_tt)
        θ̂̂ = conditional MLE under μ=0
        (μ̂, θ̂) = unconditional MLE

    Parameters
    ----------
    histogram_dictionary : PyTree[Hist1D]
        Nested histogram structure (see build_channel_data_scalar)
    channel_configurations : list[Any]
        Channel configuration objects
    parameters : Params, optional
        Initial parameter values for optimization, default:
            {"mu": 1.0, "scale_ttbar": 1.0}
    signal_strength_test_value : float, optional
        Signal strength for null hypothesis (typically 0 for discovery)

    Returns
    -------
    p_value : jnp.ndarray
        Asymptotic p-value for discovery (1-tailed)
    mle_parameters : Dict[str, jnp.ndarray]
        Maximum Likelihood Estimates under null hypothesis

    Notes
    -----
    - Uses the "evermore" library for automatic differentiation-based inference
    - Implements the q₀ test statistic from arXiv:1007.1727
    - The p-value is computed using the asymptotic approximation:
        p = 1 - Φ(√q₀)
      where Φ is the standard normal CDF
    """
    # =====================================================================
    # Step 1: Prepare data and model
    # =====================================================================
    channels, _ = build_channel_data_scalar(
        histogram_dictionary, channel_configurations
    )

    # Handle case with no valid channels
    if not channels:
        logger.error("Discovery calculation aborted: no valid channels")
        return jnp.array(0.0), {}
    
    # wrap the parameters in a dictionary of evm.Parameter
    # should probably be done by the caller already
    params = {
        "mu": evm.Parameter(value=parameters["mu"], name="mu"),
        "scale_ttbar": evm.Parameter(value=parameters["scale_ttbar"], name="scale_ttbar"),
    }

    return q0_test(
        params=params,
        channels=channels,
        test_poi=signal_strength_test_value,
        poi_where=lambda t: t["mu"],  # Default path to signal strength parameter
    )