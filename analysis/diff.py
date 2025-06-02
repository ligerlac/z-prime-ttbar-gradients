from __future__ import annotations

import glob
import logging
import os
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import cloudpickle
from pprint import pformat
from typing import Any, Literal, NamedTuple, Optional

import awkward as ak
import equinox as eqx
import evermore as evm
import optax
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree
import numpy as np
from tabulate import tabulate
import uproot
import vector
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

from analysis.base import Analysis
from utils.cuts import lumi_mask
from utils.evm_stats import calculate_significance

# -----------------------------------------------------------------------------
# Backend & Logging Setup
# -----------------------------------------------------------------------------
ak.jax.register_and_check()
vector.register_awkward()

logging.basicConfig(level=logging.INFO, format="[%(levelname)s: %(name)s] %(message)s")
logger = logging.getLogger("DiffAnalysis")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")

# colours to use in printouts
GREEN = "\033[92m"
RESET = "\033[0m"

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def merge_histograms(
    existing: dict[str, dict[str, dict[str, jnp.ndarray]]],
    new: dict[str, dict[str, dict[str, jnp.ndarray]]],
) -> dict[str, dict[str, dict[str, jnp.ndarray]]]:
    """
    Recursively merge `new` histogram dict into `existing`.
    Both must follow the structure:
    histograms[variation][region][observable] = jnp.ndarray
    """
    for variation, region_dict in new.items():
        for region, obs_dict in region_dict.items():
            for observable, array in obs_dict.items():
                existing[variation].setdefault(region, {})
                if observable in existing[variation][region]:
                    existing[variation][region][observable] += array
                else:
                    existing[variation][region][observable] = array
    return existing


def recursive_to_backend(data: Any, backend: str = "jax") -> Any:
    """
    Recursively move data structures containing Awkward Arrays to the specified
    backend.

    Parameters
    ----------
    data : Any
        Object, list, or dict containing awkward Arrays
    backend : str
        Target backend ('jax', 'cpu', etc.)

    Returns
    -------
    Any
        Data with all awkward arrays moved to the specified backend
    """
    if isinstance(data, ak.Array):
        if ak.backend(data) != backend:
            return ak.to_backend(data, backend)
        return data
    elif isinstance(data, Mapping):
        return {k: recursive_to_backend(v, backend) for k, v in data.items()}
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return [recursive_to_backend(v, backend) for v in data]
    else:
        return data


class FitResult(NamedTuple):
    params: Any
    loss: float
    uncertainties: dict[str, float]
    covariance: Optional[jnp.ndarray]

def summarize_fit_result(fit: FitResult, title: str) -> None:
    """
    Print a formatted summary of all fit parameter values and uncertainties,
    including both differentiable and static (frozen) parameters.
    Ensures each parameter appears only once and only unfrozen params get uncertainties.
    """
    from tabulate import tabulate
    import jax

    entries = []
    seen = set()
    diffable_param_names = set()

    # First: collect names of diffable params so we know which ones had gradients
    def get_diffable_names(tree, prefix=""):
        if isinstance(tree, dict):
            for k, v in tree.items():
                get_diffable_names(v, prefix + f"{k}.")
        elif hasattr(tree, "_fields"):
            for field in tree._fields:
                get_diffable_names(getattr(tree, field), prefix + f"{field}.")
        else:
            name = prefix.rstrip(".")
            diffable_param_names.add(name)

    # Then: walk both trees and build entry rows
    def collect_named_values(tree, prefix="", frozen=False):
        if isinstance(tree, dict):
            for k, v in tree.items():
                collect_named_values(v, prefix + f"{k}.", frozen)
        elif hasattr(tree, "_fields"):
            for field in tree._fields:
                collect_named_values(getattr(tree, field), prefix + f"{field}.", frozen)
        else:
            name = prefix.rstrip(".")
            if name in seen:
                return
            seen.add(name)

            raw_val = tree.value
            if raw_val is None:
                val_str = "n/a"
            else:
                val = jax.device_get(raw_val).astype(float).item()
                val_str = f"{val:+.4f}"

            # Only show uncertainty if param was in diffable during this fit
            print(name, diffable_param_names)
            if name in diffable_param_names:
                print(fit.uncertainties.get(name, None))
                unc = fit.uncertainties.get(name, None)
                unc_str = f"{jax.device_get(unc).item():.4f}" if unc is not None else "n/a"
            else:
                unc_str = "n/a"

            entries.append([name, val_str, unc_str, "✓" if frozen else ""])

    diffable, static = evm.parameter.partition(fit.params)
    get_diffable_names(diffable)
    collect_named_values(diffable, frozen=False)
    collect_named_values(static, frozen=True)

    entries.sort(key=lambda x: x[0])
    entries.append(["Final loss", f"{fit.loss:.4f}", "", ""])

    print(f"\n📋 {title}")
    print(tabulate(entries,
                   headers=["Parameter", "Value", "Uncertainty", "Frozen"],
                   tablefmt="fancy_grid"))

def get_param_names_and_values(tree, prefix=""):
    """
    Flatten the tree with matching parameter names and values,
    preserving the same traversal order as ravel_pytree.
    """
    names = []
    values = []

    if isinstance(tree, dict):
        for k in sorted(tree):  # Ensure consistent order
            subnames, subvals = get_param_names_and_values(tree[k], prefix + f"{k}.")
            names.extend(subnames)
            values.extend(subvals)
    elif hasattr(tree, "_fields"):  # NamedTuple
        for field in tree._fields:
            subnames, subvals = get_param_names_and_values(getattr(tree, field), prefix + f"{field}.")
            names.extend(subnames)
            values.extend(subvals)
    else:
        names.append(prefix.rstrip("."))
        values.append(tree)

    return names, values

# -----------------------------------------------------------------------------
# DifferentiableAnalysis Class Definition
# -----------------------------------------------------------------------------
class DifferentiableAnalysis(Analysis):
    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the DifferentiableAnalysis."""
        super().__init__(config)
        self.histograms: dict[str, dict[str, dict[str, jnp.ndarray]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )

    def set_histograms(
        self, histograms: dict[str, dict[str, dict[str, jnp.ndarray]]]
    ) -> None:
        """Set the final histograms after processing."""
        self.histograms = histograms


    # -------------------------------------------------------------------------
    # Significance Calculation
    # -------------------------------------------------------------------------
    def _calculate_significance(self) -> jnp.ndarray:
        """
        Generalized significance calculation using evermore with multi-channel,
        multi-process, and systematic-aware modeling.

        Returns
        -------
        jnp.ndarray
            Asymptotic significance (sqrt(q0)) from a profile likelihood ratio.
        """
        return calculate_significance(self.histograms, self.channels)

    # def _calculate_significance(self) -> jnp.ndarray:
    #     """
    #     Calculate signal significance using evermore for binned likelihood analysis.
    #     Returns significance as sqrt(q0) using the Asimov dataset (background-only).
    #     """
    #     from typing import NamedTuple
    #     import equinox as eqx
    #     import jax
    #     import evermore as evm
    #     import optax  # For optimization
    #     from jaxtyping import Array, PyTree

    #     jax.config.update("jax_enable_x64", True)

    #     # ------------------------------------------------------------
    #     # 1. Collect histograms from all channels and processes
    #     # ------------------------------------------------------------
    #     channel_data = []
    #     signal_kappa = self.config.jax.params.get("signal_systematic", 0.05)
    #     background_kappa = self.config.jax.params.get("background_systematic", 0.1)

    #     for channel in self.channels:
    #         if not getattr(channel, "use_in_diff", False):
    #             continue

    #         region = channel.name
    #         observable = getattr(channel, "fit_observable", None)
    #         if observable is None:
    #             logger.warning(f"[evermore] No fit_observable in {region}, skipping.")
    #             continue

    #         # Collect signal histogram
    #         signal_hist = None
    #         for signal_name in ["signal", "zprime"]:  # Add other signal names as needed
    #             if (signal_name in self.histograms and
    #                 "nominal" in self.histograms[signal_name] and
    #                 region in self.histograms[signal_name]["nominal"] and
    #                 observable in self.histograms[signal_name]["nominal"][region]):
    #                 signal_hist = self.histograms[signal_name]["nominal"][region][observable]
    #                 break

    #         if signal_hist is None:
    #             continue

    #         # Sum background histograms
    #         background_hist = jnp.zeros_like(signal_hist)
    #         for process, proc_hists in self.histograms.items():
    #             if process in ["signal", "data"]:
    #                 continue
    #             if ("nominal" in proc_hists and
    #                 region in proc_hists["nominal"] and
    #                 observable in proc_hists["nominal"][region]):
    #                 background_hist += proc_hists["nominal"][region][observable]

    #         # Get data histogram
    #         data_hist = None
    #         if ("data" in self.histograms and
    #             "nominal" in self.histograms["data"] and
    #             region in self.histograms["data"]["nominal"] and
    #             observable in self.histograms["data"]["nominal"][region]):
    #             data_hist = self.histograms["data"]["nominal"][region][observable]

    #         if data_hist is None:
    #             continue

    #         channel_data.append({
    #             "region": region,
    #             "observable": observable,
    #             "signal": signal_hist,
    #             "background": background_hist,
    #             "data": data_hist,
    #             "signal_syst": signal_kappa,
    #             "background_syst": background_kappa,
    #         })

    #     print(signal_hist)
    #     print(background_hist)
    #     print(data_hist)
    #     if not channel_data:
    #         logger.warning("[evermore] No valid channels found for significance calculation")
    #         return jnp.array(0.0)

    #     # ------------------------------------------------------------
    #     # 2. Define the statistical model
    #     # ------------------------------------------------------------
    #     class Params(NamedTuple):
    #         mu: evm.Parameter  # Signal strength (POI)
    #         # We'll create one nuisance parameter per systematic per channel
    #         theta_s: list[evm.NormalParameter]  # Signal systematics
    #         theta_b: list[evm.NormalParameter]  # Background systematics

    #     def model(params: Params, channel_data: list) -> list[jnp.ndarray]:
    #         expectations = []
    #         for i, channel in enumerate(channel_data):
    #             # Get modifiers for this channel
    #             mu_modifier = params.mu.scale()
    #             signal_modifier = params.theta_s[i].scale(offset=1.0, slope=channel["signal_syst"])
    #             bkg_modifier = params.theta_b[i].scale(offset=1.0, slope=channel["background_syst"])

    #             # Calculate expectation: μ × signal × (1 + θ_s × κ_s) + bkg × (1 + θ_b × κ_b)
    #             signal_part = mu_modifier(channel["signal"]) * signal_modifier(channel["signal"])
    #             bkg_part = bkg_modifier(channel["background"])
    #             expectations.append(signal_part + bkg_part)
    #         return expectations


    #     print(jax.device_get(model()))

    #     def loss(
    #         diffable: PyTree,
    #         static: PyTree,
    #         channel_data: list,
    #     ) -> Array:
    #         params = eqx.combine(diffable, static)
    #         expectations = model(params, channel_data)

    #         total_log_likelihood = 0.0
    #         for i, channel in enumerate(channel_data):
    #             # Poisson log-likelihood for this channel
    #             poisson_llh = evm.pdf.Poisson(lamb=expectations[i]).log_prob(channel["data"]).sum()
    #             total_log_likelihood += poisson_llh

    #             # Constraints for nuisance parameters (Gaussian)
    #             signal_constraint = evm.pdf.Normal(mean=0.0, width=1.0).log_prob(params.theta_s[i].value)
    #             bkg_constraint = evm.pdf.Normal(mean=0.0, width=1.0).log_prob(params.theta_b[i].value)
    #             total_log_likelihood += signal_constraint + bkg_constraint

    #         return -jnp.sum(total_log_likelihood)  # Negative log-likelihood for minimization

    #     # ------------------------------------------------------------
    #     # 3. Parameter setup and fitting
    #     # ------------------------------------------------------------
    #     # Initialize parameters
    #     init_params = Params(
    #         mu=evm.Parameter(1.0),
    #         theta_s=[evm.NormalParameter(0.0) for _ in channel_data],
    #         theta_b=[evm.NormalParameter(0.0) for _ in channel_data],
    #     )

    #     # Partition parameters into differentiable and static parts
    #     diffable, static = evm.parameter.partition(init_params)

    #     # Create optimizer
    #     optimizer = optax.adam(learning_rate=0.1)
    #     opt_state = optimizer.init(diffable)

    #     # Minimization function
    #     @jax.jit
    #     def step(diffable, static, opt_state):
    #         loss_value, grads = eqx.filter_value_and_grad(loss)(diffable, static, channel_data)
    #         updates, opt_state = optimizer.update(grads, opt_state)
    #         diffable = optax.apply_updates(diffable, updates)
    #         return diffable, opt_state, loss_value

    #     # ------------------------------------------------------------
    #     # 4. Fit under alternative hypothesis (μ free)
    #     # ------------------------------------------------------------
    #     print("Fitting alternative hypothesis (μ free)...")
    #     for i in range(100):  # Fixed number of iterations for simplicity
    #         diffable, opt_state, loss_value = step(diffable, static, opt_state)
    #         loss_value = jax.device_get(loss_value).item()  # Ensure loss is a scalar
    #         if i % 10 == 0:
    #             print(f"Step {i}: loss = {loss_value:.4f}")

    #     alt_loss = loss_value
    #     alt_params = eqx.combine(diffable, static)

    #     # ------------------------------------------------------------
    #     # 5. Fit under null hypothesis (μ = 0)
    #     # ------------------------------------------------------------
    #     print("Fitting null hypothesis (μ = 0)...")
    #     # Create new parameters with μ frozen at 0
    #     null_params = Params(
    #         mu=evm.Parameter(0.0, frozen=True),
    #         theta_s=alt_params.theta_s,
    #         theta_b=alt_params.theta_b,
    #     )
    #     diffable_null, static_null = evm.parameter.partition(null_params)
    #     opt_state_null = optimizer.init(diffable_null)

    #     for i in range(100):
    #         diffable_null, opt_state_null, loss_value = step(diffable_null, static_null, opt_state_null)
    #         loss_value = jax.device_get(loss_value).item()  # Ensure loss is a scalar
    #         if i % 10 == 0:
    #             print(f"Step {i}: loss = {loss_value:.4f}")

    #     null_loss = loss_value

    #     # ------------------------------------------------------------
    #     # 6. Compute test statistic and significance
    #     # ------------------------------------------------------------
    #     # q0 = -2 log(λ) = 2 * [L(null) - L(alt)]
    #     # Where L = log-likelihood (but our loss is negative log-likelihood)
    #     q0 = 2 * (null_loss - alt_loss)  # Note: loss = -logL ⇒ logL = -loss
    #     q0 = jnp.clip(q0, 0.0)  # Ensure non-negative
    #     significance = jnp.sqrt(q0)

    #     print("\n" + "="*50)
    #     print(f"Null loss: {null_loss:.4f}, Alt loss: {alt_loss:.4f}")
    #     print(f"q0 = {q0:.4f}, Significance = {significance:.4f}σ")
    #     print("="*50)

        # from typing import NamedTuple

        # import equinox as eqx
        # from jaxtyping import Array, PyTree

        # import evermore as evm

        # jax.config.update("jax_enable_x64", True)


        # # define a simple model with two processes and two parameters
        # def model(params: PyTree, hists: dict[str, Array]) -> Array:
        #     mu_modifier = params.mu.scale()
        #     syst_modifier = params.syst.scale_log(up=1.1, down=0.9)
        #     return mu_modifier(hists["signal"]) + syst_modifier(hists["background"])


        # def loss(
        #     diffable: PyTree,
        #     static: PyTree,
        #     hists: dict[str, Array],
        #     observation: Array,
        # ) -> Array:
        #     params = eqx.combine(diffable, static)
        #     expectation = model(params, hists)
        #     # Poisson NLL of the expectation and observation
        #     log_likelihood = evm.pdf.Poisson(lamb=expectation).log_prob(observation).sum()
        #     # Add parameter constraints from logpdfs
        #     constraints = evm.loss.get_log_probs(params)
        #     log_likelihood += evm.util.sum_over_leaves(constraints)
        #     return -jnp.sum(log_likelihood)


        # # setup data

        # hists = {"signal":  self.histograms["signal"]["nominal"]["CMS_WORKSHOP_JAX"]["workshop_mtt"],
        #         "background": self.histograms["ttbar_semilep"]["nominal"]["CMS_WORKSHOP_JAX"]["workshop_mtt"]
        #         }

        # observation = self.histograms["data"]["nominal"]["CMS_WORKSHOP_JAX"]["workshop_mtt"]


        # # define parameters, can be any PyTree of evm.Parameters
        # class Params(NamedTuple):
        #     mu: evm.Parameter
        #     syst: evm.NormalParameter


        # params = Params(mu=evm.Parameter(1.0), syst=evm.NormalParameter(0.0))
        # diffable, static = evm.parameter.partition(params)

        # # Calculate negative log-likelihood/loss
        # loss_val = loss(diffable, static, hists, observation)
        # # gradients of negative log-likelihood w.r.t. diffable parameters
        # grads = eqx.filter_grad(loss)(diffable, static, hists, observation)
        # print(f"{grads.mu.value=}, {grads.syst.value=}")

        return 2.5

    # -------------------------------------------------------------------------
    # Histogramming Logic
    # -------------------------------------------------------------------------
    def histogramming(
        self,
        object_copies: dict[str, ak.Array],
        events: ak.Array,
        process: str,
        variation: str,
        xsec_weight: float,
        analysis: str,
        params: dict[str, Any],
        event_syst: Optional[dict[str, Any]] = None,
        direction: Literal["up", "down", "nominal"] = "nominal",
    ) -> dict[str, jnp.ndarray]:
        """
        Apply selections and fill histograms for each observable and channel.

        Parameters
        ----------
        object_copies : dict
            Corrected event-level objects.
        events : ak.Array
            Original NanoAOD events.
        process : str
            Sample label (e.g. 'ttbar', 'data').
        variation : str
            Systematic variation label.
        xsec_weight : float
            Cross-section normalization.
        analysis : str
            Analysis name string.
        params : dict
            JAX parameters used in soft selections.
        event_syst : dict, optional
            Event-level systematic.
        direction : str, optional
            Systematic direction: 'up', 'down', 'nominal'.

        Returns
        -------
        dict[str, jnp.ndarray]
            Histogram dictionary for the channel and observables.
        """

        # ---------------------------
        # Setup and fast exits
        # ---------------------------
        jax_config = self.config.jax
        histograms = defaultdict(dict)

        if process == "data" and variation != "nominal":
            return histograms

        # ---------------------------
        # Move data to JAX backend
        # ---------------------------
        events = recursive_to_backend(events, "jax")
        object_copies = recursive_to_backend(object_copies, "jax")

        # ---------------------------
        # Compute soft selection weights using differentiable function
        # ---------------------------
        diff_selection_args = self._get_function_arguments(
            jax_config.soft_selection.use, object_copies
        )
        diff_selection_weights = jax_config.soft_selection.function(
            *diff_selection_args, params
        )

        # ---------------------------
        # Loop over channels
        # ---------------------------
        for channel in self.channels:
            if not channel.use_in_diff:
                logger.warning(f"Skipping channel {channel.name} in diff analysis")
                continue

            chname = channel["name"]
            if (
                (req_channels := self.config.general.channels)
                and chname not in req_channels
            ):
                continue

            logger.info(f"Applying selection for {chname} in {process}")

            # ---------------------------
            # Apply packed selection mask
            # ---------------------------
            mask = 1
            if (sel_fn := channel.selection.function):
                selection_args = self._get_function_arguments(
                    channel.selection.use, object_copies
                )
                packed = sel_fn(*selection_args)
                if not isinstance(packed, PackedSelection):
                    raise ValueError("Expected PackedSelection")
                mask = ak.Array(packed.all(packed.names[-1]))

            mask = recursive_to_backend(mask, "jax")

            # If data, apply good run list via lumi mask
            if process == "data":
                good_runs = lumi_mask(
                    self.config.general.lumifile,
                    object_copies["run"],
                    object_copies["luminosityBlock"],
                    jax=True,
                )
                mask = mask & ak.to_backend(good_runs, "jax")

            if ak.sum(mask) == 0:
                logger.warning(f"No events left in {chname} for {process}.")
                continue

            # ---------------------------
            # Apply selection mask to objects
            # ---------------------------
            obj_copies_ch = {
                k: v[mask] for k, v in object_copies.items()
            }

            # Compute per-event weights
            if process != "data":
                weights = (
                    events[mask].genWeight * xsec_weight
                    / abs(events[mask].genWeight)
                )
            else:
                weights = ak.Array(np.ones(ak.sum(mask)))

            # Apply event-level systematic reweighting if needed
            if event_syst and process != "data":
                weights = self.apply_event_weight_correction(
                    weights, event_syst, direction, obj_copies_ch
                )

            weights = jnp.array(weights.to_numpy())
            diff_selection_weights = diff_selection_weights[ak.to_jax(mask)]

            logger.info(f"Events in {chname}: raw={ak.sum(mask)}, weighted={ak.sum(weights)}")

            # ---------------------------
            # Fill histograms for each observable
            # ---------------------------
            for observable in channel["observables"]:
                if not observable.works_with_jax:
                    logger.warning(f"Skipping {observable['name']}, not JAX-compatible.")
                    continue

                # Evaluate observable values
                observable_args = self._get_function_arguments(
                    observable["use"], obj_copies_ch
                )
                values = observable["function"](*observable_args)
                binning = observable["binning"]

                # Parse binning string if needed
                bandwidth = jax_config.params["kde_bandwidth"]
                if isinstance(binning, str):
                    low, high, nbins = map(float, binning.split(","))
                    binning = jnp.linspace(low, high, int(nbins))
                else:
                    binning = jnp.array(binning)

                # KDE-based soft histogramming
                cdf = jax.scipy.stats.norm.cdf(
                    binning.reshape(-1, 1),
                    loc=ak.to_jax(values).reshape(1, -1),
                    scale=bandwidth,
                )
                weighted_cdf = (
                    cdf * diff_selection_weights.reshape(1, -1)
                    * weights.reshape(1, -1)
                )
                bin_weights = weighted_cdf[1:, :] - weighted_cdf[:-1, :]
                histogram = jnp.sum(bin_weights, axis=1)

                histograms[chname][observable["name"]] = histogram

        return histograms

    # -------------------------------------------------------------------------
    # Event Processing Entry Point
    # -------------------------------------------------------------------------
    def process(
        self,
        events: ak.Array,
        metadata: dict[str, Any],
        params: dict[str, Any],
    ) -> dict[str, dict[str, dict[str, jnp.ndarray]]]:
        """
        Run the full analysis logic on events from one dataset.

        Parameters
        ----------
        events : ak.Array
            Input NanoAOD events.
        metadata : dict
            Metadata with keys 'process', 'xsec', 'nevts', and 'dataset'.
        params : dict
            JAX parameter dictionary.

        Returns
        -------
        dict
            Histogram dictionary keyed by variation/channel/observable.
        """

        # ------
        # Metadata unpacking
        # ------
        all_histograms = self.histograms.copy()
        process = metadata["process"]
        variation = metadata.get("variation", "nominal")
        analysis = self.__class__.__name__
        xsec = metadata["xsec"]
        n_gen = metadata["nevts"]
        lumi = self.config["general"]["lumi"]
        xsec_weight = (xsec * lumi / n_gen) if process != "data" else 1.0

        # ------
        # Object preparation and baseline filtering
        # ------
        obj_copies = self.get_object_copies(events)

        # Use CPU backend for jagged masks
        obj_copies = self.apply_object_masks(obj_copies)

        # Move to JAX for processing
        events = recursive_to_backend(events, "jax")
        obj_copies = recursive_to_backend(obj_copies, "jax")

        # Apply baseline selection mask
        baseline_args = self._get_function_arguments(
            self.config.baseline_selection["use"], obj_copies
        )
        packed = self.config.baseline_selection["function"](*baseline_args)
        mask = ak.Array(packed.all(packed.names[-1]))
        # Move mask to JAX backend
        mask = recursive_to_backend(mask, "jax")
        obj_copies = {k: v[mask] for k, v in obj_copies.items()}

        # ------
        # Ghost observable computation and object correction
        # ------
        # Compute ghost observables
        obj_copies = self.compute_ghost_observables(obj_copies)
        # Apply object corrections (e.g. JEC, JER)
        obj_copies_corrected = self.apply_object_corrections(
            obj_copies, self.corrections, direction="nominal"
        )
        # Ensure corrected objects in JAX backend
        obj_copies_corrected = recursive_to_backend(obj_copies_corrected, "jax")

        # ------
        # Nominal histogramming
        # ------
        histograms = self.histogramming(
            obj_copies_corrected,
            events,
            process,
            "nominal",
            xsec_weight,
            analysis,
            params,
        )
        all_histograms["nominal"] = histograms

        # ------
        # Loop over systematic variations
        # ------
        if self.config.general.run_systematics:
            for syst in self.systematics + self.corrections:
                if syst["name"] == "nominal":
                    continue

                for direction in ["up", "down"]:
                    varname = f"{syst['name']}_{direction}"
                    # Move objects to CPU backend for jagged masks
                    events = recursive_to_backend(events, "cpu")
                    obj_copies = recursive_to_backend(obj_copies, "cpu")
                    obj_copies = self.apply_object_masks(obj_copies)

                    # Move back to JAX for processing
                    events = recursive_to_backend(events, "jax")
                    obj_copies = recursive_to_backend(obj_copies, "jax")

                    # Apply object corrections (e.g. JEC, JER)
                    obj_copies_corrected = self.apply_object_corrections(
                        obj_copies, [syst], direction=direction
                    )
                    # ------
                    # Variation histogramming
                    # ------
                    histograms = self.histogramming(
                        obj_copies_corrected,
                        events,
                        process,
                        varname,
                        xsec_weight,
                        analysis,
                        params,
                        event_syst=syst,
                        direction=direction,
                    )
                    all_histograms[varname] = histograms

        return all_histograms

    # -------------------------------------------------------------------------
    # Main Analysis Loop
    # -------------------------------------------------------------------------
    def run_analysis_chain(
        self,
        params: dict[str, Any],
        fileset: dict[str, Any],
        read_from_cache: bool = False,
        run_and_cache: bool = True,
        cache_dir: Optional[str] = "/tmp/gradients_analysis/",
    ) -> jnp.ndarray:
        """
        Run the full analysis on all datasets in the fileset.

        Parameters
        ----------
        params : dict
            Dictionary of analysis parameters.
        fileset : dict
            Dictionary mapping dataset names to file and metadata.

        Returns
        -------
        jnp.ndarray
            Final signal significance.
        """

        # ----------------------------
        # Initialize histograms store
        # ----------------------------
        config = self.config
        process_histograms: dict[str, dict[str, dict[str, jnp.ndarray]]] = defaultdict(dict)

        # ----------------------------
        # Iterate over datasets
        # ----------------------------
        for dataset, content in fileset.items():
            metadata = content["metadata"]
            metadata["dataset"] = dataset
            process_name = metadata["process"]

            if process_name not in process_histograms:
                process_histograms[process_name] = defaultdict(
                    lambda: defaultdict(dict)
                )

            if (
                (req_processes := config.general.processes)
                and process_name not in req_processes
            ):
                continue

            os.makedirs(f"{config.general.output_dir}/{dataset}", exist_ok=True)

            logger.info("========================================")
            logger.info(f"🚀 Processing dataset: {dataset}")

            # -----------------------------
            # Iterate over skimmed files
            # -----------------------------
            for idx, (_, tree) in enumerate(content["files"].items()):
                if (
                    config.general.max_files != -1
                    and idx >= config.general.max_files
                ):
                    continue

                output_dir = (
                    f"output/{dataset}/file__{idx}/"
                    if not config.general.preprocessed_dir
                    else f"{config.general.preprocessed_dir}/{dataset}/file__{idx}/"
                )

                skimmed_files = glob.glob(f"{output_dir}/part*.root")
                skimmed_files = [f"{f}:{tree}" for f in skimmed_files]
                remaining = sum(
                    uproot.open(f).num_entries for f in skimmed_files
                )
                logger.info(
                    f"✅ Events retained after filtering: {remaining:,}"
                )

                for skimmed in skimmed_files:
                    logger.info(f"📘 Processing skimmed file: {skimmed}")
                    # If caching is used, create cache directory and build
                    # cache file name
                    if run_and_cache or read_from_cache:
                        os.makedirs(cache_dir, exist_ok=True)
                        cache_key = hashlib.md5(skimmed.encode()).hexdigest()
                        cache_file = os.path.join(cache_dir, f"{dataset}__{cache_key}.pkl")
                    # If user asks to process data then cache it, do it
                    if run_and_cache:
                        events = NanoEventsFactory.from_root(
                            skimmed, schemaclass=NanoAODSchema, delayed=False
                        ).events()
                        with open(cache_file, "wb") as f:
                            cloudpickle.dump(events, f)
                        logger.info(f"💾 Cached events to {cache_file}")

                    # If user does not want to run then cache, they either want to
                    # read from cache, or just reprocess without caching
                    else:
                        # If user wants to read from cache
                        if read_from_cache:
                            # Check if cache file exists and read from it
                            if os.path.exists(cache_file):
                                with open(cache_file, "rb") as f:
                                    events = cloudpickle.load(f)
                                logger.info(f"🔁 Loaded cached events from {cache_file}")

                            # otherwise, reprocess the file and cache it
                            else:
                                logger.warning(
                                    f"Cache file {cache_file} not found. Reprocessing."
                                )
                                events = NanoEventsFactory.from_root(
                                    skimmed, schemaclass=NanoAODSchema, delayed=False
                                ).events()
                                with open(cache_file, "wb") as f:
                                    cloudpickle.dump(events, f)
                                logger.info(f"💾 Cached events to {cache_file}")
                        # In this case user wants nothing to do with caching
                        else:
                            events = NanoEventsFactory.from_root(
                                skimmed, schemaclass=NanoAODSchema, delayed=False
                            ).events()

                    histograms = self.process(events, metadata, params)
                    process_histograms[process_name] = merge_histograms(
                        process_histograms[process_name], dict(histograms)
                    )

            logger.info(f"✅ Finished dataset: {dataset}\n")

        # -----------------------------
        # Final aggregation and return
        # -----------------------------
        self.set_histograms(process_histograms)
        significance = self._calculate_significance()

        logger.info("✅ All datasets processed.")
        return significance

    # -------------------------------------------------------------------------
    # Run Full Chain with JAX Gradients
    # -------------------------------------------------------------------------
    def run_analysis_chain_with_gradients(
        self, fileset: dict[str, dict[str, Any]],
        read_from_cache: bool = False,
        run_and_cache: bool = True,
        cache_dir: Optional[str] = "/tmp/gradients_analysis/"
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """
        Run the full analysis chain and compute gradients w.r.t. parameters.

        Parameters
        ----------
        fileset : dict
            Dataset files and metadata

        Returns
        -------
        tuple
            (Significance, gradient dictionary)
        """


        # Compute significance from datasets
        significance, gradients = jax.value_and_grad(
                                    self.run_analysis_chain, argnums=0
                                )(self.config.jax.params, fileset,
                                  read_from_cache, run_and_cache, cache_dir)

        logger.info(f"Signal significance: {significance:.4f}")
        logger.info(f"Gradient dictionary: {pformat(gradients)}")

        return significance, gradients

    # -------------------------------------------------------------------------
    # Cut Optimization via Gradient Ascent
    # -------------------------------------------------------------------------
    def optimize_analysis_cuts(
        self, fileset: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
        """
        Optimize analysis cuts using gradient ascent.

        Parameters
        ----------
        fileset : dict
            File dictionary used for histogram generation.

        Returns
        -------
        tuple
            Optimized parameters and final significance.
        """
        logger.info("Running analysis chain with init values for all datasets...")
        logger.info(f"Processes: {list(fileset.keys())}")
        # Run the initial analysis chain to get starting significance and gradients
        cache_dir = "/tmp/gradients_analysis/"
        if self.config.general.read_from_cache:
            read_from_cache = True
            run_and_cache = False

        significance, gradients = (
            self.run_analysis_chain_with_gradients(fileset,
                                                    read_from_cache=read_from_cache,
                                                    run_and_cache=run_and_cache,
                                                    cache_dir=cache_dir)
                                                )

        params = self.config.jax.params.copy()
        if not self.config.jax.optimize:
            return params, significance

        logger.info("Starting parameter optimization...")
        logger.info(f"Initial significance: {significance:.4f}")

        def objective(params):
            return self.run_analysis_chain(params, fileset,
                                            read_from_cache=True,
                                            run_and_cache=False,
                                            cache_dir=cache_dir)

        learning_rate = self.config.jax.learning_rate

        # Save initial values for comparison
        initial_params = {k: float(v) for k, v in params.items()}

        logger.info("Starting optimization...\n")
        print(self.config.jax.max_iterations)

        for i in range(self.config.jax.max_iterations):
            # Compute significance and gradients
            significance, gradients = jax.value_and_grad(objective, argnums=0)(params)

            # Parameter update
            for key in params:
                delta = learning_rate * gradients[key]
                update_fn = self.config.jax.param_updates.get(key, lambda x, d: x + d)
                params[key] = update_fn(params[key], delta)

            # Build table with value, gradient, and % change as columns
            step_summary = [["Parameter", "Value", "Gradient", "% Change"]]

            for key in sorted(params.keys()):
                if isinstance(params[key], (int, float, jnp.ndarray)):
                    new_val = float(params[key])
                    grad = float(gradients[key])
                    old_val = initial_params[key]
                    percent_change = ((new_val - old_val) / (old_val + 1e-12)) * 100

                    colored_val = f"{GREEN}{new_val:.4f}{RESET}" \
                        if abs(new_val - old_val) > 1e-5 else f"{new_val:.4f}"
                    step_summary.append([
                        f"{key:<30}",
                        colored_val,
                        f"{grad:+.4f}",
                        f"{percent_change:+.2f}%"
                    ])

            # Add significance row
            step_summary.append(["-" * 30, "-" * 10, "-" * 10, "-" * 10])
            step_summary.append(["Significance", f"{significance:.4f}", "", ""])

            logger.info("\n" + "=" * 60)
            logger.info(f" Step {i + 1}: Optimization Summary")
            logger.info("\n" + tabulate(step_summary, tablefmt="fancy_grid", colalign=("left", "right", "right", "right")))
            logger.info("=" * 60)

        # After loop: Final summary
        final_significance = objective(params)
        improvement = ((final_significance / significance - 1) * 100)

        # Build colored final param summary
        param_summary = []
        for key in sorted(params.keys()):
            new_val = float(params[key])
            old_val = initial_params[key]
            delta = abs(new_val - old_val)
            formatted_val = f"{new_val:.4f}"
            formatted_key = f"{key:<30}"
            if delta > 1e-5:
                formatted_val = f"{GREEN}{formatted_val}{RESET}"
                formatted_key = f"{GREEN}{formatted_key}{RESET}"
            param_summary.append([formatted_key, formatted_val])

        # Significance summary
        final_stats = [
            ["Initial Significance", f"{significance:.4f}"],
            ["Final Significance", f"{final_significance:.4f}"],
            ["Improvement (%)", f"{improvement:.2f}%"],
        ]

        # Print summaries
        logger.info("\nFinal Optimized Parameters:")
        logger.info("\n" + tabulate(param_summary,
                                    headers=["Parameter", "Value"],
                                    tablefmt="fancy_grid",
                                    colalign=("left", "right")))

        logger.info("Significance Summary:")
        logger.info("\n" + tabulate(final_stats,
                                    tablefmt="fancy_grid",
                                    colalign=("left", "right")))

        return params, final_significance