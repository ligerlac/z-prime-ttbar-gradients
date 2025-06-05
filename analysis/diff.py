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
from typing import Any, Callable, Dict, Literal, List, Tuple, Optional

import awkward as ak
import equinox as eqx
import evermore as evm
import jax
import jax.numpy as jnp
import numpy as np
from tabulate import tabulate
import uproot
import vector
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

from analysis.base import Analysis
from utils.cuts import lumi_mask
from utils.evm_stats import calculate_significance, calculate_significance_relaxed, _get_valid_channels, _create_parameter_structure, _prepare_channel_data, print_significance_summary

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

def infer_processes_and_systematics(
    fileset: Dict[str, Dict[str, Any]],
    systematics: List[Dict[str, Any]],
    corrections: List[Dict[str, Any]]
) -> Tuple[List[str], List[str]]:
    """
    Infer all process names and base systematic names (without up/down) from fileset and config.

    Parameters
    ----------
    fileset : dict
        Fileset with metadata containing process names
    systematics : list
        List from config["systematics"]
    corrections : list
        List from config["corrections"]

    Returns
    -------
    Tuple[List[str], List[str]]
        Sorted list of unique processes and systematic base names (e.g. "jec")
    """
    process_set = set()
    systematics_set = set()

    for dataset_info in fileset.values():
        metadata = dataset_info.get("metadata", {})
        process = metadata.get("process")
        if process is not None:
            process_set.add(process)

    for syst in systematics + corrections:
        systematics_set.add(syst["name"])

    return sorted(process_set), sorted(systematics_set)

def extract_scalar(x):
    """
    Extract scalar float from a parameter or gradient.
    Handles evm.Parameter, jnp.ndarray, or raw float.
    """
    if isinstance(x, evm.Parameter):
        return float(x.value.astype(float)[0])
    if isinstance(x, jnp.ndarray):
        return float(x)
    return float(x)

def update_params(
    params: dict[str, Any],
    gradients: dict[str, Any],
    learning_rate: float,
    param_updates: dict[str, Callable[[Any, float], Any]],
) -> dict[str, Any]:
    """
    Update 'aux' and 'fit' parameters given gradients and learning rate.

    Parameters
    ----------
    params : dict
        Dictionary with keys 'aux' and 'fit'. 'fit' is a NamedTuple with nested dicts.
    gradients : dict
        Gradient dictionary with same structure.
    learning_rate : float
        Step size.
    param_updates : dict
        Update function mapping for aux parameters.
    """
    # --------------------
    # Update aux parameters
    # --------------------
    for key, val in params["aux"].items():
        grad = gradients["aux"][key]
        delta = learning_rate * extract_scalar(grad)
        update_fn = param_updates.get(key, lambda x, d: x + d)
        params["aux"][key] = update_fn(val, delta)

    def _update_evm(p, g, lr):
        return evm.Parameter(
                        value=p.value + lr * g.value,
                        lower=p.lower,
                        upper=p.upper,
                    )
    # --------------------
    # Update fit parameters
    # --------------------
    fit_updated = {}
    for key, val in params["fit"]._asdict().items():
        grad = gradients["fit"]._asdict()[key]

        if isinstance(val, dict):
            sub_updated = {}
            for subkey, p in val.items():
                g = grad[subkey]
                sub_updated[subkey] = _update_evm(p, g, learning_rate)
            fit_updated[key] = sub_updated
        else:
            fit_updated[key] = _update_evm(val, grad, learning_rate)

    # Replace fit NamedTuple with updated values
    params["fit"] = type(params["fit"])(**fit_updated)
    return params

def flatten_params(params: dict[str, Any]) -> dict[str, float]:
    """
    Flatten a nested parameter dictionary where 'fit' is a NamedTuple with possible
    nested dicts like 'norm' and 'nuis'.

    Produces keys like:
      'aux.cut_threshold'
      'fit.mu'
      'fit.norm.ttbar_semilep'
      'fit.nuis.JES'

    Returns
    -------
    dict[str, float]
        Fully flattened parameter dictionary with scalar values.
    """
    flat = {}

    for k, v in params["aux"].items():
        flat[f"aux.{k}"] = extract_scalar(v)

    for k, v in params["fit"]._asdict().items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"fit.{k}.{kk}"] = extract_scalar(vv)
        else:
            flat[f"fit.{k}"] = extract_scalar(v)

    return flat

def build_param_summary(
    params: dict[str, Any],
    gradients: dict[str, Any],
    initial_values: dict[str, float],
) -> list[list[str]]:
    """
    Build a summary table showing parameter values, gradients, and percent changes.

    Handles nested structures in 'fit', including subdicts like 'norm' and 'nuis'.
    """
    summary = [["Parameter", "Value", "Gradient", "% Change"]]

    def iter_group(group_name: str, value_dict: dict, grad_dict: dict):
        for k, v in value_dict.items():
            g = grad_dict.get(k)
            if isinstance(v, dict):
                for kk, vv in v.items():
                    gg = g[kk]
                    key = f"{group_name}.{k}.{kk}"
                    val = extract_scalar(vv)
                    grad = extract_scalar(gg)
                    old = initial_values.get(key, 0.0)
                    percent = ((val - old) / (old + 1e-12)) * 100
                    colored = f"{GREEN}{val:.4f}{RESET}" if abs(val - old) > 1e-5 else f"{val:.4f}"
                    yield [f"{key:<30}", colored, f"{grad:+.4f}", f"{percent:+.2f}%"]
            else:
                key = f"{group_name}.{k}"
                val = extract_scalar(v)
                grad = extract_scalar(g)
                old = initial_values.get(key, 0.0)
                percent = ((val - old) / (old + 1e-12)) * 100
                colored = f"{GREEN}{val:.4f}{RESET}" if abs(val - old) > 1e-5 else f"{val:.4f}"
                yield [f"{key:<30}", colored, f"{grad:+.4f}", f"{percent:+.2f}%"]

    summary += list(iter_group("aux", params["aux"], gradients["aux"]))
    summary += list(iter_group("fit", params["fit"]._asdict(), gradients["fit"]._asdict()))
    return summary

def format_param_summary_flat(params: dict, initial_flat: dict) -> list[list[str]]:
    """
    Compare current nested parameters to initial flat ones and format summary.

    Parameters
    ----------
    params : dict
        Nested parameter dict with 'aux' and 'fit' (NamedTuple).
    initial_flat : dict
        Flat dict of initial parameter values (e.g. from flatten_params).

    Returns
    -------
    list of [str, str]
        Table rows: [Parameter Name, Formatted Value]
    """
    rows = []

    flat_current = flatten_params(params)
    for key in sorted(flat_current.keys()):
        new_val = extract_scalar(flat_current[key])
        old_val = extract_scalar(initial_flat[key])
        delta = abs(new_val - old_val)

        formatted_val = f"{new_val:.4f}"
        formatted_key = f"{key:<30}"
        if delta > 1e-5:
            formatted_val = f"{GREEN}{formatted_val}{RESET}"
            formatted_key = f"{GREEN}{formatted_key}{RESET}"

        rows.append([formatted_key, formatted_val])

    return rows

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
    def _calculate_significance(self, params, recreate_fit_params: bool = False) -> jnp.ndarray:
        """
        Generalized significance calculation using evermore with multi-channel,
        multi-process, and systematic-aware modeling.

        Returns
        -------
        jnp.ndarray
            Asymptotic significance (sqrt(q0)) from a profile likelihood ratio.
        """
        p0 = (
            calculate_significance_relaxed(self.histograms,
                                   self.channels,
                                   params,
                                   #recreate_fit_params=recreate_fit_params
                                )
            )

        print("ccc", p0)
        #print_significance_summary(p0, -1.0)

        return p0

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
        all_histograms = defaultdict(
            lambda: defaultdict(dict)
        )
        process = metadata["process"]
        variation = metadata.get("variation", "nominal")
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
        recreate_fit_params: bool = False,
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

                    histograms = self.process(events, metadata, params["aux"])
                    process_histograms[process_name] = merge_histograms(
                        process_histograms[process_name], dict(histograms)
                    )

            logger.info(f"✅ Finished dataset: {dataset}\n")

        # -----------------------------
        # Final aggregation and return
        # -----------------------------
        self.set_histograms(process_histograms)
        significance = self._calculate_significance(params["fit"], recreate_fit_params=recreate_fit_params)

        logger.info("✅ All datasets processed.")

        return significance

    # -------------------------------------------------------------------------
    # Run Full Chain with JAX Gradients
    # -------------------------------------------------------------------------
    def run_analysis_chain_with_gradients(
        self, fileset: dict[str, dict[str, Any]],
        read_from_cache: bool = False,
        run_and_cache: bool = True,
        cache_dir: Optional[str] = "/tmp/gradients_analysis/",
        params = None,
        recreate_fit_params: bool = False
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
                                    self.run_analysis_chain, argnums=0,
                                )(params,
                                  fileset,
                                  read_from_cache,
                                  run_and_cache,
                                  cache_dir,
                                  recreate_fit_params=recreate_fit_params,
                                )


        logger.info(f"Signal significance: {significance:.4f}")
        logger.info(f"Gradient dictionary: {pformat(gradients)}")

        # Make sure we have no redundant parameters in fit
        valid_channels = _get_valid_channels(self.channels)
        _, systematics, processes = _prepare_channel_data(
            self.histograms, valid_channels
        )
        evm_params = _create_parameter_structure(processes, systematics)
        params["fit"] = evm_params

        return significance, gradients, params

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
        from functools import partial
        import optax
        from jaxopt import OptaxSolver

        logger.info("Running analysis chain with init values for all datasets...")
        logger.info(f"Processes: {list(fileset.keys())}")

        # Some book keeping
        processes, systematics = infer_processes_and_systematics(
            fileset,
            self.config.systematics,
            self.config.corrections
        )
        aux_params = self.config.jax.params.copy()
        evm_params = _create_parameter_structure(processes, systematics)
        all_params = {
            "aux": aux_params,
            "fit": evm_params,
        }


        # Run the initial analysis chain to get starting significance and gradients
        cache_dir = "/tmp/gradients_analysis/"
        if self.config.general.read_from_cache:
            read_from_cache = True
            run_and_cache = False
        else:
            read_from_cache = False
            run_and_cache = True

        # in theory once we build histograms we may
        # drop some parameters
        # significance, gradients, params = (
        #     self.run_analysis_chain_with_gradients(fileset,
        #                                             read_from_cache=read_from_cache,
        #                                             run_and_cache=run_and_cache,
        #                                             cache_dir=cache_dir,
        #                                             params = all_params,
        #                                             recreate_fit_params=True)
        #                                         )

        pvals = self.run_analysis_chain(all_params, fileset,
                                        read_from_cache,
                                        run_and_cache,
                                        cache_dir)  # map over phi grid
        # # Wrap your analysis chain into a differentiable scalar function
        # def objective(params: dict) -> jnp.ndarray:
        #     # Returns a scalar (p0 value or negative Z)
        #     p0 = self.run_analysis_chain(params, fileset,
        #                                 read_from_cache,
        #                                 run_and_cache,
        #                                 cache_dir)
        #     return p0  # minimize p-value

        # initial_params = all_params
        # solver = OptaxSolver(objective, opt=optax.adam(1e-3), tol=1e-8, maxiter=10000)
        # result = solver.run(all_params).params
        # print(
        #     f"our solution: phi={result:.5f}"
        # )

        return pvals
        if not self.config.jax.optimize:
            return params, significance

        initial_significance = significance
        logger.info("Starting parameter optimization...")
        logger.info(f"Initial significance: {initial_significance:.4f}")

        def objective(params):
            return self.run_analysis_chain(params, fileset,
                                            read_from_cache=True,
                                            run_and_cache=False,
                                            cache_dir=cache_dir)


        learning_rate = self.config.jax.learning_rate
        # Save initial values for comparison
        initial_params = flatten_params(params)
        logger.info("Starting optimization...\n")

        for i in range(self.config.jax.max_iterations):
            significance, gradients = jax.value_and_grad(objective, argnums=0)(params)

            # Update parameters
            params = update_params(params, gradients, learning_rate, self.config.jax.param_updates)

            # Summary table
            step_summary = build_param_summary(params, gradients, initial_params)
            step_summary.append([" " * 30, " " * 10, " " * 10, " " * 10])
            step_summary.append(["Significance", f"{significance:.4f}", "", ""])

            logger.info("\n" + "=" * 60)
            logger.info(f" Step {i + 1}: Optimization Summary:\n" +
                        tabulate(step_summary, tablefmt="fancy_grid", colalign=("left", "right", "right", "right")))
            logger.info("=" * 60)

        # After loop: Final summary
        final_significance = significance
        param_summary = format_param_summary_flat(params, initial_params)

        final_stats = [
            ["Initial Significance", f"{initial_significance:.4f}"],
            ["Final Significance", f"{final_significance:.4f}"],
            ["Improvement (%)", f"{((final_significance / initial_significance - 1) * 100):.2f}%"],
        ]

        logger.info("Final Optimized Parameters:\n" +
            tabulate(param_summary, headers=["Parameter", "Value"], tablefmt="fancy_grid", colalign=("left", "right")))

        logger.info("Significance Summary:\n" +
            tabulate(final_stats, tablefmt="fancy_grid", colalign=("left", "right")))

        return params, final_significance