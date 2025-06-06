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
from typing import Any, Dict, Literal, List, Tuple, Optional

import awkward as ak
import evermore as evm
import optax
import jax
import jax.numpy as jnp
import numpy as np
import uproot
import vector
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

from analysis.base import Analysis
from utils.cuts import lumi_mask
from utils.evm_stats import (
    calculate_significance_relaxed,
    #_get_valid_channels,
    #_create_parameter_structure,
    #_prepare_channel_data,
)

# -----------------------------------------------------------------------------
# Backend & Logging Setup
# -----------------------------------------------------------------------------
ak.jax.register_and_check()
vector.register_awkward()

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s: %(name)s - %(lineno)d - %(funcName)20s()] %(message)s"
)

logger = logging.getLogger("DiffAnalysis")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")

# Colours to use in printouts
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
    corrections: List[Dict[str, Any]],
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


def nested_defaultdict_to_dict(d):
    """
    Recursively converts a nested defaultdict to a dict at all levels.

    Parameters
    ----------
    d : Any
        The potentially nested defaultdict to convert.

    Returns
    -------
    dict
        A regular dict with all defaultdicts converted.
    """
    if isinstance(d, defaultdict):
        d = {k: nested_defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: nested_defaultdict_to_dict(v) for k, v in d.items()}
    return d


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
    def _calculate_significance(
        self, process_histograms, params, recreate_fit_params: bool = False
    ) -> jnp.ndarray:
        """
        Generalized significance calculation using evermore with multi-channel,
        multi-process, and systematic-aware modeling.

        Returns
        -------
        jnp.ndarray
            Asymptotic significance (sqrt(q0)) from a profile likelihood ratio.
        """
        histograms = nested_defaultdict_to_dict(process_histograms).copy()
        p0 = calculate_significance_relaxed(histograms, self.channels, params)
        return p0

    # -------------------------------------------------------------------------
    # Histogramming Logic
    # -------------------------------------------------------------------------
    def histogramming(
        self,
        proced,
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
        proced : dict
            Preprocessed channel data for events and objects.
        process : str
            Sample label (e.g. 'ttbar', 'data').
        variation : str
            Systematic variation label.
        xsec_weight : float
            Cross-section normalization.
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
        jax_config = self.config.jax
        histograms = defaultdict(dict)

        if process == "data" and variation != "nominal":
            return histograms

        for channel in self.channels:
            if not channel.use_in_diff:
                logger.warning(f"Skipping channel {channel.name}")
                continue

            chname = channel["name"]
            if (req := self.config.general.channels) and chname not in req:
                continue

            obj_copies_ch = proced[chname]["objects"]
            obj_copies_ch = recursive_to_backend(obj_copies_ch, "jax")
            events_ch = proced[chname]["events"]
            events_ch = recursive_to_backend(events_ch, "jax")
            nevents = proced[chname]["nevents"]

            diff_args = self._get_function_arguments(
                jax_config.soft_selection.use, obj_copies_ch
            )
            diff_weights = jax_config.soft_selection.function(
                *diff_args, params
            )

            if process != "data":
                weights = (events_ch.genWeight * xsec_weight) / abs(events_ch.genWeight)
            else:
                weights = ak.Array(np.ones(nevents))

            if event_syst and process != "data":
                weights = self.apply_event_weight_correction(
                    weights, event_syst, direction, obj_copies_ch
                )

            weights = jnp.asarray(ak.to_jax(weights))
            logger.info(f"Events in {chname}: raw={nevents}, weighted={ak.sum(weights)}")

            for observable in channel["observables"]:
                if not observable.works_with_jax:
                    logger.warning(f"Skipping {observable['name']}, not JAX-compatible.")
                    continue

                obs_args = self._get_function_arguments(observable["use"], obj_copies_ch)
                values = jnp.asarray(ak.to_jax(observable["function"](*obs_args)))
                binning = observable["binning"]

                bandwidth = jax_config.params["kde_bandwidth"]
                if isinstance(binning, str):
                    low, high, nbins = map(float, binning.split(","))
                    binning = jnp.linspace(low, high, int(nbins))
                else:
                    binning = jnp.array(binning)

                cdf = jax.scipy.stats.norm.cdf(
                    binning.reshape(-1, 1),
                    loc=values.reshape(1, -1),
                    scale=bandwidth,
                )
                weighted_cdf = cdf * diff_weights.reshape(1, -1) * weights.reshape(1, -1)
                bin_weights = weighted_cdf[1:, :] - weighted_cdf[:-1, :]
                histogram = jnp.sum(bin_weights, axis=1)

                histograms[chname][observable["name"]] = jnp.asarray(histogram)

        return histograms

    def get_channel_data(
        self,
        object_copies: dict[str, ak.Array],
        events: ak.Array,
        process: str,
        variation: str,
    ) -> dict[str, jnp.ndarray]:
        """
        Apply per-channel event selection and return masks/events for histogramming.

        Parameters
        ----------
        object_copies : dict[str, ak.Array]
            Corrected event-level objects.
        events : ak.Array
            Original NanoAOD events.
        process : str
            Sample label.
        variation : str
            Systematic variation label.

        Returns
        -------
        dict[str, jnp.ndarray]
            Channel-wise dictionary with keys 'mask', 'objects', 'events', and 'nevents'.
        """
        histograms = defaultdict(dict)

        if process == "data" and variation != "nominal":
            return histograms

        events = recursive_to_backend(events, "cpu")
        object_copies = recursive_to_backend(object_copies, "cpu")

        per_channel = defaultdict(dict)
        for channel in self.channels:
            if not channel.use_in_diff:
                logger.warning(f"Skipping channel {channel.name} in diff analysis")
                continue

            chname = channel["name"]
            if (req := self.config.general.channels) and chname not in req:
                continue

            logger.info(f"Applying selection for {chname} in {process}")

            mask = 1
            if sel_fn := channel.selection.function:
                sel_args = self._get_function_arguments(channel.selection.use, object_copies)
                packed = sel_fn(*sel_args)
                if not isinstance(packed, PackedSelection):
                    raise ValueError("Expected PackedSelection")
                mask = ak.Array(packed.all(packed.names[-1]))

            mask = recursive_to_backend(mask, "cpu")

            if process == "data":
                good_runs = lumi_mask(
                    self.config.general.lumifile,
                    object_copies["run"],
                    object_copies["luminosityBlock"],
                    jax=True,
                )
                mask = mask & ak.to_backend(good_runs, "cpu")

            if ak.sum(mask) == 0:
                logger.warning(f"No events left in {chname} for {process}.")
                continue

            object_copies_ch = {
                k: v[mask] for k, v in object_copies.items()
            }
            per_channel[chname] = {
                "objects": object_copies_ch,
                "events": events[mask],
                "nevents": ak.sum(mask),
            }

        return per_channel

    def untraced_process(
        self,
        events: ak.Array,
        process: str,
    ) -> dict[str, dict[str, dict[str, jnp.ndarray]]]:
        """
        Preprocess objects and events without JAX tracing: apply baseline masks,
        ghost observables, and object corrections to prepare per-channel data.

        Parameters
        ----------
        events : ak.Array
            Raw NanoAOD events.
        process : str
            Sample label.

        Returns
        -------
        dict
            Dict of per-variation channel data for histogramming.
        """
        proced: dict[str, dict[str, dict[str, jnp.ndarray]]] = {}

        obj_copies = self.get_object_copies(events)
        obj_copies = self.apply_object_masks(obj_copies)

        events = recursive_to_backend(events, "cpu")
        obj_copies = recursive_to_backend(obj_copies, "cpu")

        baseline_args = self._get_function_arguments(
            self.config.baseline_selection["use"], obj_copies
        )
        packed = self.config.baseline_selection["function"](*baseline_args)
        mask = ak.Array(packed.all(packed.names[-1]))
        mask = recursive_to_backend(mask, "cpu")
        obj_copies = {k: v[mask] for k, v in obj_copies.items()}

        obj_copies = self.compute_ghost_observables(obj_copies)
        obj_copies_corrected = self.apply_object_corrections(
            obj_copies, self.corrections, direction="nominal"
        )
        obj_copies_corrected = recursive_to_backend(obj_copies_corrected, "cpu")

        channels_data = self.get_channel_data(
            obj_copies_corrected,
            events[mask],
            process,
            "nominal",
        )
        proced["nominal"] = channels_data

        if self.config.general.run_systematics:
            for syst in self.systematics + self.corrections:
                if syst["name"] == "nominal":
                    continue
                for direction in ["up", "down"]:
                    varname = f"{syst['name']}_{direction}"
                    events = recursive_to_backend(events, "cpu")
                    obj_copies = recursive_to_backend(obj_copies, "cpu")
                    obj_copies = self.apply_object_masks(obj_copies)

                    events = recursive_to_backend(events, "cpu")
                    obj_copies = recursive_to_backend(obj_copies, "cpu")
                    obj_copies_corrected = self.apply_object_corrections(
                        obj_copies, [syst], direction=direction
                    )

                    channels_data = self.get_channel_data(
                        obj_copies_corrected,
                        events[mask],
                        process,
                        varname,
                    )
                    proced[varname] = channels_data

        return proced

    # -------------------------------------------------------------------------
    # Event Processing Entry Point
    # -------------------------------------------------------------------------
    def collect_histograms(
        self,
        proced: dict[str, dict[str, ak.Array]],
        metadata: dict[str, Any],
        params: dict[str, Any],
    ) -> dict[str, dict[str, dict[str, jnp.ndarray]]]:
        """
        Run the full analysis logic on events from one dataset.

        Parameters
        ----------
        proced : dict
            Per-variation channel data from `untraced_process`.
        metadata : dict
            Metadata with keys 'process', 'xsec', 'nevts', and 'dataset'.
        params : dict
            JAX parameter dictionary (aux parameters).

        Returns
        -------
        dict
            Histogram dictionary keyed by variation/channel/observable.
        """
        all_histograms: dict[str, dict[str, dict[str, jnp.ndarray]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        process = metadata["process"]
        xsec = metadata["xsec"]
        n_gen = metadata["nevts"]
        lumi = self.config["general"]["lumi"]
        xsec_weight = (xsec * lumi / n_gen) if process != "data" else 1.0

        histograms = self.histogramming(
            proced["nominal"],
            process,
            "nominal",
            xsec_weight,
            params,
        )
        all_histograms["nominal"] = histograms

        if self.config.general.run_systematics:
            for syst in self.systematics + self.corrections:
                if syst["name"] == "nominal":
                    continue
                for direction in ["up", "down"]:
                    varname = f"{syst['name']}_{direction}"
                    histograms = self.histogramming(
                        proced[varname],
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
    def run_analysis_processing(
        self,
        params: dict[str, Any],
        fileset: dict[str, Any],
        read_from_cache: bool = False,
        run_and_cache: bool = True,
        cache_dir: Optional[str] = "/tmp/gradients_analysis/",
        recreate_fit_params: bool = False,
    ) -> dict[str, dict[str, dict[str, Any]]]:
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
        dict
            Nested dict of preprocessed per-file, per-dataset events ready for histogramming.
        """
        config = self.config
        all_events: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(dict))

        for dataset, content in fileset.items():
            metadata = content["metadata"]
            metadata["dataset"] = dataset
            process_name = metadata["process"]

            if (req := config.general.processes) and process_name not in req:
                continue

            os.makedirs(f"{config.general.output_dir}/{dataset}", exist_ok=True)

            logger.info("========================================")
            logger.info(f"🚀 Processing dataset: {dataset}")

            for idx, (_, tree) in enumerate(content["files"].items()):
                if config.general.max_files != -1 and idx >= config.general.max_files:
                    continue

                output_dir = (
                    f"output/{dataset}/file__{idx}/"
                    if not config.general.preprocessed_dir
                    else f"{config.general.preprocessed_dir}/{dataset}/file__{idx}/"
                )
                skimmed_files = glob.glob(f"{output_dir}/part*.root")
                skimmed_files = [f"{f}:{tree}" for f in skimmed_files]
                remaining = sum(uproot.open(f).num_entries for f in skimmed_files)
                logger.info(f"✅ Events retained after filtering: {remaining:,}")

                for skimmed in skimmed_files:
                    logger.info(f"📘 Processing skimmed file: {skimmed}")
                    if run_and_cache or read_from_cache:
                        os.makedirs(cache_dir, exist_ok=True)
                        cache_key = hashlib.md5(skimmed.encode()).hexdigest()
                        cache_file = os.path.join(cache_dir, f"{dataset}__{cache_key}.pkl")

                    if run_and_cache:
                        events = NanoEventsFactory.from_root(
                            skimmed, schemaclass=NanoAODSchema, delayed=False
                        ).events()
                        with open(cache_file, "wb") as f:
                            cloudpickle.dump(events, f)
                        logger.info(f"💾 Cached events to {cache_file}")
                    else:
                        if read_from_cache:
                            if os.path.exists(cache_file):
                                with open(cache_file, "rb") as f:
                                    events = cloudpickle.load(f)
                                logger.info(f"🔁 Loaded cached events from {cache_file}")
                            else:
                                logger.warning(f"Cache file {cache_file} not found. Reprocessing.")
                                events = NanoEventsFactory.from_root(
                                    skimmed, schemaclass=NanoAODSchema, delayed=False
                                ).events()
                                with open(cache_file, "wb") as f:
                                    cloudpickle.dump(events, f)
                                logger.info(f"💾 Cached events to {cache_file}")
                        else:
                            events = NanoEventsFactory.from_root(
                                skimmed, schemaclass=NanoAODSchema, delayed=False
                            ).events()

                    proced = self.untraced_process(events, process_name)
                    all_events[f"{dataset}___{process_name}"][f"file__{idx}"][skimmed] = (proced, metadata)

            logger.info(f"✅ Finished dataset: {dataset}\n")

        return all_events

    def run_histogram_and_significance(
        self,
        params: dict[str, Any],
        proced_events,
    ) -> jnp.ndarray:
        """
        Load preprocessed events from disk, collect histograms, and compute significance.

        Parameters
        ----------
        params : dict
            Nested dict containing 'aux' and 'fit' parameters.

        Returns
        -------
        jnp.ndarray
            Final asymptotic significance.
        """

        process_histograms: dict[str, dict[str, dict[str, jnp.ndarray]]] = defaultdict(dict)
        for dataset, files in proced_events.items():
            logger.info(f"Processing dataset {dataset} with {len(files)} files")

            process_name = dataset.split("___")[1]
            if process_name not in process_histograms:
                process_histograms[process_name] = defaultdict(lambda: defaultdict(dict))

            for file_key, skim in files.items():
                for proced, metadata in skim.values():
                    logger.info(
                        f"Processing histograms from {file_key} in dataset {dataset}"
                    )
                    histograms = self.collect_histograms(proced, metadata, params["aux"])
                    process_histograms[process_name] = merge_histograms(
                        process_histograms[process_name], dict(histograms)
                    )

        significance = self._calculate_significance(process_histograms, params["fit"])
        logger.info("✅ All datasets processed.")
        return significance

    # -------------------------------------------------------------------------
    # Cut Optimization via Gradient Ascent
    # -------------------------------------------------------------------------
    def optimize_analysis_cuts(
        self, fileset: dict[str, dict[str, Any]]
    ) -> Tuple[dict[str, jnp.ndarray], jnp.ndarray]:
        """
        Optimize analysis cuts using gradient ascent.

        Parameters
        ----------
        fileset : dict
            File dictionary used for histogram generation.

        Returns
        -------
        tuple
            (Optimized parameters, final significance)
        """
        from functools import partial
        import optax
        from jaxopt import OptaxSolver

        logger.info("Running analysis chain with init values for all datasets...")
        logger.info(f"Processes: {list(fileset.keys())}")

        processes, systematics = infer_processes_and_systematics(
            fileset, self.config.systematics, self.config.corrections
        )
        aux_params = self.config.jax.params.copy()
        #evm_params = _create_parameter_structure(processes, systematics)
        all_params = {"aux": aux_params, "fit": {
                                                 "mu": 1.0,
                                                 "norm_ttbar_semilep": 1.0
                                                }
        }

        all_params["aux"] = {
            k: jnp.array(v) for k, v in all_params["aux"].items()
        }
        all_params["fit"] = {
            k: jnp.array(v) for k, v in all_params["fit"].items()
        }

        cache_dir = "/tmp/gradients_analysis/"
        if self.config.general.read_from_cache:
            read_from_cache = True
            run_and_cache = False
        else:
            read_from_cache = False
            run_and_cache = True

        proced_events = self.run_analysis_processing(
            all_params,
            fileset,
            read_from_cache=read_from_cache,
            run_and_cache=run_and_cache,
            cache_dir=cache_dir,
        )

        # pvals = self.run_histogram_and_significance(all_params, proced_events

        #                                             )
        # print(f"Initial p-value: {pvals}")

        # pvals, gradients = jax.value_and_grad(
        #     self.run_histogram_and_significance, argnums=0
        # )(all_params, proced_events)

        initial_params = all_params
        def objective(params):
            return -1*self.run_histogram_and_significance(
                                params, proced_events
                    )

        # 2) Make the “clamp” function from the config:
        clamp_fn = make_apply_param_updates(self.config.jax.param_updates)
        if (config_lr := self.config.jax.learning_rates) is not None:
            make_builder = make_lr_transform(config_lr, default_lr=1e-2, fit_lr=1e-3)
            # 2) Call the builder on your initial `all_params` to get (tx, label_pytree):
            tx, _ = make_builder(all_params)

        else:
            tx = optax.adam(learning_rate=1e-1)

        def optimize_and_log(n_steps: int = 100):
            # all_params is your initial pytree: {"aux": {...}, "fit": {...}}
            pars = initial_params
            opt = optax.adam(learning_rate=0.1)
            # value_and_grad=True makes solver.state contain .value (scalar) and .grad (pytree)
            solver = OptaxSolver(fun=objective, opt=tx, jit=False, has_aux=False, value_and_grad=False)
            state = solver.init_state(pars)

            logger.info("Starting gradient ascent optimization...")
            for step in range(n_steps):
                temp_pars, state = solver.update(pars, state)
                pars = clamp_fn(pars, temp_pars)

                # Extract current p-value and gradient pytree out of state:
                pval = state.value              # JAX array scalar
                #grads = state.grad              # same pytree structure as all_params

                # (Optional) Convert JAX arrays to Python floats/ndarrays for logging:
                #pval_f = float(jax.device_get(pval))
                # Flatten grads into a dictionary of small floats for easy printout:
                #grads_flat, _ = jax.flatten_util.ravel_pytree(grads)
                #grads_arr = jax.device_get(grads_flat)

                if step % 1 == 0:
                    jax.debug.print('Step {:3d}: p-value = {:.4f}', step, pval)
                    logger.info(f"    parameters  = {pars}")
                    #logger.info(f"    gradients  (flat) = {grads_arr}")

            # At the end, return final gradients and final p-value
            return 1.0, state.value
        # Set up optimizer
        gradients, final_pval = optimize_and_log(n_steps=30)

        return gradients, final_pval

def make_apply_param_updates(param_update_rules):
    """
    Given a dict param_update_rules:
      { param_name: (lambda old_x, delta: new_x), … }
    we return a function that “projects” all_params as follows:
        new_params["aux"][name] = param_update_rules[name](
                                 old_params["aux"][name],
                                 new_params["aux"][name] - old_params["aux"][name]
                               )
    and leaves everything else unchanged.
    """
    def apply_rules(old_params, tentative_new_params):
        # old_params and tentative_new_params are assumed to share the same pytree structure:
        #   {
        #     "aux": { key1: float, key2: float, … },
        #     "fit": { … },
        #   }
        #
        # We only rewrite the leaves under "aux" that appear in param_update_rules.
        aux_old = old_params["aux"]
        aux_new_temp = tentative_new_params["aux"]

        # Create a new “aux” dict by applying each rule in turn.
        new_aux = {}
        for key, x_temp in aux_new_temp.items():
            if key in param_update_rules:
                # old value:
                x_old = aux_old[key]
                # “proposed delta” = (x_temp − x_old)
                delta = x_temp - x_old
                # run the user’s clamp function:
                new_aux[key] = param_update_rules[key](x_old, delta)
            else:
                # If no special rule, just accept x_temp
                new_aux[key] = x_temp

        # Everything under “fit” is untouched here. We simply pass it through.
        return {"aux": new_aux, "fit": tentative_new_params["fit"]}

    return apply_rules

# ========================
# 2) Build make_lr_transform(...) correctly
# ========================

def make_lr_transform(lr_map, default_lr=1e-2, fit_lr=1e-3):
    """
    Build an Optax `GradientTransformation` that applies:
      - Adam(lr_map[key]) whenever path = ("aux", key) and key is in lr_map
      - Adam(default_lr) whenever path = ("aux", key) and key not in lr_map
      - Adam(fit_lr)      whenever path = ("fit", <anything>)
    """'''
    # 1) Create each sub‐optimizer and give it a name.
    #    We'll later have these three “buckets”:
    #      - "aux__<key>"    for every aux‐key in lr_map
    #      - "aux__default"  for all other “aux” leaves
    #      - "fit__default"  for all “fit” leaves
    sub_transforms = {}

    # For each aux‐key in lr_map, build an Adam with that lr.
    for key, lr in lr_map.items():
        sub_transforms[f"aux__{key}"] = optax.adam(learning_rate=lr)

    # A fallback Adam(default_lr) for any other “aux” parameter not in lr_map
    sub_transforms["aux__default"] = optax.adam(learning_rate=default_lr)

    # A single transform for all “fit” leaves
    sub_transforms["fit__default"] = optax.adam(learning_rate=fit_lr)

    # 2) Now build a “label tree” whose structure mirrors the structure of `all_params`.
    #    Recall: `all_params = {"aux": {...}, "fit": {...}}`
    def make_label_pytree(params):
        """
        Walk over `params` (a dict with keys "aux" and "fit"), and return a pytree of the
        same shape whose leaves are strings naming which sub‐transform should apply.
        """
        labels = {"aux": {}, "fit": {}}

        # Fill in the “aux” branch:
        for aux_key, aux_val in params["aux"].items():
            if aux_key in lr_map:
                labels["aux"][aux_key] = f"aux__{aux_key}"
            else:
                labels["aux"][aux_key] = "aux__default"

        # Fill in the “fit” branch (all fit‐leaves use "fit__default"):
        for fit_key, fit_val in params["fit"].items():
            labels["fit"][fit_key] = "fit__default"

        return labels

    # 3) Return a function that will create (transform, labels) when given a param‐tree:
    def builder(params):
        """
        Given a parameters pytree `params = {"aux": {...}, "fit": {...}}`,
        returns (tx, label_pytree), where:
          - tx = optax.multi_transform(sub_transforms, label_pytree)
          - label_pytree has the same structure as params but with leaves = strings
        """
        label_pytree = make_label_pytree(params)
        tx = optax.multi_transform(sub_transforms, label_pytree)
        return tx, label_pytree

    return builder