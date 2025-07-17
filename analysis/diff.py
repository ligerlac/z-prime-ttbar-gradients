from __future__ import annotations

import glob
import logging
import os
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import pickle
import cloudpickle
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Literal, List, Tuple, Optional

import awkward as ak
import optax
import jax
import jax.numpy as jnp
from jaxopt import OptaxSolver
import numpy as np
import uproot
import vector
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

from analysis.base import Analysis
from utils.cuts import lumi_mask
from utils.mva import JAXNetwork, TFNetwork
from utils.preproc import pre_process_dak, pre_process_uproot
from utils.jax_stats import (
    calculate_significance_relaxed,
    build_allbkg_channel_data_scalar
)
from utils.plot import plot_cms_histogram, plot_params_per_iter, plot_pval_history

# -----------------------------------------------------------------------------
# Backend & Logging Setup
# -----------------------------------------------------------------------------
# Register JAX backend for Awkward Arrays and vector operations
ak.jax.register_and_check()
vector.register_awkward()

# Configure logging with informative formatting
logger = logging.getLogger("DiffAnalysis")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s: %(name)s - %(lineno)d - %(funcName)20s()] %(message)s"
)
# Suppress noisy JAX warnings
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

# Disable Coffea warnings
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")

# Console color codes
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
    Recursively merge new histograms into existing structure.

    Structure: histograms[variation][region][observable] = jnp.ndarray

    Parameters
    ----------
    existing : dict
        Existing histogram structure to merge into
    new : dict
        New histogram data to merge

    Returns
    -------
    dict
        Merged histogram structure
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
    Recursively move data structures containing Awkward Arrays to specified backend.

    Parameters
    ----------
    data : Any
        Object, list, or dict containing awkward Arrays
    backend : str
        Target backend ('jax', 'cpu', etc.)

    Returns
    -------
    Any
        Data with arrays moved to specified backend
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
    Infer all process names and systematic names from fileset and config.

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
        Sorted list of unique processes and systematic base names
    """
    process_set = set()
    systematics_set = set()

    # Extract processes from fileset metadata
    for dataset_info in fileset.values():
        metadata = dataset_info.get("metadata", {})
        process = metadata.get("process")
        if process is not None:
            process_set.add(process)

    # Extract systematic names from config
    for syst in systematics + corrections:
        systematics_set.add(syst["name"])

    return sorted(process_set), sorted(systematics_set)


def nested_defaultdict_to_dict(d) -> dict:
    """
    Recursively convert nested defaultdict to regular dict.

    Parameters
    ----------
    d : Any
        Potentially nested defaultdict to convert

    Returns
    -------
    dict
        Regular dict with all defaultdicts converted
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
    """Differentiable statistical analysis implementation."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the DifferentiableAnalysis."""
        super().__init__(config)
        # Histogram storage: variation -> region -> observable -> jnp.ndarray
        self.histograms: dict[str, dict[str, dict[str, jnp.ndarray]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict)))

    def set_histograms(
        self, histograms: dict[str, dict[str, dict[str, jnp.ndarray]]]
    ) -> None:
        """Set the final histograms after processing."""
        self.histograms = histograms

    def _prepare_dirs(self):
            # 1) always make sure output/ exists
            super()._prepare_dirs()

            # 2) cache for gradients
            cache = Path(self.config.general.cache_dir or "/tmp/gradients_analysis/")
            cache.mkdir(parents=True, exist_ok=True)

            # 3) preprocessed files (if used)
            preproc = self.config.general.get("preprocessed_dir")
            if preproc:
                Path(preproc).mkdir(parents=True, exist_ok=True)

            # 4) MVA models under output
            mva = self.dirs["output"] / "mva_models"
            mva.mkdir(parents=True, exist_ok=True)

            # store
            self.dirs.update({
                "cache":       cache,
                "preproc":     Path(preproc) if preproc else None,
                "mva_models":  mva,
            })
    # -------------------------------------------------------------------------
    # Significance Calculation
    # -------------------------------------------------------------------------
    def _calculate_significance(
        self,
        process_histograms: dict,
        params: dict,
        recreate_fit_params: bool = False
    ) -> jnp.ndarray:
        """
        Calculate asymptotic significance using evermore with multi-channel modeling.

        Parameters
        ----------
        process_histograms : dict
            Histograms organized by process, variation, region and observable
        params : dict
            Fit parameters for significance calculation

        Returns
        -------
        jnp.ndarray
            Asymptotic significance (sqrt(q0)) from profile likelihood ratio
        """
        logger.info("📊 Calculating significance from histograms...")
        histograms = nested_defaultdict_to_dict(process_histograms).copy()
        significance, mle_pars= calculate_significance_relaxed(histograms, self.channels, params)
        logger.info(f"✅ Significance calculation complete\n")
        return significance, mle_pars

    # -------------------------------------------------------------------------
    # Histogramming Logic
    # -------------------------------------------------------------------------
    def histogramming(
        self,
        processed_data: dict,
        mva_instances: dict[str, Any],
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
        processed_data : dict
            Preprocessed channel data
        process : str
            Sample label (e.g., 'ttbar', 'data')
        variation : str
            Systematic variation label
        xsec_weight : float
            Cross-section normalization factor
        params : dict
            JAX parameters for soft selections
        event_syst : dict, optional
            Event-level systematic information
        direction : str, optional
            Systematic direction ('up', 'down', 'nominal')

        Returns
        -------
        dict[str, jnp.ndarray]
            Histograms for the channel and observables
        """
        jax_config = self.config.jax
        histograms = defaultdict(dict)

        # Skip systematic variations for data
        if process == "data" and variation != "nominal":
            logger.debug(f"Skipping {variation} for data")
            return histograms

        for channel in self.channels:
            # Skip channels not used in differentiable analysis
            if not channel.use_in_diff:
                logger.info(f"Skipping channel {channel.name} (use_in_diff=False)")
                continue

            chname = channel["name"]
            # Skip channels not in requested list
            if (req := self.config.general.channels) and chname not in req:
                logger.debug(f"Skipping channel {chname} (not in requested channels)")
                continue

            logger.debug(f"Processing channel: {chname}")

            # Prepare channel data
            obj_copies_ch = processed_data[chname]["objects"]
            #print("checkpoint 3", obj_copies_ch["mva"])

            # we can comptue the MVA features at this point
            for mva_name, mva_instance in mva_instances.items():
                mva_features = mva_instance._extract_features(obj_copies_ch, mva_instance.mva_cfg.features)
                #print("checkpoint 4", mva_features)
                obj_copies_ch[mva_name] = {"features": mva_features, "instance": mva_instance}

            obj_copies_ch = recursive_to_backend(obj_copies_ch, "jax")
            events_ch = processed_data[chname]["events"]
            events_ch = recursive_to_backend(events_ch, "jax")
            nevents = processed_data[chname]["nevents"]
            logger.debug(f"Channel {chname} has {nevents} events")

            # Compute differentiable weights
            diff_args = self._get_function_arguments(
                jax_config.soft_selection.use, obj_copies_ch
            )
            diff_weights = jax_config.soft_selection.function(
                *diff_args, params
            )
            logger.debug("Computed differentiable weights")

            # Prepare event weights
            if process != "data":
                weights = (events_ch.genWeight * xsec_weight) / abs(events_ch.genWeight)
                logger.debug(f"MC weights: xsec_weight={xsec_weight:.4f}")
            else:
                weights = ak.Array(np.ones(nevents))
                logger.debug("Using unit weights for data")

            # Apply systematic variations
            if event_syst and process != "data":
                weights = self.apply_event_weight_correction(
                    weights, event_syst, direction, obj_copies_ch
                )
                logger.debug(f"Applied {event_syst['name']} {direction} correction")

            weights = jnp.asarray(ak.to_jax(weights))
            logger.info(
                f"Events in {chname}: raw={nevents}, weighted={ak.sum(weights):.2f}"
            )

            # Fill histograms for each observable
            for observable in channel["observables"]:
                obs_name = observable["name"]
                # Skip non-JAX compatible observables
                if not observable.works_with_jax:
                    logger.warning(f"Skipping {obs_name} - not JAX-compatible")
                    continue

                logger.info(f"Processing observable: {obs_name}")

                # Compute observable values
                obs_args = self._get_function_arguments(observable["use"], obj_copies_ch)
                values = jnp.asarray(ak.to_jax(observable["function"](*obs_args)))
                logger.debug(f"Computed {obs_name} values")
                # Binning
                binning = observable["binning"]
                if isinstance(binning, str):
                    low, high, nbins = map(float, binning.split(","))
                    binning = jnp.linspace(low, high, int(nbins))
                else:
                    binning = jnp.array(binning)

                # Prepare binning
                bandwidth = jax_config.params["kde_bandwidth"]
                # Compute KDE-based histogram
                cdf = jax.scipy.stats.norm.cdf(
                    binning.reshape(-1, 1),
                    loc=values.reshape(1, -1),
                    scale=bandwidth,
                )
                weighted_cdf = cdf * diff_weights.reshape(1, -1) * weights.reshape(1, -1)
                bin_weights = weighted_cdf[1:, :] - weighted_cdf[:-1, :]
                histogram = jnp.sum(bin_weights, axis=1)

                histograms[chname][obs_name] = (jnp.asarray(histogram), binning)
                logger.info(f"Filled histogram for {obs_name} in {chname}")

        return histograms

    def get_channel_data(
        self,
        object_copies: dict[str, ak.Array],
        events: ak.Array,
        process: str,
        variation: str,
    ) -> dict[str, jnp.ndarray]:
        """
        Apply per-channel event selection and prepare data for histogramming.

        Parameters
        ----------
        object_copies : dict
            Corrected event-level objects
        events : ak.Array
            Original NanoAOD events
        process : str
            Sample label
        variation : str
            Systematic variation label

        Returns
        -------
        dict[str, jnp.ndarray]
            Channel-wise data with keys:
            - 'objects': selected objects
            - 'events': selected events
            - 'nevents': number of selected events
        """
        # Skip systematic variations for data
        if process == "data" and variation != "nominal":
            logger.debug(f"Skipping {variation} for data")
            return {}

        events = recursive_to_backend(events, "cpu")
        object_copies = recursive_to_backend(object_copies, "cpu")
        per_channel = defaultdict(dict)

        for channel in self.channels:
            # Skip channels not used in differentiable analysis
            if not channel.use_in_diff:
                logger.info(f"Skipping channel {channel.name} in diff analysis")
                continue

            chname = channel["name"]
            # Skip channels not in requested list
            if (req := self.config.general.channels) and chname not in req:
                logger.debug(f"Skipping channel {chname} (not in requested channels)")
                continue

            logger.info(f"Applying selection for {chname} in {process}")

            # Apply channel selection
            mask = 1
            if sel_fn := channel.selection.function:
                sel_args = self._get_function_arguments(channel.selection.use, object_copies)
                packed = sel_fn(*sel_args)
                if not isinstance(packed, PackedSelection):
                    raise ValueError("Selection function must return PackedSelection")
                mask = ak.Array(packed.all(packed.names[-1]))
            else:
                logger.warning(f"No selection function for channel {chname}")

            mask = recursive_to_backend(mask, "cpu")

            # Apply luminosity mask for data
            if process == "data":
                good_runs = lumi_mask(
                    self.config.general.lumifile,
                    object_copies["run"],
                    object_copies["luminosityBlock"],
                    jax=True,
                )
                mask = mask & ak.to_backend(good_runs, "cpu")
                logger.debug("Applied luminosity mask for data")

            # Check if any events survive selection
            n_events_after = ak.sum(mask)
            if n_events_after == 0:
                logger.warning(f"No events left in {chname} for {process} after selection")
                continue

            logger.info(
                f"Events in {chname}: before={len(mask)}, after={n_events_after}"
            )

            # Prepare selected data
            object_copies_ch = {k: v[mask] for k, v in object_copies.items()}
            per_channel[chname] = {
                "objects": object_copies_ch,
                "events": events[mask],
                "nevents": n_events_after,
            }

        return per_channel

    # def _extract_mva_features(
    #         self,
    #         obj_copies: dict[str, ak.Array],
    #         feat_cfgs: List[FeatureConfig],
    #     ) -> np.ndarray:
    #         """
    #         Stack all configured features into an (n_events, n_features) array.
    #         Applies any optional per-feature scaling.
    #         """
    #         arrays: list[np.ndarray] = []
    #         for feat in feat_cfgs:
    #             # grab the raw awkward array(s)
    #             args = self._get_function_arguments(feat.use, obj_copies)
    #             vals = feat.function(*args)
    #             # apply optional scaling
    #             if feat.scale is not None:
    #                 vals = feat.scale(vals)
    #             arrays.append(np.array(vals))

    #         # each arrays[i] is shape (n_events,)  → stack as columns
    #         X = np.stack(arrays, axis=1).astype(float)
    #         return X

    # def _extract_mva_labels(
    #         self,
    #         n_events: int,
    #         process: str,
    #         classes: List[str],
    #     ) -> np.ndarray:
    #         """
    #         Return a 1-D integer array of length `n_events` equal to classes.index(process).
    #         """
    #         try:
    #             lbl = classes.index(process)
    #         except ValueError:
    #             raise RuntimeError(f"Process `{process}` not in MVAConfig.classes={classes}")
    #         return np.full(n_events, lbl, dtype=int)


    def run_mva_training(
            self,
            events_per_process: dict[str, list[Tuple[dict, int]]]
        ) -> dict[str, Any]:
            """
            Train each MVAConfig using the pre‐collected object copies and event counts.
            Returns a dict mapping mva_cfg.name -> trained (model or params).
            """
            trained = {}
            nets = {}
            for mva_cfg in self.config.mva:
                # # 1) Gather X,y per class
                # X_list, y_list = [], []
                # for cls_idx, proc in enumerate(mva_cfg.classes):
                #     entries = events_per_process.get(proc, [])
                #     if not entries:
                #         logger.warning(f"No events found for class '{proc}'.")
                #         continue
                #     for obj_copies, n_events in entries:
                #         X = self._extract_mva_features(obj_copies, mva_cfg.features)
                #         y = self._extract_mva_labels(n_events, proc, mva_cfg.classes)
                #         X_list.append(X)
                #         y_list.append(y)

                # X_all = np.vstack(X_list)
                # y_all = np.concatenate(y_list)

                # # 2) Balance
                # X_bal, y_bal, class_weights = balance_dataset(
                #     X_all, y_all,
                #     strategy=mva_cfg.balance_strategy,
                #     random_state=mva_cfg.random_state,
                # )

                # # 3) Split train/val
                # X_train, X_val, y_train, y_val = split_train_test(
                #     X_bal, y_bal,
                #     test_size=mva_cfg.validation_split,
                #     random_state=mva_cfg.random_state,
                #     shuffle=True,
                #     stratify=y_bal if mva_cfg.balance_strategy!="none" else None,
                # )

                # 4) Fit
                if mva_cfg.framework == "jax":
                    net = JAXNetwork(mva_cfg)
                    X_train, y_train, X_val, y_val, class_weights= net.prepare_inputs(events_per_process)
                    Xtr, Xvl = jnp.array(X_train), jnp.array(X_val)
                    ytr, yvl = jnp.array(y_train), jnp.array(y_val)
                    net.init_network()
                    params = net.train(Xtr, ytr, Xvl, yvl)
                    trained[mva_cfg.name] = params
                    nets[mva_cfg.name] = net

                else:
                    net = TFNetwork(mva_cfg)
                    # Prepare inputs from events_per_process
                    X_train, y_train, X_val, y_val, class_weights = net.prepare_inputs(events_per_process)
                    net.init_network()
                    sw_train = None
                    if class_weights and mva_cfg.balance_strategy=="class_weight":
                        sw_train = np.vectorize(class_weights.get)(y_train)
                    fit_kwargs = {}
                    if sw_train is not None:
                        fit_kwargs["sample_weight"] = sw_train
                    model = net.train(X_train, y_train, X_val, y_val, **fit_kwargs)
                    trained[mva_cfg.name] = model
                    nets[mva_cfg.name] = net

                logger.info(f"Finished training MVA '{mva_cfg.name}'")

            return trained, nets


    def untraced_process(
        self,
        events: ak.Array,
        process: str,
    ) -> dict[str, dict[str, dict[str, jnp.ndarray]]]:
        """
        Preprocess events without JAX tracing for systematic variations.

        Parameters
        ----------
        events : ak.Array
            Raw NanoAOD events
        process : str
            Sample label

        Returns
        -------
        dict
            Per-variation channel data for histogramming
        """
        processed_data = {}
        logger.info(f"Starting untraced processing for {process}")

        # Prepare object copies and apply baseline masks
        obj_copies = self.get_object_copies(events)
        obj_copies = self.apply_object_masks(obj_copies)
        logger.debug("Created object copies and applied masks")

        events = recursive_to_backend(events, "cpu")
        obj_copies = recursive_to_backend(obj_copies, "cpu")

        # Apply baseline selection
        baseline_args = self._get_function_arguments(
            self.config.baseline_selection["use"], obj_copies
        )
        packed = self.config.baseline_selection["function"](*baseline_args)
        mask = ak.Array(packed.all(packed.names[-1]))
        mask = recursive_to_backend(mask, "cpu")
        obj_copies = {k: v[mask] for k, v in obj_copies.items()}
        logger.info(
            f"Baseline selection: before={len(mask)}, after={ak.sum(mask)} events"
        )

        # Compute ghost observables and apply corrections
        obj_copies = self.compute_ghost_observables(obj_copies)
        logger.debug("Computed ghost observables")

        obj_copies_corrected = self.apply_object_corrections(
            obj_copies, self.corrections, direction="nominal"
        )
        obj_copies_corrected = recursive_to_backend(obj_copies_corrected, "cpu")
        logger.debug("Applied object corrections")

        # Get nominal channel data
        channels_data = self.get_channel_data(
            obj_copies_corrected,
            events[mask],
            process,
            "nominal",
        )
        processed_data["nominal"] = channels_data
        logger.info("Prepared nominal channel data")

        # Process systematic variations if enabled
        if self.config.general.run_systematics:
            logger.info("Processing systematic variations...")
            for syst in self.systematics + self.corrections:
                if syst["name"] == "nominal":
                    continue

                logger.info(f"Processing {syst['name']} systematics")

                for direction in ["up", "down"]:
                    varname = f"{syst['name']}_{direction}"
                    logger.debug(f"Processing variation: {varname}")

                    # Reset backend
                    events = recursive_to_backend(events, "cpu")
                    obj_copies = recursive_to_backend(obj_copies, "cpu")

                    # Apply systematic correction
                    obj_copies_corrected = self.apply_object_corrections(
                        obj_copies, [syst], direction=direction
                    )

                    # Get channel data for this variation
                    channels_data = self.get_channel_data(
                        obj_copies_corrected,
                        events[mask],
                        process,
                        varname,
                    )
                    processed_data[varname] = channels_data
                    logger.info(f"Prepared channel data for {varname}")

        return processed_data

    # -------------------------------------------------------------------------
    # Event Processing Entry Point
    # -------------------------------------------------------------------------
    def collect_histograms(
        self,
        processed_data: dict[str, dict[str, ak.Array]],
        metadata: dict[str, Any],
        params: dict[str, Any],
    ) -> dict[str, dict[str, dict[str, jnp.ndarray]]]:
        """
        Run full analysis logic on events from one dataset.

        Parameters
        ----------
        processed_data : dict
            Per-variation channel data from `untraced_process`
        metadata : dict
            Dataset metadata with keys:
            - 'process': process name
            - 'xsec': cross-section
            - 'nevts': number of generated events
            - 'dataset': dataset name
        params : dict
            JAX parameters for histogramming

        Returns
        -------
        dict
            Histograms keyed by variation/channel/observable
        """
        all_histograms = defaultdict(lambda: defaultdict(dict))
        process = metadata["process"]
        xsec = metadata["xsec"]
        n_gen = metadata["nevts"]
        lumi = self.config["general"]["lumi"]

        # Calculate cross-section weight for MC
        xsec_weight = (xsec * lumi / n_gen) if process != "data" else 1.0
        logger.debug(
            f"Process: {process}, xsec: {xsec}, n_gen: {n_gen}, "
            f"lumi: {lumi}, weight: {xsec_weight:.6f}"
        )

        # Process nominal variation
        logger.debug(f"Processing nominal variation for {process}")
        #print("Checkpoint 2", processed_data.keys())
        histograms = self.histogramming(
            processed_data["nominal"],
            processed_data["mva_nets"],
            process,
            "nominal",
            xsec_weight,
            params,
        )
        all_histograms["nominal"] = histograms

        # Process systematics if enabled
        if self.config.general.run_systematics:
            logger.info(f"Processing systematics for {process}")
            for syst in self.systematics + self.corrections:
                if syst["name"] == "nominal":
                    continue

                for direction in ["up", "down"]:
                    varname = f"{syst['name']}_{direction}"
                    logger.info(f"Processing {varname} for {process}")

                    histograms = self.histogramming(
                        processed_data[varname],
                        processed_data["mva_nets"],
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
        Run full analysis on all datasets in fileset with caching support.

        Parameters
        ----------
        params : dict
            Analysis parameters
        fileset : dict
            Dictionary mapping dataset names to file and metadata
        read_from_cache : bool
            Read preprocessed events from cache
        run_and_cache : bool
            Process events and cache results
        cache_dir : str, optional
            Directory for cached events

        Returns
        -------
        dict
            Preprocessed events keyed by dataset and file
        """
        config = self.config
        all_events = defaultdict(lambda: defaultdict(dict))
        mva_data: dict[str, list[Tuple[dict, int]]] = {cls: [] for cfg in (config.mva or []) for cls in cfg.classes}

        for dataset, content in fileset.items():
            metadata = content["metadata"]
            metadata["dataset"] = dataset
            process_name = metadata["process"]

            # Skip processes not in requested list
            if (req := config.general.processes) and process_name not in req:
                logger.info(f"Skipping {dataset} (process {process_name} not in requested)")
                continue

            logger.info("========================================")
            logger.info(f"🚀 Processing dataset: {dataset} ({process_name})")

            for idx, (file_path, tree) in enumerate(content["files"].items()):

                # Respect file limit
                if config.general.max_files != -1 and idx >= config.general.max_files:
                    logger.info(f"Reached max files limit ({config.general.max_files})")
                    break

                # Prepare output directory
                output_dir = (
                    f"output/{dataset}/file__{idx}/"
                    if not config.general.preprocessed_dir
                    else f"{config.general.preprocessed_dir}/{dataset}/file__{idx}/"
                )
                logger.info(f"Preprocessed files directory: {output_dir}")

                if config.general.run_preprocessing:
                    logger.info(f"🔍 Preprocessing input file: {file_path}")
                    logger.info(f"➡️  Writing to: {output_dir}")
                    if config.general.preprocessor == "uproot":
                        pre_process_uproot(
                            file_path,
                            tree,
                            output_dir,
                            config,
                            is_mc=(dataset != "data"),
                        )
                    elif config.general.preprocessor == "dask":
                        pre_process_dak(
                            file_path,
                            tree,
                            output_dir + f"/part{idx}.root",
                            config,
                            is_mc=(dataset != "data"),
                        )

                # Find skimmed files
                skimmed_files = glob.glob(f"{output_dir}/part*.root")
                skimmed_files = [f"{f}:{tree}" for f in skimmed_files]
                remaining = sum(uproot.open(f).num_entries for f in skimmed_files)
                logger.info(f"📘 Events retained after filtering: {remaining:,}")

                for skimmed in skimmed_files:
                    logger.info(f"📘 Processing skimmed file: {skimmed}")
                    cache_key = hashlib.md5(skimmed.encode()).hexdigest()
                    cache_file = os.path.join(cache_dir, f"{dataset}__{cache_key}.pkl")

                    # Cache handling logic
                    if run_and_cache:
                        logger.info(f"Processing and caching: {skimmed}")
                        events = NanoEventsFactory.from_root(
                            skimmed, schemaclass=NanoAODSchema, delayed=False
                        ).events()
                        with open(cache_file, "wb") as f:
                            cloudpickle.dump(events, f)
                        logger.info(f"💾 Cached events to {cache_file}")
                    elif read_from_cache:
                        if os.path.exists(cache_file):
                            with open(cache_file, "rb") as f:
                                events = cloudpickle.load(f)
                            logger.info(f"🔁 Loaded cached events from {cache_file}")
                        else:
                            logger.warning(f"Cache file not found: {cache_file}")
                            events = NanoEventsFactory.from_root(
                                skimmed, schemaclass=NanoAODSchema, delayed=False
                            ).events()
                            with open(cache_file, "wb") as f:
                                cloudpickle.dump(events, f)
                            logger.info(f"💾 Created new cache: {cache_file}")
                    else:
                        events = NanoEventsFactory.from_root(
                            skimmed, schemaclass=NanoAODSchema, delayed=False
                        ).events()

                    # Process events
                    processed_data = self.untraced_process(events, process_name)
                    all_events[f"{dataset}___{process_name}"][f"file__{idx}"][skimmed] = (processed_data, metadata)

                    # get nominal objects for MVA training ───
                    if config.mva and config.general.run_mva_training:
                        # pick the "nominal" channel entries
                        nominal_ch = processed_data.get("nominal", {})
                        # you might combine multiple channels or pick one—
                        # here we just concatenate *all* channels for this dataset
                        for ch_name, ch_dict in nominal_ch.items():
                            obj_copies = ch_dict["objects"]
                            n_events   = int(ch_dict["nevents"])
                            mva_data.setdefault(process_name, []).append((obj_copies, n_events))

            logger.info(f"✅ Finished dataset: {dataset}\n")

        # Train any MVAs that need pre-training
        if self.config.general.run_mva_training and (mva_cfg := self.config.mva) is not None:
            logger.info("Executing MVA pre-training")
            models, nets = self.run_mva_training(mva_data)

        for model_name in models.keys():
            model = models[model_name]
            net = nets[model_name]
            model_path = self.dirs["mva_models"] / f"{model_name}.pkl"
            net_path   = self.dirs["mva_models"] / f"{model_name}_network.pkl"
            with open(model_path, "wb") as f:
                cloudpickle.dump(model, f)
            with open(net_path, "wb") as f:
                cloudpickle.dump(net, f)
            logger.info(f"Saved MVA model '{model_name}' to {model_path}")
            logger.info(f"Saved model network '{model_name}' to {net_path}")

            for dataset_files in all_events.values():                # each dataset
                for file_dict in dataset_files.values():             # each file__idx
                    for skim_key, (processed_data, metadata) in file_dict.items():
                        # add a key 'mva_nets' pointing to your nets dict
                        processed_data['mva_nets'] = nets


        return all_events, models

    def run_histogram_and_significance(
        self,
        params: dict[str, Any],
        processed_data_events: dict,
    ) -> jnp.ndarray:
        """
        Collect histograms from preprocessed events and compute significance.

        Parameters
        ----------
        params : dict
            Parameters containing 'aux' and 'fit' sub-dictionaries
        processed_data_events : dict
            Preprocessed events from run_analysis_processing

        Returns
        -------
        jnp.ndarray
            Final asymptotic significance
        """
        logger.info("📊 Starting histogram collection and significance calculation...")
        process_histograms = defaultdict(dict)

        aux_and_nn_params = {}
        for param_group, param_vals in params.items():
            if 'nn_' in param_group or param_group == "aux":
                # TODO: at schema level enforce different parameter names? or here add prefix?
                # TODO: check if this system works with the parameter updates?
                aux_and_nn_params.update(param_vals)

        # Process each dataset
        for dataset, files in processed_data_events.items():
            process_name = dataset.split("___")[1]
            logger.info(
                f"Processing dataset {dataset} ({process_name}) with {len(files)} files"
            )

            if process_name not in process_histograms:
                process_histograms[process_name] = defaultdict(lambda: defaultdict(dict))

            # Process each file
            for file_key, skim in files.items():
                for processed_data, metadata in skim.values():
                    logger.debug(
                        f"Processing histograms from {file_key} in {dataset}"
                    )
                    # Collect histograms for this file
                    # print("Checkpoint 1", processed_data, aux_and_nn_params.keys())
                    histograms = self.collect_histograms(processed_data, metadata, aux_and_nn_params)
                    # Merge with existing histograms
                    process_histograms[process_name] = merge_histograms(
                        process_histograms[process_name], dict(histograms)
                    )

        # Calculate final significance
        logger.info(f"✅ Histogramming complete")
        significance, mle_pars = self._calculate_significance(process_histograms, params["fit"])
        jax.lax.stop_gradient(self.set_histograms(process_histograms))
        return significance, mle_pars

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
            File dictionary for histogram generation

        Returns
        -------
        tuple
            Optimized parameters and final significance
        """
        # Configure caching
        cache_dir = "/tmp/gradients_analysis/"
        read_from_cache = self.config.general.read_from_cache
        run_and_cache = not read_from_cache

        # Initialize parameters
        aux_params = self.config.jax.params.copy()

        all_params = {
            "aux": aux_params,
            "fit": {"mu": 1.0, "norm_ttbar_semilep": 1.0},
        }

        # Preprocess events
        processed_data_events, mva_models = self.run_analysis_processing(
            all_params,
            fileset,
            read_from_cache=read_from_cache,
            run_and_cache=run_and_cache,
            cache_dir=cache_dir,
        )

        for model_name, model in mva_models.items():
            initial_nn_params = jax.tree.map(jnp.array, model)
            all_params[f"nn_{model_name}"] = initial_nn_params

        for k, v in all_params["aux"].items():
            all_params["aux"][k] = jnp.array(v)
        all_params["fit"] = {k: jnp.array(v) for k, v in all_params["fit"].items()}
        logger.info(f"Initial parameters: {pformat(all_params)}")


        logger.info("✅ Event preprocessing complete\n")

        logger.info(" === Running untraced siginificance calculation to get initial histograms.. ===")
        init_pval, init_mle_pars = self.run_histogram_and_significance(
            all_params, processed_data_events
        )
        init_histograms = self.histograms

        if not self.config.general.run_plots_only:
            logger.info("Starting cut optimization...")

            # Infer processes and systematics
            processes, systematics = infer_processes_and_systematics(
                fileset, self.config.systematics, self.config.corrections
            )
            logger.info(f"Processes: {processes}")
            logger.info(f"Systematics: {systematics}")

            logger.info(" === Running initial siginificance calculation to get gradients.. ===")
            (_, _), gradients = jax.value_and_grad(
                self.run_histogram_and_significance,
                has_aux=True,
                argnums=0,
            )(all_params, processed_data_events)

            # Define objective function (negative significance)
            def objective(params):
                p0, mle_pars = self.run_histogram_and_significance(params, processed_data_events)
                return -p0, mle_pars

            # Create parameter clamping function
            clamp_fn = make_apply_param_updates(self.config.jax.param_updates)

            # Configure learning rates
            if (config_lr := self.config.jax.learning_rates) is not None:
                make_builder = make_lr_and_clamp_transform(config_lr, default_lr=1e-2, fit_lr=1e-3, nn_lr=0.0005, clamp_fn=clamp_fn)
                tx, _ = make_builder(all_params)
            else:
                tx = optax.adam(learning_rate=self.config.jax.learning_rate)

            logger.info(f"== Starting gradient-based optimisation ===")

            initial_params = all_params.copy()
            print(initial_params)
            # Optimization loop
            pvals_history = []
            aux_history = {k: [] for k in all_params["aux"]}
            mle_history = {k: [] for k in init_mle_pars}
            def optimize_and_log(n_steps: int = 100):
                # all_params is the initial pytree: {"aux": {...}, "fit": {...}}
                pars = initial_params
                solver = OptaxSolver(fun=objective, opt=tx,
                                    jit=False, has_aux=True,
                                    value_and_grad=False,
                                    maxiter=self.config.jax.max_iterations,
                                    tol=0.0,)

                state = solver.init_state(pars)
                logger.info("Starting gradient-based optimization...")

                if self.config.jax.explicit_optimization:
                    logger.info("Using explicit optimization loop")
                    for step in range(n_steps):
                        temp_pars, state = solver.update(pars, state)
                        #pars = clamp_fn(pars, temp_pars)
                        pars = temp_pars

                        # Extract current p-value out of sate
                        pval = state.value              # JAX array scalar
                        mle_pars = state.aux            # MLE parameters from aux

                        if step % 10 == 0:
                            jax.debug.print('\n\nStep {:3d}: p-value = {:.4f}', step, pval)
                            logger.info(f"    parameters  = {pars['aux']}")
                            logger.info(f"    MLE parameters    = {mle_pars}")

                        pvals_history.append(float(state.value))
                        for k, v in pars["aux"].items():
                            aux_history[k].append(float(v))
                        # auxiliary returns are the MLE fit parameters
                        for k, v in state.aux.items():
                            mle_history[k].append(float(v))

                else:
                    pars, state = solver.run(pars)
                    mle_pars = state.aux

                # At the end, return final gradients and final p-value
                return -state.value, mle_pars, pars

            # Set up optimizer
            final_pval, mle_pars, pars = optimize_and_log(n_steps=self.config.jax.max_iterations)
            logger.info(f"Initial p-value before optimization: {init_pval:.4f}")
            logger.info(f"Final p-value after optimization: {final_pval:.4f}")
            logger.info(f"Improvement in p-value: {(final_pval-init_pval)*100/init_pval:.4f}%")
            logger.info(f"Initial parameters: {initial_params}")
            logger.info(f"Final parameters: {pars}")
            logger.info(f"Gradients: {gradients}")

            _ = self.run_histogram_and_significance(
                all_params, processed_data_events
            )
            with open(f"{cache_dir}/cached_result.pkl", "wb") as f:
                cloudpickle.dump({
                    "params": pars,
                    "mle_pars": mle_pars,
                    "significance": final_pval,
                    "histograms": self.histograms,
                    "pvals_history": pvals_history,
                    "aux_history": aux_history,
                    "mle_history": mle_history,
                    "gradients": gradients,
                }, f)
            for model_name, model in mva_models.items():
                numpy_params = jax.tree.map(np.array, pars[f"nn_{model_name}"])
                opt_model = self.dirs["mva_models"] / f"{model_name}_optimised.pkl"
                with open(opt_model, 'wb') as f:
                    pickle.dump(numpy_params, f)

        # ——————————————————————————————————————————————————————————————
        # Plotting of results
        # ——————————————————————————————————————————————————————————————
        with open(f"{cache_dir}/cached_result.pkl", "rb") as f:
            cached_result = cloudpickle.load(f)
            pars = cached_result["params"]
            mle_pars = cached_result["mle_pars"]
            final_pval = cached_result["significance"]
            pvals_history = cached_result["pvals_history"]
            aux_history = cached_result["aux_history"]
            mle_history = cached_result["mle_history"]
            gradients = cached_result["gradients"]
            histograms = cached_result["histograms"]

        plot_settings = self.config.plotting
        if self.config.jax.explicit_optimization:
            if (self.config.jax.learning_rates) is  None:
                lrs = {p: self.config.jax.learning_rate for p in pars["aux"]}
            else:
                lrs = self.config.jax.learning_rates

            plot_pval_history(pvals_history, aux_history, mle_history, gradients, lrs, plot_settings=plot_settings)
            plot_params_per_iter(pvals_history, aux_history, mle_history,  gradients, lrs, plot_settings=plot_settings)

        # ————————————————————————————————————————————————————————————————
        # 1) Prepare your inputs
        #    • `histograms` is your nested dict of KDE-based jnp.ndarrays
        #    • `channels` is your list of config objects with `.name`, `.fit_observable`,
        #       `.use_in_diff`, and now `.binning`
        # ————————————————————————————————————————————————————————————————

        # Build the ChannelData objects
        channel_data_list, _ = build_allbkg_channel_data_scalar(
            histograms,
            self.config.channels,
        )

        # Build the ChannelData objects
        init_channel_data_list, _ = build_allbkg_channel_data_scalar(
            init_histograms,
            self.config.channels,
        )

        # ————————————————————————————————————————————————————————————————
        # 2) Loop over channels and make pre-fit plots
        # ————————————————————————————————————————————————————————————————
        for ch_device in channel_data_list:
            ch = jax.device_get(ch_device)
            channel_settings = [cfg_ch for cfg_ch in self.config.channels if cfg_ch.name == ch.name][0]
            fit_obs = channel_settings.fit_observable
            obs_label = [obss["label"] for obss in channel_settings["observables"] if obss["name"] == fit_obs][0]

            fig_prefit = plot_cms_histogram(
                bin_edges    = ch.binning,
                data         = ch.data_counts,
                templates    = ch.processes,
                plot_settings = plot_settings,
                xlabel        = obs_label,
                #title        = f"Pre-Fit: {ch.binning.shape[0]-1} bins in {ch}",
            )
            fig_prefit.savefig(f"prefit_{ch.name}.png", dpi=150)


        # ————————————————————————————————————————————————————————————————
        # 3) Post-fit plots with the MLE parameters
        # ————————————————————————————————————————————————————————————————
        for ch in channel_data_list:
            channel_settings = [cfg_ch for cfg_ch in self.config.channels if cfg_ch.name == ch.name][0]
            fit_obs = channel_settings.fit_observable
            obs_label = [obss["label"] for obss in channel_settings["observables"] if obss["name"] == fit_obs][0]

            fig_postfit = plot_cms_histogram(
                bin_edges     = ch.binning,
                data          = ch.data_counts,
                templates     = ch.processes,
                fitted_pars   = init_mle_pars,
                plot_settings = plot_settings,
                xlabel        = obs_label,
                #title         = f"Post-Fit: {ch}",
            )
            fig_postfit.savefig(f"postfit_{ch.name}.png", dpi=150)

        # ————————————————————————————————————————————————————————————————
        # 4) Post-fit plots with the initial MLE parameters and histograms
        # ————————————————————————————————————————————————————————————————
        for ch in init_channel_data_list:
            channel_settings = [cfg_ch for cfg_ch in self.config.channels if cfg_ch.name == ch.name][0]
            fit_obs = channel_settings.fit_observable
            obs_label = [obss["label"] for obss in channel_settings["observables"] if obss["name"] == fit_obs][0]

            fig_postfit = plot_cms_histogram(
                bin_edges     = ch.binning,
                data          = ch.data_counts,
                templates     = ch.processes,
                fitted_pars   = init_mle_pars,
                plot_settings = plot_settings,
                xlabel        = obs_label,
                #title         = f"Post-Fit: {ch}",
            )
            fig_postfit.savefig(f"init_postfit_{ch.name}.png", dpi=150)

def make_apply_param_updates(param_update_rules: dict) -> callable:
    """
    Create a function to apply parameter update rules during optimization.

    Parameters
    ----------
    param_update_rules : dict
        Mapping of parameter names to update functions:
        {param_name: (lambda old_x, delta: new_x), …}

    Returns
    -------
    callable
        Function that applies update rules to parameters
    """
    def apply_rules(old_params, tentative_new_params):
        new_aux = {}
        aux_old = old_params["aux"]
        aux_new_temp = tentative_new_params["aux"]

        # Apply rules to each parameter
        for key, x_temp in aux_new_temp.items():
            if key in param_update_rules:
                x_old = aux_old[key]
                delta = x_temp - x_old
                new_aux[key] = param_update_rules[key](x_old, delta)
            else:
                new_aux[key] = x_temp

        return {"aux": new_aux, "fit": tentative_new_params["fit"], **{nn: tentative_new_params[nn] for nn in tentative_new_params if nn.startswith("nn_")}}

    return apply_rules


def make_clamp_transform(clamp_fn):
    """An Optax transform that projects parameters after every update."""
    def init_fn(params):
        return None  # no extra state

    def update_fn(updates, state, params=None):
        # 1) apply the raw updates to get tentative new params
        new_params = optax.apply_updates(params, updates)
        # 2) clamp into your allowed region
        new_params = clamp_fn(params, new_params)
        # 3) compute “effective” updates = new − old
        new_updates = jax.tree_util.tree_map(lambda n, o: n - o, new_params, params)
        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def make_lr_and_clamp_transform(
    lr_map: dict, default_lr: float, fit_lr: float, nn_lr: float, clamp_fn: callable
) -> callable:
    """
    Combines your per-parameter LR multi-transform with a clamp-after-step transform.
    Returns a builder(params) -> (tx, labels) just like before.
    """
    # first, your existing sub_transforms and label builder:
    sub_transforms = {
        **{f"aux__{k}": optax.adam(lr) for k, lr in lr_map.items()},
        "aux__default":    optax.adam(default_lr),
        "fit__default":    optax.adam(fit_lr),
        "nn__default":     optax.adam(nn_lr),
    }

    def make_label_pytree(params):
        labels = {"aux": {}, "fit": {}, **{nn: {} for nn in params if nn.startswith("nn_")}}
        for aux_key in params["aux"]:
            labels["aux"][aux_key] = (
                f"aux__{aux_key}" if aux_key in lr_map else "aux__default"
            )

        for key in params.keys():
            if not key.startswith("nn_"): continue
            for nn_key in params[key]:
                labels[key][nn_key] = (
                f"aux__{nn_key}" if nn_key in lr_map else "nn__default"
                )

        for fit_key in params["fit"]:
            labels["fit"][fit_key] = "fit__default"

        return labels

    # clamp‐after‐update transform
    clamp_tx = make_clamp_transform(clamp_fn)

    def builder(params):
        label_pytree = make_label_pytree(params)
        lr_tx = optax.multi_transform(sub_transforms, label_pytree)
        # chain them: first do lr updates, then project
        tx = optax.chain(lr_tx, clamp_tx)
        return tx, label_pytree

    return builder
