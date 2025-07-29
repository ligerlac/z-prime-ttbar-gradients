from __future__ import annotations

# =============================================================================
# Standard Library Imports
# =============================================================================
import glob
import hashlib
import logging
import os
import pickle
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Literal, List, Tuple, Optional

# =============================================================================
# Third-Party Imports
# =============================================================================
import cloudpickle
import numpy as np
import jax
import jax.numpy as jnp
from jaxopt import OptaxSolver
import optax
import awkward as ak
import uproot
import vector
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

# =============================================================================
# Project Imports
# =============================================================================
from analysis.base import Analysis
from utils.cuts import lumi_mask
from utils.mva import JAXNetwork, TFNetwork
from utils.preproc import pre_process_dak, pre_process_uproot
from utils.jax_stats import (
    compute_discovery_pvalue,
    build_channel_data_scalar
)
from utils.plot import (create_cms_histogram,
                        plot_parameters_over_iterations,
                        plot_pvalue_vs_parameters)

# =============================================================================
# Backend & Logging Setup
# =============================================================================
ak.jax.register_and_check()
vector.register_awkward()

logger = logging.getLogger("DiffAnalysis")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s: %(name)s - %(lineno)d - %(funcName)20s()] %(message)s"
)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")

GREEN = "\033[92m"
RESET = "\033[0m"


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def merge_histograms(
    existing_histograms: dict[str, dict[str, dict[str, jnp.ndarray]]],
    new_histograms: dict[str, dict[str, dict[str, jnp.ndarray]]],
) -> dict[str, dict[str, dict[str, jnp.ndarray]]]:
    """
    Recursively merge new histograms into an existing nested histogram structure.

    Each histogram is assumed to follow the structure:
    histograms[variation][region][observable] = jnp.ndarray

    Parameters
    ----------
    existing_histograms : dict
        Existing nested histogram structure to be updated.
    new_histograms : dict
        New histogram data to merge in.

    Returns
    -------
    dict
        The updated nested histogram structure with new data merged in.
    """
    for variation, region_data in new_histograms.items():
        for region, observable_data in region_data.items():
            for observable, array in observable_data.items():
                # Ensure region exists under the current variation
                existing_histograms[variation].setdefault(region, {})
                # Merge or initialise the observable histogram
                if observable in existing_histograms[variation][region]:
                    existing_histograms[variation][region][observable] += array
                else:
                    existing_histograms[variation][region][observable] = array
    return existing_histograms


def recursive_to_backend(data_structure: Any, backend: str = "jax") -> Any:
    """
    Recursively convert all Awkward Arrays in a data structure to the specified backend.

    Parameters
    ----------
    data_structure : Any
        Input data structure possibly containing Awkward Arrays.
    backend : str
        Target backend to convert arrays to (e.g. 'jax', 'cpu').

    Returns
    -------
    Any
        Data structure with Awkward Arrays converted to the desired backend.
    """
    if isinstance(data_structure, ak.Array):
        # Convert only if not already on the target backend
        return ak.to_backend(data_structure, backend) if ak.backend(data_structure) != backend else data_structure
    elif isinstance(data_structure, Mapping):
        # Recurse into dictionary values
        return {key: recursive_to_backend(value, backend) for key, value in data_structure.items()}
    elif isinstance(data_structure, Sequence) and not isinstance(data_structure, (str, bytes)):
        # Recurse into list or tuple elements
        return [recursive_to_backend(value, backend) for value in data_structure]
    else:
        # Leave unchanged if not an Awkward structure
        return data_structure


def infer_processes_and_systematics(
    fileset: dict[str, dict[str, Any]],
    systematics_config: list[dict[str, Any]],
    corrections_config: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """
    Extract all unique process and systematic names from the config and fileset.

    Parameters
    ----------
    fileset : dict
        Dataset structure with 'metadata' dictionaries including process names.
    systematics_config : list
        Configuration entries for systematic variations.
    corrections_config : list
        Configuration entries for object-level corrections.

    Returns
    -------
    tuple[list[str], list[str]]
        Sorted list of process names and systematic variation base names.
    """
    # Pull out all process names from the fileset metadata
    process_names = {
        metadata.get("process")
        for dataset in fileset.values()
        if (metadata := dataset.get("metadata")) and metadata.get("process")
    }

    # Extract systematic names from both systematics and corrections configs
    systematic_names = {syst["name"] for syst in systematics_config + corrections_config}

    return sorted(process_names), sorted(systematic_names)


def nested_defaultdict_to_dict(nested_structure: Any) -> dict:
    """
    Recursively convert any nested defaultdicts into standard Python dictionaries.

    Parameters
    ----------
    nested_structure : Any
        A nested structure possibly containing defaultdicts.

    Returns
    -------
    dict
        Fully converted structure using built-in dict.
    """
    if isinstance(nested_structure, defaultdict):
        return {key: nested_defaultdict_to_dict(value) for key, value in nested_structure.items()}
    elif isinstance(nested_structure, dict):
        return {key: nested_defaultdict_to_dict(value) for key, value in nested_structure.items()}
    return nested_structure

# -----------------------------------------------------------------------------
# Optimisation helper functions
# -----------------------------------------------------------------------------

def make_apply_param_updates(parameter_update_rules: dict) -> callable:
    """
    Build a function that updates parameters using user-defined rules.

    Parameters
    ----------
    parameter_update_rules : dict
        Mapping from parameter name to update function.
        Each function should have the form: lambda old_value, delta -> new_value

    Returns
    -------
    callable
        Function that applies update rules to a given pair of parameter dictionaries.
    """
    def apply_rules(old_params: dict, tentative_params: dict) -> dict:
        updated_auxiliary = {}
        aux_old = old_params["aux"]
        aux_new_candidate = tentative_params["aux"]

        # Apply the user-defined update rule for each parameter
        for param_name, candidate_value in aux_new_candidate.items():
            if param_name in parameter_update_rules:
                delta = candidate_value - aux_old[param_name]
                updated_auxiliary[param_name] = parameter_update_rules[param_name](aux_old[param_name], delta)
            else:
                updated_auxiliary[param_name] = candidate_value

        return {"aux": updated_auxiliary, "fit": tentative_params["fit"]}

    return apply_rules


def make_clamp_transform(clamp_function: callable) -> optax.GradientTransformation:
    """
    Create an Optax transformation that clamps parameters after each update step.

    Parameters
    ----------
    clamp_function : callable
        A function that takes (old_params, new_params) and returns clamped new_params.

    Returns
    -------
    optax.GradientTransformation
        An Optax transformation object with clamping logic.
    """
    def init_fn(initial_params: dict):
        return None

    def update_fn(
        parameter_updates: dict,
        state: None,
        current_params: Optional[dict] = None
    ) -> tuple[dict, None]:
        # Apply standard update
        updated_params = optax.apply_updates(current_params, parameter_updates)
        # Clamp values to allowed region
        clamped_params = clamp_function(current_params, updated_params)
        # Compute difference between clamped and original values
        effective_updates = jax.tree_util.tree_map(lambda new, old: new - old, clamped_params, current_params)
        return effective_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def make_lr_and_clamp_transform(
    auxiliary_lr_map: dict,
    default_auxiliary_lr: float,
    default_fit_lr: float,
    neural_net_lr_map: dict,
    clamp_function: callable,
    frozen_parameter_keys: Optional[set[str]] = None,
) -> callable:
    """
    Create an optimiser builder combining learning-rate scheduling and clamping.

    Parameters
    ----------
    auxiliary_lr_map : dict
        Custom learning rates for specific auxiliary parameters.
    default_auxiliary_lr : float
        Default learning rate for auxiliary parameters not in lr_map.
    default_fit_lr : float
        Learning rate for 'fit' parameters.
    neural_net_lr_map : dict
        Learning rates for neural network parameters.
    clamp_function : callable
        Projection function that clamps parameters within valid bounds.
    frozen_parameter_keys : set[str], optional
        Set of parameter names to freeze (prevent updates).

    Returns
    -------
    callable
        A builder function that returns (optax_transform, label_mapping).
    """
    frozen_parameter_keys = frozen_parameter_keys or set()

    # Build learning-rate schedule map for different parameter groups
    sub_transforms = {
        **{f"aux__{key}": optax.adam(lr) for key, lr in auxiliary_lr_map.items()},
        **{f"NN__{key}": optax.adam(lr) for key, lr in neural_net_lr_map.items()},
        "aux__default": optax.adam(default_auxiliary_lr),
        "NN__default": optax.adam(default_auxiliary_lr),
        "fit__default": optax.adam(default_fit_lr),
        "no_update": optax.set_to_zero(),
    }

    def make_label_pytree(parameter_tree: dict) -> dict:
        """Create a label tree assigning each parameter to a named sub-transform."""
        label_tree = {"aux": {}, "fit": {}}

        for param_name in parameter_tree["aux"]:
            if param_name in frozen_parameter_keys:
                label_tree["aux"][param_name] = "no_update"
            elif "__NN" in param_name:
                # Use specific NN LR if matched, otherwise fall back to default
                label_tree["aux"][param_name] = next(
                    (f"NN__{name}" for name in neural_net_lr_map if name in param_name),
                    "NN__default"
                )
            elif param_name in auxiliary_lr_map:
                label_tree["aux"][param_name] = f"aux__{param_name}"
            else:
                label_tree["aux"][param_name] = "aux__default"

        for param_name in parameter_tree["fit"]:
            label_tree["fit"][param_name] = (
                "no_update" if param_name in frozen_parameter_keys else "fit__default"
            )

        return label_tree

    def optimiser_builder(initial_params: dict) -> tuple[optax.GradientTransformation, dict]:
        # Map each parameter to its optimiser label
        label_mapping = make_label_pytree(initial_params)
        # Chain the learning rate transform with clamping logic
        optimiser = optax.chain(
            optax.multi_transform(sub_transforms, label_mapping),
            make_clamp_transform(clamp_function)
        )
        return optimiser, label_mapping

    return optimiser_builder


# -----------------------------------------------------------------------------
# DifferentiableAnalysis Class Definition
# -----------------------------------------------------------------------------
class DifferentiableAnalysis(Analysis):
    """
    Differentiable statistical analysis implementation.

    This class extends the base `Analysis` class to support differentiable workflows,
    including:
    - Managing histograms with JAX-compatible data structures.
    - Preparing directory structure for outputs, caches, and plots.
    - Applying selections per analysis channel for both analysis and MVA data branches.
    - Training MVA models using JAX or TensorFlow frameworks.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialise the DifferentiableAnalysis with configuration.

        Parameters
        ----------
        config : dict
            Analysis configuration dictionary.
        """
        super().__init__(config)

        # Histogram storage:
        # histograms[variation][region][observable] = jnp.ndarray
        # Used for storing the final outputs from the differentiable histogramming step.
        self.histograms: dict[str, dict[str, dict[str, jnp.ndarray]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )
        self._prepare_dirs()


    def _set_histograms(
        self, histograms: dict[str, dict[str, dict[str, jnp.ndarray]]]
    ) -> None:
        """
        Set final histograms after processing.

        Parameters
        ----------
        histograms : dict
            Histogram dictionary to store.
        """
        # Replace the current histogram store with new results
        self.histograms = histograms


    def _prepare_dirs(self):
        """
        Create necessary output directories for analysis results.

        This includes:
        - General output directory structure
        - Cache directory for storing gradients
        - Optional preprocessed input directory
        - Directories for storing MVA models, optimisation plots, and fit plots
        """
        # Ensure the output/ directory and common structure are prepared
        self.dirs = super()._prepare_dirs()

        # Create cache directory for gradient checkpoints and intermediate results
        cache = Path(self.config.general.cache_dir or "/tmp/gradients_analysis/")
        cache.mkdir(parents=True, exist_ok=True)

        # Optional: directory to store preprocessed inputs for later reuse
        preproc = self.config.general.get("preprocessed_dir")
        if preproc:
            Path(preproc).mkdir(parents=True, exist_ok=True)

        # Directory for trained MVA models
        mva = self.dirs["output"] / "mva_models"
        mva.mkdir(parents=True, exist_ok=True)

        # Output plots from analysis cut optimisation step
        optimisation_plots = self.dirs["output"] / "plots" / "optimisation"
        optimisation_plots.mkdir(parents=True, exist_ok=True)

        # Output plots from fit or profiling stage
        fit_plots = self.dirs["output"] / "plots" / "fit"
        fit_plots.mkdir(parents=True, exist_ok=True)

        # Register the created paths in the analysis directory registry
        self.dirs.update({
            "cache":       cache,
            "preproc":     Path(preproc) if preproc else None,
            "mva_models":  mva,
            "optimisation_plots": optimisation_plots,
            "fit_plots": fit_plots,
        })



    # -------------------------------------------------------------------------
    # Data categorisation (channels)
    # -------------------------------------------------------------------------
    def _get_channel_data(
        self,
        object_copies_analysis: dict[str, ak.Array],
        events_analysis: ak.Array,
        object_copies_mva: dict[str, ak.Array],
        events_mva: ak.Array,
        process: str,
        variation: str,
    ) -> dict[str, dict[str, Any]]:
        """
        Apply per-channel selection to the corrected object and event collections,
        and return selected data for both physics analysis and MVA pre-training.

        This function performs selection using the analysis branch and returns:
        - selected objects and events for histogramming (from analysis branch)
        - selected objects and events for MVA pre-training (from MVA branch)
        - total number of selected events from each branch
        - unmasked inputs in a '__presel' entry

        See docstring inside `apply_selection` for further details.
        """
        if process == "data" and variation != "nominal":
            logger.debug(f"Skipping {variation} for data")
            return {}

        def apply_selection(
            channel,
            object_copies: dict[str, ak.Array],
            events: ak.Array,
            label: str,
        ) -> tuple[dict[str, ak.Array], ak.Array, ak.Array]:
            """
            Apply selection function to the input objects and events.

            This runs the channel-specific selection function (typically returning a
            PackedSelection object) and constructs the event mask to use.

            For data samples, applies the certified luminosity mask afterwards.

            Parameters
            ----------
            channel : object
                Analysis channel containing selection information.
            object_copies : dict[str, ak.Array]
                Corrected object collections.
            events : ak.Array
                Corresponding events.
            label : str
                Context label for logging.

            Returns
            -------
            tuple
                (Selected objects, events, and mask array)
            """
            # Ensure data is not backed by JAX (selection not traced)
            events = recursive_to_backend(events, "cpu")
            object_copies = recursive_to_backend(object_copies, "cpu")

            # Check for presence of selection logic
            if not channel.selection.function:
                logger.warning(f"[{label}] No selection function for channel {channel.name}")
                return object_copies, events, ak.ones_like(events, dtype=bool)

            # Extract arguments used by selection function
            sel_args = self._get_function_arguments(channel.selection.use, object_copies)
            packed = channel.selection.function(*sel_args)

            if not isinstance(packed, PackedSelection):
                raise ValueError("Selection function must return PackedSelection")

            # Generate mask from PackedSelection object
            mask = ak.Array(packed.all(packed.names[-1]))
            mask = recursive_to_backend(mask, "cpu")

            # Apply certified luminosity filter for data
            if process == "data":
                good_runs = lumi_mask(
                    self.config.general.lumifile,
                    object_copies["run"],
                    object_copies["luminosityBlock"],
                    jax=True,
                )
                mask = mask & ak.to_backend(good_runs, "cpu")
                logger.debug(f"[{label}] Applied luminosity mask for data")

            return object_copies, events, mask

        # Initialise dictionary for channel-wise outputs, and store unmasked input
        per_channel = defaultdict(dict)
        per_channel["__presel"] = {
            "objects": object_copies_analysis,
            "events": events_analysis,
            "mva_objects": object_copies_mva,
            "mva_events": events_mva,
        }

        # Iterate over all channels defined in configuration
        for channel in self.channels:
            if not channel.use_in_diff:
                logger.info(f"Skipping channel {channel.name} in diff analysis")
                continue

            chname = channel["name"]

            # If a subset of channels is requested, skip the rest
            if (req := self.config.general.channels) and chname not in req:
                logger.debug(f"Skipping channel {chname} (not in requested channels)")
                continue

            logger.info(f"Applying selection for {chname} in {process}")

            # Run selection and get event-level mask
            _, events_analysis_cpu, mask = apply_selection(
                channel, object_copies_analysis, events_analysis, label="analysis"
            )

            # Count number of events passing selection
            n_events = ak.sum(mask)
            if n_events == 0:
                logger.warning(f"No events left in {chname} for {process} after selection")
                continue

            logger.info(f"Events in {chname}: before={len(mask)}, after={n_events}")

            # Apply mask to objects and events for both branches
            obj_analysis = {k: v[mask] for k, v in object_copies_analysis.items()}
            evt_analysis = events_analysis_cpu[mask]
            obj_mva = {k: v[mask] for k, v in object_copies_mva.items()}
            evt_mva = events_mva[mask]

            # Save results to output structure
            per_channel[chname] = {
                "objects": obj_analysis,
                "events": evt_analysis,
                "nevents": n_events,
                "mva_objects": obj_mva,
                "mva_events": evt_mva,
                "mva_nevents": len(evt_mva),
            }

        return per_channel


    # -------------------------------------------------------------------------
    # Running training of MVA models
    # -------------------------------------------------------------------------
    def _run_mva_training(
        self,
        events_per_process: dict[str, list[Tuple[dict, int]]]
    ) -> dict[str, Any]:
        """
        Train each MVAConfig using the pre-collected object copies and event counts.

        Parameters
        ----------
        events_per_process : dict
            Mapping from process name to list of (input dictionary, event count).

        Returns
        -------
        dict[str, Any]
            Mapping of MVAConfig name to trained model or parameters.
        """
        trained = {}
        nets = {}

        for mva_cfg in self.config.mva:
            # JAX-based model
            if mva_cfg.framework == "jax":
                net = JAXNetwork(mva_cfg)
                X_train, y_train, X_val, y_val, class_weights = net.prepare_inputs(events_per_process)
                Xtr, Xvl = jnp.array(X_train), jnp.array(X_val)
                ytr, yvl = jnp.array(y_train), jnp.array(y_val)
                net.init_network()
                params = net.train(Xtr, ytr, Xvl, yvl)
                trained[mva_cfg.name] = params
                nets[mva_cfg.name] = net

            # TensorFlow-based model
            else:
                net = TFNetwork(mva_cfg)
                X_train, y_train, X_val, y_val, class_weights = net.prepare_inputs(events_per_process)
                net.init_network()
                sw_train = None

                # Optional: apply class-balancing weights to training
                if class_weights and mva_cfg.balance_strategy == "class_weight":
                    sw_train = np.vectorize(class_weights.get)(y_train)

                fit_kwargs = {"sample_weight": sw_train} if sw_train is not None else {}
                model = net.train(X_train, y_train, X_val, y_val, **fit_kwargs)
                trained[mva_cfg.name] = model
                nets[mva_cfg.name] = net

            logger.info(f"Finished training MVA '{mva_cfg.name}'")

        return trained, nets


    # -------------------------------------------------------------------------
    # Preparing data for JAX (untraced)
    # -------------------------------------------------------------------------
    def _prepare_data_for_tracing(
        self,
        events: ak.Array,
        process: str,
    ) -> dict[str, dict[str, dict[str, jnp.ndarray]]]:
        """
        Preprocess events without JAX tracing for systematic variations.

        This method performs baseline selection and object correction, then
        returns all data needed to fill histograms per systematic variation.

        Parameters
        ----------
        events : ak.Array
            Raw NanoAOD events.
        process : str
            Sample label (e.g. "ttbar", "data").

        Returns
        -------
        dict
            Dictionary mapping variation names to channel-wise data structures.
        """
        processed_data = {}
        logger.info(f"Starting untraced processing for {process}")

        def prepare_full_obj_copies(events, mask_set: str, label: str):
            """
            Perform object copying, masking, baseline selection, correction,
            and ghost observable computation.

            Parameters
            ----------
            events : ak.Array
                Raw event data.
            mask_set : str
                Name of mask set to apply.
            label : str
                Label for logging.

            Returns
            -------
            tuple
                (masked object copies, corrected objects, selection mask)
            """
            # Copy objects and apply good-object masks
            obj_copies = self.get_object_copies(events)
            obj_copies = self.apply_object_masks(obj_copies, mask_set=mask_set)
            logger.debug(f"[{label}] Created object copies and applied '{mask_set}' masks")

            # Move all arrays to CPU for baseline selection
            events_cpu = recursive_to_backend(events, "cpu")
            obj_copies = recursive_to_backend(obj_copies, "cpu")

            # Apply baseline selection mask
            baseline_args = self._get_function_arguments(
                self.config.baseline_selection["use"], obj_copies
            )
            packed = self.config.baseline_selection["function"](*baseline_args)
            mask = ak.Array(packed.all(packed.names[-1]))
            mask = recursive_to_backend(mask, "cpu")

            # Apply selection mask to object copies
            obj_copies = {k: v[mask] for k, v in obj_copies.items()}
            logger.info(f"[{label}] Baseline selection: before={len(mask)}, after={ak.sum(mask)} events")

            # Compute ghost observables before corrections
            obj_copies = self.compute_ghost_observables(obj_copies)
            logger.debug(f"[{label}] Computed ghost observables")

            # Apply object-level corrections (nominal direction only)
            obj_copies_corrected = self.apply_object_corrections(
                obj_copies, self.corrections, direction="nominal"
            )
            obj_copies_corrected = recursive_to_backend(obj_copies_corrected, "cpu")
            logger.debug(f"[{label}] Applied object corrections")

            return obj_copies, obj_copies_corrected, mask

        # Choose mask set for MVA branch ("mva" or fallback to "analysis")
        mva_mask_set = "mva" if self.config.good_object_masks.get("mva", []) else "analysis"

        # Prepare corrected object copies and masks for both branches
        obj_copies_analysis, obj_copies_corrected_analysis, mask_analysis = prepare_full_obj_copies(
            events, mask_set="analysis", label="analysis"
        )
        obj_copies_mva, obj_copies_corrected_mva, mask_mva = prepare_full_obj_copies(
            events, mask_set=mva_mask_set, label="mva"
        )

        # Extract nominal per-channel data using corrected objects
        channels_data = self._get_channel_data(
            obj_copies_corrected_analysis,
            events[mask_analysis],
            obj_copies_corrected_mva,
            events[mask_mva],
            process,
            "nominal",
        )
        channels_data["__presel"]["nevents"] = ak.sum(mask_analysis)
        channels_data["__presel"]["mva_nevents"] = ak.sum(mask_mva)
        processed_data["nominal"] = channels_data
        logger.info("Prepared nominal channel data")

        # Loop over systematics if enabled
        if self.config.general.run_systematics:
            logger.info("Processing systematic variations...")
            for syst in self.systematics + self.corrections:
                if syst["name"] == "nominal":
                    continue

                logger.info(f"Processing {syst['name']} systematics")
                for direction in ["up", "down"]:
                    varname = f"{syst['name']}_{direction}"
                    logger.debug(f"Processing variation: {varname}")

                    # Apply systematic correction to object copies
                    obj_copies_corrected_analysis = self.apply_object_corrections(
                        obj_copies_analysis, [syst], direction=direction
                    )
                    obj_copies_corrected_mva = self.apply_object_corrections(
                        obj_copies_mva, [syst], direction=direction
                    )

                    # Get channel data for the variation
                    channels_data = self._get_channel_data(
                        obj_copies_corrected_analysis,
                        events[mask_analysis],
                        obj_copies_corrected_mva,
                        events[mask_mva],
                        process,
                        varname,
                    )
                    channels_data["__presel"]["nevents"] = ak.sum(mask_analysis)
                    channels_data["__presel"]["mva_nevents"] = ak.sum(mask_mva)
                    processed_data[varname] = channels_data
                    logger.info(f"Prepared channel data for {varname}")

        return processed_data


    # -------------------------------------------------------------------------
    # Histogram building
    # -------------------------------------------------------------------------
    def _histogramming(
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
        mva_instances : dict[str, Any]
            Trained MVA instances per MVA name
        process : str
            Sample label (e.g., 'ttbar', 'data')
        variation : str
            Systematic variation label
        xsec_weight : float
            Cross-section normalisation factor
        params : dict
            JAX parameters for soft selections and KDE bandwidth
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

        # Skip non-nominal variations for data
        if process == "data" and variation != "nominal":
            logger.debug(f"Skipping {variation} for data")
            return histograms

        for channel in self.channels:
            # Skip channels not participating in differentiable analysis
            if not channel.use_in_diff:
                logger.info(f"Skipping channel {channel.name} (use_in_diff=False)")
                continue

            chname = channel["name"]

            # Skip if channel is not listed in requested channels
            if (req := self.config.general.channels) and chname not in req:
                logger.debug(f"Skipping channel {chname} (not in requested channels)")
                continue

            logger.debug(f"Processing channel: {chname}")

            # Extract object and event data for this channel
            obj_copies_ch = processed_data[chname]["objects"]
            events_ch = processed_data[chname]["events"]
            nevents = processed_data[chname]["nevents"]

            # Compute MVA features for this channel
            for mva_name, mva_instance in mva_instances.items():
                mva_features = mva_instance._extract_features(
                    obj_copies_ch,
                    mva_instance.mva_cfg.features
                )
                obj_copies_ch[mva_name] = {
                    "features": mva_features,
                    "instance": mva_instance,
                }

            # Move data to JAX backend
            obj_copies_ch = recursive_to_backend(obj_copies_ch, "jax")
            events_ch = recursive_to_backend(events_ch, "jax")
            logger.debug(f"Channel {chname} has {nevents} events")

            # Compute differentiable selection weights using soft cut function
            diff_args = self._get_function_arguments(
                jax_config.soft_selection.use, obj_copies_ch
            )
            diff_weights = jax_config.soft_selection.function(*diff_args, params)
            logger.debug("Computed differentiable weights")

            # Compute cross-section normalised event weights
            if process != "data":
                weights = (events_ch.genWeight * xsec_weight) / abs(events_ch.genWeight)
                logger.debug(f"MC weights: xsec_weight={xsec_weight:.4f}")
            else:
                weights = ak.Array(np.ones(nevents))
                logger.debug("Using unit weights for data")

            # Apply event-level systematic correction if available
            if event_syst and process != "data":
                weights = self.apply_event_weight_correction(
                    weights, event_syst, direction, obj_copies_ch
                )
                logger.debug(f"Applied {event_syst['name']} {direction} correction")

            weights = jnp.asarray(ak.to_jax(weights))
            logger.info(
                f"Events in {chname}: raw={nevents}, weighted={ak.sum(weights):.2f}"
            )

            # Loop over observables and compute KDE-based histograms
            for observable in channel["observables"]:
                obs_name = observable["name"]

                # Skip observables that are not JAX-compatible
                if not observable.works_with_jax:
                    logger.warning(f"Skipping {obs_name} - not JAX-compatible")
                    continue

                logger.info(f"Processing observable: {obs_name}")

                # Evaluate observable function
                obs_args = self._get_function_arguments(observable["use"], obj_copies_ch)
                values = jnp.asarray(ak.to_jax(observable["function"](*obs_args)))
                logger.debug(f"Computed {obs_name} values")

                # Parse binning specification
                binning = observable["binning"]
                if isinstance(binning, str):
                    low, high, nbins = map(float, binning.split(","))
                    binning = jnp.linspace(low, high, int(nbins))
                else:
                    binning = jnp.array(binning)

                # Kernel bandwidth for KDE
                bandwidth = params["kde_bandwidth"]

                # Compute cumulative density function (CDF) for each bin edge
                cdf = jax.scipy.stats.norm.cdf(
                    binning.reshape(-1, 1),
                    loc=values.reshape(1, -1),
                    scale=bandwidth,
                )

                # Multiply CDF by per-event weights and compute bin contributions
                weighted_cdf = cdf * diff_weights.reshape(1, -1) * weights.reshape(1, -1)
                bin_weights = weighted_cdf[1:, :] - weighted_cdf[:-1, :]
                histogram = jnp.sum(bin_weights, axis=1)

                # Store histogram and binning
                histograms[chname][obs_name] = (jnp.asarray(histogram), binning)
                logger.info(f"Filled histogram for {obs_name} in {chname}")

        return histograms


    # -------------------------------------------------------------------------
    # Histogram collection
    # -------------------------------------------------------------------------
    def _collect_histograms(
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
            Per-variation channel data from `._prepare_data_for_tracing`
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

        # Calculate cross-section weight for MC (unit weight for data)
        xsec_weight = (xsec * lumi / n_gen) if process != "data" else 1.0
        logger.debug(
            f"Process: {process}, xsec: {xsec}, n_gen: {n_gen}, "
            f"lumi: {lumi}, weight: {xsec_weight:.6f}"
        )

        # Nominal histogramming
        logger.debug(f"Processing nominal variation for {process}")
        histograms = self._histogramming(
            processed_data["nominal"],
            processed_data["mva_nets"],
            process,
            "nominal",
            xsec_weight,
            params,
        )
        all_histograms["nominal"] = histograms

        # Loop over systematics (if enabled)
        if self.config.general.run_systematics:
            logger.info(f"Processing systematics for {process}")
            for syst in self.systematics + self.corrections:
                if syst["name"] == "nominal":
                    continue

                for direction in ["up", "down"]:
                    varname = f"{syst['name']}_{direction}"
                    logger.info(f"Processing {varname} for {process}")

                    histograms = self._histogramming(
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
    # pvalue Calculation
    # -------------------------------------------------------------------------
    def _calculate_pvalue(
        self,
        process_histograms: dict,
        params: dict,
        recreate_fit_params: bool = False
    ) -> jnp.ndarray:
        """
        Calculate asymptotic p-value using evermore with multi-channel modelling.

        Parameters
        ----------
        process_histograms : dict
            Histograms organised by process, variation, region and observable.
        params : dict
            Fit parameters for p-value calculation.

        Returns
        -------
        jnp.ndarray
            Asymptotic p-value (sqrt(q0)) from profile likelihood ratio.
        """
        logger.info("📊 Calculating p-value from histograms...")

        # Convert histograms to standard Python dictionaries (from nested defaultdicts)
        histograms = nested_defaultdict_to_dict(process_histograms).copy()

        # Use relaxed to compute p-value and maximum likelihood estimators
        pval, mle_pars = compute_discovery_pvalue(histograms, self.channels, params)

        logger.info("✅ p-value calculation complete\n")
        return pval, mle_pars


    # -------------------------------------------------------------------------
    # The analysis workflow being optimised [histograms + statistics]
    # -------------------------------------------------------------------------
    def _run_traced_analysis_chain(
        self,
        params: dict[str, Any],
        processed_data_events: dict,
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        """
        Collect histograms from preprocessed events and compute the final statistical p-value.

        This function iterates over the preprocessed events, builds histograms for each dataset
        and process, merges them, and then computes the overall p-value using a profile
        likelihood fit. The resulting histograms are also stored for later access (e.g. plotting).

        Parameters
        ----------
        params : dict
            Dictionary of model parameters containing:
            - 'aux': Auxiliary parameters (e.g. selection thresholds, nuisance parameters)
            - 'fit': Fit parameters for statistical inference (e.g. normalization factors)
        processed_data_events : dict
            Nested dictionary containing preprocessed events produced by `run_analysis_processing`.
            Structure:
            {
                "<dataset>": {
                    "<file_key>": {
                        "<variation>": (processed_data, metadata),
                        ...
                    },
                    ...
                },
                ...
            }

        Returns
        -------
        tuple
            pvalue : jnp.ndarray
                The computed asymptotic p-value.
            mle_params : dict
                The maximum-likelihood estimated fit parameters obtained from the p-value fit.
        """
        logger.info("📊 Starting histogram collection and p-value calculation...")
        histograms_by_process = defaultdict(dict)

        # -------------------------------------------------------------------------
        # Loop over datasets
        # -------------------------------------------------------------------------
        for dataset_name, dataset_files in processed_data_events.items():
            process_name = dataset_name.split("___")[1]

            logger.info(
                f"🔍 Processing dataset: {dataset_name} "
                f"(process: {process_name}, files: {len(dataset_files)})"
            )

            # Ensure process histogram container exists
            if process_name not in histograms_by_process:
                histograms_by_process[process_name] = defaultdict(lambda: defaultdict(dict))

            # ---------------------------------------------------------------------
            # Loop over files in the dataset
            # ---------------------------------------------------------------------
            for file_key, variations in dataset_files.items():
                for variation_name, (processed_data, metadata) in variations.items():
                    logger.debug(f"  • Collecting histograms for file: {file_key} ({variation_name})")

                    # Build histograms for this file and variation
                    file_histograms = self._collect_histograms(processed_data, metadata, params["aux"])

                    # Merge file histograms into the global container for this process
                    histograms_by_process[process_name] = merge_histograms(
                        histograms_by_process[process_name], dict(file_histograms)
                    )

        # -------------------------------------------------------------------------
        # Compute statistical p-value from histograms
        # -------------------------------------------------------------------------
        logger.info("✅ Histogram collection complete. Starting p-value calculation...")
        pvalue, mle_params = self._calculate_pvalue(histograms_by_process, params["fit"])

        # Store histograms for later plotting or debugging
        self._set_histograms(histograms_by_process)

        return pvalue, mle_params


    # -------------------------------------------------------------------------
    # Run the data processing code
    # -------------------------------------------------------------------------
    def _prepare_data(
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
            Analysis parameters.
        fileset : dict
            Dictionary mapping dataset names to file and metadata.
        read_from_cache : bool
            Read preprocessed events from cache.
        run_and_cache : bool
            Process events and cache results.
        cache_dir : str, optional
            Directory for cached events.

        Returns
        -------
        dict
            Preprocessed events keyed by dataset and file.
        """
        config = self.config
        all_events = defaultdict(lambda: defaultdict(dict))

        # Prepare dictionary to collect MVA training data
        mva_data: dict[str, list[Tuple[dict, int]]] = {}
        for mva_cfg in config.mva or []:
            for cls in mva_cfg.classes:
                if isinstance(cls, str):
                    mva_data.setdefault(cls, [])
                elif isinstance(cls, dict):
                    class_name = next(iter(cls.keys()))
                    mva_data.setdefault(class_name, [])

        # Loop over datasets in the fileset
        for dataset, content in fileset.items():
            metadata = content["metadata"]
            metadata["dataset"] = dataset
            process_name = metadata["process"]

            # Skip datasets not explicitly requested in config
            if (req := config.general.processes) and process_name not in req:
                logger.info(f"Skipping {dataset} (process {process_name} not in requested)")
                continue

            logger.info("========================================")
            logger.info(f"🚀 Processing dataset: {dataset} ({process_name})")

            # Loop over ROOT files associated with the dataset
            for idx, (file_path, tree) in enumerate(content["files"].items()):

                # Honour file limit if set in configuration
                if config.general.max_files != -1 and idx >= config.general.max_files:
                    logger.info(f"Reached max files limit ({config.general.max_files})")
                    break

                # Determine output directory for preprocessed files
                output_dir = (
                    f"output/{dataset}/file__{idx}/"
                    if not config.general.preprocessed_dir
                    else f"{config.general.preprocessed_dir}/{dataset}/file__{idx}/"
                )
                logger.info(f"Preprocessed files directory: {output_dir}")

                # Preprocess ROOT files into skimmed format using uproot or dask
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

                # Discover skimmed files and summarise retained events
                skimmed_files = glob.glob(f"{output_dir}/part*.root")
                skimmed_files = [f"{f}:{tree}" for f in skimmed_files]
                remaining = sum(uproot.open(f).num_entries for f in skimmed_files)
                logger.info(f"📘 Events retained after filtering: {remaining:,}")

                # Loop over skimmed files for further processing and caching
                for skimmed in skimmed_files:
                    logger.info(f"📘 Processing skimmed file: {skimmed}")
                    cache_key = hashlib.md5(skimmed.encode()).hexdigest()
                    cache_file = os.path.join(cache_dir, f"{dataset}__{cache_key}.pkl")

                    # Handle caching: process and cache, read from cache, or skip
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

                    # Run preprocessing pipeline and store processed results
                    processed_data = self._prepare_data_for_tracing(events, process_name)
                    all_events[f"{dataset}___{process_name}"][f"file__{idx}"][skimmed] = (processed_data, metadata)

                    # Collect training data for MVA, if enabled
                    if config.mva and config.general.run_mva_training:
                        for mva_cfg in config.mva:
                            for cls in mva_cfg.classes:
                                if isinstance(cls, str):
                                    class_name = cls
                                    proc_names = [cls]
                                elif isinstance(cls, dict):
                                    class_name, proc_names = next(iter(cls.items()))
                                else:
                                    continue

                                if process_name not in proc_names:
                                    continue

                                nominal_channels = processed_data.get("nominal", {})
                                if nominal_channels:
                                    presel_ch = nominal_channels["__presel"]

                                    logger.debug(
                                        f"Adding {presel_ch['mva_nevents']} events from process '{process_name}' "
                                        f"to MVA training class '{class_name}'."
                                    )

                                    mva_data[class_name].append(
                                        (presel_ch["mva_objects"], presel_ch["mva_nevents"])
                                    )

            logger.info(f"✅ Finished dataset: {dataset}\n")

        # Run MVA training after all datasets are processed
        models = {}
        nets = {}
        # Attach empty dict of MVA nets to each file’s processed data
        for dataset_files in all_events.values():
            for file_dict in dataset_files.values():
                for skim_key, (processed_data, metadata) in file_dict.items():
                    processed_data['mva_nets'] = nets
        if self.config.general.run_mva_training and (mva_cfg := self.config.mva) is not None:
            logger.info("Executing MVA pre-training")
            models, nets = self._run_mva_training(mva_data)

            # Save trained models and attach to processed data
            for model_name in models.keys():
                model = models[model_name]
                net = nets[model_name]
                model_path = self.dirs["mva_models"] / f"{model_name}.pkl"
                net_path = self.dirs["mva_models"] / f"{model_name}_network.pkl"
                with open(model_path, "wb") as f:
                    cloudpickle.dump(model, f)
                with open(net_path, "wb") as f:
                    cloudpickle.dump(net, f)
                logger.info(f"Saved MVA model '{model_name}' to {model_path}")
                logger.info(f"Saved model network '{model_name}' to {net_path}")

            # Attach MVA nets to each file’s processed data
            for dataset_files in all_events.values():
                for file_dict in dataset_files.values():
                    for skim_key, (processed_data, metadata) in file_dict.items():
                        processed_data['mva_nets'] = nets

        return all_events, models



    # -------------------------------------------------------------------------
    # Cut Optimisation via Gradient Ascent
    # -------------------------------------------------------------------------
    def run_analysis_optimisation(
        self, fileset: dict[str, dict[str, Any]]
    ) -> Tuple[dict[str, jnp.ndarray], jnp.ndarray]:
        """
        Perform gradient-based optimisation of analysis selection cuts and MVA parameters.

        This function runs the entire workflow:
        - Reads and preprocesses events
        - Runs the initial (unoptimised) analysis and caches the result
        - Computes gradients and optimises selection and fit parameters
        - Saves optimised MVA parameters and produces pre/post-fit plots

        Parameters
        ----------
        fileset : dict
            Mapping of dataset names to metadata and ROOT file paths

        Returns
        -------
        tuple
            - Dictionary of final optimised KDE-based histograms
            - Final JAX scalar p-value
        """
        # ---------------------------------------------------------------------
        # 1. Configure caching and initialise analysis parameters
        # ---------------------------------------------------------------------
        cache_dir = "/tmp/gradients_analysis/"
        read_from_cache = self.config.general.read_from_cache
        run_and_cache = not read_from_cache

        # Copy soft selection and NN parameters from config
        auxiliary_parameters = self.config.jax.params.copy()

        all_parameters = {
            "aux": auxiliary_parameters,
            "fit": {"mu": 1.0, "scale_ttbar": 1.0},
        }

        # ---------------------------------------------------------------------
        # 2. Preprocess events and extract MVA models (if any)
        # ---------------------------------------------------------------------
        processed_data, mva_models = self._prepare_data(
            all_parameters,
            fileset,
            read_from_cache=read_from_cache,
            run_and_cache=run_and_cache,
            cache_dir=cache_dir,
        )

        # Add MVA model parameters to aux tree (flattened by name)
        for model_name, model in mva_models.items():
            model_parameters = jax.tree.map(jnp.array, model)
            for name, value in model_parameters.items():
                all_parameters["aux"][name] = value

        # Ensure all parameters are JAX arrays
        all_parameters["aux"] = {k: jnp.array(v) for k, v in all_parameters["aux"].items()}
        all_parameters["fit"] = {k: jnp.array(v) for k, v in all_parameters["fit"].items()}
        logger.info(f"Initial parameters: {pformat(all_parameters)}")

        logger.info("✅ Event preprocessing complete\n")

        # ---------------------------------------------------------------------
        # 3. Run initial traced analysis to compute KDE histograms
        # ---------------------------------------------------------------------
        logger.info("=== Running initial p-value computation (traced)... ===")
        initial_pvalue, mle_parameters = self._run_traced_analysis_chain(
            all_parameters, processed_data
        )
        initial_histograms = self.histograms

        # ---------------------------------------------------------------------
        # 4. If not just plotting, begin gradient-based optimisation
        # ---------------------------------------------------------------------
        if not self.config.general.run_plots_only:
            logger.info("=== Beginning cut and NN parameter optimisation ===")

            # Collect relevant processes and systematics
            processes, systematics = infer_processes_and_systematics(
                fileset, self.config.systematics, self.config.corrections
            )
            logger.info(f"Processes: {processes}")
            logger.info(f"Systematics: {systematics}")

            # Compute initial gradients to seed optimiser
            (_, _), gradients = jax.value_and_grad(
                self._run_traced_analysis_chain,
                has_aux=True,
                argnums=0,
            )(all_parameters, processed_data)

            # Define objective for optimiser (p-value to minimise)
            def objective(params):
                return self._run_traced_analysis_chain(params, processed_data)

            # Build parameter update/clamping logic
            clamp_fn = make_apply_param_updates(self.config.jax.param_updates)

            # Configure learning rates
            global_lr = self.config.jax.learning_rate
            manual_lrs = self.config.jax.learning_rates or {}
            nn_config_lr = {}
            frozen_keys = set()

            for mva_cfg in self.config.mva:
                if not mva_cfg.grad_optimisation.optimise:
                    for name in all_parameters["aux"]:
                        if "__NN" in name and mva_cfg.name in name:
                            frozen_keys.add(name)
                else:
                    nn_config_lr[mva_cfg.name] = mva_cfg.grad_optimisation.learning_rate


            # Construct optimiser with clamping
            tx, _ = make_lr_and_clamp_transform(
                manual_lrs,
                default_auxiliary_lr=global_lr,
                default_fit_lr=1e-3,
                neural_net_lr_map=nn_config_lr,
                clamp_function=clamp_fn,
                frozen_parameter_keys=frozen_keys,
            )(all_parameters)

            # Set up optimisation loop
            logger.info(f"== Starting gradient-based optimisation ==")
            initial_params = all_parameters.copy()
            pval_history = []
            aux_history = {k: [] for k in all_parameters["aux"] if "__NN" not in k}
            mle_history = {k: [] for k in mle_parameters}

            def optimise_and_log(n_steps: int = 100):
                parameters = initial_params
                solver = OptaxSolver(
                    fun=objective,
                    opt=tx,
                    jit=False,
                    has_aux=True,
                    value_and_grad=False,
                    maxiter=self.config.jax.max_iterations,
                    tol=0.0,
                )
                state = solver.init_state(parameters)
                logger.info("Starting explicit optimisation loop...")

                for step in range(n_steps):
                    new_parameters, state = solver.update(parameters, state)
                    parameters = new_parameters

                    # Record progress
                    pval = state.value
                    mle = state.aux
                    pval_history.append(float(pval))
                    for name, value in parameters["aux"].items():
                        if "__NN" not in name:
                            aux_history[name].append(float(value))
                    for name, value in mle.items():
                        mle_history[name].append(float(value))

                    if step % 10 == 0:
                        jax.debug.print("\nStep {:3d}: p = {:.4f}", step, pval)
                        logger.info(f"  Aux: {parameters['aux']}")
                        logger.info(f"  MLE: {mle}")

                return state.value, state.aux, parameters

            # Run optimiser and collect result
            final_pval, final_mle_pars, final_params = optimise_and_log(n_steps=self.config.jax.max_iterations)

            logger.info(f"Initial p-value: {initial_pvalue:.4f}")
            logger.info(f"Final   p-value: {final_pval:.4f}")
            logger.info(f"Improvement: {(final_pval - initial_pvalue) * 100 / initial_pvalue:.2f}%")

            # Re-run analysis with final parameters to update histograms
            _ = self._run_traced_analysis_chain(final_params, processed_data)

            # Cache optimisation results for later inspection
            with open(f"{cache_dir}/cached_result.pkl", "wb") as f:
                cloudpickle.dump({
                    "params": final_params,
                    "mle_pars": final_mle_pars,
                    "pvalue": final_pval,
                    "histograms": self.histograms,
                    "pvals_history": pval_history,
                    "aux_history": aux_history,
                    "mle_history": mle_history,
                    "gradients": gradients,
                }, f)

            # Save optimised MVA parameters
            for model_name, model in mva_models.items():
                optimised_nn_params = {
                    p: v for p, v in final_params["aux"].items()
                    if "__NN" in p and model_name in p
                }
                path = self.dirs["mva_models"] / f"{model_name}_optimised.pkl"
                with open(path, "wb") as f:
                    pickle.dump(jax.tree.map(np.array, optimised_nn_params), f)

        # ---------------------------------------------------------------------
        # 5. Reload results and generate summary plots
        # ---------------------------------------------------------------------
        with open(f"{cache_dir}/cached_result.pkl", "rb") as f:
            results = cloudpickle.load(f)

        final_params = results["params"]
        mle_pars = results["mle_pars"]
        final_pval = results["pvalue"]
        pval_history = results["pvals_history"]
        aux_history = results["aux_history"]
        mle_history = results["mle_history"]
        gradients = results["gradients"]
        histograms = results["histograms"]

        # Generate optimisation progress plots
# Updated optimization history plots
        if self.config.jax.explicit_optimization:
            lrs = self.config.jax.learning_rates or {k: self.config.jax.learning_rate for k in final_params["aux"]}
            plot_pvalue_vs_parameters(
                pvalue_history=pval_history,
                auxiliary_history=aux_history,
                mle_history=mle_history,
                gradients=gradients,
                learning_rates=lrs,
                plot_settings=self.config.plotting,
                filename=f"{self.dirs['optimisation_plots']}/pvalue_vs_parameters.pdf"
            )

            plot_parameters_over_iterations(
                pvalue_history=pval_history,
                auxiliary_history=aux_history,
                mle_history=mle_history,
                gradients=gradients,
                learning_rates=lrs,
                plot_settings=self.config.plotting,
                filename=f"{self.dirs['optimisation_plots']}/parameters_vs_iterations.pdf"
            )

        # ---------------------------------------------------------------------
        # 6. Generate pre- and post-fit plots
        # ---------------------------------------------------------------------
        def make_pre_and_post_fit_plots(histograms_dict, label: str, fitted_pars):
            channel_data_list, _ = build_channel_data_scalar(histograms_dict, self.config.channels)
            for channel_data in channel_data_list:
                channel_cfg = next(c for c in self.config.channels if c.name == channel_data.name)
                fit_obs = channel_cfg.fit_observable
                obs_label = next(o["label"] for o in channel_cfg["observables"] if o["name"] == fit_obs)
                fig = create_cms_histogram(
                    bin_edges=channel_data.bin_edges,
                    data=channel_data.observed_counts,
                    templates=channel_data.templates,
                    fitted_params=fitted_pars,
                    plot_settings=self.config.plotting,
                    xlabel=obs_label,
                )
                fig.savefig(f"{self.dirs['fit_plots']}/{label}_{channel_data.name}.pdf", dpi=300)

        # Plot post-fit (final) and post-fit (initial) comparison
        make_pre_and_post_fit_plots(histograms, "postopt_postfit", mle_pars)
        make_pre_and_post_fit_plots(initial_histograms, "preopt_postfit", mle_pars)
        make_pre_and_post_fit_plots(initial_histograms, "preopt_prefit", all_parameters["fit"])

        return histograms, final_pval


