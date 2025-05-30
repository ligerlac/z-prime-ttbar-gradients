#!/usr/bin/env python3

"""
ZprimeAnalysis framework for applying object and event-level systematic corrections
on NanoAOD ROOT files and producing histograms of observables like mtt. Supports both
correctionlib-based and function-based corrections.
"""

import copy
from collections import defaultdict
from functools import reduce
import glob
import gzip
import logging
import operator
import os
import sys
from typing import Any, Callable, Literal, Optional, Union
import warnings

import awkward as ak
import cabinetry
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from correctionlib import CorrectionSet
import hist
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import mplhep
import numpy as np
import uproot
import vector


from utils.configuration import config as ZprimeConfig
from utils.cuts import lumi_mask
from utils.input_files import construct_fileset
from utils.output_files import save_histograms, pkl_histograms, unpkl_histograms
from utils.preproc import pre_process_dak, pre_process_uproot
from utils.schema import Config, load_config_with_restricted_cli, GoodObjectMasksConfig
from utils.stats import get_cabinetry_rebinning_router


# -----------------------------
# Register backends
# -----------------------------
ak.jax.register_and_check()
vector.register_awkward()

# -----------------------------
# Logging Configuration
# -----------------------------

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ZprimeAnalysis")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")


def is_jagged(arraylike) -> bool:
    try:
        return ak.num(arraylike, axis=1) is not None
    except Exception:
        return False

# -----------------------------
# Base class
# -----------------------------
class Analysis:
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize ZprimeAnalysis with configuration for systematics, corrections,
        and channels.

        Parameters
        ----------
        config : dict
            Configuration dictionary with 'systematics', 'corrections', 'channels',
            and 'general'.
        """
        self.config = config
        self.channels = config.channels
        self.systematics = config.systematics
        self.corrections = config.corrections
        self.corrlib_evaluators = self._load_correctionlib()
        self.nD_hists_per_region = self._init_histograms()


    def _load_correctionlib(self) -> dict[str, CorrectionSet]:
        """
        Load correctionlib JSON files into evaluators.

        Returns
        -------
        dict
            Dictionary of correction name to CorrectionSet evaluator.
        """
        evaluators = {}
        for systematic in self.corrections:
            if not systematic.get("use_correctionlib"):
                continue
            name = systematic["name"]
            path = systematic["file"]

            if path.endswith(".json.gz"):
                with gzip.open(path, "rt") as f:
                    evaluators[name] = CorrectionSet.from_string(
                        f.read().strip()
                    )
            elif path.endswith(".json"):
                evaluators[name] = CorrectionSet.from_file(path)
            else:
                raise ValueError(f"Unsupported correctionlib format: {path}")

        return evaluators

    def get_object_copies(self, events: ak.Array) -> dict[str, ak.Array]:
        """
        Extract a dictionary of objects from the NanoEvents array.

        Parameters
        ----------
        events : ak.Array
            Input events.

        Returns
        -------
        dict
            Dictionary of field name to awkward array.
        """
        return {field: events[field] for field in events.fields}

    def get_good_objects(
        self, object_copies: dict[str, ak.Array],
        masks: list[GoodObjectMasksConfig] = []
    ) -> dict[str, ak.Array]:

        good_objects = {}
        for obj_mask in masks:
            mask_args = self._get_function_arguments(
                obj_mask.use, object_copies
            )
            mask = obj_mask.function(*mask_args)
            if not isinstance(mask, ak.Array):
                raise TypeError(
                    f"Expected mask to be an awkward array, got {type(mask)}"
                )

            good_objects[obj_mask.object] = object_copies[obj_mask.object][mask]

        return good_objects

    def apply_correctionlib(
        self,
        name: str,
        key: str,
        direction: Literal["up", "down", "nominal"],
        correction_arguments: list[ak.Array],
        target: Optional[Union[ak.Array, list[ak.Array]]] = None,
        op: Optional[str] = None,
        transform: Optional[Callable[..., Any]] = None,
    ) -> Union[ak.Array, list[ak.Array]]:
        """
        Apply a correction using correctionlib.
        """
        logger.info(
            f"Applying correctionlib correction: name={name}, "
            f"key={key}, direction={direction}"
        )
        if transform is not None:
            correction_arguments = transform(*correction_arguments)

        flat_args, counts_to_unflatten = [], []
        for arg in correction_arguments:
            if is_jagged(arg):
                flat_args.append(ak.flatten(arg))
                counts_to_unflatten.append(ak.num(arg))
            else:
                flat_args.append(arg)

        correction = self.corrlib_evaluators[name][key].evaluate(
            *flat_args, direction
        )

        if counts_to_unflatten:
            correction = ak.unflatten(correction, counts_to_unflatten[0])

        if target is not None and op is not None:
            if isinstance(target, list):
                correction = ak.to_backend(correction, ak.backend(target[0]))
                return [self.apply_op(op, t, correction) for t in target]
            else:
                correction = ak.to_backend(correction, ak.backend(target))
                return self.apply_op(op, target, correction)

        return correction

    def apply_syst_fn(
        self,
        name: str,
        fn: Callable[..., ak.Array],
        args: list[ak.Array],
        affects: Union[ak.Array, list[ak.Array]],
        op: str,
    ) -> Union[ak.Array, list[ak.Array]]:
        """
        Apply function-based systematic variation.
        """
        logger.debug(f"Applying function-based systematic: {name}")
        correction = fn(*args)
        if isinstance(affects, list):
            return [self.apply_op(op, a, correction) for a in affects]
        else:
            return self.apply_op(op, affects, correction)

    def apply_op(self, op: str, lhs: ak.Array, rhs: ak.Array) -> ak.Array:
        """
        Apply a binary operation.
        """
        if op == "add":
            return lhs + rhs
        elif op == "mult":
            return lhs * rhs
        else:
            raise ValueError(f"Unsupported operation: {op}")

    def _get_function_arguments(
        self,
        use: list[tuple[str, Optional[str]]],
        object_copies: dict[str, ak.Array],
    ) -> list[ak.Array]:
        """
        Extract correction arguments from object_copies.
        """
        return [
            object_copies[obj][var] if var is not None else object_copies[obj]
            for obj, var in use
        ]

    def _get_targets(
        self,
        target: Union[tuple[str, str], list[tuple[str, str]]],
        object_copies: dict[str, ak.Array],
    ) -> list[ak.Array]:
        """
        Extract one or more target arrays from object_copies.
        """
        targets = target if isinstance(target, list) else [target]
        return [object_copies[obj][var] for obj, var in targets]

    def _set_targets(
        self,
        target: Union[tuple[str, str], list[tuple[str, str]]],
        object_copies: dict[str, ak.Array],
        new_values: Union[ak.Array, list[ak.Array]],
    ) -> None:
        """
        Set corrected values in object_copies.
        """
        targets = target if isinstance(target, list) else [target]
        for (obj, var), val in zip(targets, new_values):
            object_copies[obj][var] = val

    def apply_object_corrections(
        self,
        object_copies: dict[str, ak.Array],
        corrections: list[dict[str, Any]],
        direction: Literal["up", "down", "nominal"] = "nominal",
    ) -> dict[str, ak.Array]:
        """
        Apply object-level corrections to input object copies.
        """
        for corr in corrections:
            if corr["type"] != "object":
                continue
            args = self._get_function_arguments(corr["use"], object_copies)
            targets = self._get_targets(corr["target"], object_copies)
            op = corr["op"]
            key = corr.get("key")
            transform = corr.get("transform", lambda *x: x)
            dir_map = corr.get("up_and_down_idx", ["up", "down"])
            corr_dir = (
                dir_map[0 if direction == "up" else 1]
                if direction in ["up", "down"]
                else "nominal"
            )

            if corr.get("use_correctionlib", False):
                corrected = self.apply_correctionlib(
                    corr["name"], key, corr_dir, args, targets, op, transform
                )
            else:
                fn = corr.get(f"{direction}_function")
                corrected = (
                    self.apply_syst_fn(corr["name"], fn, args, targets, op)
                    if fn
                    else targets
                )

            self._set_targets(corr["target"], object_copies, corrected)

        return object_copies

    def apply_event_weight_correction(
        self,
        weights: ak.Array,
        systematic: dict[str, Any],
        direction: Literal["up", "down"],
        object_copies: dict[str, ak.Array],
    ) -> ak.Array:
        """
        Apply event-level correction to weights.
        """
        if systematic["type"] != "event":
            return weights

        args = self._get_function_arguments(systematic["use"], object_copies)
        op = systematic["op"]
        key = systematic.get("key")
        transform = systematic.get("transform", lambda *x: x)
        dir_map = systematic.get("up_and_down_idx", ["up", "down"])
        corr_dir = dir_map[0 if direction == "up" else 1]

        if systematic.get("use_correctionlib", False):
            return self.apply_correctionlib(
                systematic["name"], key, corr_dir, args, weights, op, transform
            )
        else:
            fn = systematic.get(f"{direction}_function")
            return (
                self.apply_syst_fn(systematic["name"], fn, args, weights, op)
                if fn
                else weights
            )


    def compute_ghost_observables(
        self, obj_copies: dict[str, ak.Array], jax=False,
    ) -> dict[str, ak.Array]:
        for ghost in self.config.ghost_observables:
            if not ghost.works_with_jax and jax:
                logger.warning(
                    f"Ghost observable {ghost.names} does not work with JAX, skipping."
                )
                continue

            logger.info(f"Computing ghost observables {ghost.names}")
            ghost_args = self._get_function_arguments(ghost["use"], obj_copies)
            ghost_outputs = ghost["function"](*ghost_args)

            if not isinstance(ghost_outputs, (list, tuple)):
                ghost_outputs = [ghost_outputs]

            names = (
                ghost.names if isinstance(ghost.names, list) else [ghost.names]
            )
            colls = (
                ghost.collections
                if isinstance(ghost.collections, list)
                else [ghost.collections] * len(names)
            )
            # update object_copies with ghost outputs
            for out, name, coll in zip(ghost_outputs, names, colls):
                if (
                    isinstance(out, ak.Array)
                    and len(ak.fields(out)) == 1
                    and name in out.fields
                ):
                    out = out[name]
                if coll in obj_copies:
                    try:
                        # add new field to existing awkward array
                        obj_copies[coll][name] = out
                    # happens if we are adding a field to an awkward
                    # array with no fields (scalar fields)
                    except ValueError as e:
                        raise e
                else:
                    # create new awkward array with single field
                    obj_copies[coll] = ak.Array({name: out})
        return obj_copies

# -----------------------------
# ZprimeAnalysis Class Definition
# -----------------------------
class NonDiffAnalysis(Analysis):

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize ZprimeAnalysis with configuration for systematics, corrections,
        and channels.

        Parameters
        ----------
        config : dict
            Configuration dictionary with 'systematics', 'corrections', 'channels',
            and 'general'.
        """
        super().__init__(config)
        self.nD_hists_per_region = self._init_histograms()

    def _init_histograms(self) -> dict[str, dict[str, hist.Hist]]:
        """
        Initialize histograms for each analysis channel based on configuration.

        Returns
        -------
        dict
            Dictionary of channel name to hist.Hist object.
        """
        histograms = defaultdict(dict)
        for channel in self.channels:
            chname = channel.name
            if (req_channels := self.config.general.channels) is not None:
                if chname not in req_channels:
                    continue

            for observable in channel["observables"]:

                observable_label = observable["label"]
                observable_binning = observable["binning"]
                observable_name = observable["name"]

                if isinstance(observable_binning, str):
                    low, high, nbins = map(
                        float, observable_binning.split(",")
                    )
                    axis = hist.axis.Regular(
                        int(nbins),
                        low,
                        high,
                        name="observable",
                        label=observable_label,
                    )
                else:
                    axis = hist.axis.Variable(
                        observable_binning,
                        name="observable",
                        label=observable_label,
                    )

                histograms[chname][observable_name] = hist.Hist(
                    axis,
                    hist.axis.StrCategory([], name="process", growth=True),
                    hist.axis.StrCategory([], name="variation", growth=True),
                    storage=hist.storage.Weight(),
                )
        return histograms


    def histogramming(
        self,
        object_copies: dict[str, ak.Array],
        events: ak.Array,
        process: str,
        variation: str,
        xsec_weight: float,
        analysis: str,
        event_syst: Optional[dict[str, Any]] = None,
        direction: Literal["up", "down", "nominal"] = "nominal",
    ) -> Optional[ak.Array]:
        """
        Apply physics selections and fill histograms.

        Parameters
        ----------
        object_copies : dict
            Corrected event-level objects.
        events : ak.Array
            Original NanoAOD event collection.
        process : str
            Sample name.
        variation : str
            Systematic variation label.
        xsec_weight : float
            Normalization weight.
        analysis : str
            Analysis name string.
        event_syst : dict, optional
            Event-level systematic to apply.
        direction : str, optional
            Systematic direction: 'up', 'down', or 'nominal'.

        Returns
        -------
        dict
            Updated histogram dictionary.
        """
        if process == "data" and variation != "nominal":
            return

        for channel in self.channels:
            chname = channel["name"]
            if (req_channels := self.config.general.channels) is not None:
                if chname not in req_channels:
                    continue
            logger.info(
                f"Applying selection for {chname} in {process}")
            mask = 1
            if (
                selection_funciton := channel["selection_function"]
            ) is not None:
                selection_args = self._get_function_arguments(
                    channel["selection_use"], object_copies
                )
                packed_selection = selection_funciton(*selection_args)
                if not isinstance(packed_selection, PackedSelection):
                    raise ValueError(
                        f"PackedSelection expected, got {type(packed_selection)}"
                    )
                mask = ak.Array(
                    packed_selection.all(packed_selection.names[-1])
                )

            if process == "data":
                mask = mask & lumi_mask(
                    self.config.general.lumifile,
                    object_copies["run"],
                    object_copies["luminosityBlock"],
                )

            if ak.sum(mask) == 0:
                logger.warning(
                    f"{analysis}:: No events left in {chname} for {process} with "
                    + "variation {variation}"
                )
                continue

            object_copies_channel = {
                                collection: variable[mask]
                                for collection, variable in object_copies.items()
                            }

            # WIP:: This should be fully removed (no need anymore, all cuts equal footing)
            soft_mask = 1.0
            if (
                soft_selection_funciton := channel[
                    "soft_selection_function"
                ]
            ) is not None:
                soft_selection_args = self._get_function_arguments(
                    channel["soft_selection_use"], object_copies_channel
                )
                soft_selection_dict = soft_selection_funciton(
                    *soft_selection_args
                )

                soft_mask = reduce(
                    operator.and_, soft_selection_dict.values()
                )

            if not isinstance(soft_mask, float):
                mask = mask[mask] & soft_mask
                object_copies_channel = {
                    collection: variable[mask]
                    for collection, variable in object_copies_channel.items()
                }

            if process != "data":
                weights = (
                    events[mask].genWeight
                    * xsec_weight
                    / abs(events[mask].genWeight)
                )
            else:
                weights = np.ones(ak.sum(mask))

            if event_syst and process != "data":
                weights = self.apply_event_weight_correction(
                    weights, event_syst, direction, object_copies_channel
                )

            logger.info(f"Number of weighted events in {chname}: {ak.sum(weights):.2f}")
            logger.info(f"Number of raw events in {chname}: {ak.sum(mask)}")
            for observable in channel["observables"]:
                logger.info(f"Computing observable {observable['name']}")
                observable_name = observable["name"]
                observable_args = self._get_function_arguments(
                    observable["use"], object_copies_channel
                )
                observable_vals = observable["function"](*observable_args)
                self.nD_hists_per_region[chname][observable_name].fill(
                    observable=observable_vals,
                    process=process,
                    variation=variation,
                    weight=weights,
                )


    def process(
        self, events: ak.Array, metadata: dict[str, Any],
    ) -> None:
        """
        Run the full analysis logic on a batch of events.

        Parameters
        ----------
        events : ak.Array
            Input NanoAOD events.
        metadata : dict
            Metadata with keys 'process', 'xsec', 'nevts', and 'dataset'.

        Returns
        -------
        dict
            Histogram dictionary after processing.
        """
        analysis = self.__class__.__name__

        process = metadata["process"]
        variation = metadata.get("variation", "nominal")
        logger.debug(f"Processing {process} with variation {variation}")
        xsec = metadata["xsec"]
        n_gen = metadata["nevts"]

        lumi = self.config["general"]["lumi"]
        xsec_weight = (xsec * lumi / n_gen) if process != "data" else 1.0

        # Nominal processing
        obj_copies = self.get_object_copies(events)
        # Filter objects
        # Get object masks from configuration:
        if (obj_masks := self.config.good_object_masks) != []:
            filtered_objs = self.get_good_objects(obj_copies, obj_masks)
            for obj, filtered in filtered_objs.items():
                if obj not in obj_copies:
                    raise KeyError(f"Object {obj} not found in object_copies")
                obj_copies[obj] = filtered

        # Apply baseline selection
        baseline_args = self._get_function_arguments(
            self.config.baseline_selection["use"], obj_copies
        )

        packed_selection = self.config.baseline_selection["function"](
            *baseline_args
        )
        mask = ak.Array(packed_selection.all(packed_selection.names[-1]))
        obj_copies = {
            collection: variable[mask]
            for collection, variable in obj_copies.items()
        }
        # apply ghost observables
        obj_copies = self.compute_ghost_observables(
            obj_copies,
        )

        # apply event-level corrections
        # apply nominal corrections
        obj_copies_corrected = self.apply_object_corrections(
            obj_copies, self.corrections, direction="nominal"
        )
        # apply selection and fill histograms
        obj_copies_corrected = {
            obj: var
            for obj, var in obj_copies_corrected.items()
        }
        self.histogramming(
            obj_copies_corrected,
            events,
            process,
            "nominal",
            xsec_weight,
            analysis,
        )

        # Systematic variations
        for syst in self.systematics + self.corrections:
            if syst["name"] == "nominal":
                continue
            for direction in ["up", "down"]:
                # Filter objects
                # Get object masks from configuration:
                if (obj_masks := self.config.good_object_masks) != []:
                    filtered_objs = self.get_good_objects(obj_copies, obj_masks)
                    for obj, filtered in filtered_objs.items():
                        if obj not in obj_copies:
                            raise KeyError(f"Object {obj} not found in object_copies")
                        obj_copies[obj] = filtered

                # apply corrections
                obj_copies_corrected = self.apply_object_corrections(
                    obj_copies, [syst], direction=direction
                )
                varname = f"{syst['name']}_{direction}"
                self.histogramming(
                    obj_copies_corrected,
                    events,
                    process,
                    varname,
                    xsec_weight,
                    analysis,
                    event_syst=syst,
                    direction=direction,
                )


    def run_fit(
        self, cabinetry_config: dict[str, Any]
    ) -> tuple[Any, Any, Any, Any]:
        """
        Run the fit using cabinetry.

        Parameters
        ----------
        cabinetry_config : dict
            Configuration for cabinetry.
        """

        # what do we do with this
        rebinning_router = get_cabinetry_rebinning_router(
            cabinetry_config, rebinning=slice(110j, None, hist.rebin(2))
        )
        # build the templates
        cabinetry.templates.build(cabinetry_config, router=rebinning_router)
        # optional post-processing (e.g. smoothing, symmetrise)
        cabinetry.templates.postprocess(cabinetry_config)
        # build the workspace
        ws = cabinetry.workspace.build(cabinetry_config)
        # save the workspace
        workspace_path = self.config.general.output_dir + "/statistics/"
        os.makedirs(workspace_path, exist_ok=True)
        workspace_path += "workspace.json"
        cabinetry.workspace.save(ws, workspace_path)
        # build the model and data
        model, data = cabinetry.model_utils.model_and_data(ws)
        # get pre-fit predictions
        prefit_prediction = cabinetry.model_utils.prediction(model)
        # perform the fit
        results = cabinetry.fit.fit(
            model,
            data,
        )  # perform the fit
        postfit_prediction = cabinetry.model_utils.prediction(
            model, fit_results=results
        )

        return data, results, prefit_prediction, postfit_prediction

# --------------------------------
# Differnetiable Analysis
# --------------------------------
class DifferentiableAnalysis(Analysis):

    def __init__(self, config):
        super.__init__(config)
        # 4 indices
        self.histograms = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(dict))
            )

    def set_histograms(
        self,
        histograms: dict[str, dict[str, dict[str,jnp.Array]]]
    ) -> None:
        """
        Set the histograms for the analysis for all processes, variations, channels,
        and observables.

        Parameters
        ----------
        histograms : dict
            Dictionary of histograms with channel, observable, and variation.
        """
        self.histograms = histograms

    def histogramming(
        self,
        object_copies: dict[str, ak.Array],
        events: ak.Array,
        process: str,
        variation: str,
        xsec_weight: float,
        analysis: str,
        event_syst: Optional[dict[str, Any]] = None,
        direction: Literal["up", "down", "nominal"] = "nominal",
    ) -> Optional[ak.Array]:
        """
        Apply physics selections and fill histograms.

        Parameters
        ----------
        object_copies : dict
            Corrected event-level objects.
        events : ak.Array
            Original NanoAOD event collection.
        process : str
            Sample name.
        variation : str
            Systematic variation label.
        xsec_weight : float
            Normalization weight.
        analysis : str
            Analysis name string.
        event_syst : dict, optional
            Event-level systematic to apply.
        direction : str, optional
            Systematic direction: 'up', 'down', or 'nominal'.

        Returns
        -------
        dict
            Updated histogram dictionary.
        """
        histograms = defaultdict(dict)
        if process == "data" and variation != "nominal":
            return histograms

        jax_config = self.config.jax

        diff_selection_args = self._get_function_arguments(
                    jax_config.soft_selection.use, object_copies
                )
        diff_selection_weights = (
            jax_config.soft_selection.function(*diff_selection_args,
                                                self.config.params)
        )

        for channel in self.channels:
            # skip channels in non-differentiable analysis
            if not channel.use_in_diff:
                continue

            chname = channel["name"]
            if (req_channels := self.config.general.channels) is not None:
                if chname not in req_channels:
                    continue
            logger.info(
                f"Applying selection for {chname} in {process}")
            mask = 1
            if (
                selection_funciton := channel["selection_function"]
            ) is not None:
                selection_args = self._get_function_arguments(
                    channel["selection_use"], object_copies
                )
                packed_selection = selection_funciton(*selection_args)
                if not isinstance(packed_selection, PackedSelection):
                    raise ValueError(
                        f"PackedSelection expected, got {type(packed_selection)}"
                    )
                mask = ak.Array(
                    packed_selection.all(packed_selection.names[-1])
                )

            if process == "data":
                mask = mask & lumi_mask(
                    self.config.general.lumifile,
                    object_copies["run"],
                    object_copies["luminosityBlock"],
                )

            if ak.sum(mask) == 0:
                logger.warning(
                    f"{analysis}:: No events left in {chname} for {process} with "
                    + "variation {variation}"
                )
                continue

            object_copies_channel = {
                                collection: variable[mask]
                                for collection, variable in object_copies.items()
                            }

            if process != "data":
                weights = (
                    events[mask].genWeight
                    * xsec_weight
                    / abs(events[mask].genWeight)
                )
            else:
                weights = np.ones(ak.sum(mask))

            weights = jnp.array(weights.to_numpy())

            if event_syst and process != "data":
                weights = self.apply_event_weight_correction(
                    weights, event_syst, direction, object_copies_channel
                )

            logger.info(f"Number of weighted events in {chname}: {ak.sum(weights):.2f}")
            logger.info(f"Number of raw events in {chname}: {ak.sum(mask)}")

            for observable in channel["observables"]:
                # check if observable works with JAX
                if not observable.works_with_jax:
                    continue

                logger.info(f"Computing observable {observable['name']}")
                observable_name = observable["name"]
                observable_label = observable["label"]

                observable_args = self._get_function_arguments(
                    observable["use"], object_copies_channel
                )
                observable_vals = observable["function"](*observable_args)
                observable_binning = observable["binning"]

                # WIP:: need to enforce presence of this in config
                bandwidth = jax_config.params['kde_bandwidth']
                if isinstance(observable_binning, str):
                    low, high, nbins = map(
                        float, observable_binning.split(",")
                    )
                    nbins = int(nbins)
                    observable_binning = jnp.linspace(low, high, nbins)
                else:
                    observable_binning = jnp.array(observable_binning)

                # KDE-style soft binning
                cdf = jax.scipy.stats.norm.cdf(
                    observable_binning.reshape(-1, 1),
                    loc=observable_vals.reshape(1, -1),
                    scale=bandwidth
                )

                # Weight each event's contribution by selection weight
                weighted_cdf = cdf * diff_selection_weights.reshape(1, -1) * weights.reshape(-1, 1)
                bin_weights = weighted_cdf[1:, :] - weighted_cdf[:-1, :]
                histogram = jnp.sum(bin_weights, axis=1)
                histograms[chname][observable_name] = histogram


    def process(
        self, events: ak.Array, metadata: dict[str, Any],
    ) -> None:
        """
        Run the full analysis logic on a batch of events.

        Parameters
        ----------
        events : ak.Array
            Input NanoAOD events.
        metadata : dict
            Metadata with keys 'process', 'xsec', 'nevts', and 'dataset'.

        Returns
        -------
        dict
            Histogram dictionary after processing.
        """
        all_histograms = copy.deepcopy(self.histograms)
        analysis = self.__class__.__name__

        process = metadata["process"]
        variation = metadata.get("variation", "nominal")
        logger.debug(f"Processing {process} with variation {variation}")
        xsec = metadata["xsec"]
        n_gen = metadata["nevts"]

        lumi = self.config["general"]["lumi"]
        xsec_weight = (xsec * lumi / n_gen) if process != "data" else 1.0

        # move everything to JAX backend
        ak.to_backend(events, "jax")
        # Nominal processing
        obj_copies = self.get_object_copies(events)
        # Filter objects
        # Get object masks from configuration:
        if (obj_masks := self.config.good_object_masks) != []:
            filtered_objs = self.get_good_objects(obj_copies, obj_masks)
            for obj, filtered in filtered_objs.items():
                if obj not in obj_copies:
                    raise KeyError(f"Object {obj} not found in object_copies")
                obj_copies[obj] = filtered

        # Apply baseline selection
        baseline_args = self._get_function_arguments(
            self.config.baseline_selection["use"], obj_copies
        )

        packed_selection = self.config.baseline_selection["function"](
            *baseline_args
        )
        mask = ak.Array(packed_selection.all(packed_selection.names[-1]))
        obj_copies = {
            collection: variable[mask]
            for collection, variable in obj_copies.items()
        }
        # apply ghost observables
        obj_copies = self.compute_ghost_observables(
            obj_copies,
        )

        # apply event-level corrections
        # apply nominal corrections
        obj_copies_corrected = self.apply_object_corrections(
            obj_copies, self.corrections, direction="nominal"
        )
        # apply selection and fill histograms
        obj_copies_corrected = {
            obj: var
            for obj, var in obj_copies_corrected.items()
        }
        histograms = self.histogramming(
            obj_copies_corrected,
            events,
            process,
            "nominal",
            xsec_weight,
            analysis,
        )
        all_histograms["nominal"] = histograms

        # Systematic variations
        for syst in self.systematics + self.corrections:
            if syst["name"] == "nominal":
                continue
            for direction in ["up", "down"]:
                # Filter objects
                # Get object masks from configuration:
                if (obj_masks := self.config.good_object_masks) != []:
                    filtered_objs = self.get_good_objects(obj_copies, obj_masks)
                    for obj, filtered in filtered_objs.items():
                        if obj not in obj_copies:
                            raise KeyError(f"Object {obj} not found in object_copies")
                        obj_copies[obj] = filtered

                # apply corrections
                obj_copies_corrected = self.apply_object_corrections(
                    obj_copies, [syst], direction=direction
                )
                varname = f"{syst['name']}_{direction}"
                histograms = self.histogramming(
                    obj_copies_corrected,
                    events,
                    process,
                    varname,
                    xsec_weight,
                    analysis,
                    event_syst=syst,
                    direction=direction,
                )
                all_histograms[varname] = histograms


        return all_histograms

    def _calculate_significance(self, ) -> jnp.ndarray:
        """
        Calculate signal significance using KDE-smoothed nominal histograms.

        Parameters
        ----------
        variation : str, optional
            Systematic variation key to use. Default is "nominal".

        Returns
        -------
        jnp.ndarray
            Signal significance (S / sqrt(B + δS² + δB²)).
        """
        params = self.config.jax.params
        signal_yield_total = 0.0
        background_yield_total = 0.0
        variation = "nominal" # nominal only test
        for channel in self.channels:
            if not getattr(channel, "use_in_diff", False):
                continue

            region = channel.name
            observable = getattr(channel, "fit_observable", None)
            if observable is None:
                logger.warning(f"[Significance] No fit_observable in {region}, skipping.")
                continue

            bin_vals_signal = None
            bin_vals_bkg = None

            for process, proc_hists in self.histograms.items():
                # Skip non-nominal or missing data
                if variation not in proc_hists:
                    continue
                if region not in proc_hists[variation]:
                    continue
                if observable not in proc_hists[variation][region]:
                    continue

                hist = proc_hists[variation][region][observable]

                # Lazy init based on first matching histogram
                if bin_vals_signal is None:
                    bin_vals_signal = jnp.zeros_like(hist)
                    bin_vals_bkg = jnp.zeros_like(hist)

                if process in {"signal", "zprime"}:
                    bin_vals_signal += hist
                elif process != "data":
                    bin_vals_bkg += hist

            if bin_vals_signal is None:
                logger.warning(f"[Significance] No histograms found for {region}/{observable}")
                continue

            signal_yield = jnp.sum(bin_vals_signal)
            background_yield = jnp.sum(bin_vals_bkg)

            signal_yield_total += signal_yield
            background_yield_total += background_yield

        # Systematic uncertainties
        signal_syst = params.get("signal_systematic", 0.05) * signal_yield_total
        background_syst = params.get("background_systematic", 0.1) * background_yield_total

        denom = jnp.sqrt(background_yield_total + signal_syst**2 + background_syst**2 + 1e-6)
        significance = signal_yield_total / denom

        return significance

    def analysis_chain(self, ):
        """
        Run the full analysis chain for differentiable analysis.

        Returns
        -------
        jnp.ndarray
            Signal significance.
        """
        histograms = self.histograms
        # Calculate significance
        significance = self._calculate_significance()

        # Return the significance value
        return significance

    def process_with_gradients(self,):
        """
        Modified version that processes multiple processes and calculates significance.

        Parameters:
        -----------
        process_data_dict : dict
            Dictionary with process names as keys and JAX data as values
            e.g., {'signal': jax_data, 'ttbar': jax_data, 'wjets': jax_data}
        """

        print("Running differentiable event loop for multiple processes...")
        print(f"Processes: {list(process_data_dict.keys())}")

        # Compute significance
        significance = self.differentiable_event_loop(self.params, process_data_dict)

        # Create gradient function
        grad_fn = jax.grad(self.differentiable_event_loop, argnums=0)
        gradients = grad_fn(self.params, process_data_dict)

        print(f"Signal significance: {significance}")
        print(f"Parameter gradients: {gradients}")

        return significance, gradients



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
    # ✅ You now have a fully validated config object
    logger.info(f"Luminosity: {config.general.lumi}")

    nondiff_analysis = NonDiffAnalysis(config)
    diff_analysis = DifferentiableAnalysis(config)

    fileset = construct_fileset(
        n_files_max_per_sample=config.general.max_files
    )

    process_histograms = {}
    for dataset, content in fileset.items():
        metadata = content["metadata"]
        metadata["dataset"] = dataset
        process_name = metadata["process"]
        if (req_processes := config.general.processes) is not None:
            if process_name not in req_processes:
                continue

        os.makedirs(f"{config.general.output_dir}/{dataset}", exist_ok=True)

        logger.info("========================================")
        logger.info(f"🚀 Processing dataset: {dataset}")

        for idx, (file_path, tree) in enumerate(content["files"].items()):
            output_dir = (
                f"output/{dataset}/file__{idx}/"
                if not config.general.preprocessed_dir
                else f"{config.general.preprocessed_dir}/{dataset}/file__{idx}/"
            )
            if (
                config.general.max_files != -1
                and idx >= config.general.max_files
            ):
                continue
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

            skimmed_files = glob.glob(f"{output_dir}/part*.root")
            skimmed_files = [f"{f}:{tree}" for f in skimmed_files]
            remaining = sum(uproot.open(f).num_entries for f in skimmed_files)
            logger.info(f"✅ Events retained after filtering: {remaining:,}")
            if config.general.run_histogramming:
                for skimmed in skimmed_files:
                    logger.info(f"📘 Processing skimmed file: {skimmed}")
                    logger.info("📈 Processing for non-differentiable analysis")
                    events = NanoEventsFactory.from_root(
                        skimmed, schemaclass=NanoAODSchema, delayed=False
                    ).events()
                    nondiff_analysis.process(events, metadata)
                    logger.info("📈 Non-differentiable histogram-filling complete.")
                    # Differentiable histograms
                    logger.info("📈 Processing for differentiable analysis")
                    diff_histograms = diff_analysis.process(events, metadata)
                    process_histograms[process_name] = diff_histograms
                    logger.info("📈 Differentiable histogram (kde) compute.")

        logger.info(f"🏁 Finished dataset: {dataset}\n")

    # Set histograms for differentiable analysis
    diff_analysis.set_histograms(process_histograms)
    diff_analysis._calculate_significance()


    # Report end of processing
    logger.info("✅ All datasets processed.")

    # Save histograms for non-differentiable analysis
    if config.general.run_histogramming:
        save_histograms(
            nondiff_analysis.nD_hists_per_region,
            output_file=f"{config.general.output_dir}/histograms/histograms.root",
        )
        pkl_histograms(
            nondiff_analysis.nD_hists_per_region,
            output_file=f"{config.general.output_dir}/histograms/histograms.pkl",
        )
    # Run statistics for non-differentiable analysis
    if config.general.run_statistics:
        cabinetry_config = cabinetry.configuration.load(
            config.statistics.cabinetry_config
        )
        data, fit_results, pre_fit_predictions, postfit_predictions = (
            nondiff_analysis.run_fit(cabinetry_config=cabinetry_config)
        )
        cabinetry.visualize.data_mc(
            pre_fit_predictions,
            data,
            close_figure=False,
            config=cabinetry_config,
        )
        cabinetry.visualize.data_mc(
            postfit_predictions,
            data,
            close_figure=False,
            config=cabinetry_config,
        )
        cabinetry.visualize.pulls(fit_results, close_figure=False)

    plot_nominal_histograms("output/histograms/histograms.root")
    plot_cms_style(histograms_file="output/histograms/histograms.pkl")

def plot_cms_style(histograms_file: str):
    """
    Plot histograms in CMS style.
    Parameters
    ----------
    histograms_file : str
        Path to the histograms pkl file.
    """
    if ".pkl" not in histograms_file:
        raise ValueError("histograms_file must be a .pkl file")
    histograms = unpkl_histograms(histograms_file)
    os.makedirs("output/plots_TEST", exist_ok=True)
    mplhep.style.use("CMS")

    process_colors = {
        "ttbar": "#FF6600",    # orange
        "wjets": "#33CCFF",    # blue
        "other bkgs.": "#CC00FF",    # magenta
        "data": "black",
        "signal": "black",
    }

    for region, obs_dict in histograms.items():
        for observable, nD_histogram in obs_dict.items():
            for variation in nD_histogram.axes["variation"]:
                if variation != "nominal":
                    continue
                logger.info(
                f"Plotting {observable} for {region} and {variation}")
                fig, ax = plt.subplots(figsize=(12, 10), layout="constrained")
                mplhep.cms.label(
                                    label="Preliminary",
                                    lumi=35.6,
                                    data=True,
                                    loc=1,
                                )
                combined_hist = {}
                for process in nD_histogram.axes["process"]:
                    if process.startswith("ttbar_"):
                        if "ttbar" not in combined_hist:
                            combined_hist["ttbar"] = nD_histogram[:, process, variation]
                        else:
                            combined_hist["ttbar"] += nD_histogram[:, process, variation]
                    elif process.startswith("wjets") or process.startswith("signal") or process.startswith("data"):
                        combined_hist[process] = nD_histogram[:, process, variation]
                    else:
                        combined_hist["other bkgs."] = nD_histogram[:, process, variation]

                # Move signal histogram to be last plotted
                if "signal" in combined_hist:
                    signal_hist = combined_hist.pop("signal")
                    combined_hist["signal"] = signal_hist

                total_histogram = 0.0
                for process, histogram in combined_hist.items():
                    total_histogram += histogram
                    color = process_colors.get(process)
                    histtype = ("errorbar" if process == "data"
                                else "step" if process == "signal"
                                else "fill")
                    linestyle = "none" if process == "data" else "--" if process == "signal" else "-"
                    stack = True if process != "data" else False
                    linewidth = 1.2 if process == "signal" else None

                    mplhep.histplot(histogram, histtype=histtype, label=process, ax=ax, linestyle=linestyle, linewidth=linewidth, color=color, stack=stack)

                ax.set_xlabel(observable)
                ax.set_ylabel("Events")
                yscale = "linear"
                ax.set_yscale(yscale)
                max_events = max(total_histogram.values())
                ax.set_ylim(None, max_events*10 if yscale=="log" else max_events*1.2)
                ax.legend(loc="upper right", fontsize=12)
                plt.savefig(
                    f"output/plots_TEST/{region}_{observable}_{variation}.png"
                )


def plot_nominal_histograms(
    hist_file: str, output_dir: str = "output/plots/nominal_mva_inputs"
):
    """
    Generate normalized histograms for each observable, region, and process from the
    ROOT file.

    Parameters
    ----------
    hist_file : str
        Path to the histogram ROOT file.
    output_dir : str
        Directory to save output plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    file = uproot.open(hist_file)
    histograms = defaultdict(lambda: defaultdict(dict))

    for key in file.keys():
        name = key.strip(";1")
        parts = name.split("__")

        if parts[0] != "baseline":
            continue  # only interested in 'baseline' channel

        region = parts[0]
        observable = parts[1]
        process = "_".join(parts[2:])

        if "up" in process or "down" in process:
            continue  # skip systematic variations

        hist = file[name]
        values = hist.values()
        edges = hist.axis().edges()

        histograms[region][observable][process] = (values, edges)

    for region, obs_dict in histograms.items():
        for observable, proc_dict in obs_dict.items():
            plt.figure(figsize=(6, 4))
            for proc, (vals, edges) in proc_dict.items():
                norm_vals = vals / vals.sum() if vals.sum() > 0 else vals
                centers = 0.5 * (edges[1:] + edges[:-1])
                plt.step(centers, norm_vals, where="mid", label=proc)

            plt.xlabel(observable)
            plt.ylabel("Normalized Entries")
            plt.title(f"{observable} — {region}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{region}_{observable}.png")
            plt.close()

    logger.info(f"Saved plots to: {output_dir}")



if __name__ == "__main__":
    main()
