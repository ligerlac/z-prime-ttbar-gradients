#!/usr/bin/env python3

"""
ZprimeAnalysis framework for applying object and event-level systematic corrections
on NanoAOD ROOT files and producing histograms of observables like mtt. Supports both
correctionlib-based and function-based corrections.
"""

from collections import defaultdict
import glob
import gzip
import logging
import os
import sys
import warnings

import awkward as ak
import cabinetry
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from correctionlib import CorrectionSet
import hist
import jax
import jax.numpy as jnp
import numpy as np
import uproot
import vector

from utils.configuration import config as ZprimeConfig
from utils.cuts import lumi_mask
from utils.input_files import construct_fileset
from utils.output_files import save_histograms
from utils.preproc import pre_process_dak, pre_process_uproot
from utils.schema import Config, load_config_with_restricted_cli
from utils.stats import get_cabinetry_rebinning_router
from utils.mva import decorate_mva_scores


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
# ZprimeAnalysis Class Definition
# -----------------------------
class ZprimeAnalysis:
    def __init__(self, config):
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
        self.channels = config["channels"]
        self.systematics = config["systematics"]
        self.corrections = config["corrections"]
        self.corrlib_evaluators = self._load_correctionlib()
        self.nD_hists_per_region = self._init_histograms()

    def _init_histograms(self):
        """
        Initialize histograms for each analysis channel based on configuration.

        Returns
        -------
        dict
            Dictionary of channel name to hist.Hist object.
        """
        histograms = defaultdict(dict)
        for channel in self.channels:
            chname = channel["name"]
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

    def _load_correctionlib(self):
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

    def get_object_copies(self, events):
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

    def get_good_objects(self, object_copies):

        muons, jets, fatjets, met = (
            object_copies["Muon"],
            object_copies["Jet"],
            object_copies["FatJet"],
            object_copies["PuppiMET"],
        )
        muons = muons[
            (muons.pt > 55)
            & (abs(muons.eta) < 2.4)
            & muons.tightId
            & (muons.miniIsoId > 1)
        ]
        jets = jets[
            (jets.pt > 30)
            & (abs(jets.eta) < 2.4)
            & jets.isTightLeptonVeto
            & (jets.jetId >= 4)
        ]
        fatjets = fatjets[
            (fatjets.pt > 200)
            & (abs(fatjets.eta) < 2.4)
            & (fatjets.particleNet_TvsQCD > 0.5)
        ]

        return muons, jets, fatjets, met

    def apply_correctionlib(
        self,
        name,
        key,
        direction,
        correction_arguments,
        target=None,
        op=None,
        transform=None,
    ):
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

    def apply_syst_fn(self, name, fn, args, affects, op):
        """
        Apply function-based systematic variation.
        """
        logger.debug(f"Applying function-based systematic: {name}")
        correction = fn(*args)
        if isinstance(affects, list):
            return [self.apply_op(op, a, correction) for a in affects]
        else:
            return self.apply_op(op, affects, correction)

    def apply_op(self, op, lhs, rhs):
        """
        Apply a binary operation.
        """
        if op == "add":
            return lhs + rhs
        elif op == "mult":
            return lhs * rhs
        else:
            raise ValueError(f"Unsupported operation: {op}")

    def _get_function_arguments(self, use, object_copies):
        """
        Extract correction arguments from object_copies.
        """
        return [
            object_copies[obj][var] if var is not None else object_copies[obj]
            for obj, var in use
        ]

    def _get_targets(self, target, object_copies):
        """
        Extract one or more target arrays from object_copies.
        """
        targets = target if isinstance(target, list) else [target]
        return [object_copies[obj][var] for obj, var in targets]

    def _set_targets(self, target, object_copies, new_values):
        """
        Set corrected values in object_copies.
        """
        targets = target if isinstance(target, list) else [target]
        for (obj, var), val in zip(targets, new_values):
            object_copies[obj][var] = val

    def apply_object_corrections(
        self, object_copies, corrections, direction="nominal"
    ):
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
        self, weights, systematic, direction, object_copies
    ):
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

    def apply_selection_and_fill(
        self,
        object_copies,
        events,
        process,
        variation,
        xsec_weight,
        analysis,
        met_cut,
        event_syst=None,
        direction="nominal",
        tracing=False,
    ):
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
                mask = mask & lumi_mask(self.config.general.lumifile, events)

            if ak.sum(mask) == 0:
                logger.warning(
                    f"{analysis}:: No events left in {chname} for {process} with "
                    + "variation {variation}"
                )
                continue

            if tracing:
                mask = ak.to_backend(mask, "jax")
                object_copies = {
                    collection: variable[mask]
                    for collection, variable in object_copies.items()
                }

                region_muons = object_copies["Muon"]
                region_jets = object_copies["Jet"]
                region_met = object_copies["PuppiMET"]
                region_fatjets = object_copies["FatJet"]

                region_lep_ht = region_muons.pt + region_met.pt
                soft_cuts = {
                    "atleast_1b": ak.sum(region_jets.btagDeepB > 0.5, axis=1) > 0,
                    # "met_cut": met.pt > 50,
                    # "met_cut": 0.5*jnp.tanh((ak.to_jax(region_met.pt)-50)/100)+0.5,
                    "met_cut": jax.nn.sigmoid(
                        (ak.to_jax(region_met.pt) - met_cut) / met_cut
                    ),
                    "lep_ht_cut": ak.fill_none(
                        ak.firsts(region_lep_ht) > 150, False
                    ),
                }

                # Convert selections to JAX arrays with float dtype
                soft_cuts = {
                    k: jnp.array(ak.to_jax(v), dtype=float)
                    for k, v in soft_cuts.items()
                }

                weights = jnp.prod(jnp.stack(list(soft_cuts.values())), axis=0)
                logger.info(f"Weights:: {weights} ")
            else:
                object_copies = {
                    collection: variable[mask]
                    for collection, variable in object_copies.items()
                }

                region_muons = object_copies["Muon"]
                region_jets = object_copies["Jet"]
                region_met = object_copies["PuppiMET"]
                region_lep_ht = region_muons.pt + region_met.pt

                soft_cuts = {
                                "atleast_1b": ak.sum(region_jets.btagDeepB > 0.5, axis=1) > 0,
                                "met_cut": region_met.pt > 50,
                                "lep_ht_cut": ak.fill_none(
                                    ak.firsts(region_lep_ht) > 150, False
                                ),
                                }

                mask = mask[mask] & (soft_cuts["atleast_1b"]) & (soft_cuts["met_cut"]) & (soft_cuts["lep_ht_cut"])
                object_copies = {
                    collection: variable[mask]
                    for collection, variable in object_copies.items()
                }
                weights = 1.


            if process != "data":
                weights *= (
                    events[mask].genWeight
                    * xsec_weight
                    / abs(events[mask].genWeight)
                )
            else:
                weights *= np.ones(len(ak.count_nonzero(mask)))

            if event_syst and process != "data":
                weights = self.apply_event_weight_correction(
                    weights, event_syst, direction, object_copies
                )

            for observable in channel["observables"]:
                logger.info(
                    f"Filling histogram for {observable['name']} in {chname}"
                )
                observable_name = observable["name"]
                observable_args = self._get_function_arguments(
                    observable["use"], object_copies
                )
                observable_vals = observable["function"](*observable_args)
                if not tracing:
                    self.nD_hists_per_region[chname][observable_name].fill(
                        observable=observable_vals,
                        process=process,
                        variation=variation,
                        weight=jax.lax.stop_gradient(weights),
                    )
                else:
                    return ak.sum(
                        weights
                    )  # return some dummy value to test auto-diff

    def run_fit(self, cabinetry_config):
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

    def process(self, events, metadata):
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
        # filter objects
        muons, jets, fatjets, met = self.get_good_objects(obj_copies)
        (
            obj_copies["Muon"],
            obj_copies["Jet"],
            obj_copies["FatJet"],
            obj_copies["PuppiMET"],
        ) = (muons, jets, fatjets, met)
        # apply nominal corrections

        obj_copies = self.apply_object_corrections(
            obj_copies, self.corrections, direction="nominal"
        )
        apply_selection_and_fill_grad = jax.value_and_grad(
            self.apply_selection_and_fill, argnums=6, has_aux=False
        )
        val, grad = apply_selection_and_fill_grad(
            obj_copies,
            events,
            process,
            variation,
            xsec_weight,
            analysis,
            50.0,
            tracing=True,
        )
        logger.info(f"val: {val}, grad: {grad}")

        events = ak.to_backend(events, "cpu")
        obj_copies = {obj: ak.to_backend(var, "cpu") for obj, var in obj_copies.items()}
        self.apply_selection_and_fill(
            obj_copies,
            events,
            process,
            "nominal",
            xsec_weight,
            analysis,
            50.0,
            tracing=False,
        )

        # Systematic variations
        for syst in self.systematics + self.corrections:
            if syst["name"] == "nominal":
                continue
            for direction in ["up", "down"]:
                obj_copies = self.get_object_copies(events)
                # filter objects
                muons, jets, fatjets, met = self.get_good_objects(obj_copies)
                (
                    obj_copies["Muon"],
                    obj_copies["Jet"],
                    obj_copies["FatJet"],
                    obj_copies["PuppiMET"],
                ) = (muons, jets, fatjets, met)
                # apply corrections
                obj_copies = self.apply_object_corrections(
                    obj_copies, [syst], direction=direction
                )
                varname = f"{syst['name']}_{direction}"
                self.apply_selection_and_fill(
                    obj_copies,
                    events,
                    process,
                    varname,
                    xsec_weight,
                    analysis,
                    50.0,
                    event_syst=syst,
                    direction=direction,
                    tracing=False,
                )


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

    analysis = ZprimeAnalysis(config)
    fileset = construct_fileset(
        n_files_max_per_sample=config.general.max_files
    )

    for dataset, content in fileset.items():
        metadata = content["metadata"]
        metadata["dataset"] = dataset

        if (req_processes := config.general.processes) is not None:
            if dataset.split("__")[0] not in req_processes:
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
            if config.general.decorate_mva_score:
                for skimmed in skimmed_files:
                    logger.info(f"📘 Decorating MVA score to skimmed file: {skimmed}")
                    decorate_mva_scores(skimmed, config.general.mva_model_path)
                    logger.info("✅ MVA variables added.")
            if config.general.run_histogramming:
                for skimmed in skimmed_files:
                    logger.info(f"📘 Processing skimmed file: {skimmed}")
                    events = NanoEventsFactory.from_root(
                        skimmed, schemaclass=NanoAODSchema, delayed=False
                    ).events()
                    events = ak.to_backend(events, "jax")
                    analysis.process(events, metadata)
                    logger.info("📈 Histogram filling complete.")

        logger.info(f"🏁 Finished dataset: {dataset}\n")

    logger.info("✅ All datasets processed.")
    if config.general.run_histogramming:
        save_histograms(
            analysis.nD_hists_per_region,
            output_file=f"{config.general.output_dir}/histograms/histograms.root",
        )

    if config.general.run_statistics:
        cabinetry_config = cabinetry.configuration.load(
            config.statistics.cabinetry_config
        )
        data, fit_results, pre_fit_predictions, postfit_predictions = (
            analysis.run_fit(cabinetry_config=cabinetry_config)
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


if __name__ == "__main__":
    main()
