#!/usr/bin/env python3

"""
ZprimeAnalysis framework for applying object and event-level systematic corrections
on NanoAOD ROOT files and producing histograms of observables like mtt. Supports both
correctionlib-based and function-based corrections.
"""

import os
import glob
import gzip
import logging
import warnings
import copy

import numpy as np
import awkward as ak
import uproot
import hist
import dask_awkward as dak

from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from correctionlib import CorrectionSet

import utils
from utils.input_files import construct_fileset

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger("ZprimeAnalysis")

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")

RUN_PREPROCESS = False
RUN_HISTOGRAMS = True
MAX_FILES_PER_SAMPLE = 1

def is_jagged(arraylike) -> bool:
    try:
        return ak.num(arraylike, axis=1) is not None
    except Exception:
        return False

# -----------------------------
# Branch Selection
# -----------------------------
def build_branches_to_keep():
    """
    Define the branches to retain during pre-processing.

    Returns
    -------
    dict
        Dictionary mapping object names to lists of branch names.
    """
    return {
        "Muon": ["pt", "eta", "phi", "mass", "miniIsoId", "tightId", "charge"],
        "FatJet": ["particleNet_TvsQCD", "pt", "eta", "phi", "mass", "charge"],
        "Jet": ["btagDeepB", "jetId", "pt", "eta", "phi", "mass", "charge"],
        "PuppiMET": ["pt", "phi"],
        "HLT": ["TkMu50"],
        "Pileup": ["nTrueInt"],
        "event": ["genWeight", "event", "run", "luminosityBlock"],
    }

# -----------------------------
# Preprocessing Logic
# -----------------------------
def pre_process(input_path, tree, output_path, step_size=100_000):
    """
    Preprocess input ROOT file by applying basic filtering and reducing branches.

    Parameters
    ----------
    input_path : str
        Path to the input ROOT file.
    tree : str
        Name of the TTree inside the file.
    output_path : str
        Destination directory for filtered output.
    step_size : int
        Chunk size to load events incrementally.

    Returns
    -------
    int
        Total number of input events before filtering.
    """
    with uproot.open(f"{input_path}:{tree}") as f:
        total_events = f.num_entries

    logger.info("========================================")
    logger.info(f"📂 Preprocessing file: {input_path} with {total_events:,} events")

    branches = build_branches_to_keep()
    selected = None

    for start in range(0, total_events, step_size):
        stop = min(start + step_size, total_events)

        events = NanoEventsFactory.from_root(
            {input_path: tree},
            schemaclass=NanoAODSchema,
            entry_start=start,
            entry_stop=stop,
            delayed=True
        ).events()

        mu_sel = (
            (events.Muon.pt > 55) &
            (abs(events.Muon.eta) < 2.4) &
            events.Muon.tightId &
            (events.Muon.miniIsoId > 1)
        )
        muon_count = ak.sum(mu_sel, axis=1)
        mask = (
            events.HLT.TkMu50 &
            (muon_count == 1) &
            (events.PuppiMET.pt > 50)
        )

        filtered = events[mask]

        subset = {}
        for obj, obj_branches in branches.items():
            if obj == "event":
                subset.update({br: filtered[br] for br in obj_branches if br in filtered.fields})
            elif obj in filtered.fields:
                subset.update({f"{obj}_{br}": filtered[obj][br] for br in obj_branches if br in filtered[obj].fields})

        compact = dak.zip(subset, depth_limit=1)
        selected = compact if selected is None else ak.concatenate([selected, compact])

    logger.info(f"💾 Writing skimmed output to: {output_path}")
    uproot.dask_write(selected, destination=output_path, compute=True, tree_name=tree)
    return total_events

# -----------------------------
# ZprimeAnalysis Class Definition
# -----------------------------

class ZprimeAnalysis:
    def __init__(self, config):
        """
        Initialize ZprimeAnalysis with configuration for systematics, corrections, and channels.

        Parameters
        ----------
        config : dict
            Configuration dictionary with 'systematics', 'corrections', 'channels', and 'general'.
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
        histograms = {}
        for channel in self.channels:
            name = channel["name"]
            label = channel["observable_label"]
            binning = channel["observable_binning"]

            if isinstance(binning, str):
                low, high, nbins = map(float, binning.split(","))
                axis = hist.axis.Regular(int(nbins), low, high, name="observable", label=label)
            else:
                axis = hist.axis.Variable(binning, name="observable", label=label)

            histograms[name] = hist.Hist(
                axis,
                hist.axis.StrCategory([], name="process", growth=True),
                hist.axis.StrCategory([], name="variation", growth=True),
                storage=hist.storage.Weight()
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
                    evaluators[name] = CorrectionSet.from_string(f.read().strip())
            elif path.endswith(".json"):
                evaluators[name] = CorrectionSet.from_file(path)
            else:
                raise ValueError(f"Unsupported correctionlib format: {path}")

        return evaluators

    def apply_op(self, op, lhs, rhs):
        """
        Apply a mathematical operation.

        Parameters
        ----------
        op : str
            Operation name, 'add' or 'mult'.
        lhs : array-like
            Left-hand side operand.
        rhs : array-like
            Right-hand side operand.

        Returns
        -------
        array-like
            Result of the operation.
        """
        if op == "add":
            return lhs + rhs
        elif op == "mult":
            return lhs * rhs
        else:
            raise ValueError(f"Unsupported operation: {op}")

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

        muons, jets, fatjets, met = object_copies["Muon"], object_copies["Jet"], object_copies["FatJet"], object_copies["PuppiMET"]
        muons = muons[(muons.pt > 55) & (abs(muons.eta) < 2.4) & muons.tightId & (muons.miniIsoId > 1)]
        jets = jets[(jets.pt > 30) & (abs(jets.eta) < 2.4) & jets.isTightLeptonVeto & (jets.jetId >= 4)]
        fatjets = fatjets[(fatjets.pt > 200) & (abs(fatjets.eta) < 2.4) & (fatjets.particleNet_TvsQCD > 0.5)]

        return muons, jets, fatjets, met

    def apply_syst_fn(self, name, fn, args, affects, op):
        """
        Apply function-based systematic variation.

        Parameters
        ----------
        name : str
            Name of the systematic.
        fn : callable
            Function to compute the correction.
        args : list
            Arguments to pass to the function.
        affects : array-like
            The quantity to apply the correction to.
        op : str
            Operation to apply ('add' or 'mult').

        Returns
        -------
        array-like
            Corrected result.
        """
        logger.debug(f"Applying function-based systematic: {name}")
        return self.apply_op(op, affects, fn(*args))

    def apply_correctionlib(self, name, key, direction, correction_arguments, target=None, op=None, transform=None):
        """
        Apply a correction using correctionlib.

        Parameters
        ----------
        name : str
            The name of the correction set.
        key : str
            The key within the correction set.
        direction : str
            Direction of the variation ('up', 'down', or 'nominal').
        correction_arguments : list
            List of arguments to pass to the correction.
        target : array-like, optional
            Target array to apply the correction to (if using an operation).
        op : str, optional
            Operation to apply ('add' or 'mult').
        transform : callable, optional
            Transformation function to apply to correction_arguments before evaluation.

        Returns
        -------
        array-like
            The corrected array or correction result.
        """
        logger.info(f"Applying correctionlib correction: name={name}, key={key}, direction={direction}")

        if transform is not None:
            correction_arguments = transform(*correction_arguments)

        flat_args, counts_to_unflatten = [], []
        for arg in correction_arguments:
            if is_jagged(arg):
                flat_args.append(ak.flatten(arg))
                counts_to_unflatten.append(ak.num(arg))
            else:
                flat_args.append(arg)

        correction = self.corrlib_evaluators[name][key].evaluate(*flat_args, direction)

        if counts_to_unflatten != []:
            unflattened_corr = ak.unflatten(correction, counts_to_unflatten[0])
        else:
            unflattened_corr = correction

        if target is not None and op is not None:
            return self.apply_op(op, target, unflattened_corr)
        return unflattened_corr

    def apply_object_corrections(self, object_copies, corrections, direction="nominal"):
        """
        Apply object-level corrections to input object copies.

        Parameters
        ----------
        object_copies : dict
            Dictionary of objects.
        corrections : list
            List of systematic/correction dicts.
        direction : str
            'nominal', 'up', or 'down'.

        Returns
        -------
        dict
            Corrected object copies.
        """
        for corr in corrections:
            if corr["type"] != "object":
                continue
            use_lib = corr.get("use_correctionlib", False)
            correction_arguments = []
            for use_from_object in corr.get("use", []):
                if isinstance(use_from_object, tuple) and len(use_from_object) == 2:
                    correction_arguments.append(object_copies[use_from_object[0]][use_from_object[1]])
                else:
                    raise ValueError(f"Invalid 'use' field: {use_from_object}")

            key = corr.get("key")
            op = corr.get("op")
            target = object_copies[corr["target"][0]][corr["target"][1]]
            transform = corr.get("transform", lambda *x: x)
            dir_map = corr.get("up_and_down_idx", ["up", "down"])
            corr_dir = dir_map[0 if direction == "up" else 1] if direction in ["up", "down"] else "nominal"
            if use_lib:
                object_copies[corr["target"][0]][corr["target"][1]] = self.apply_correctionlib(
                    corr["name"], key, corr_dir, correction_arguments, target, op, transform
                )
            else:
                fn = corr.get(f"{direction}_function")
                if fn:
                    object_copies[corr["target"][0]][corr["target"][1]] = self.apply_syst_fn(
                        corr["name"], fn, correction_arguments, target, op
                    )
        return object_copies

    def apply_event_weight_correction(self, weights, systematic, direction, object_copies):
        """
        Apply event-level correction to weights.

        Parameters
        ----------
        weights : ak.Array
            Event weights.
        systematic : dict
            Systematic configuration.
        direction : str
            Direction: 'up' or 'down'.
        object_copies : dict
            Object copies dictionary.

        Returns
        -------
        ak.Array
            Corrected event weights.
        """
        if systematic["type"] != "event":
            return weights
        use_correctionlib = systematic.get("use_correctionlib", False)
        correction_arguments = []
        for use_from_object in systematic.get("use", []):
            if isinstance(use_from_object, tuple) and len(use_from_object) == 2:
                correction_arguments.append(object_copies[use_from_object[0]][use_from_object[1]])
            else:
                raise ValueError(f"Invalid 'use' field: {use_from_object}")


        correction_key = systematic.get("key")
        correction_operation = systematic.get("op")
        direction_map = systematic.get("up_and_down_idx", ["up", "down"])
        correctionlib_direction = direction_map[0 if direction == "up" else 1]
        transform_function = systematic.get("transform", lambda *x: x)

        if use_correctionlib:
            return self.apply_correctionlib(
                systematic["name"],
                correction_key,
                correctionlib_direction,
                correction_arguments,
                target=weights,
                op=correction_operation,
                transform=transform_function
            )
        else:
            correction_function = systematic.get(f"{direction}_function")
            return self.apply_syst_fn(systematic["name"], correction_function, correction_arguments, weights, correction_operation) if correction_function else weights

    def apply_selection_and_fill(self, object_copies, events, process, variation, hist_dict, xsec_weight, analysis, event_syst=None, direction="nominal"):
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
        hist_dict : dict
            Dictionary of output histograms.
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

        muons, jets, fatjets, met = object_copies["Muon"], object_copies["Jet"], object_copies["FatJet"], object_copies["PuppiMET"]

        lep_ht = muons.pt + met.pt

        selections = PackedSelection(dtype='uint64')
        selections.add("dummy", ak.num(muons) > -1)
        selections.add("exactly_1mu", ak.num(muons) == 1)
        selections.add("pass_mu_trigger", events.HLT.TkMu50)
        selections.add("atleast_1b", ak.sum(jets.btagDeepB > 0.5, axis=1) > 0)
        selections.add("met_cut", met.pt > 50)
        selections.add("lep_ht_cut", lep_ht[:, 0] > 150)
        selections.add("exactly_1fatjet", ak.num(fatjets) == 1)
        selections.add("Zprime_channel", selections.all("pass_mu_trigger", "exactly_1mu", "met_cut", "exactly_1fatjet", "lep_ht_cut", "atleast_1b"))
        selections.add("preselection", selections.all("dummy"))


        for channel in self.channels:
            chname = channel["name"]
            mask = selections.all(chname)
            if ak.sum(mask) == 0:
                logger.warning(f"{analysis}:: No events left in {chname} for {process} with variation {variation}")
                continue

            object_copies = {collection: variable[mask] for collection, variable in object_copies.items()}
            region_muons, region_fatjets, region_jets, region_met = object_copies["Muon"], object_copies["FatJet"], object_copies["Jet"], object_copies["PuppiMET"]
            object_copies["Muon"], object_copies["FatJet"], object_copies["Jet"], object_copies["PuppiMET"] = region_muons, region_fatjets, region_jets, region_met
            region_muons_4vec, region_jets_4vec, region_jets_4vec = [ak.zip({"pt": o.pt, "eta": o.eta, "phi": o.phi, "mass": o.mass}, with_name="Momentum4D") for o in [region_muons, region_fatjets, region_jets[:, 0]]]
            region_met_4vec = ak.zip(
                {"pt": region_met.pt, "eta": 0 * region_met.pt, "phi": region_met.phi, "mass": 0},
                with_name="Momentum4D",
            )

            mtt = ak.flatten((region_muons_4vec + region_jets_4vec + region_jets_4vec + region_met_4vec).mass)

            if process != "data":
                weights = events[mask].genWeight * xsec_weight / abs(events[mask].genWeight)
            else:
                weights = np.ones(len(region_met))

            if event_syst and process != "data":
                weights = self.apply_event_weight_correction(weights, event_syst, direction, object_copies)

            self.nD_hists_per_region[chname].fill(observable=mtt, process=process, variation=variation, weight=weights)
            hist_dict[chname].fill(observable=mtt, process=process, variation=variation, weight=weights)

        return hist_dict

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
        hist_dict = copy.deepcopy(self.nD_hists_per_region)

        process = metadata["process"]
        variation = metadata.get("variation", "nominal")
        xsec = metadata["xsec"]
        n_gen = metadata["nevts"]
        lumi = self.config["general"]["lumi"]
        xsec_weight = (xsec * lumi / n_gen) if process != "data" else 1.0

        # Nominal processing
        obj_copies = self.get_object_copies(events)
        # filter objects
        muons, jets, fatjets, met = self.get_good_objects(obj_copies)
        obj_copies["Muon"], obj_copies["Jet"], obj_copies["FatJet"], obj_copies["PuppiMET"] = muons, jets, fatjets, met
        # apply nominal corrections
        obj_copies = self.apply_object_corrections(obj_copies, self.corrections, direction="nominal")
        hist_dict = self.apply_selection_and_fill(obj_copies, events, process, "nominal", hist_dict, xsec_weight, analysis)

        # Systematic variations
        for syst in self.systematics + self.corrections:
            if syst["name"] == "nominal":
                continue
            for direction in ["up", "down"]:
                obj_copies = self.get_object_copies(events)
                # filter objects
                muons, jets, fatjets, met = self.get_good_objects(obj_copies)
                obj_copies["Muon"], obj_copies["Jet"], obj_copies["FatJet"], obj_copies["PuppiMET"] = muons, jets, fatjets, met
                # apply corrections
                obj_copies = self.apply_object_corrections(obj_copies, [syst], direction=direction)
                varname = f"{syst['name']}_{direction}"
                hist_dict = self.apply_selection_and_fill(obj_copies, events, process, varname, hist_dict, xsec_weight, analysis, event_syst=syst, direction=direction)

        return hist_dict

def save_histograms(hist_dict, output_file="outputs/histograms/histograms.root", add_offset=False):
    """
    Save histograms to a specified directory.

    Parameters
    ----------
    hist_dict : dict
        Dictionary of histograms to save.
    output_dir : str
        Directory to save the histograms.
    """

    with uproot.recreate(output_file) as f:
        # save all available histograms to disk
        for channel, histogram in hist_dict.items():
            # optionally add minimal offset to avoid completely empty bins
            # (useful for the ML validation variables that would need binning adjustment
            # to avoid those)
            if add_offset:
                histogram += 1e-6
                # reference count for empty histogram with floating point math tolerance
                empty_hist_yield = histogram.axes[0].size*(1e-6)*1.01
            else:
                empty_hist_yield = 0

            for sample in histogram.axes[1]:
                for variation in histogram[:, sample, :].axes[1]:
                    variation_string = "" if variation == "nominal" else f"_{variation}"
                    current_1d_hist = histogram[:, sample, variation]

                    if sum(current_1d_hist.values()) > empty_hist_yield:
                        # only save histograms containing events
                        f[f"{channel}_{sample}{variation_string}"] = current_1d_hist

# -----------------------------
# Main Driver
# -----------------------------
def main():
    """
    Main driver function for running the Zprime analysis framework.
    Loads configuration, runs preprocessing, and dispatches analysis over datasets.
    """
    config = utils.configuration.config
    analysis = ZprimeAnalysis(config)
    fileset = construct_fileset(n_files_max_per_sample=MAX_FILES_PER_SAMPLE)

    for dataset, content in fileset.items():
        os.makedirs(f"output/{dataset}", exist_ok=True)
        metadata = content["metadata"]
        metadata["dataset"] = dataset

        logger.info("========================================")
        logger.info(f"🚀 Processing dataset: {dataset}")

        for idx, (file_path, tree) in enumerate(content["files"].items()):
            output_dir = f"output/{dataset}/file__{idx}/"
            if idx >= MAX_FILES_PER_SAMPLE:  continue

            if RUN_PREPROCESS:
                logger.info(f"🔍 Preprocessing input file: {file_path}")
                logger.info(f"➡️  Writing to: {output_dir}")
                pre_process(file_path, tree, output_dir)

            skimmed_files = glob.glob(f"{output_dir}/part*.root")
            skimmed_files = [f"{f}:{tree}" for f in skimmed_files]
            remaining = sum(uproot.open(f).num_entries for f in skimmed_files)
            logger.info(f"✅ Events retained after filtering: {remaining:,}")
            if RUN_HISTOGRAMS:
                for skimmed in skimmed_files:
                    logger.info(f"📘 Processing skimmed file: {skimmed}")
                    events = NanoEventsFactory.from_root(skimmed, schemaclass=NanoAODSchema, delayed=False).events()
                    result = analysis.process(events, metadata)
                    logger.info("📈 Histogram filling complete.")

        logger.info(f"🏁 Finished dataset: {dataset}\n")

    logger.info("✅ All datasets processed.")
    save_histograms(analysis.nD_hists_per_region, output_file="output/histograms/histograms.root")

    # logger.info("📊 Final histogram (Zprime_channel, signal):")
    # logger.info(analysis.nD_hists_per_region["Zprime_channel"][:, "signal", "nominal"])
    # logger.info(analysis.nD_hists_per_region["Zprime_channel"][:, "signal", "muon_id_sf_up"])
    # logger.info(analysis.nD_hists_per_region["Zprime_channel"][:, "signal", "pu_weight_up"])



if __name__ == "__main__":
    main()
