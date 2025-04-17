#!/usr/bin/env python3

import os
import json
import gzip
import copy
import numpy as np
import awkward as ak
import uproot
import glob
import hist
from correctionlib import CorrectionSet
from coffea.analysis_tools import PackedSelection

import utils  # assuming utils/configuration.py and utils/input_files.py are available
from utils.input_files import construct_fileset
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")
import dask_awkward as dak
NanoAODSchema.warn_missing_crossrefs = False

RUN_PREPROCESS = False

def build_branches_to_keep():
    branches_to_keep = {
    "Muon": [
        "pt",
        "eta",
        "phi",
        "mass",
        "miniIsoId",
        "tightId",
        "charge",
    ],
    "FatJet": [
        "particleNet_TvsQCD",
        "pt",
        "eta",
        "phi",
        "mass",
        "charge",
    ],
    "Jet": [
        "btagDeepB",
        "jetId",
        "pt",
        "eta",
        "phi",
        "mass",
        "charge",
    ],
    "PuppiMET": [
        "pt",
        "phi",
    ],
    "HLT": [
        "TkMu50",
    ],
    "PileUp":
        ["nTrueInt"],
    "event": [
        "genWeight",
        "event",
        "run",
        "luminosityBlock",
    ],
}

    return branches_to_keep

def pre_process(input_path, tree, output_path, step_size=100_000):


    with uproot.open(input_path + f":{tree}") as f:
        total_events = f.num_entries
        events = f
    print("=========")
    print("Total number of events:", total_events)

    branches_to_keep = build_branches_to_keep()

    selected = None
    for start in range(0, total_events, step_size):
        stop = min(start + step_size, total_events)

        events = NanoEventsFactory.from_root(
            {input_path: tree},
            schemaclass=NanoAODSchema,
            entry_start=start,
            entry_stop=stop,
            delayed=True,
        ).events()

        muons = events.Muon
        met = events.PuppiMET
        pass_trigger = events.HLT.TkMu50
        mu_sel = (muons.pt > 55) & (abs(muons.eta) < 2.4) & muons.tightId & (muons.miniIsoId > 1)
        n_mu = ak.sum(mu_sel, axis=1)
        exactly_1mu = (n_mu == 1)
        met_above_50 = (met.pt > 50)

        mask = pass_trigger & exactly_1mu & met_above_50
        filtered = events[mask]

        subset = {}
        for obj, obj_branches_to_keep in branches_to_keep.items():
            if obj == "event":
                subset.update({
                    br: filtered[br]
                    for br in obj_branches_to_keep if br in filtered.fields
                })
            else:
                if obj in filtered.fields:
                    subset.update({
                        f"{obj}_{br}": filtered[obj][br]
                        for br in obj_branches_to_keep if br in filtered[obj].fields
                    })

        filtered = dak.zip(subset, depth_limit=1)
        selected = filtered if selected is None else ak.concatenate([selected, filtered])

    uproot.dask_write(selected, destination=output_path, compute=True,tree_name=tree)

    return total_events

class ZprimeAnalysis:
    def __init__(self, config):
        self.config = config
        self.channels = config["channels"]
        self.systematics = config["systematics"]
        self.corrections = config["corrections"]
        self.corrlib_evaluators = self._load_correctionlib()
        self.nD_hists_per_region = self._init_histograms()

    def _init_histograms(self):
        nD_hists_per_region = {}
        for channel in self.channels:
            channel_name = channel["name"]
            channel_binning = channel["observable_binning"]
            if isinstance(channel_binning, str):
                channel_binning = list(map(float, channel_binning.split(",")))
                if len(channel_binning) != 3:
                    raise ValueError(f"Invalid binning format for {channel_name}: {channel_binning}. Expected format: 'low,high,n_bins'")
                nD_hist = hist.Hist(
                    hist.axis.Regular(int(channel_binning[2]), channel_binning[0], channel_binning[1], name="observable", label=channel["observable_label"]),
                    hist.axis.StrCategory([], name="process", label="Process", growth=True),
                    hist.axis.StrCategory([], name="variation", label="Systematic variation", growth=True),
                    storage=hist.storage.Weight()
                )
            else:
                nD_hist = hist.Hist(
                    hist.axis.Variable(channel_binning, name="observable", label=channel["observable_label"]),
                    hist.axis.StrCategory([], name="process", label="Process", growth=True),
                    hist.axis.StrCategory([], name="variation", label="Systematic variation", growth=True),
                    storage=hist.storage.Weight()
                )
            nD_hists_per_region[channel_name] = nD_hist
        return nD_hists_per_region

    def _load_correctionlib(self):
        evaluators = {}
        for systematic in self.corrections:
            if not systematic.get("use_correctionlib"):
                continue
            name = systematic["name"]
            filename = systematic["file"]
            if filename.endswith(".json.gz"):
                with gzip.open(filename, 'rt') as f:
                    data = f.read().strip()
                    evaluators[name] = CorrectionSet.from_string(data)
            elif filename.endswith(".json"):
                evaluators[name] = CorrectionSet.from_file(filename)
            else:
                raise ValueError(f"Unsupported correctionlib file format: {filename}")
        return evaluators

    def apply_op(self, op, lhs, rhs):
        if op == "add":
            return lhs + rhs
        elif op == "mult":
            return lhs * rhs
        else:
            raise ValueError(f"Unsupported operation: {op}")

    def apply_syst_fn(self, syst_name, syst_fn, syst_fn_args, syst_fn_affects, syst_fn_op):
        return self.apply_op(syst_fn_op, syst_fn_affects, syst_fn(*syst_fn_args))

    def apply_correctionlib(self, corr_name, corr_key, direction, corr_args, corr_target=None, corr_op=None):
        corr = self.corrlib_evaluators[corr_name][corr_key].evaluate(*corr_args, direction)
        if corr_target is not None and corr_op is not None:
            return self.apply_op(corr_op, corr_target, corr)
        return corr

    def process(self, events, metadata):

        analysis = self.__class__.__name__
        hist_dict = copy.deepcopy(self.nD_hists_per_region)

        process = metadata["process"]
        variation = metadata["variation"]
        print("=========")
        print(f"{analysis}:: Processing {process} with variation {variation}")

        x_sec = metadata["xsec"]
        n_generated_events = metadata["nevts"]
        lumi = 16400  # /pb
        if process != "data":
            xsec_weight = x_sec * lumi / n_generated_events

        for systematic_source in self.systematics + self.corrections + [{"name": "nominal"}]:
            object_copies = {field: events[field] for field in events.fields}

            use_correctionlib = systematic_source.get("use_correctionlib", False)
            syst_fn_args = []

            if process != "data" and systematic_source != "nominal":
                for use in systematic_source.get("use", []):
                    if isinstance(use, tuple) and len(use) == 2:
                        syst_fn_args.append(object_copies[use[0]][use[1]])
                    else:
                        raise ValueError(f"Invalid 'use' field: {use}")

                target = systematic_source.get("target")
                syst_fn_affects = object_copies[target[0]][target[1]] if target else None
                syst_fn_op = systematic_source.get("op")

            directions = ["up", "down"] if systematic_source["name"] != "nominal" else ["nominal"]
            for direction in directions:
                suffix = f"_{direction}" if systematic_source['name'] != "nominal" else ""
                syst_variation_hist_name = f"{systematic_source['name']}{suffix}"

                if systematic_source["name"] != "nominal" and systematic_source["type"] == "object" and process != "data":
                    if use_correctionlib:
                        object_copies[target[0]][target[1]] = self.apply_correctionlib(systematic_source["name"], systematic_source["key"], direction, syst_fn_args, syst_fn_affects, syst_fn_op)
                    else:
                        fn = systematic_source.get(f"{direction}_function")
                        if fn:
                            object_copies[target[0]][target[1]] = self.apply_syst_fn(systematic_source["name"], fn, syst_fn_args, syst_fn_affects, syst_fn_op)

                muons = object_copies["Muon"]
                jets = object_copies["Jet"]
                fatjets = object_copies["FatJet"]
                met = object_copies["PuppiMET"]

                lep_ht = muons.pt + met.pt
                muons_reqs = ((muons.pt > 55) & (np.abs(muons.eta) < 2.4) & (muons.tightId) & (muons.miniIsoId > 1))
                jets_reqs = ((jets.pt > 30) & (np.abs(jets.eta) < 2.4) & (jets.isTightLeptonVeto) & (jets.jetId >= 4))
                fatjets_reqs = ((fatjets.pt > 200) & (np.abs(fatjets.eta) < 2.4) & (fatjets.particleNet_TvsQCD > 0.5))

                muons = muons[muons_reqs]
                jets = jets[jets_reqs]
                fatjets = fatjets[fatjets_reqs]

                selections = PackedSelection(dtype='uint64')
                selections.add("exactly_1mu", ak.num(muons, axis=1) == 1)
                selections.add("pass_mu_trigger", events.HLT.TkMu50)
                selections.add("atleast_1b", ak.sum(jets.btagDeepB > 0.5, axis=1) > 0)
                selections.add("met_cut", met.pt > 50)
                selections.add("lep_ht_cut", lep_ht[:,0] > 150)
                selections.add("exactly_1fatjet", ak.num(fatjets, axis=1) == 1)
                selections.add("dummy", ak.num(muons, axis=1) > -1)

                selections.add("Zprime_channel", selections.all("pass_mu_trigger", "exactly_1mu", "met_cut", "exactly_1fatjet", "lep_ht_cut", "atleast_1b"))
                selections.add("preselection", selections.all("dummy"))


                for channel in self.channels:
                    channel_name = channel["name"]
                    region_selection = selections.all(channel_name)

                    if ak.sum(region_selection) == 0:
                        print(f"{analysis}:: No events left in {channel_name} for {process} with variation {variation}")
                        continue

                    region_muons = muons[region_selection]
                    region_met = met[region_selection]
                    region_jets = jets[region_selection]
                    region_fatjets = fatjets[region_selection]

                    muons_4vec = ak.zip(
                        {
                            "pt": region_muons.pt,
                            "eta": region_muons.eta,
                            "phi": region_muons.phi,
                            "mass": region_muons.mass,
                        },
                        with_name="Momentum4D",
                    )
                    fatjets_4vec = ak.zip(
                        {
                            "pt": region_fatjets.pt,
                            "eta": region_fatjets.eta,
                            "phi": region_fatjets.phi,
                            "mass": region_fatjets.mass,
                        },
                        with_name="Momentum4D",
                    )
                    jets_4vec = ak.zip(
                        {
                            "pt": region_jets.pt,
                            "eta": region_jets.eta,
                            "phi": region_jets.phi,
                            "mass": region_jets.mass,
                        },
                        with_name="Momentum4D",
                    )
                    met_4vec = ak.zip(
                            {
                                "pt": region_met.pt,
                                "eta": 0*region_met.pt,
                                "phi": region_met.phi,
                                "mass": 0,
                            },
                            with_name="Momentum4D",
                        )

                    if process != "data":
                        event_weight = events[region_selection].genWeight
                        region_weights = event_weight / np.abs(event_weight) * xsec_weight
                    else:
                        region_weights = np.ones(len(region_met))

                    p4mu,p4fj,p4j,p4met = ak.unzip(ak.cartesian([muons_4vec, fatjets_4vec, jets_4vec[:,0], met_4vec]))
                    p4tot = p4mu + p4fj + p4j + p4met
                    mtt = p4tot.mass
                    observable = ak.flatten(mtt)

                    if systematic_source["name"] != "nominal" and systematic_source["type"] == "event" and process != "data":
                        if use_correctionlib:
                            region_weights = self.apply_correctionlib(systematic_source["name"], systematic_source["key"], direction, syst_fn_args, region_weights, syst_fn_op)
                        else:
                            fn = systematic_source.get(f"{direction}_function")
                            if fn:
                                region_weights = self.apply_syst_fn(systematic_source["name"], fn, syst_fn_args, region_weights, syst_fn_op)

                    if process == "data" and direction != "nominal":
                        continue

                    self.nD_hists_per_region[channel_name].fill(observable=observable, process=process, variation=syst_variation_hist_name, weight=region_weights)
                    # do we need a fill?
                    hist_dict[channel_name].fill(observable=observable, process=process, variation=syst_variation_hist_name, weight=region_weights)

        return {"nevents": {metadata["dataset"]: len(events)}, "hist_dict": hist_dict}

def main():

    config = utils.configuration.config
    analysis = ZprimeAnalysis(config)

    fileset = construct_fileset(n_files_max_per_sample=2)

    for dataset, content in fileset.items():
        os.makedirs(f"output/{dataset}", exist_ok=True)

        metadata = content["metadata"]
        metadata["dataset"] = dataset
        print(f"==========================")
        print(f"Processing {dataset}")
        for f_idx, (file_path, tree) in enumerate(content["files"].items()):
            if RUN_PREPROCESS:
                # ============================================
                # Pre-processing
                # ============================================
                preproc_output_path = f"output/{dataset}/file__{f_idx}/"
                print(f"==========================")
                print(f"Pre-processing {file_path}")
                print(f"Writing skimmed data to {preproc_output_path}")
                pre_process(file_path, tree, f"{preproc_output_path}")


            # ======
            # compute remaining number of events
            # ======
            skimmed_files = glob.glob(f"output/{dataset}/file__{f_idx}/part*.root")
            skimmed_files = [f"{f}: {tree}" for f in skimmed_files]

            remaining_num_events = 0
            for f in skimmed_files:
                with uproot.open(f) as f:
                    remaining_num_events += f.num_entries

            print("=========")
            print(f"Number of events to be processed (after pre-processing): {remaining_num_events}")
            # for fraction we need number per file
            #print("Fraction of events kept relative to before pre-processing:", f"{(remaining_num_events*100 / total_num_events):.2f}%")
            print("=========")
            for file in skimmed_files:
                events = NanoEventsFactory.from_root(
                    file,
                    schemaclass=NanoAODSchema,
                    delayed=False,
                ).events()
                print("=========")
                print(f"Processing the skimmed file: {file}")
                result = analysis.process(events, metadata)
                print("=========")
                print("\n")
            print("==========================")
            print("\n\n")

        print("\n\n\n")

    print(analysis.nD_hists_per_region["Zprime_channel"][:, "signal", "nominal"])

if __name__ == "__main__":
    main()
