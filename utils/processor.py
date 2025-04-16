from coffea import processor
from coffea.analysis_tools import PackedSelection
import correctionlib
import hist
import gzip
import copy
import awkward as ak
import dask_awkward as dak
import numpy as np
import utils
import hist.dask as hda

class ZprimeAnalysis(processor.ProcessorABC):
    def __init__(self):
        self.nD_hists_per_region = {}
        for channel in utils.configuration.config["channels"]:
            channel_name = channel["name"]
            channel_binning =  channel["observable_binning"]
            if isinstance(channel_binning, str):
                channel_binning = channel_binning.split(",")
                channel_binning = [float(i) for i in channel_binning]
                if len(channel_binning) != 3:
                    raise ValueError(f"Invalid binning format for {channel_name}: {channel_binning}. Expected format: 'low,high,n_bins'")

                nD_hist = (hda.Hist.new.Reg(int(channel_binning[2]),
                                    channel_binning[0], channel_binning[1],
                                    name="observable",
                                    label=channel["observable_label"]
                                )
                                .StrCat([], name="process", label="Process", growth=True)
                                .StrCat([], name="variation", label="Systematic variation", growth=True)
                                .Weight() )
            else:
                nD_hist = (hda.Hist.new.Variable(channel_binning,
                                        name="observable",
                                        label=channel["observable_label"]
                                )
                                .StrCat([], name="process", label="Process", growth=True)
                                .StrCat([], name="variation", label="Systematic variation", growth=True)
                                .Weight() )


            self.nD_hists_per_region[channel_name]= nD_hist

        self.systematics = utils.configuration.config["systematics"]
        self.corrections = utils.configuration.config["corrections"]
        self.corrlib_evaluators = {}
        for systematic in self.corrections:
            if not systematic["use_correctionlib"]: continue # like what?
            correction_filename =  systematic["file"]
            correction_name = systematic["name"]
            if correction_filename.endswith(".json.gz"):
                with gzip.open(correction_filename,'rt') as file:
                    correction_data = file.read().strip()
                    self.corrlib_evaluators[correction_name] = correctionlib.CorrectionSet.from_string(correction_data)
            elif correction_filename.endswith(".json"):
                self.corrlib_evaluators[correction_name] = correctionlib.CorrectionSet.from_file(correction_filename)
            else:
                raise ValueError(f"Unsupported file format for {correction_filename}. Supported formats: .json, .json.gz")


    def apply_op(self, op, lhs, rhs):
        """
        Operate by rhs on lhs
        """
        if op == "add":
            result = lhs + rhs
        elif op == "mult":
            result = lhs*rhs

        return lhs

    def apply_syst_fn(self, syst_name, syst_fn, syst_fn_args, syst_fn_affects, syst_fn_op):
        """
        Apply the systematic function to the arguments
        """
        return self.apply_op(syst_fn_op, syst_fn_affects, syst_fn(*syst_fn_args))

    def apply_correctionlib(self, corr_name, corr_key, direction, corr_args, corr_target=None, corr_op=None):
        """
        Apply the correctionlib function to the arguments
        """
        print(corr_name, corr_key, corr_args)
        corr = self.corrlib_evaluators[corr_name][corr_key].evaluate(
                            *corr_args,
                            direction,
                        )

        if corr_target is not None and corr_op is not None:
            corr_target = self.apply_op(corr_op, corr_target, corr)
            return corr
        else:
            return corr

    def process(self, events):

        # create copies of histogram objects
        hist_dict = copy.deepcopy(self.nD_hists_per_region)

        process = events.metadata["process"]  # "ttbar" etc.

        variation = events.metadata["variation"]  # "nominal" etc.

        print(f"Processing {process} with variation {variation}")
        print(f"Fields: {events.fields}")

        # normalisation for MC
        x_sec = events.metadata["xsec"]
        n_generated_events = events.metadata["nevts"]
        lumi = 16400 # /pb
        if process != "data":
            xsec_weight = x_sec * lumi / n_generated_events

        # ============== Corrections to event weights ==============

        #=============== Systematics =========================
        for systematic_source in self.systematics+self.corrections:

            # create copies of objects to modify in systematic variations
            # without touching the original objects
            object_copies = {field: events[field] for field in events.fields}
            #     "Electron": events.Electron,
            #     "Muon": events.Muon,
            #     "Jet": events.Jet,
            #     "FatJet": events.FatJet,
            #     "PuppiMET": events.PuppiMET,
            # }
            # ========================= Workout what and how to vary ======================== #

            if process != "data":
                # Get arguments for corrections function
                systematics_fn_str_args = systematic_source.get("use", [])
                syst_fn_args = []
                for i, use in enumerate(systematics_fn_str_args):
                    if isinstance(use, tuple) and len(use) == 2:
                        # the argument is a copy
                        arg = object_copies[use[0]][use[1]]
                    else:
                        raise ValueError(f"Invalid argument for systematic {systematic_source['name']}: {use}")
                    print("x", arg)
                    syst_fn_args.append(arg)

                # Get target branch (only one allowed) for corrections function
                # (only used for object systematics)
                syst_fn_target = systematic_source.get("target", None)
                if syst_fn_target is not None:
                    if isinstance(syst_fn_target, tuple) and len(syst_fn_target) == 2:
                        syst_fn_affects = object_copies[syst_fn_target[0]][syst_fn_target[1]]
                    else:
                        raise ValueError(f"Invalid target for systematic {systematic_source['name']}: {syst_fn_target}")
                else:
                    syst_fn_affects = None

                # Get the operation to apply to the target branch as a result of the systematic
                # e.g. add, mult, subtract
                # (only used for object systematics)
                syst_fn_op = systematic_source.get("op", None) # how fn output affects target branch
                if syst_fn_op is not None and syst_fn_op not in ["add", "mult"]:
                    raise ValueError(f"Invalid operation for systematic {systematic_source['name']}: {syst_fn_op}")

                # workout if the systematic is applied via correctionlib
                use_correctionlib = systematic_source.get("use_correctionlib", False)

            # ====== Object systematics first because they affect cuts ========== #
            either_variations_present = True
            for direction in ["up", "down", "nominal"]:
                suffix = direction if direction != "nominal" else ""
                syst_variation_hist_name = f"{systematic_source['name']}_{suffix}"

                if systematic_source["type"] == "object" and process != "data":
                    if use_correctionlib:
                        object_copies[syst_fn_target[0]][syst_fn_target[1]] = self.apply_correctionlib(
                            systematic_source["name"],
                            systematic_source["key"],
                            direction,
                            syst_fn_args,
                            syst_fn_affects,
                            syst_fn_op,
                        )

                    else:
                        # Get the function to apply
                        syst_variation_fn = systematic_source.get(f"{direction}_function", None)
                        either_variations_present *= syst_variation_fn is not None
                        if syst_variation_fn is None: continue

                        object_copies[syst_fn_target[0]][syst_fn_target[1]] = self.apply_syst_fn(
                            systematic_source["name"],
                            syst_variation_fn,
                            syst_fn_args,
                            syst_fn_affects,
                            syst_fn_op
                        )


                # We now varied an objects kinematics, we can apply cuts and fill histograms
                # we use the object copies which we have modified
                muons = object_copies["Muon"]
                jets = object_copies["Jet"]
                fatjets = object_copies["FatJet"]
                met = object_copies["PuppiMET"]

                lep_ht = muons.pt + met.pt
                muons_reqs = ((muons.pt > 55) &
                            (np.abs(muons.eta) > 2.4) &
                            (muons.tightId) &
                            (muons.miniIsoId > 1) #&
                            #(events["ht_lep"] > 150)
                            )
                jets_reqs = ((jets.pt > 30) &
                            (np.abs(jets.eta) < 2.4) &
                            (jets.isTightLeptonVeto) &
                            (jets.jetId >= 4)
                        )
                fatjets_reqs = ((fatjets.pt > 200) &
                                (np.abs(fatjets.eta) < 2.4) &
                                (fatjets.particleNet_TvsQCD > 0.5)
                                )

                muons = muons[muons_reqs]
                jets = jets[jets_reqs]
                fatjets = fatjets[fatjets_reqs]

                #========== Store boolean masks with PackedSelection ============#
                selections = PackedSelection(dtype='uint64')
                # Basic selection criteria
                selections.add("exactly_1mu", ak.num(muons) == 1)
                selections.add("pass_mu_trigger", events.HLT.TkMu50)
                selections.add("atleast_1b", ak.sum(jets.btagDeepB > 0.5, axis=1) > 0)
                selections.add("met_cut", met.pt > 50)
                #selections.add("lep_ht_cut", lep_ht > 150)
                selections.add("exactly_1fatjet", ak.num(fatjets) == 1)

                # Complex selection criteria
                selections.add("Zprime_channel", selections.all("pass_mu_trigger", "exactly_1mu", "met_cut", "atleast_1b", "exactly_1fatjet")) #, "lep_ht_cut"

                for channel in utils.configuration.config["channels"]:
                    channel_name = channel["name"]
                    region_selection = selections.all(channel_name)

                    region_jets = jets[region_selection]
                    region_muons = muons[region_selection]
                    region_fatjets = fatjets[region_selection]
                    region_met = met[region_selection]

                    print(region_met.fields)

                    if process != "data":
                        event_weight = events[region_selection]["genWeight"]
                        region_weights = event_weight/np.abs(event_weight) * xsec_weight
                    else:
                        region_weights = dak.ones_like(region_met)

                    # compute the region observable
                    p4mu,p4fj,p4j = ak.unzip(ak.cartesian([region_muons, region_fatjets, region_jets[:,0]]))
                    p4tot = p4mu + p4fj + p4j #+ p4met
                    observable = dak.flatten(p4tot.mass)

                    # if this is a systematic affects event weight
                    if systematic_source["type"] == "event" and process != "data":
                        if use_correctionlib:
                            region_weights = self.apply_correctionlib(
                                systematic_source["name"],
                                systematic_source["key"],
                                direction,
                                syst_fn_args,
                                region_weights,
                                syst_fn_op
                            )
                        else:
                            # Get the function to apply
                            syst_variation_fn = systematic_source.get(f"{direction}_function", None)
                            either_variations_present *= syst_variation_fn is not None
                            if syst_variation_fn is None: continue

                            region_weights = self.apply_syst_fn(
                                systematic_source["name"],
                                syst_variation_fn,
                                syst_fn_args,
                                region_weights,
                                syst_fn_op
                            )

                    print("Filling histogram: ", syst_variation_hist_name, process)
                    if process == "data" and direction != "nominal":
                        # skip data for systematic variations
                        continue
                    elif process == "data" and direction == "nominal":
                        syst_variation_hist_name = ""

                    hist_dict[channel_name].fill(
                        observable=observable, process=process,
                        variation=syst_variation_hist_name, weight=region_weights
                    )

                if not either_variations_present:
                    raise ValueError(f"Systematic function not found for either up/down variations of {systematic_source['name']}")

        output = {"nevents": {events.metadata["dataset"]: len(events)}, "hist_dict": hist_dict}
        return output

    def postprocess(self, accumulator):
        return accumulator


# def numpy_from_dak(item):
#     seeds = (
#         ak.typetracer.length_one_if_typetracer(item).to_numpy()[[0, -1]].view("i4")
#     )
#     randomstate = ak.random.Generator(numpy.random.PCG64(seeds))

#     def getfunction(layout, depth, **kwargs):
#         if isinstance(layout, ak.contents.NumpyArray) or not isinstance(
#             layout, (ak.contents.Content,)
#         ):
#             return ak.contents.NumpyArray(
#                 randomstate.normal(size=len(layout)).astype(np.float32)
#             )
#         return None

#     out = ak.transform(
#         getfunction,
#         ak.typetracer.length_zero_if_typetracer(item),
#         behavior=item.behavior,
#     )
#     if ak.backend(item) == "typetracer":
#         out = awkward.Array(
#             out.layout.to_typetracer(forget_length=True), behavior=out.behavior
#         )

#     assert out is not None
#     return out