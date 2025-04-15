from coffea import processor
from coffea.nanoevents import NanoAODSchema
from coffea.nanoevents import PackedSelection
import correctionlib
import hist
import gzip
import copy
import awkward as ak
import numpy as np

class ZprimeAnalysis(processor.ProcessorABC):
    def __init__(self):
        self.nD_hists_per_region = {}
        for channel in utils.configuration["channels"]:
            channel_name = channel["name"]
            channel_binning =  channel["observable_binning"]
            if isinstance(channel_binning, str):
                channel_binning = channel_binning.split(",")
                channel_binning = [float(i) for i in channel_binning]
                if len(channel_binning) != 3:
                    raise ValueError(f"Invalid binning format for {channel_name}: {channel_binning}. Expected format: 'low,high,n_bins'")

                nD_hist = (hist.Hist.new.Reg(channel_binning[2],
                                    channel_binning[0], channel_binning[1],
                                    name="observable",
                                    label=channel["observable_label"]
                                )
                                .StrCat([], name="process", label="Process", growth=True)
                                .StrCat([], name="variation", label="Systematic variation", growth=True)
                                .StrCat([], name="region", label="Region", growth=True)
                                .Weight() )
            else:
                nD_hist = (hist.Hist.new.Variable(channel_binning,
                                        name="observable",
                                        label=channel["observable_label"]
                                )
                                .StrCat([], name="process", label="Process", growth=True)
                                .StrCat([], name="variation", label="Systematic variation", growth=True)
                                .StrCat([], name="region", label="Region", growth=True)
                                .Weight() )


            self.Nd_hists_per_region[channel_name]= nD_hist

        self.systematics = utils.configuration["systematics"]
        self.corrlib_evaluators = {}
        for systematic in self.corrections:
            if not systematic["use_correctionlib"]: continue # like what?
            correction_filename =  systematic["file"]
            correction_name = systematic["name"]
            if correction_filename.endswith(".json.gz"):
                with gzip.open("./muon_Z.json.gz",'rt') as file:
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
            lhs += rhs
        elif op == "mult":
            lhs *= rhs

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
        hist_dict = copy.deepcopy(self.hist_dict)

        process = events.metadata["process"]  # "ttbar" etc.
        variation = events.metadata["variation"]  # "nominal" etc.

        # normalisation for MC
        x_sec = events.metadata["xsec"]
        n_generated_events = events.metadata["nevts"]
        lumi = 16400 # /pb
        if process != "data":
            xsec_weight = x_sec * lumi / n_generated_events
        else:
            xsec_weight = 1

        event_weight = events["genWeight"]

        # ============== Corrections to event weights ==============

        #=============== Systematics =========================
        for systematic_source in self.systematics+self.corrections+["nominal"]:

            if systematic_source == "nominal":
                continue # for now, this needs to do processing still

            # create copies of objects to modify in systematic variations
            # without touching the original objects
            object_copies = {
                "Electron": events.Electron,
                "Muon": events.Muon,
                "Jet": events.Jet,
                "FatJet": events.FatJet,
                "MET": events.MET,
            }
            # ========================= Workout what and how to vary ======================== #

            # Get arguments for corrections function
            systematics_fn_str_args = systematic_source.get("use", [])
            syst_fn_args = []
            for i, use in enumerate(systematics_fn_str_args):
                if isinstance(use, tuple) and len(use) == 2:
                    # the argument is a copy
                    arg = object_copies[use[0]][use[1]]
                else:
                    raise ValueError(f"Invalid argument for systematic {systematic_source['name']}: {use}")
                syst_fn_args.append(arg)

            # Get target branch (only one allowed) for corrections function
            # (only used for object systematics)
            syst_fn_target = systematic_source.get("target", None)
            if syst_fn_target is not None:
                if isinstance(use, tuple) and len(use) == 2:
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
            if systematic_source["type"] == "object":
                either_variations_present = True
                for direction in ["up", "down", "nominal"]:
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
                    electrons = object_copies["Electron"]
                    jets = object_copies["Jet"]
                    fatjets = object_copies["FatJet"]
                    met = object_copies["MET"]

                    lep_ht = muons.pt + met.pt
                    muons_reqs = ((muons.pt > 55) &
                                (ak.abs(muons.eta) > 2.4) &
                                (muons.tightId) &
                                (muons.iso > 1) #&
                                #(events["ht_lep"] > 150)
                                )
                    jets_reqs = ((jets.pt > 30) &
                                (ak.abs(jets.eta) < 2.4) &
                                (jets.isTightLeptonVeto) &
                                (jets.jetid >= 4)
                            )
                    fatjets_reqs = ((fatjets.pt > 200) &
                                    (ak.abs(fatjets.eta) < 2.4) &
                                    (fatjets.tag > 0.5)
                                    )

                    muons = muons[muons_reqs]
                    jets = jets[jets_reqs]
                    fatjets = fatjets[fatjets_reqs]

                    #========== Store boolean masks with PackedSelection ============#
                    selections = PackedSelection(dtype='uint64')
                    # Basic selection criteria
                    selections.add("exactly_1mu", ak.num(muons) == 1)
                    selections.add("pass_mu_trigger", events["HLT_TkMu50"])
                    selections.add("atleast_1b", ak.sum(jets.btag > 0.5, axis=1) > 0)
                    selections.add("met_cut", met.pt > 50)
                    selections.add("lep_ht_cut", lep_ht > 150)
                    selections.add("exactly_1fatjet", ak.num(fatjets) == 1)

                    # Complex selection criteria
                    selections.add("Zprime_channel", selections.all("pass_mu_trigger", "exactly_1mu", "met_cut", "lep_ht_cut", "atleast_1b", "exactly_1fatjet"))

                    for channel in utils.configuration["channels"]:
                        channel_name = channel["name"]
                        region_selection = selections.all(channel_name)

                        region_jets = jets[region_selection]
                        region_muons = muons[region_selection]
                        region_fatjets = fatjets[region_selection]
                        region_met = met[region_selection]
                        region_weights = np.ones(len(region_jets)) * xsec_weight



                if not either_variations_present:
                    raise ValueError(f"Systematic function not found for either up/down variations of {systematic_source['name']}")







        #=====================================================



        # jet energy scale / resolution systematics
        # need to adjust schema to instead use coffea add_systematic feature, especially for ServiceX
        # cannot attach pT variations to events.jet, so attach to events directly
        # and subsequently scale pT by these scale factors
        events["pt_scale_up"] = 1.03
        events["pt_res_up"] = utils.systematics.jet_pt_resolution(events.Jet.pt)

        syst_variations = ["nominal"]
        jet_kinematic_systs = ["pt_scale_up", "pt_res_up"]
        event_systs = [f"btag_var_{i}" for i in range(4)]
        if process == "wjets":
            event_systs.append("scale_var")

        # Only do systematics for nominal samples, e.g. ttbar__nominal
        if variation == "nominal":
            syst_variations.extend(jet_kinematic_systs)
            syst_variations.extend(event_systs)

        # for pt_var in pt_variations:
        for syst_var in syst_variations:
            ### event selection
            # very very loosely based on https://arxiv.org/abs/2006.13076

            # Note: This creates new objects, distinct from those in the 'events' object
            elecs = events.Electron
            muons = events.Muon
            jets = events.Jet
            if syst_var in jet_kinematic_systs:
                # Replace jet.pt with the adjusted values
                jets["pt"] = jets.pt * events[syst_var]



            electron_reqs = (elecs.pt > 30) & (np.abs(elecs.eta) < 2.1) & (elecs.cutBased == 4) & (elecs.sip3d < 4)
            muon_reqs = ((muons.pt > 30) & (np.abs(muons.eta) < 2.1) & (muons.tightId) & (muons.sip3d < 4) &
                         (muons.pfRelIso04_all < 0.15))
            jet_reqs = (jets.pt > 30) & (np.abs(jets.eta) < 2.4) & (jets.isTightLeptonVeto)

            # Only keep objects that pass our requirements
            elecs = elecs[electron_reqs]
            muons = muons[muon_reqs]
            jets = jets[jet_reqs]

            if self.use_inference:
                even = (events.event%2==0)  # whether events are even/odd

            B_TAG_THRESHOLD = 0.5

            ######### Store boolean masks with PackedSelection ##########
            selections = PackedSelection(dtype='uint64')
            # Basic selection criteria
            selections.add("exactly_1l", (ak.num(elecs) + ak.num(muons)) == 1)
            selections.add("atleast_4j", ak.num(jets) >= 4)
            selections.add("exactly_1b", ak.sum(jets.btagCSVV2 > B_TAG_THRESHOLD, axis=1) == 1)
            selections.add("atleast_2b", ak.sum(jets.btagCSVV2 > B_TAG_THRESHOLD, axis=1) >= 2)
            # Complex selection criteria
            selections.add("4j1b", selections.all("exactly_1l", "atleast_4j", "exactly_1b"))
            selections.add("4j2b", selections.all("exactly_1l", "atleast_4j", "atleast_2b"))

            for region in ["4j1b", "4j2b"]:
                region_selection = selections.all(region)
                region_jets = jets[region_selection]
                region_elecs = elecs[region_selection]
                region_muons = muons[region_selection]
                region_weights = np.ones(len(region_jets)) * xsec_weight
                if self.use_inference:
                    region_even = even[region_selection]

                if region == "4j1b":
                    observable = ak.sum(region_jets.pt, axis=-1)

                elif region == "4j2b":

                    # reconstruct hadronic top as bjj system with largest pT
                    trijet = ak.combinations(region_jets, 3, fields=["j1", "j2", "j3"])  # trijet candidates
                    trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3  # calculate four-momentum of tri-jet system
                    trijet["max_btag"] = np.maximum(trijet.j1.btagCSVV2, np.maximum(trijet.j2.btagCSVV2, trijet.j3.btagCSVV2))
                    trijet = trijet[trijet.max_btag > B_TAG_THRESHOLD]  # at least one-btag in trijet candidates
                    # pick trijet candidate with largest pT and calculate mass of system
                    trijet_mass = trijet["p4"][ak.argmax(trijet.p4.pt, axis=1, keepdims=True)].mass
                    observable = ak.flatten(trijet_mass)

                    if sum(region_selection)==0:
                        continue

                    if self.use_inference:
                        features, perm_counts = utils.ml.get_features(
                            region_jets,
                            region_elecs,
                            region_muons,
                            max_n_jets=utils.config["ml"]["MAX_N_JETS"],
                        )
                        even_perm = np.repeat(region_even, perm_counts)

                        # calculate ml observable
                        if self.use_triton:
                            results = utils.ml.get_inference_results_triton(
                                features,
                                even_perm,
                                triton_client,
                                utils.config["ml"]["MODEL_NAME"],
                                utils.config["ml"]["MODEL_VERSION_EVEN"],
                                utils.config["ml"]["MODEL_VERSION_ODD"],
                            )

                        else:
                            results = utils.ml.get_inference_results_local(
                                features,
                                even_perm,
                                utils.ml.model_even,
                                utils.ml.model_odd,
                            )

                        results = ak.unflatten(results, perm_counts)
                        features = ak.flatten(ak.unflatten(features, perm_counts)[
                            ak.from_regular(ak.argmax(results,axis=1)[:, np.newaxis])
                        ])
                syst_var_name = f"{syst_var}"
                # Break up the filling into event weight systematics and object variation systematics
                if syst_var in event_systs:
                    for i_dir, direction in enumerate(["up", "down"]):
                        # Should be an event weight systematic with an up/down variation
                        if syst_var.startswith("btag_var"):
                            i_jet = int(syst_var.rsplit("_",1)[-1])   # Kind of fragile
                            wgt_variation = self.cset["event_systematics"].evaluate("btag_var", direction, region_jets.pt[:,i_jet])
                        elif syst_var == "scale_var":
                            # The pt array is only used to make sure the output array has the correct shape
                            wgt_variation = self.cset["event_systematics"].evaluate("scale_var", direction, region_jets.pt[:,0])
                        syst_var_name = f"{syst_var}_{direction}"
                        hist_dict[region].fill(
                            observable=observable, process=process,
                            variation=syst_var_name, weight=region_weights * wgt_variation
                        )
                        if region == "4j2b" and self.use_inference:
                            for i in range(len(utils.config["ml"]["FEATURE_NAMES"])):
                                ml_hist_dict[utils.config["ml"]["FEATURE_NAMES"][i]].fill(
                                    observable=features[..., i], process=process,
                                    variation=syst_var_name, weight=region_weights * wgt_variation
                                )
                else:
                    # Should either be 'nominal' or an object variation systematic
                    if variation != "nominal":
                        # This is a 2-point systematic, e.g. ttbar__scaledown, ttbar__ME_var, etc.
                        syst_var_name = variation
                    hist_dict[region].fill(
                        observable=observable, process=process,
                        variation=syst_var_name, weight=region_weights
                    )
                    if region == "4j2b" and self.use_inference:
                        for i in range(len(utils.config["ml"]["FEATURE_NAMES"])):
                            ml_hist_dict[utils.config["ml"]["FEATURE_NAMES"][i]].fill(
                                observable=features[..., i], process=process,
                                variation=syst_var_name, weight=region_weights
                            )


        output = {"nevents": {events.metadata["dataset"]: len(events)}, "hist_dict": hist_dict}
        if self.use_inference:
            output["ml_hist_dict"] = ml_hist_dict

        return {}

    def postprocess(self, accumulator):
        return accumulator