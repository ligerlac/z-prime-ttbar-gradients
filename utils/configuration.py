import numpy as np

from utils.cuts import (
    Zprime_baseline,
    Zprime_hardcuts,
    Zprime_softcuts_nonjax,
    Zprime_softcuts_nonjax_workshop,
    Zprime_softcuts_CR1,
    Zprime_softcuts_CR2,
    Zprime_softcuts_SR_notag,
    Zprime_softcuts_SR_tag,
)

from utils.observables import (
    chi2_from_ttbar_reco,
    compute_mva_scores,
    get_deltaR,
    get_deltaR_times_pt,
    get_leading_jet_btag_score,
    get_leading_jet_mass,
    get_mtt,
    get_mva_scores,
    get_mva_vars,
    get_n_jet,
    get_pt_rel,
    get_S_zz,
    get_st,
    get_subleading_jet_btag_score,
    get_subleading_jet_mass,
    mtt_from_ttbar_reco,
    ttbar_reco,
)
from utils.systematics import jet_pt_resolution, jet_pt_scale
LIST_OF_VARS = [
    # {
    #                 "name": "mva_score",
    #                 "binning": "-1,1,50",
    #                 "label": r"NN score",
    #                 "function": get_mva_scores,
    #                 "use": [
    #                     ("mva", None),
    #                 ],
    #                 "works_with_jax": False,
    #             },
                # {
                #     "name": "n_jet",
                #     "label": r"N_{\mathrm{jets}}",
                #     "binning": "0,14,10",
                #     "function": get_n_jet,
                #     "use": [("mva", None)],
                #     "works_with_jax": False,
                # },
                # {
                #     "name": "leading_jet_mass",
                #     "label": r"m_{j_1} [GeV]",
                #     "binning": "0,600,40",
                #     "function": get_leading_jet_mass,
                #     "use": [("mva", None)],
                #     "works_with_jax": False,
                # },
                # {
                #     "name": "subleading_jet_mass",
                #     "label": r"m_{j_2} [GeV]",
                #     "binning": "0,600,40",
                #     "function": get_subleading_jet_mass,
                #     "use": [("mva", None)],
                #     "works_with_jax": False,
                # },
                # {
                #     "name": "st",
                #     "label": r"S_T [GeV]",
                #     "binning": "0,3000,50",
                #     "function": get_st,
                #     "use": [("mva", None)],
                #     "works_with_jax": False,
                # },
                # {
                #     "name": "leading_jet_btag_score",
                #     "label": r"b-tag(j_1)",
                #     "binning": "-1,1,50",
                #     "function": get_leading_jet_btag_score,
                #     "use": [("mva", None)],
                #     "works_with_jax": False,
                # },
                # {
                #     "name": "subleading_jet_btag_score",
                #     "label": r"b-tag(j_2)",
                #     "binning": "-1,1,50",
                #     "function": get_subleading_jet_btag_score,
                #     "use": [("mva", None)],
                #     "works_with_jax": False,
                # },
                # {
                #     "name": "S_zz",
                #     "label": r"S_{zz}",
                #     "binning": "0,1,50",
                #     "function": get_S_zz,
                #     "use": [("mva", None)],
                #     "works_with_jax": False,
                # },
                # {
                #     "name": "deltaR",
                #     "label": r"\Delta R(\mu,\mathrm{jet})",
                #     "binning":  "0,7,50",
                #     "function": get_deltaR,
                #     "use": [("mva", None)],
                #     "works_with_jax": False,
                # },
                # {
                #     "name": "pt_rel",
                #     "label": r"p_T^{\mathrm{rel}} [GeV]",
                #     "binning": "0,500,50",
                #     "function": get_pt_rel,
                #     "use": [("mva", None)],
                #     "works_with_jax": False,
                # },
                # {
                #     "name": "deltaR_times_pt",
                #     "label": r"\Delta R \times p_T^{\mathrm{jet}}",
                #     "binning": "0,500,50",
                #     "function": get_deltaR_times_pt,
                #     "use": [("mva", None)],
                #     "works_with_jax": False,
                # },
                {
                    "name": "workshop_mtt",
                    "binning": "0,3000,50",
                    "label": r"M(t\bar{t}) [GeV]",
                    "function": get_mtt,
                    "use": [
                        ("Muon", None),
                        ("Jet", None),
                        ("FatJet", None),
                        ("PuppiMET", None),
                    ],
                    "works_with_jax": True,
                },
                # {
                #     "name": "ttbar_chi2",
                #     "binning": "0,200,50",
                #     "label": r"\chi^2(t\bar{t})",
                #     "function": chi2_from_ttbar_reco,
                #     "use": [
                #         ("ttbar_reco", None),
                #     ],
                #     "works_with_jax": True,
                # },
                # {
                #     "name": "mtt_chi2",
                #     "binning": "0,3000,50",
                #     "label": r"\chi^2(t\bar{t})",
                #     "function": mtt_from_ttbar_reco,
                #     "use": [
                #         ("ttbar_reco", None),
                #     ],
                #     "works_with_jax": True,
                # },
]

config = {
    "general": {
        "lumi": 16400,
        "weights_branch": "genWeight",
        "max_files": -1,
        "run_preprocessing": False,
        "run_histogramming": False,
        "run_statistics": True,
        "output_dir": "output/",
        "preprocessed_dir": "./preproc_uproot/z-prime-ttbar-data/",
        "processor": "uproot",
        "lumifile": "./corrections/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",
    },
    "baseline_selection": {
        "function": Zprime_baseline,
        "use": [
            ("Muon", None),
            ("Jet", None),
            ("FatJet", None),
            ("PuppiMET", None),
        ],
    },
    "good_object_masks": [
        {
            "object": "Muon",
            "function": lambda muons:   ((muons.pt > 55)
                                        & (abs(muons.eta) < 2.4)
                                        & (muons.tightId)
                                        & (muons.miniIsoId > 1)),
            "use": [("Muon", None)],
        },
        {
            "object": "Jet",
            "function": lambda jets: ((jets.jetId >= 4) & (jets.btagDeepB > 0.5)),
            "use": [("Jet", None)],
        },
        {
            "object": "FatJet",
            "function": lambda fatjets: ((fatjets.pt > 500)
                                         & (fatjets.particleNet_TvsQCD > 0.5)),
            "use": [("FatJet", None)],
        },
    ],
    "preprocess": {
        "branches": {
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
            ],
            "Jet": [
                "btagDeepB",
                "jetId",
                "pt",
                "eta",
                "phi",
                "mass",
            ],
            "PuppiMET": ["pt", "phi"],
            "HLT": ["TkMu50"],
            "Pileup": ["nTrueInt"],
            "event": ["genWeight", "run", "luminosityBlock"],
        },
        "ignore_missing": False,  # is this implemented?
        "mc_branches": {
            "event": ["genWeight", "luminosityBlock"],
            "Pileup": ["nTrueInt"],
        },
    },
    "statistics": {"cabinetry_config": "cabinetry/cabinetry_config.yaml"},
    "channels": [
        {
            "name": "CMS_WORKSHOP",
            "fit_observable": "workshop_mtt",
            "observables": LIST_OF_VARS,
            "selection_function": Zprime_hardcuts,
            "selection_use": [
                ("Muon", None),
            ],
            "soft_selection_function": Zprime_softcuts_nonjax_workshop,
            "soft_selection_use": [
                ("Muon", None),
                ("Jet", None),
                ("FatJet", None),
                ("PuppiMET", None),
            ],
        },
        # {
        #     "name": "baseline",
        #     "fit_observable": "mtt_chi2",
        #     "observables": LIST_OF_VARS,
        #     "selection_function": Zprime_baseline,
        #     "selection_use": [
        #         ("Muon", None),
        #         ("Jet", None),
        #         ("FatJet", None),
        #         ("PuppiMET", None),
        #     ],
        # },
    ],
    "ghost_observables": [
        # {
        #     "names": ("chi2", "mtt"),
        #     "collections": "ttbar_reco",
        #     "function": ttbar_reco,
        #     "use": [
        #         ("Muon", None),
        #         ("Jet", None),
        #         ("FatJet", None),
        #         ("PuppiMET", None),
        #     ],
        #     "works_with_jax": False,
        # },
        # {
        #     "names": "nn_score",
        #     "collections": "mva",
        #     "function": compute_mva_scores,
        #     "use": [
        #         ("Muon", None),
        #         ("Jet", None),
        #     ],
        # },
        # {
        #     "names": (
        #         "n_jet",
        #         "leading_jet_mass",
        #         "subleading_jet_mass",
        #         "st",
        #         "leading_jet_btag_score",
        #         "subleading_jet_btag_score",
        #         "S_zz",
        #         "deltaR",
        #         "pt_rel",
        #         "deltaR_times_pt",
        #     ),
        #     "collections": "mva",
        #     "function": get_mva_vars,
        #     "use": [
        #         ("Muon", None),
        #         ("Jet", None),
        #     ],
        #     "works_with_jax": False,
        # },
    ],
    "corrections": [
        {
            "name": "pu_weight",
            "file": "corrections/puWeights.json.gz",
            "type": "event",  # event or object
            "use": [("Pileup", "nTrueInt")],
            "op": "mult",  # or add or subtract
            "key": "Collisions16_UltraLegacy_goldenJSON",
            "use_correctionlib": True,
        },
        {
            "name": "muon_id_sf",
            "file": "corrections/muon_Z.json.gz",
            "use": [("Muon", "eta"), ("Muon", "pt")],
            "transform": lambda eta, pt: (np.abs(eta)[:, 0], pt[:, 0]),
            "type": "event",
            "key": "NUM_TightID_DEN_TrackerMuons",
            "use_correctionlib": True,
            "op": "mult",
            "up_and_down_idx": ["systup", "systdown"],
        },
    ],
    "systematics": [
        {
            "name": "jet_pt_resolution",
            "up_function": jet_pt_resolution,
            "target": ("Jet", "pt"),
            "use": [("Jet", "pt")],
            "symmetrise": True,  # not implemented
            "op": "mult",  # or add or subtract
            "type": "object",
        },
        {
            "name": "jet_pt_scale",
            "up_function": jet_pt_scale,
            "target": ("Jet", "pt"),
            "symmetrise": True,  # not implemented
            "op": "mult",  # or add or subtract
            "type": "object",
        },
    ],
}
