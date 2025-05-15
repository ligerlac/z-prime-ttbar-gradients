import numpy as np

from utils.cuts import (
    Zprime_baseline,
    Zprime_hardcuts,
    Zprime_softcuts_nonjax,
    Zprime_workshop_selection,
)
from utils.observables import (get_mva_vars, compute_mva_scores, mtt_from_ttbar_reco,
                               ttbar_reco, get_mva_scores, get_n_jet, get_leading_jet_mass,
                               get_subleading_jet_mass, get_st, get_leading_jet_btag_score,
                               get_subleading_jet_btag_score, get_S_zz, get_deltaR, get_pt_rel,
                               get_deltaR_times_pt)
from utils.systematics import jet_pt_resolution, jet_pt_scale

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
            "name": "Zprime_channel",
            "fit_observable": "mtt_chi2",
            "observables": [
                {
                    "name": "mtt_chi2",
                    "binning": "0,5000,50",
                    "label": r"m_{t\bar{t}}(\chi^2_{t\bar{t}})",
                    "function": mtt_from_ttbar_reco,
                    "use": [
                        ("ttbar_reco", None),
                    ],
                    "works_with_jax": True,
                },
                {
                    "name": "mva_score",
                    "binning": "0,1,50",
                    "label": r"NN score",
                    "function": get_mva_scores,
                    "use": [
                        ("mva", None),
                    ],
                    "works_with_jax": True,
                },
                {
                    "name": "n_jet",
                    "label": r"N_{\mathrm{jets}}",
                    "binning": "0,10,10",
                    "function": get_n_jet,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                },
                {
                    "name": "leading_jet_mass",
                    "label": r"m_{j_1} [GeV]",
                    "binning": "0,200,40",
                    "function": get_leading_jet_mass,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                },
                {
                    "name": "subleading_jet_mass",
                    "label": r"m_{j_2} [GeV]",
                    "binning": "0,200,40",
                    "function": get_subleading_jet_mass,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                },
                {
                    "name": "st",
                    "label": r"S_T [GeV]",
                    "binning": "0,2000,50",
                    "function": get_st,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                },
                {
                    "name": "leading_jet_btag_score",
                    "label": r"b-tag(j_1)",
                    "binning": "0,1,50",
                    "function": get_leading_jet_btag_score,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                },
                {
                    "name": "subleading_jet_btag_score",
                    "label": r"b-tag(j_2)",
                    "binning": "0,1,50",
                    "function": get_subleading_jet_btag_score,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                },
                {
                    "name": "S_zz",
                    "label": r"S_{zz}",
                    "binning": "0,1,50",
                    "function": get_S_zz,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                },
                {
                    "name": "deltaR",
                    "label": r"\Delta R(\mu,\mathrm{jet})",
                    "binning":  np.arange(0, 0.2, 0.002),
                    "function": get_deltaR,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                },
                {
                    "name": "pt_rel",
                    "label": r"p_T^{\mathrm{rel}} [GeV]",
                    "binning": np.arange(0, 10, 0.2),
                    "function": get_pt_rel,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                },
                {
                    "name": "deltaR_times_pt",
                    "label": r"\Delta R \times p_T^{\mathrm{jet}}",
                    "binning": np.arange(0, 10, 0.2),
                    "function": get_deltaR_times_pt,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                }
            ],
            "selection_function": Zprime_hardcuts,
            "selection_use": [
                ("Muon", None),
                ("Jet", None),
                ("FatJet", None),
                ("PuppiMET", None),
                ("ttbar_reco", None),
            ],
            "soft_selection_function": Zprime_softcuts_nonjax,
            "soft_selection_use": [
                ("Muon", None),
                ("Jet", None),
                ("FatJet", None),
                ("PuppiMET", None),
            ],
        },
        {
            "name": "baseline",
            "fit_observable": "mva_score",
            "observables": [
                                {
                    "name": "mva_score",
                    "binning": "0,1,50",
                    "label": r"NN score",
                    "function": get_mva_scores,
                    "use": [
                        ("mva", None),
                    ],
                    "works_with_jax": True,
                },
                {
                    "name": "n_jet",
                    "label": r"N_{\mathrm{jets}}",
                    "binning": "0,10,10",
                    "function": get_n_jet,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                },
                {
                    "name": "leading_jet_mass",
                    "label": r"m_{j_1} [GeV]",
                    "binning": "0,200,40",
                    "function": get_leading_jet_mass,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                },
                {
                    "name": "subleading_jet_mass",
                    "label": r"m_{j_2} [GeV]",
                    "binning": "0,200,40",
                    "function": get_subleading_jet_mass,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                },
                {
                    "name": "st",
                    "label": r"S_T [GeV]",
                    "binning": "0,2000,50",
                    "function": get_st,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                },
                {
                    "name": "leading_jet_btag_score",
                    "label": r"b-tag(j_1)",
                    "binning": "-1,1,50",
                    "function": get_leading_jet_btag_score,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                },
                {
                    "name": "subleading_jet_btag_score",
                    "label": r"b-tag(j_2)",
                    "binning": "-1,1,50",
                    "function": get_subleading_jet_btag_score,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                },
                {
                    "name": "S_zz",
                    "label": r"S_{zz}",
                    "binning": "0,1,50",
                    "function": get_S_zz,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                },
                {
                    "name": "deltaR",
                    "label": r"\Delta R(\mu,\mathrm{jet})",
                    "binning":  np.arange(0, 0.2, 0.002),
                    "function": get_deltaR,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                },
                {
                    "name": "pt_rel",
                    "label": r"p_T^{\mathrm{rel}} [GeV]",
                    "binning": np.arange(0, 10, 0.2),
                    "function": get_pt_rel,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                },
                {
                    "name": "deltaR_times_pt",
                    "label": r"\Delta R \times p_T^{\mathrm{jet}}",
                    "binning": np.arange(0, 10, 0.2),
                    "function": get_deltaR_times_pt,
                    "use": [("mva", None)],
                    "works_with_jax": False,
                }
            ],
            "selection_function": Zprime_baseline,
            "selection_use": [
                ("Muon", None),
                ("Jet", None),
                ("FatJet", None),
                ("PuppiMET", None),
            ],
        }
    ],
    "ghost_observables": [
        {
            "names": ("chi2", "mtt"),
            "collections": "ttbar_reco",
            "function": ttbar_reco,
            "use": [
                ("Muon", None),
                ("Jet", None),
                ("FatJet", None),
                ("PuppiMET", None),
            ],
            "works_with_jax": False,
        },
        {
            "names": "nn_score",
            "collections": "mva",
            "function": compute_mva_scores,
            "use": [
                ("Muon", None),
                ("Jet", None),
            ],
        },
        {
            "names": (
                "n_jet",
                "leading_jet_mass",
                "subleading_jet_mass",
                "st",
                "leading_jet_btag_score",
                "subleading_jet_btag_score",
                "S_zz",
                "deltaR",
                "pt_rel",
                "deltaR_times_pt",
            ),
            "collections": "mva",
            "function": get_mva_vars,
            "use": [
                ("Muon", None),
                ("Jet", None),
            ],
            "works_with_jax": False
        }
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
