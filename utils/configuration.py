import numpy as np

from utils.cuts import (
    Zprime_baseline,
    Zprime_workshop_selection,
)
from utils.observables import mtt_from_chi2, ttbar_chi2
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
                    "binning": "0,3000,50",
                    "label": r"m_{t\bar{t}}(\chi^2_{t\bar{t}})",
                    "function": mtt_from_chi2,
                    "use": [
                        ("Muon", None),
                        ("Jet", None),
                        ("FatJet", None),
                        ("PuppiMET", None),
                    ],
                    "works_with_jax": False,
                }
            ],
            "selection_function": Zprime_workshop_selection,
            "selection_use": [
                ("Muon", None),
                ("Jet", None),
                ("FatJet", None),
                ("PuppiMET", None),
                ("reco", None),
            ],
        },
    ],
    "ghost_observables": [
        {
            "names": "ttbar_chi2",
            "collections": "reco",
            "function": ttbar_chi2,
            "use": [
                ("Muon", None),
                ("Jet", None),
                ("FatJet", None),
                ("PuppiMET", None),
            ],
            "works_with_jax": False,
        },
        {
            "names": "ttbar_chi22",
            "collections": "reco",
            "function": ttbar_chi2,
            "use": [
                ("Muon", None),
                ("Jet", None),
                ("FatJet", None),
                ("PuppiMET", None),
            ],
            "works_with_jax": False,
        },
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
