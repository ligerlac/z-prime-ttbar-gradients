import numpy as np
import jax.numpy as jnp

from utils.cuts import (
    Zprime_hardcuts,
    Zprime_softcuts_jax_workshop,
    Zprime_workshop_cuts
)
from utils.observables import (
    get_mtt,
)
from utils.systematics import (
    jet_pt_resolution,
    jet_pt_scale
)

LIST_OF_VARS = [
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
]

config = {
    "general": {
        "lumi": 16400,
        "weights_branch": "genWeight",
        "max_files": -1,
        "analysis": "nondiff",
        "run_preprocessing": False,
        "run_histogramming": False,
        "run_statistics": True,
        "run_systematics": False,
        "read_from_cache": True,
        "output_dir": "output/",
        "preprocessed_dir": "./preproc_uproot/z-prime-ttbar-data/",
        "processor": "uproot",
        "lumifile": "./corrections/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt",

    },
    "preprocess": {
        "branches": {
            "Muon": ["pt", "eta", "phi", "mass", "miniIsoId", "tightId", "charge"],
            "FatJet": ["particleNet_TvsQCD", "pt", "eta", "phi", "mass"],
            "Jet": ["btagDeepB", "jetId", "pt", "eta", "phi", "mass"],
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
    "jax": {
        "optimize": True,
        "learning_rate": 0.01,
        "max_iterations": 5,
        "soft_selection": {
            "function": Zprime_softcuts_jax_workshop,
            "use": [
                ("Muon", "pt"),
                ("Jet", "btagDeepB"),
                ("FatJet", "pt"),
                ("PuppiMET", "pt"),
            ],
        },
        "params": {
            'met_threshold': 50.0,
            'met_scale': 25.0,
            'btag_threshold': 0.5,
            'lep_ht_threshold': 150.0,
            'muon_weight': 1.0,
            'jet_weight': 0.1,
            'met_weight': 1.0,
            'kde_bandwidth': 10.0,
            # Process-specific scales (cross-section * luminosity / n_events)
            'signal_scale': 1.0,
            'ttbar_scale': 1.0,
            'wjets_scale': 1.0,
            'other_scale': 1.0,
            # Systematic uncertainties
            'signal_systematic': 0.05,  # 5% on signal
            'background_systematic': 0.1,  # 10% on background
        },
        "param_updates": {
            # Thresholds: clip within physics-motivated bounds
            "met_threshold": lambda x, d: jnp.clip(x + d, 20.0, 150.0),
            "btag_threshold": lambda x, d: jnp.clip(x + d, 0.1, 0.9),
            "lep_ht_threshold": lambda x, d: jnp.clip(x + d, 50.0, 300.0),

            # Sigmoid scaling factors: constrain to non-degenerate reasonable positive values
            "met_scale": lambda x, d: jnp.clip(x + d, 1.0, 100.0),

            # Feature weights: allow any positive value, enforce minimum
            "muon_weight": lambda x, d: jnp.maximum(x + d, 0.01),
            "jet_weight": lambda x, d: jnp.maximum(x + d, 0.01),
            "met_weight": lambda x, d: jnp.maximum(x + d, 0.01),

            # KDE smoothing: keep bandwidth strictly positive and reasonably sized
            "kde_bandwidth": lambda x, d: jnp.clip(x + d, 1.0, 50.0),

            # Process normalizations: float freely, but constrain to positive domain
            "signal_scale": lambda x, d: jnp.maximum(x + d, 0.01),
            "ttbar_scale": lambda x, d: jnp.maximum(x + d, 0.01),
            "wjets_scale": lambda x, d: jnp.maximum(x + d, 0.01),
            "other_scale": lambda x, d: jnp.maximum(x + d, 0.01),

            # Systematic uncertainties: must remain in [0, 1]
            "signal_systematic": lambda x, d: jnp.clip(x + d, 0.0, 1.0),
            "background_systematic": lambda x, d: jnp.clip(x + d, 0.0, 1.0),
        }
    },
    "baseline_selection": {
        "function": Zprime_hardcuts,
        "use": [
            ("Muon", None),
            ("Jet", None),
            ("FatJet", None),
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
    "channels": [
        {
            "name": "CMS_WORKSHOP_JAX",
            "fit_observable": "workshop_mtt",
            "observables": LIST_OF_VARS,
            "selection": {
                "function": Zprime_hardcuts,
                "use": [
                    ("Muon", None),
                    ("Jet", None),
                    ("FatJet", None),
                ],
            },
            "use_in_diff": True,
        },
        {
            "name": "CMS_WORKSHOP",
            "fit_observable": "workshop_mtt",
            "observables": LIST_OF_VARS,
            "selection":{
                "function": Zprime_workshop_cuts,
                "use": [
                    ("Muon", None),
                    ("Jet", None),
                    ("FatJet", None),
                    ("PuppiMET", None),
                ],
            },
            "use_in_diff": False,
        },
    ],
    "ghost_observables": [],
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
    "statistics": {
        "cabinetry_config": "cabinetry/cabinetry_config.yaml"
    },
}
