import jax.numpy as jnp
import numpy as np

from user.cuts import (
    Zprime_hardcuts,
    Zprime_hardcuts_no_fj,
    Zprime_softcuts_jax_workshop,
    Zprime_workshop_cuts,
)
from user.observables import get_mtt, get_mva_vars
from user.systematics import jet_pt_resolution, jet_pt_scale
from user.skim import dataset_manager_config, skimming_config


# ==============================================================================
#  Observables Definition
# ==============================================================================

LIST_OF_VARS = [
    {
        "name": "workshop_mtt",
        "binning": "200,3000,20",
        "label": r"$M(t\bar{t})$ [GeV]",
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

# ==============================================================================
#  General Configuration
# ==============================================================================

general_config = {
<<<<<<< HEAD
    "lumi": 16400,
    "weights_branch": "genWeight",
    "max_files": -1,
    "analysis": "diff",
    "run_preprocessing": False,
    "run_histogramming": False,
    "run_statistics": False,
    "run_systematics": False,
    "run_plots_only": False,
    "run_mva_training": True,
    "read_from_cache": True,
    "output_dir": "outputs/traced_zprime_with_jax_nn/",
    "preprocessed_dir": "./preproc_uproot/z-prime-ttbar-data/",
    "processor": "uproot",
    "lumifile": "./corrections/Cert_271036-284044_13TeV_Legacy2016_"\
        "Collisions16_JSON.txt",
    "cache_dir": "/tmp/gradients_analysis/",
=======
        "lumi": 16400,
        "weights_branch": "genWeight",
        "max_files": -1,
        "analysis": "diff",
        "run_skimming": False,
        "run_histogramming": False,
        "run_statistics": False,
        "run_systematics": False,
        "run_plots_only": False,
        "run_mva_training": True,
        "run_metadata_generation": True,
        "read_from_cache": True,
        "output_dir": "outputs/traced_zprime_with_jax_nn/",
        "processor": "uproot",
        "lumifile": "./corrections/Cert_271036-284044_13TeV_Legacy2016_"\
            "Collisions16_JSON.txt",
        "cache_dir": "/tmp/gradients_analysis/",
        "processes": ["signal"]
>>>>>>> bfd419e (first go at improving skimming setup to work out of box)
}

# ==============================================================================
#  Preprocessing Configuration
# ==============================================================================

preprocess_config = {
<<<<<<< HEAD
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
=======
        "branches": {
            "Muon": ["pt", "eta", "phi", "mass", "miniIsoId", "tightId", "charge"],
            "FatJet": ["particleNet_TvsQCD", "pt", "eta", "phi", "mass"],
            "Jet": ["btagDeepB", "jetId", "pt", "eta", "phi", "mass"],
            "PuppiMET": ["pt", "phi"],
            "HLT": ["TkMu50"],
            "Pileup": ["nTrueInt"],
            "event": ["genWeight", "run", "luminosityBlock", "event"],
        },
        "ignore_missing": False,  # is this implemented?
        "mc_branches": {
            "event": ["genWeight", "luminosityBlock"],
            "Pileup": ["nTrueInt"],
        },
        "skimming": skimming_config,
>>>>>>> bfd419e (first go at improving skimming setup to work out of box)
}

# ==============================================================================
#  JAX / Differentiable Analysis Configuration
# ==============================================================================

jax_config = {
    "optimise": True,
    "learning_rate": 0.01,  # default learning rate
    "max_iterations": 50,
    "explicit_optimisation": True,
    "soft_selection": {
        "function": Zprime_softcuts_jax_workshop,
        "use": [
            ("Muon", "pt"),
            ("Jet", "btagDeepB"),
            ("PuppiMET", "pt"),
            ("wjets_vs_ttbar_nn", None),
        ],
    },
    "params": {
        "met_threshold": 50.0,
        "btag_threshold": 0.5,
        "lep_ht_threshold": 150.0,
        "kde_bandwidth": 10.0,
    },
    "param_updates": {
        # Thresholds: clip within physics-motivated bounds
        "met_threshold": lambda x, d: jnp.clip(x + d, 20.0, 150.0),
        "btag_threshold": lambda x, d: jnp.clip(x + d, 0.0, 3.0),
        "lep_ht_threshold": lambda x, d: jnp.clip(x + d, 50.0, 300.0),
        # KDE smoothing: keep bandwidth strictly positive and reasonably sized
        "kde_bandwidth": lambda x, d: jnp.clip(x + d, 1.0, 50.0),
    },
    "learning_rates": {
        "met_threshold": 1.0,
        "btag_threshold": 0.01,
        "lep_ht_threshold": 1.0,
        "kde_bandwidth": 0.1,
        "nn": 0.0005,
    },
}

# ==============================================================================
#  Baseline Selections & Masks
# ==============================================================================

baseline_selection_config = {
    "function": Zprime_hardcuts_no_fj,
    "use": [
        ("Muon", None),
        ("Jet", None),
    ],
}

good_object_masks_config = {
    "analysis": [
        {
            "object": "Muon",
            "function": lambda muons: (
                (muons.pt > 55)
                & (abs(muons.eta) < 2.4)
                & (muons.tightId)
                & (muons.miniIsoId > 1)
            ),
            "use": [("Muon", None)],
        },
        {
            "object": "Jet",
            "function": lambda jets: (
                (jets.jetId >= 4) & (jets.btagDeepB > 0.5)
            ),
            "use": [("Jet", None)],
        },
        {
            "object": "FatJet",
            "function": lambda fatjets: (
                (fatjets.pt > 500) & (fatjets.particleNet_TvsQCD > 0.5)
            ),
            "use": [("FatJet", None)],
        },
    ],
    "mva": [
        {
            "object": "Muon",
            "function": lambda muons: (
                (muons.pt > 55)
                & (abs(muons.eta) < 2.4)
                & (muons.tightId)
                & (muons.miniIsoId > 1)
            ),
            "use": [("Muon", None)],
        },
        {
            "object": "FatJet",
            "function": lambda fatjets: (
                (fatjets.pt > 500) & (fatjets.particleNet_TvsQCD > 0.5)
            ),
            "use": [("FatJet", None)],
        },
    ],
}

# ==============================================================================
#  Analysis Channels
# ==============================================================================

channels_config = [
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
        "selection": {
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
]

# ==============================================================================
#  Ghost Observables
# ==============================================================================

ghost_observables_config = [
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
        "works_with_jax": True,
    },
]

# ==============================================================================
#  MVA Configuration
# ==============================================================================

mva_config = [
    {
        "name": "wjets_vs_ttbar_nn",
        "use_in_diff": True,
        "epochs": 1000,
        "framework": "jax",  # keras/tf/... if TF need more info
        # (e.g. Model: Sequential layers: Dense)
        "validation_split": 0.2,
        "random_state": 42,
        "batch_size": None,
        "grad_optimisation": {
            "optimise": True,  # this will add weights to set of optimised parameters
            "learning_rate": 0.0005,  # learning rate for the MVA optimisation
        },
        "classes": [
            "wjets",
            {"ttbar": ("ttbar_semilep", "ttbar_had", "ttbar_lep")},
        ],
        "plot_classes": ["wjets", "ttbar", "signal"],
        "balance_strategy": "undersample",
        "layers": [
            {
                "ndim": 16,
                "activation": lambda x, w, b: jnp.tanh(
                    jnp.dot(x, w) + b
                ),  # if using TF, this should be a string (e.g. "relu")
                "weights": "W1",
                "bias": "b1",
            },
            {
                "ndim": 16,
                "activation": lambda x, w, b: jnp.tanh(jnp.dot(x, w) + b),
                "weights": "W2",
                "bias": "b2",
            },
            {
                "ndim": 1,
                "activation": lambda x, w, b: jnp.dot(x, w) + b,
                "weights": "W3",
                "bias": "b3",
            },
        ],
        "loss": lambda pred, y: (
            np.mean(
                jnp.maximum(pred, 0)
                - pred * y
                + jnp.log(1 + jnp.exp(-jnp.abs(pred)))
            )
        ),
        "features": [
            {
                "name": "n_jet",
                "label": r"$N_{jets}$",
                "function": lambda mva: mva.n_jet,
                "use": [("mva", None)],
                "scale": lambda x: x / 10.0,  # scale to [0, 1]
                "binning": "0,10,10",  # optional binning for pre-scaling
                # data scaled by "scale" for post-scaling data
            },
            {
                "name": "leading_jet_mass",
                "label": r"$m_{j_1}$ [GeV]",
                "function": lambda mva: mva.leading_jet_mass,
                "use": [("mva", None)],
                "scale": lambda x: x / 20.0,  # scale to [0, 1]
                "binning": "0,100,50",  # optional binning for pre-scaling
                # data scaled by "scale" for post-scaling data
            },
            {
                "name": "subleading_jet_mass",
                "label": r"$m_{j_2}$ [GeV]",
                "function": lambda mva: mva.subleading_jet_mass,
                "use": [("mva", None)],
                "binning": "0,50,25",  # optional binning for pre-scaling
                # data scaled by "scale" for post-scaling data
            },
        ],
    }
]

# ==============================================================================
#  Corrections & Systematics
# ==============================================================================

corrections_config = [
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
]

systematics_config = [
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
]

# ==============================================================================
#  Statistics Configuration
# ==============================================================================

statistics_config = {"cabinetry_config": "cabinetry/cabinetry_config.yaml"}

# ==============================================================================
#  Plotting Configuration
# ==============================================================================

plotting_config = {
    "output_dir": "plots/",
    "process_colors": {
        "ttbar_semilep": "#907AD6",
        "signal": "#DABFFF",
        "ttbar_lep": "#7FDEFF",
        "ttbar_had": "#2C2A4A",
        "wjets": "#72A1E5",
        "ttbar": "#907AD6",
    },
    "process_labels": {
        "ttbar_semilep": r"$t\bar{t}\,\textrm{(lepton+jets)}$",
        "signal": r"$Z^{\prime} \rightarrow t\bar{t}$",
        "ttbar_lep": r"$t\bar{t}\,\textrm{(leptonic)}$",
        "ttbar_had": r"$t\bar{t}\,\textrm{(hadronic)}$",
        "wjets": r"$W+\textrm{jets}$",
        "ttbar": r"$t\bar{t}$",
    },
    "process_order": [
        "ttbar",
        "ttbar_had",
        "ttbar_lep",
        "ttbar_semilep",
        "wjets",
        "signal",
    ],
    "jax": {
        "aux_param_labels": {
            "met_threshold": r"$E_{T}^{miss}$ threshold",
            "btag_threshold": r"$b$-tagging threshold",
            "lep_ht_threshold": r"$H_{T}^{lep}$ threshold",
            "kde_bandwidth": r"KDE bandwidth",
        },
        "fit_param_labels": {
            "mu": r"$\mu$",
            "scale_ttbar": r"$\kappa_{t\bar{t}}$",
        },
    },
}

# ==============================================================================
#  Final Configuration Assembly
# ==============================================================================

config = {
    "general": general_config,
    "preprocess": preprocess_config,
    "jax": jax_config,
    "baseline_selection": baseline_selection_config,
    "good_object_masks": good_object_masks_config,
    "channels": channels_config,
    "ghost_observables": ghost_observables_config,
    "mva": mva_config,
    "corrections": corrections_config,
    "systematics": systematics_config,
    "statistics": statistics_config,
    "plotting": plotting_config,
    "datasets": dataset_manager_config,
}
