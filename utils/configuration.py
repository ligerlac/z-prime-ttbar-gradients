from utils.systematics import jet_pt_resolution
config = {
    "general": {
        "lumi": 16400,
        "weights_branch": "genWeight",
    },
    "channels": [
        {
        "name" : "Zprime_channel",
        "observable_name" : "m_tt",
        "observable_binning" : "0,3000,50",
        "observable_label" : r"$m_{t\bar{t}}$ [GeV]",
     }
    ],
    "corrections":[
        {
            "name": "pu_weight",
            "file": "puWeights.json.gz",
            "type": "event", # event or object
            "target": None, # if object this must be specfied
            "op": "mult", # or add or subtract
            "key": "Collisions16_UltraLegacy_goldenJSON",
            "use_correctionlib": True,
        },
        {
            "name": "muon_id_sf",
            "file": "muon_Z.json.gz",
            "type": "event_weight",
            "key": "NUM_TightID_DEN_TrackerMuons",
            "use_correctionlib": True,
        },
    ],
    "systematics": [
        {
            "name": "jet_pt_resolution",
            "up_function": jet_pt_resolution,
            "target": ("Jet", "pt")
            "use": [("Jet", "pt")]
            "symmetrise": True,
            "op": "mult", # or add or subtract
            "type": : "object"
        },
        {
            "name": "jet_pt_scale",
            "up_function": jet_pt_scale,
            "target": ("Jet", "pt")
            "symmetrise": True,
            "op": "mult", # or add or subtract
            "type": : "object"
        },
    ]
}