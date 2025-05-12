# load the data from the uproot files
import uproot
import warnings
import numpy as np
import awkward as ak
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory


def get_mva_vars(jets: ak.Array, muons: ak.Array) -> dict[str, np.ndarray]:
    """Get MVA variables from jets and muons.
    Args:
        jets (ak.Array): Jets array.
        muons (ak.Array): Muons array.
    Returns:
        dict[str, np.ndarray]: Dictionary of MVA variables (flat numpy arrays)
    """
    assert ak.all(ak.num(jets, axis=1) >= 2), "Require at least 2 jets for MVA variables"
    assert ak.all(ak.num(muons, axis=1) == 1), "Require exactly 1 muon for MVA variables"

    d = {}

    # number of jets
    d["n_jet"] = ak.num(jets, axis=1).to_numpy()

    # leading and subleading jet mass
    d["leading_jet_mass"] = jets.mass[:, 0].to_numpy()
    d["subleading_jet_mass"] = jets.mass[:, 1].to_numpy()

    # scalar sum ST
    d["st"] = (
        ak.sum(jets.pt, axis=1) + ak.sum(muons.pt, axis=1)
    ).to_numpy()

    # leading and subleading jet b-tag score
    d["leading_jet_btag_score"] = jets.btagDeepB[:, 0].to_numpy()
    d["subleading_jet_btag_score"] = jets.btagDeepB[:, 1].to_numpy()

    # Sphericity tensor (only zz component)
    denominator = ak.sum(jets.px**2 + jets.py**2 + jets.pz**2, axis=1)
    s_zz = ak.sum(jets.pz * jets.pz, axis=1) / denominator
    d["S_zz"] = s_zz.to_numpy()

    # deltaR between muon and closest jet
    muon_in_pair, jet_in_pair = ak.unzip(ak.cartesian([muons, jets]))
    delta_r = muon_in_pair.deltaR(jet_in_pair)
    min_delta_r = ak.min(delta_r, axis=1)
    d["deltaR"] = min_delta_r.to_numpy()

    # transverse momentum of the muon w.r.t. the axis of the nearest jet (pt_rel)
    min_delta_r_indices = ak.argmin(delta_r, axis=1, keepdims=True)
    angle = muons.deltaangle(jet_in_pair[min_delta_r_indices])
    d["pt_rel"] = (muons.p * np.sin(angle)).to_numpy().flatten()

    # deltaR between muon and closest jet times the jet pt
    min_delta_r_indices = ak.argmin(delta_r, axis=1, keepdims=True)
    closest_jet_pt = jet_in_pair.pt[min_delta_r_indices]
    d["deltaR_times_pt"] = (min_delta_r * closest_jet_pt).to_numpy().flatten()

    return d


if __name__ == "__main__":

    NanoAODSchema.warn_missing_crossrefs = False
    warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")

    file_dict = {
        "wjets": "preproc_uproot/z-prime-ttbar-data/wjets__nominal/file__0/part0.root",
        "ttbar_had": "preproc_uproot/z-prime-ttbar-data/ttbar_had__nominal/file__0/part0.root",
        "ttbar_lep": "preproc_uproot/z-prime-ttbar-data/ttbar_lep__nominal/file__0/part0.root",
        "ttbar_semilep": "preproc_uproot/z-prime-ttbar-data/ttbar_semilep__nominal/file__0/part0.root",
    }

    events = NanoEventsFactory.from_root(
        f"{file_dict['wjets']}:Events", schemaclass=NanoAODSchema, delayed=False,
    ).events()

    events = events[ak.num(events.Jet, axis=1) >= 2]
    events = events[ak.num(events.Muon, axis=1) == 1]

    mva_vars = get_mva_vars(events.Jet, events.Muon)
    print(mva_vars)
