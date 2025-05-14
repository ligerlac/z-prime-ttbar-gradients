import numpy as np
import awkward as ak
import vector

from tensorflow.keras.models import load_model  # type: ignore

# -----------------------------
# Register backends
# -----------------------------
ak.jax.register_and_check()
vector.register_awkward()


def get_mtt(
    muons: ak.Array,
    jets: ak.Array,
    fatjets: ak.Array,
    met: ak.Array,
) -> ak.Array:
    """
    Calculate the invariant mass of the top quark pair (m_tt) from input objects.

    Parameters
    ----------
    muons : ak.Array
        Array of muon candidates.
    jets : ak.Array
        Array of jet candidates.
    fatjets : ak.Array
        Array of fatjet candidates.
    met : ak.Array
        Array of missing transverse energy (MET) information.

    Returns
    -------
    ak.Array
        Flattened array of reconstructed m_tt values per event.
    """
    region_muons_4vec, region_fatjets_4vec, region_jets_4vec = [
        ak.zip(
            {"pt": o.pt, "eta": o.eta, "phi": o.phi, "mass": o.mass},
            with_name="Momentum4D",
        )
        for o in [muons, fatjets, jets[:, 0]]
    ]
    region_met_4vec = ak.zip(
        {
            "pt": met.pt,
            "eta": 0 * met.pt,
            "phi": met.phi,
            "mass": 0,
        },
        with_name="Momentum4D",
    )

    mtt = ak.flatten(
        (
            region_muons_4vec
            + region_fatjets_4vec
            + region_jets_4vec
            + region_met_4vec
        ).mass
    )

    return mtt


def ttbar_reco(
    muons: ak.Array,
    jets: ak.Array,
    fatjets: ak.Array,
    met: ak.Array,
) -> tuple[ak.Array, ak.Array]:
    """
    Perform semi-leptonic ttbar reconstruction using fatjets if available,
    falling back to combinatoric jet assignment otherwise.

    Parameters
    ----------
    muons : ak.Array
        Array of muon candidates.
    jets : ak.Array
        Array of jet candidates.
    fatjets : ak.Array
        Array of fatjet candidates.
    met : ak.Array
        Array of MET information.

    Returns
    -------
    tuple[ak.Array, ak.Array]
        Tuple of (chi2, mtt) values for the best ttbar reconstruction per event.
    """
    # Define Gaussian means and widths for chi2 computation
    mean_mlep = 172.5
    sigma_mlep = 20.0
    mean_mhad = 172.5
    sigma_mhad = 20.0

    # Define which events have at least one fatjet
    has_fatjet = ak.num(fatjets, axis=1) > 0
    no_fatjet = ~has_fatjet

    # ============================================================
    # Fatjet-based reconstruction
    # ============================================================

    # Mask inputs to only fatjet-containing events
    fatjets_masked = ak.mask(fatjets, has_fatjet)
    jets_masked = ak.mask(jets, has_fatjet)
    muons_masked = ak.mask(muons, has_fatjet)
    met_masked = ak.mask(met, has_fatjet)

    # Use leading fatjet as hadronic top candidate
    leading_fatjet = fatjets_masked[:, 0]
    had_top_fj = ak.zip(
        {
            "pt": leading_fatjet.pt,
            "eta": leading_fatjet.eta,
            "phi": leading_fatjet.phi,
            "mass": leading_fatjet.mass,
        },
        with_name="Momentum4D",
    )
    # Use leading muon
    leading_muon = muons_masked[:, 0]

    # Broadcast other objects to jet axis
    fatjet_broadcast = ak.broadcast_arrays(jets_masked, leading_fatjet)[1]
    muon_broadcast = ak.broadcast_arrays(jets_masked, leading_muon)[1]
    met_broadcast = ak.broadcast_arrays(jets_masked, met_masked)[1]

    # Reject jets too close to fatjet (likely overlap with substructure)
    delta_r = jets_masked.deltaR(fatjet_broadcast)
    jet_mask = delta_r > 1.2

    valid_jets = jets_masked[jet_mask]
    valid_muons = muon_broadcast[jet_mask]
    valid_met = met_broadcast[jet_mask]

    # Construct 4-vectors for leptonic top components
    valid_muons_4vec, valid_jets_4vec = [
        ak.zip(
            {"pt": obj.pt, "eta": obj.eta, "phi": obj.phi, "mass": obj.mass},
            with_name="Momentum4D",
        )
        for obj in [valid_muons, valid_jets]
    ]
    valid_met_4vec = ak.zip(
        {
            "pt": valid_met.pt,
            "eta": 0 * valid_met.pt,
            "phi": valid_met.phi,
            "mass": 0,
        },
        with_name="Momentum4D",
    )

    # Leptonic top: muon + MET + jet
    lep_top_fj = valid_muons_4vec + valid_met_4vec + valid_jets_4vec
    had_mass_broadcast = ak.broadcast_arrays(lep_top_fj, had_top_fj)[1].mass

    # Chi² for each combination
    chi2_fj = ((lep_top_fj.mass - mean_mlep) / sigma_mlep) ** 2 + (
        (had_mass_broadcast - mean_mhad) / sigma_mhad
    ) ** 2

    # ============================================================
    # Combinatoric AK4 reconstruction (no fatjets)
    # ============================================================

    jets_nofj = ak.mask(jets, no_fatjet)
    muons_nofj = ak.mask(muons, no_fatjet)[:, 0]
    met_nofj = ak.mask(met, no_fatjet)

    # All jet pairs -> one jet for leptonic top, two for hadronic top
    combs = ak.combinations(jets_nofj, 2, axis=1, replacement=False)
    lepjet, hadjets = combs["0"], combs["1"]

    # Broadcast muon and MET to pair structure
    muon_broadcast = ak.broadcast_arrays(lepjet, muons_nofj)[1]
    met_broadcast = ak.broadcast_arrays(lepjet, met_nofj)[1]

    # Leptonic top 4-vectors
    lepjet_4vec = ak.zip(
        {"pt": lepjet.pt, "eta": lepjet.eta, "phi": lepjet.phi, "mass": lepjet.mass},
        with_name="Momentum4D",
    )
    muon_4vec = ak.zip(
        {
            "pt": muon_broadcast.pt,
            "eta": muon_broadcast.eta,
            "phi": muon_broadcast.phi,
            "mass": muon_broadcast.mass,
        },
        with_name="Momentum4D",
    )
    met_4vec = ak.zip(
        {
            "pt": met_broadcast.pt,
            "eta": 0 * met_broadcast.pt,
            "phi": met_broadcast.phi,
            "mass": 0,
        },
        with_name="Momentum4D",
    )

    lep_top_nofj = lepjet_4vec + muon_4vec + met_4vec

    # Hadronic top = dijet system
    hadjets_4vec = ak.zip(
        {"pt": hadjets.pt, "eta": hadjets.eta, "phi": hadjets.phi, "mass": hadjets.mass},
        with_name="Momentum4D",
    )
    had_top_nofj = ak.sum(hadjets_4vec, axis=1)
    had_top_nofj = ak.zip(
        {
            "pt": had_top_nofj.pt,
            "eta": had_top_nofj.eta,
            "phi": had_top_nofj.phi,
            "mass": had_top_nofj.mass,
        },
        with_name="Momentum4D",
    )

    chi2_nofj = ((lep_top_nofj.mass - mean_mlep) / sigma_mlep) ** 2 + (
        (had_top_nofj.mass - mean_mhad) / sigma_mhad
    ) ** 2

    # ============================================================
    # Select best candidate and compute mtt
    # ============================================================

    best_idx_fj = ak.argmin(chi2_fj, axis=1, keepdims=True)
    best_idx_nofj = ak.argmin(chi2_nofj, axis=1, keepdims=True)

    # Use chi2 to select best pairing
    lep_top_fj_best = lep_top_fj[best_idx_fj]
    had_top_fj_best = ak.broadcast_arrays(lep_top_fj, had_top_fj)[1][best_idx_fj]

    lep_top_nofj_best = lep_top_nofj[best_idx_nofj]
    had_top_nofj_best = ak.broadcast_arrays(lep_top_nofj, had_top_nofj)[1][best_idx_nofj]

    # Choose reconstruction based on presence of fatjet
    lep_top = ak.flatten(ak.where(has_fatjet, lep_top_fj_best, lep_top_nofj_best))
    had_top = ak.flatten(ak.where(has_fatjet, had_top_fj_best, had_top_nofj_best))
    mtt = ak.fill_none((lep_top + had_top).mass, -1.0)

    # Final chi2 score
    chi2 = ak.where(has_fatjet, chi2_fj, chi2_nofj)
    best_idx = ak.argmin(chi2, axis=1, keepdims=True)
    best_chi2 = ak.fill_none(ak.flatten(chi2[best_idx]), 9999.0)

    return best_chi2, mtt


def mtt_from_ttbar_reco(ttbar_reco: ak.Array) -> ak.Array:
    """
    Extract mtt from the ttbar reconstruction result.

    Parameters
    ----------
    ttbar_reco : ak.Array
        Output of `ttbar_reco`, expected to be awkward array with an 'mtt' field.

    Returns
    -------
    ak.Array
        Array of mtt values.
    """
    return ttbar_reco.mtt

def compute_mva_vars(muons: ak.Array, jets: ak.Array) -> dict[str, np.ndarray]:
    """
    Extract MVA input features from muons and jets.

    Parameters
    ----------
    jets : ak.Array
        Array of jet candidates.
    muons : ak.Array
        Array of muon candidates.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping feature names to NumPy arrays.
    """
    assert ak.all(ak.num(jets, axis=1) >= 2), "Require at least 2 jets"
    assert ak.all(ak.num(muons, axis=1) == 1), "Require exactly 1 muon"

    d = {}
    d["n_jet"] = ak.num(jets, axis=1).to_numpy()
    d["leading_jet_mass"] = jets.mass[:, 0].to_numpy()
    d["subleading_jet_mass"] = jets.mass[:, 1].to_numpy()
    d["st"] = (ak.sum(jets.pt, axis=1) + ak.sum(muons.pt, axis=1)).to_numpy()
    d["leading_jet_btag_score"] = jets.btagDeepB[:, 0].to_numpy()
    d["subleading_jet_btag_score"] = jets.btagDeepB[:, 1].to_numpy()

    denominator = ak.sum(jets.px**2 + jets.py**2 + jets.pz**2, axis=1)
    s_zz = ak.sum(jets.pz * jets.pz, axis=1) / denominator
    d["S_zz"] = s_zz.to_numpy()

    muon_in_pair, jet_in_pair = ak.unzip(ak.cartesian([muons, jets]))
    delta_r = muon_in_pair.deltaR(jet_in_pair)
    min_delta_r = ak.min(delta_r, axis=1)
    d["deltaR"] = min_delta_r.to_numpy()

    min_delta_r_indices = ak.argmin(delta_r, axis=1, keepdims=True)
    angle = muons.deltaangle(jet_in_pair[min_delta_r_indices])
    d["pt_rel"] = (muons.p * np.sin(angle)).to_numpy().flatten()

    closest_jet_pt = jet_in_pair.pt[min_delta_r_indices]
    d["deltaR_times_pt"] = (min_delta_r * closest_jet_pt).to_numpy().flatten()

    for var, vals in d.items():
        print(f"{var} mean:: ", ak.mean(vals))

    return d

def get_mva_vars(muons: ak.Array, jets: ak.Array) -> ak.Array:
    d = compute_mva_vars(muons, jets)
    return tuple(d.values())

def get_n_jet(mva: ak.Array) -> ak.Array:
    return mva.n_jet

def get_leading_jet_mass(mva: ak.Array) -> ak.Array:
    return mva.leading_jet_mass

def get_subleading_jet_mass(mva: ak.Array) -> ak.Array:
    return mva.subleading_jet_mass

def get_st(mva: ak.Array) -> ak.Array:
    return mva.st

def get_leading_jet_btag_score(mva: ak.Array) -> ak.Array:
    return mva.leading_jet_btag_score

def get_subleading_jet_btag_score(mva: ak.Array) -> ak.Array:
    return mva.subleading_jet_btag_score

def get_S_zz(mva: ak.Array) -> ak.Array:
    return mva.S_zz

def get_deltaR(mva: ak.Array) -> ak.Array:
    return mva.deltaR

def get_pt_rel(mva: ak.Array) -> ak.Array:
    return mva.pt_rel

def get_deltaR_times_pt(mva: ak.Array) -> ak.Array:
    return mva.deltaR_times_pt

def compute_mva_scores(
    muons: ak.Array,
    jets: ak.Array,
    model_path: str = "output/model.keras",
) -> np.ndarray:
    """
    Compute MVA scores using a pre-trained Keras model.

    Parameters
    ----------
    muons : ak.Array
        Array of muon candidates.
    jets : ak.Array
        Array of jet candidates.
    model_path : str, optional
        Path to the Keras model file, by default "output/model.keras".

    Returns
    -------
    np.ndarray
        Flat array of MVA scores per event.
    """


    model = load_model(model_path, compile=False)
    scores = np.full(len(jets), -1.0, dtype=np.float32)

    idx = ((ak.num(jets, axis=1) >= 2) & (ak.num(muons, axis=1) == 1)).to_numpy()
    mva_vars = compute_mva_vars(muons[idx], jets[idx])
    print("before predict")
    for var, vals in mva_vars.items():
        print(var, ak.mean(vals))

    X = np.column_stack(list(mva_vars.values())).astype(float)
    print("X mean:: ", ak.mean(X, axis=0))
    scores[idx] = ak.flatten(model.predict(X, batch_size=1024))
    print("score mean:: ", ak.mean(scores))
    return scores


def get_mva_scores(mva: ak.Array) -> ak.Array:
    """
    Extract neural network score from an MVA output object.

    Parameters
    ----------
    mva : object
        Object containing the `nn_score` attribute.

    Returns
    -------
    ak.Array
        Neural network score array.
    """
    return mva.nn_score