import numpy as np
import awkward as ak
import vector

from tensorflow.keras.models import load_model

# -----------------------------
# Register backends
# -----------------------------
ak.jax.register_and_check()
vector.register_awkward()


def get_mtt(muons, jets, fatjets, met):
    """
    Calculate the invariant mass of the top quark pair (m_tt) using the
    four-momenta of the muons, jets, fatjets, and MET.
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


def get_mtt_sq(muons, jets, fatjets, met):
    """
    Calculate the invariant mass of the top quark pair (m_tt) using the
    four-momenta of the muons, jets, fatjets, and MET.
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

    return mtt**2

def get_mva_score(
    jets: ak.Array,
    muons: ak.Array,
    model_path: str = "output/model.keras",
) -> ak.array:
    """Get MVA score from jets and muons.
    Args:
        jets (ak.Array): Jets array.
        muons (ak.Array): Muons array.
        model_path (str): Path to the MVA model.
    Returns:
        np.ndarray: MVA score (flat numpy array)
    """
    def _get_mva_vars(jets: ak.Array, muons: ak.Array) -> dict[str, np.ndarray]:
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

    model = load_model(model_path)

    scores = np.zeros(len(jets), dtype=np.float32) - 1

    idx = ((ak.num(jets, axis=1) >= 2) & (ak.num(muons, axis=1) == 1)).to_numpy()

    mva_vars = _get_mva_vars(jets[idx], muons[idx])
    X = np.column_stack(list(mva_vars.values())).astype(float)

    scores[idx] = model.predict(X, batch_size=1024).flatten()

    return scores

def mtt_from_ttbar_reco(ttbar_reco):
    return ttbar_reco.mtt

def ttbar_reco(muons, jets, fatjets, met):
    # ==========================
    # ttbar reconstruction settings
    # ==========================
    mean_mlep = 172.5
    sigma_mlep = 20.0
    mean_mhad = 172.5
    sigma_mhad = 20.0

    has_fatjet = ak.num(fatjets, axis=1) > 0
    no_fatjet = ~has_fatjet

    # ==========================
    # Fatjet-based reconstruction
    # ==========================
    fatjets_masked = ak.mask(fatjets, has_fatjet)
    jets_masked = ak.mask(jets, has_fatjet)
    muons_masked = ak.mask(muons, has_fatjet)
    met_masked = ak.mask(met, has_fatjet)

    leading_fatjet = fatjets_masked[:, 0]
    leading_fatjet = ak.zip(
        {
            "pt": leading_fatjet.pt,
            "eta": leading_fatjet.eta,
            "phi": leading_fatjet.phi,
            "mass": leading_fatjet.mass,
        },
        with_name="Momentum4D",
    )
    had_top_fj = leading_fatjet
    leading_muon = muons_masked[:, 0]

    fatjet_broadcast = ak.broadcast_arrays(jets_masked, leading_fatjet)[1]
    muon_broadcast = ak.broadcast_arrays(jets_masked, leading_muon)[1]
    met_broadcast = ak.broadcast_arrays(jets_masked, met_masked)[1]

    delta_r = jets_masked.deltaR(fatjet_broadcast)
    jet_mask = delta_r > 1.2

    valid_jets = jets_masked[jet_mask]
    valid_muons = muon_broadcast[jet_mask]
    valid_met = met_broadcast[jet_mask]

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

    lep_top_fj = valid_muons_4vec + valid_met_4vec + valid_jets_4vec
    had_mass_broadcast = ak.broadcast_arrays(lep_top_fj, had_top_fj)[1].mass

    chi2_fj = ((lep_top_fj.mass - mean_mlep) / sigma_mlep) ** 2 + (
        (had_mass_broadcast - mean_mhad) / sigma_mhad
    ) ** 2

    # ==========================
    # Combinatoric reconstruction (no fatjet)
    # ==========================
    jets_nofj = ak.mask(jets, no_fatjet)
    muons_nofj = ak.mask(muons, no_fatjet)[:, 0]
    met_nofj = ak.mask(met, no_fatjet)

    combs = ak.combinations(jets_nofj, 2, axis=1, replacement=False)
    lepjet = combs["0"]
    hadjets = ak.unzip(combs)[1]

    muon_broadcast = ak.broadcast_arrays(lepjet, muons_nofj)[1]
    met_broadcast = ak.broadcast_arrays(lepjet, met_nofj)[1]

    lepjet_4vec = ak.zip(
        {
            "pt": lepjet.pt,
            "eta": lepjet.eta,
            "phi": lepjet.phi,
            "mass": lepjet.mass,
        },
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

    hadjets_4vec = ak.zip(
        {
            "pt": hadjets.pt,
            "eta": hadjets.eta,
            "phi": hadjets.phi,
            "mass": hadjets.mass,
        },
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

    # ==========================
    # Combine both cases using ak.fill_none
    # ==========================
    best_idx_fj = ak.argmin(chi2_fj, axis=1, keepdims=True)
    best_idx_nofj = ak.argmin(chi2_nofj, axis=1, keepdims=True)

    # Broadcast had_top_fj to match structure of lep_top_fj
    had_top_fj_broadcasted = ak.broadcast_arrays(lep_top_fj, had_top_fj)[1]
    had_top_nofj_broadcasted = ak.broadcast_arrays(lep_top_nofj, had_top_nofj)[
        1
    ]

<<<<<<< HEAD
    return mtt_combined
=======
    # Best per-event combinations using best_idx (axis=1)
    lep_top_fj_best = lep_top_fj[best_idx_fj]
    had_top_fj_best = had_top_fj_broadcasted[best_idx_fj]

    lep_top_nofj_best = lep_top_nofj[best_idx_nofj]
    had_top_nofj_best = had_top_nofj_broadcasted[best_idx_nofj]

    # Select final top candidates per event
    lep_top = ak.flatten(
        ak.where(has_fatjet, lep_top_fj_best, lep_top_nofj_best)
    )
    had_top = ak.flatten(
        ak.where(has_fatjet, had_top_fj_best, had_top_nofj_best)
    )
    mtt = (lep_top + had_top).mass
    mtt = ak.fill_none(mtt, -1.0)

<<<<<<< HEAD
    return mtt
>>>>>>> 1ad5dfd (big one: implement ghost observables, baseline cuts and fixt ttbar reco method)
=======
    chi2 = ak.where(has_fatjet, chi2_fj, chi2_nofj)
    best_idx = ak.argmin(chi2, axis=1, keepdims=True)
    best_chi2 = ak.fill_none(ak.flatten(chi2[best_idx]), 9999.0)

    return best_chi2, mtt

def ttbar_reco_single_out(muons, jets, fatjets, met):
    # ==========================
    # ttbar reconstruction settings
    # ==========================
    mean_mlep = 172.5
    sigma_mlep = 20.0
    mean_mhad = 172.5
    sigma_mhad = 20.0

    has_fatjet = ak.num(fatjets, axis=1) > 0
    no_fatjet = ~has_fatjet

    # ==========================
    # Fatjet-based reconstruction
    # ==========================
    fatjets_masked = ak.mask(fatjets, has_fatjet)
    jets_masked = ak.mask(jets, has_fatjet)
    muons_masked = ak.mask(muons, has_fatjet)
    met_masked = ak.mask(met, has_fatjet)

    leading_fatjet = fatjets_masked[:, 0]
    leading_fatjet = ak.zip(
        {
            "pt": leading_fatjet.pt,
            "eta": leading_fatjet.eta,
            "phi": leading_fatjet.phi,
            "mass": leading_fatjet.mass,
        },
        with_name="Momentum4D",
    )
    had_top_fj = leading_fatjet
    leading_muon = muons_masked[:, 0]

    fatjet_broadcast = ak.broadcast_arrays(jets_masked, leading_fatjet)[1]
    muon_broadcast = ak.broadcast_arrays(jets_masked, leading_muon)[1]
    met_broadcast = ak.broadcast_arrays(jets_masked, met_masked)[1]

    delta_r = jets_masked.deltaR(fatjet_broadcast)
    jet_mask = delta_r > 1.2

    valid_jets = jets_masked[jet_mask]
    valid_muons = muon_broadcast[jet_mask]
    valid_met = met_broadcast[jet_mask]

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

    lep_top_fj = valid_muons_4vec + valid_met_4vec + valid_jets_4vec
    had_mass_broadcast = ak.broadcast_arrays(lep_top_fj, had_top_fj)[1].mass

    chi2_fj = ((lep_top_fj.mass - mean_mlep) / sigma_mlep) ** 2 + (
        (had_mass_broadcast - mean_mhad) / sigma_mhad
    ) ** 2

    # ==========================
    # Combinatoric reconstruction (no fatjet)
    # ==========================
    jets_nofj = ak.mask(jets, no_fatjet)
    muons_nofj = ak.mask(muons, no_fatjet)[:, 0]
    met_nofj = ak.mask(met, no_fatjet)

    combs = ak.combinations(jets_nofj, 2, axis=1, replacement=False)
    lepjet = combs["0"]
    hadjets = ak.unzip(combs)[1]

    muon_broadcast = ak.broadcast_arrays(lepjet, muons_nofj)[1]
    met_broadcast = ak.broadcast_arrays(lepjet, met_nofj)[1]

    lepjet_4vec = ak.zip(
        {
            "pt": lepjet.pt,
            "eta": lepjet.eta,
            "phi": lepjet.phi,
            "mass": lepjet.mass,
        },
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

    hadjets_4vec = ak.zip(
        {
            "pt": hadjets.pt,
            "eta": hadjets.eta,
            "phi": hadjets.phi,
            "mass": hadjets.mass,
        },
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

    # ==========================
    # Combine both cases using ak.fill_none
    # ==========================
    best_idx_fj = ak.argmin(chi2_fj, axis=1, keepdims=True)
    best_idx_nofj = ak.argmin(chi2_nofj, axis=1, keepdims=True)

    # Broadcast had_top_fj to match structure of lep_top_fj
    had_top_fj_broadcasted = ak.broadcast_arrays(lep_top_fj, had_top_fj)[1]
    had_top_nofj_broadcasted = ak.broadcast_arrays(lep_top_nofj, had_top_nofj)[
        1
    ]

    # Best per-event combinations using best_idx (axis=1)
    lep_top_fj_best = lep_top_fj[best_idx_fj]
    had_top_fj_best = had_top_fj_broadcasted[best_idx_fj]

    lep_top_nofj_best = lep_top_nofj[best_idx_nofj]
    had_top_nofj_best = had_top_nofj_broadcasted[best_idx_nofj]

    # Select final top candidates per event
    lep_top = ak.flatten(
        ak.where(has_fatjet, lep_top_fj_best, lep_top_nofj_best)
    )
    had_top = ak.flatten(
        ak.where(has_fatjet, had_top_fj_best, had_top_nofj_best)
    )
    mtt = (lep_top + had_top).mass
    mtt = ak.fill_none(mtt, -1.0)

    chi2 = ak.where(has_fatjet, chi2_fj, chi2_nofj)
    best_idx = ak.argmin(chi2, axis=1, keepdims=True)
    best_chi2 = ak.fill_none(ak.flatten(chi2[best_idx]), 9999.0)

    return best_chi2
>>>>>>> 386d831 (optimise ghost ttbar reco)
