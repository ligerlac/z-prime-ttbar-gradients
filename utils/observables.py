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

def ttbar_chi2(muons, jets, fatjets, met):
    # ==========================
    # ttbar reconstruction
    # ==========================
    mean_mlep = 172.5
    sigma_mlep = 20.0
    mean_mhad = 172.5
    sigma_mhad = 20.0

    # Split into events with and without fatjets
    has_fatjet = ak.num(fatjets, axis=1) > 0
    no_fatjet = ~has_fatjet

    # ==========================
    # Fatjet-based reconstruction
    # ==========================

    fatjets_sel = fatjets[has_fatjet]
    jets_sel = jets[has_fatjet]
    muons_sel = muons[has_fatjet]
    met_sel = met[has_fatjet]

    leading_fatjet_sel = fatjets_sel[:, 0]
    had_top_fj = leading_fatjet_sel
    leading_muon = muons_sel[:, 0]

    fatjet_broadcast = ak.broadcast_arrays(jets_sel, leading_fatjet_sel)[1]
    muon_broadcast = ak.broadcast_arrays(jets_sel, leading_muon)[1]
    met_broadcast = ak.broadcast_arrays(jets_sel, met_sel)[1]

    delta_r = jets_sel.deltaR(fatjet_broadcast)
    jet_mask = delta_r > 1.2

    valid_jets = jets_sel[jet_mask]
    valid_muons = muon_broadcast[jet_mask]
    valid_met = met_broadcast[jet_mask]

    valid_muons_4vec, valid_jets_4vec = [
        ak.zip(
            {"pt": o.pt, "eta": o.eta, "phi": o.phi, "mass": o.mass},
            with_name="Momentum4D",
        )
        for o in [valid_muons, valid_jets]
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

    chi2_fj = ((lep_top_fj.mass - mean_mlep) / sigma_mlep) ** 2 + \
              ((had_mass_broadcast - mean_mhad) / sigma_mhad) ** 2

    best_idx_fj = ak.argmin(chi2_fj, axis=1, keepdims=True)
    best_chi2_fj = ak.fill_none(ak.flatten(chi2_fj[best_idx_fj]), 9999.0)


    # ==========================
    # Combinatoric reconstruction (no fatjet)
    # ==========================
    # Events without fatjets (no cut on number of jets)
    jets_nofj = jets[no_fatjet]
    muons_nofj = muons[no_fatjet][:, 0]
    met_nofj = met[no_fatjet]

    # Assign each jet once as the lep jet, rest as had jets
    combs = ak.combinations(jets_nofj, 2, axis=1, replacement=False)
    lepjet = combs["0"]
    hadjets = jets_nofj[
        ak.local_index(jets_nofj) != ak.local_index(lepjet)
    ]

    # Now broadcast muon and met to lepjet
    muon_broadcast = ak.broadcast_arrays(lepjet, muons_nofj)[1]
    met_broadcast = ak.broadcast_arrays(lepjet, met_nofj)[1]

    # Vectorize inputs
    lepjet_4vec = ak.zip(
        {"pt": lepjet.pt, "eta": lepjet.eta, "phi": lepjet.phi, "mass": lepjet.mass},
        with_name="Momentum4D",
    )
    muon_4vec = ak.zip(
        {"pt": muon_broadcast.pt, "eta": muon_broadcast.eta,
        "phi": muon_broadcast.phi, "mass": muon_broadcast.mass},
        with_name="Momentum4D",
    )
    met_4vec = ak.zip(
        {"pt": met_broadcast.pt, "eta": 0 * met_broadcast.pt,
        "phi": met_broadcast.phi, "mass": 0},
        with_name="Momentum4D",
    )
    lep_top_nofj = lepjet_4vec + muon_4vec + met_4vec

    # Sum remaining jets as hadronic top (can be 0–3 jets)
    hadjets = ak.unzip(combs)[1]  # the second item is all the remaining jets
    hadjets_4vec = ak.zip(
        {"pt": hadjets.pt, "eta": hadjets.eta, "phi": hadjets.phi, "mass": hadjets.mass},
        with_name="Momentum4D"
    )
    had_top_nofj = ak.sum(hadjets_4vec, axis=1)

    # Compute chi2
    chi2_nofj = ((lep_top_nofj.mass - mean_mlep) / sigma_mlep) ** 2 + \
                ((had_top_nofj.mass - mean_mhad) / sigma_mhad) ** 2

    best_idx_nofj = ak.argmin(chi2_nofj, axis=1, keepdims=True)
    best_chi2_nofj = ak.fill_none(ak.flatten(chi2_nofj[best_idx_nofj]), 9999.0)

    # ==========================
    # Combine both cases
    # ==========================
    best_chi2_combined = ak.full_like(has_fatjet, 9999.0, dtype=float).to_numpy()
    best_chi2_combined[has_fatjet] = best_chi2_fj.to_numpy()
    best_chi2_combined[no_fatjet] = best_chi2_nofj.to_numpy()
    # Convert back to awkward array
    best_chi2_combined = ak.from_numpy(best_chi2_combined)

    return best_chi2_combined

def mtt_from_chi2(muons, jets, fatjets, met):

    # ==========================
    # ttbar reconstruction
    # ==========================
    mean_mlep = 172.5
    sigma_mlep = 20.0
    mean_mhad = 172.5
    sigma_mhad = 20.0

    # Split into events with and without fatjets
    has_fatjet = ak.num(fatjets, axis=1) > 0
    no_fatjet = ~has_fatjet

    # ==========================
    # Fatjet-based reconstruction
    # ==========================

    fatjets_sel = fatjets[has_fatjet]
    jets_sel = jets[has_fatjet]
    muons_sel = muons[has_fatjet]
    met_sel = met[has_fatjet]

    leading_fatjet_sel = fatjets_sel[:, 0]
    had_top_fj = leading_fatjet_sel
    leading_muon = muons_sel[:, 0]

    fatjet_broadcast = ak.broadcast_arrays(jets_sel, leading_fatjet_sel)[1]
    muon_broadcast = ak.broadcast_arrays(jets_sel, leading_muon)[1]
    met_broadcast = ak.broadcast_arrays(jets_sel, met_sel)[1]

    delta_r = jets_sel.deltaR(fatjet_broadcast)
    jet_mask = delta_r > 1.2

    valid_jets = jets_sel[jet_mask]
    valid_muons = muon_broadcast[jet_mask]
    valid_met = met_broadcast[jet_mask]

    valid_muons_4vec, valid_jets_4vec = [
        ak.zip(
            {"pt": o.pt, "eta": o.eta, "phi": o.phi, "mass": o.mass},
            with_name="Momentum4D",
        )
        for o in [valid_muons, valid_jets]
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

    chi2_fj = ((lep_top_fj.mass - mean_mlep) / sigma_mlep) ** 2 + \
              ((had_mass_broadcast - mean_mhad) / sigma_mhad) ** 2

    best_idx_fj = ak.argmin(chi2_fj, axis=1, keepdims=True)
    best_chi2_fj = ak.fill_none(ak.flatten(chi2_fj[best_idx_fj]), 9999.0)
    best_lep_top_fj = lep_top_fj[best_idx_fj]
    best_had_top_fj = ak.broadcast_arrays(best_lep_top_fj, had_top_fj)[1]
    best_had_top_fj_4vec = ak.zip(
        {"pt": best_had_top_fj.pt, "eta": best_had_top_fj.eta,
        "phi": best_had_top_fj.phi, "mass": best_had_top_fj.mass},
        with_name="Momentum4D",
    )

    mtt_fj = ak.flatten((best_lep_top_fj + best_had_top_fj_4vec).mass)

    # ==========================
    # Combinatoric reconstruction (no fatjet)
    # ==========================
    # Events without fatjets (no cut on number of jets)
    jets_nofj = jets[no_fatjet]
    muons_nofj = muons[no_fatjet][:, 0]
    met_nofj = met[no_fatjet]

    # Assign each jet once as the lep jet, rest as had jets
    combs = ak.combinations(jets_nofj, 2, axis=1, replacement=False)
    lepjet = combs["0"]
    hadjets = jets_nofj[
        ak.local_index(jets_nofj) != ak.local_index(lepjet)
    ]

    # Now broadcast muon and met to lepjet
    muon_broadcast = ak.broadcast_arrays(lepjet, muons_nofj)[1]
    met_broadcast = ak.broadcast_arrays(lepjet, met_nofj)[1]

    # Vectorize inputs
    lepjet_4vec = ak.zip(
        {"pt": lepjet.pt, "eta": lepjet.eta, "phi": lepjet.phi, "mass": lepjet.mass},
        with_name="Momentum4D",
    )
    muon_4vec = ak.zip(
        {"pt": muon_broadcast.pt, "eta": muon_broadcast.eta,
        "phi": muon_broadcast.phi, "mass": muon_broadcast.mass},
        with_name="Momentum4D",
    )
    met_4vec = ak.zip(
        {"pt": met_broadcast.pt, "eta": 0 * met_broadcast.pt,
        "phi": met_broadcast.phi, "mass": 0},
        with_name="Momentum4D",
    )
    lep_top_nofj = lepjet_4vec + muon_4vec + met_4vec

    # Sum remaining jets as hadronic top (can be 0–3 jets)
    hadjets = ak.unzip(combs)[1]  # the second item is all the remaining jets
    hadjets_4vec = ak.zip(
        {"pt": hadjets.pt, "eta": hadjets.eta, "phi": hadjets.phi, "mass": hadjets.mass},
        with_name="Momentum4D"
    )
    had_top_nofj = ak.sum(hadjets_4vec, axis=1)

    # Compute chi2
    chi2_nofj = ((lep_top_nofj.mass - mean_mlep) / sigma_mlep) ** 2 + \
                ((had_top_nofj.mass - mean_mhad) / sigma_mhad) ** 2

    best_idx_nofj = ak.argmin(chi2_nofj, axis=1, keepdims=True)
    best_chi2_nofj = ak.fill_none(ak.flatten(chi2_nofj[best_idx_nofj]), 9999.0)
    mtt_nofj = (
        ak.flatten((lep_top_nofj[best_idx_nofj] + had_top_nofj[best_idx_nofj]).mass)
        if len(lep_top_nofj) > 0
        else ak.Array([])
    )
    mtt_nofj = ak.fill_none(mtt_nofj, 9999.0)

    # ==========================
    # Combine both cases
    # ==========================
    best_chi2_combined = ak.full_like(has_fatjet, 9999.0, dtype=float).to_numpy()
    best_chi2_combined[has_fatjet] = best_chi2_fj.to_numpy()
    best_chi2_combined[no_fatjet] = best_chi2_nofj.to_numpy()
    # Convert back to awkward array
    best_chi2_combined = ak.from_numpy(best_chi2_combined)

    mtt_combined = ak.full_like(has_fatjet, 9999.0, dtype=float).to_numpy()
    print(mtt_combined, mtt_fj)
    mtt_combined[has_fatjet] = mtt_fj.to_numpy()
    mtt_combined[no_fatjet] = mtt_nofj.to_numpy()
    # Convert back to awkward array
    mtt_combined = ak.from_numpy(mtt_combined)

    return mtt_combined
