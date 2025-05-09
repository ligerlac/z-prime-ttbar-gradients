import awkward as ak
import vector
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
    print("xx")
    print(
        ak.backend(region_muons_4vec),
        ak.backend(region_fatjets_4vec),
        ak.backend(region_jets_4vec),
        ak.backend(region_met_4vec),
    )
    print(
        ak.backend(muons),
        ak.backend(fatjets),
        ak.backend(jets),
        ak.backend(met),
    )
    print(
        region_muons_4vec.type.show(),
        region_fatjets_4vec.type.show(),
        region_jets_4vec.type.show(),
        region_met_4vec.type.show(),
    )

    print(region_muons_4vec, region_fatjets_4vec, region_jets_4vec, region_met_4vec)

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