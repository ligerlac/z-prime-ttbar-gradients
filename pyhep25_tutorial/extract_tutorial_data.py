"""
Extract a small subset of the GRAEP data for the tutorial.
This creates lightweight numpy files suitable for a Kaggle tutorial.
"""
from collections import defaultdict
import uproot
import numpy as np
import awkward as ak
import vector
from pathlib import Path
import warnings
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")

# Register awkward with vector for 4-vector operations
ak.jax.register_and_check()
vector.register_awkward()

def extract_and_process_subset(file_path, process_name, xsec, ngen, frac_events=0.1):
    """
    Extract a subset of events and compute physics features with proper normalization.

    Parameters:
    -----------
    file_path : str
        Path to the ROOT file
    process_name : str
        Name of the physics process (wjets, ttbar, signal, data)
    xsec : float
        Cross-section in pb
    ngen : float
        Total number of generated events for this process
    frac_events : float
        Fraction of events to extract (between 0 and 1)

    Returns:
    --------
    dict : Dictionary with features and properly normalized weights
    """

    # Load events
    events = NanoEventsFactory.from_root(
        f"{file_path}:Events",
        schemaclass=NanoAODSchema,
        # delayed=False,  # Important: disable delayed loading
    ).events()

    # Apply basic object selections
    jets = events.Jet
    muons = events.Muon[
        (events.Muon.pt > 55) &
        (abs(events.Muon.eta) < 2.4) &
        events.Muon.tightId &
        (events.Muon.miniIsoId > 1)
    ]
    fatjets = events.FatJet[
        (events.FatJet.pt > 500) &
        (events.FatJet.particleNet_TvsQCD > 0.5)
    ]

    # Event selection: at least 2 jets, exactly 1 muon, and at least 1 fatjet
    mask = (ak.num(jets, axis=1) >= 2) & (ak.num(muons, axis=1) == 1) & (ak.num(fatjets, axis=1) >= 1)

    # Apply mask
    muons = muons[mask]
    jets = jets[mask]
    fatjets = fatjets[mask]
    met = events.PuppiMET[mask]

    # Compute arrays to materialize them (this triggers the lazy evaluation)
    jets = jets.compute() if hasattr(jets, 'compute') else jets
    muons = muons.compute() if hasattr(muons, 'compute') else muons
    fatjets = fatjets.compute() if hasattr(fatjets, 'compute') else fatjets
    met = met.compute() if hasattr(met, 'compute') else met

    # Now we can get the length
    n_selected = len(met)
    n_events = int(n_selected * frac_events)

    #print(f"  Events passing selection: {n_selected}, taking {n_events} ({frac_events*100:.0f}%)")

    muons = muons[:n_events]
    jets = jets[:n_events]
    fatjets = fatjets[:n_events]
    met = met[:n_events]

    # Extract features
    features = {}

    # Event weights with cross-section normalization: weight = genWeight * xsec / N_gen
    if process_name == "data":
        # Real data has no generator weight, use 1.0
        features["weight"] = np.ones(n_events, dtype=np.float32)
    else:
        # Extract genWeight for all selected events (before taking subset)
        genWeight_masked = events.genWeight[mask]
        genWeight_masked = genWeight_masked.compute() if hasattr(genWeight_masked, 'compute') else genWeight_masked
        genWeight_array = genWeight_masked.to_numpy()

        # N_gen = sum of absolute genWeights (needed for proper xsec normalization)
        n_gen = float(np.sum(np.abs(genWeight_array)))
        n_gen = ngen


        # Compute normalized weight: (genWeight / |genWeight|) * xsec / N_gen
        # This preserves the sign of genWeight while applying xsec normalization
        xsec_weight = xsec * 16400 / n_gen # Using lumi = 16400 pb^-1
        normalized_weights = (genWeight_array / np.abs(genWeight_array)) * xsec_weight

        # Store weights for the subset
        features["weight"] = normalized_weights[:n_events].astype(np.float32)


    # Number of jets
    features["n_jet"] = ak.num(jets, axis=1).to_numpy()

    # Leading and subleading jet features
    features["leading_jet_mass"] = jets.mass[:, 0].to_numpy()
    features["subleading_jet_mass"] = jets.mass[:, 1].to_numpy()
    features["leading_jet_pt"] = jets.pt[:, 0].to_numpy()
    features["subleading_jet_pt"] = jets.pt[:, 1].to_numpy()

    # Scalar sum HT
    features["st"] = (ak.sum(jets.pt, axis=1) + ak.sum(muons.pt, axis=1)).to_numpy()

    # B-tagging scores
    features["leading_jet_btag"] = jets.btagDeepB[:, 0].to_numpy()
    features["subleading_jet_btag"] = jets.btagDeepB[:, 1].to_numpy()

    # MET
    features["met_pt"] = met.pt.to_numpy()

    # Muon pT
    features["muon_pt"] = muons.pt[:, 0].to_numpy()


    # Sphericity (only zz component)
    denominator = ak.sum(jets.px**2 + jets.py**2 + jets.pz**2, axis=1)
    S_zz = ak.sum(jets.pz * jets.pz, axis=1) / denominator
    features["S_zz"] = S_zz.to_numpy()

    # DeltaR between muon and closest jet
    muon_in_pair, jet_in_pair = ak.unzip(ak.cartesian([muons, jets]))
    delta_r = muon_in_pair.deltaR(jet_in_pair)
    min_delta_r = ak.min(delta_r, axis=1)
    features["min_deltaR_mu_jet"] = min_delta_r.to_numpy()


    # Compute m_tt (invariant mass of ttbar system)
    # Following get_mtt from user/observables.py: m_tt = (muon + fatjet + jet + MET).mass
    # Build 4-vectors
    muon_4v = ak.zip({
        "pt": muons.pt[:, 0],
        "eta": muons.eta[:, 0],
        "phi": muons.phi[:, 0],
        "mass": muons.mass[:, 0],
    }, with_name="Momentum4D")

    fatjet_4v = ak.zip({
        "pt": fatjets.pt[:, 0],
        "eta": fatjets.eta[:, 0],
        "phi": fatjets.phi[:, 0],
        "mass": fatjets.mass[:, 0],
    }, with_name="Momentum4D")

    jet_4v = ak.zip({
        "pt": jets.pt[:, 0],
        "eta": jets.eta[:, 0],
        "phi": jets.phi[:, 0],
        "mass": jets.mass[:, 0],
    }, with_name="Momentum4D")

    met_4v = ak.zip({
        "pt": met.pt,
        "eta": 0 * met.pt,
        "phi": met.phi,
        "mass": 0 * met.pt,
    }, with_name="Momentum4D")

    # m_tt = invariant mass of the system
    ttbar_system = muon_4v + fatjet_4v + jet_4v + met_4v
    features["m_ttbar"] = ttbar_system.mass.to_numpy()

    return features


def main():
    """
    Extract tutorial data from preprocessed ROOT files.

    This script:
    1. Loads events from GRAEP preprocessed ROOT files
    2. Applies basic event selection (2+ jets, 1 muon)
    3. Computes physics features (MET, HT, b-tagging, etc.)
    4. Applies proper MC normalization: weight = genWeight × xsec / N_gen
    5. Saves to tutorial_data/ directory

    Output files are ready to use with luminosity scaling in the tutorial.
    """

    # Define file paths (relative to script location)
    script_dir = Path(__file__).parent
    output_dir = script_dir / "tutorial_data"
    output_dir.mkdir(exist_ok=True)

    # Source data location
    #base_dir = Path("/Users/moaly/work/iris-hep/autodiff-analysis-MODE/fork/graep/example/outputs/skimmed/")

    # file paths
    if (Path.cwd() / "preproc_uproot").exists():
        base_dir = Path("preproc_uproot/z-prime-ttbar-data/")
    else:
        base_dir = Path("GRAEP/preproc_uproot/z-prime-ttbar-data/")

    print("="*70)
    print("GRAEP TUTORIAL DATA EXTRACTION")
    print("="*70)
    print(f"Source: {base_dir.absolute()}")
    print(f"Output: {output_dir.absolute()}")
    print()

    # Cross-sections (pb)
    xsec_map = {
        "data": 1.0,  # Data doesn't have xsec
        "wjets": 61526.7,
        "ttbar_had": 831.76 * 0.457,
        "ttbar_semilep": 831.76 * 0.438,
        "ttbar_lep": 831.76 * 0.105,
        "signal": 1.0,
    }

    ngen_map = {
        "signal": 527279,
        "ttbar_semilep": 144722000,
        "ttbar_lep": 43546000,
        "ttbar_had": 107067000,
        "wjets": 80958227,
        "data": 323952013
    }


    #  file paths
    files = {
        "data": [ base_dir / "data__nominal/file__0/part0.root"],
        "wjets": [ base_dir / "wjets__nominal/file__0/part0.root"],
        "ttbar_had": [ base_dir / "ttbar_had__nominal/file__0/part0.root"],
        "ttbar_semilep": [ base_dir / "ttbar_semilep__nominal/file__0/part0.root"],
        "ttbar_lep": [ base_dir / "ttbar_lep__nominal/file__0/part0.root"],
        "signal": [ base_dir / "signal__nominal/file__0/part0.root"],
    }

    # Extract data for each process
    extracted_data = defaultdict(list)
    for process_name, file_paths in files.items():
        print(f"Processing {process_name}...")
        for file_path in file_paths:
            if not file_path.exists():
                print(f"Warning: {file_path} not found, skipping...")
                continue

            extracted_data[process_name].append(extract_and_process_subset(
                str(file_path),
                process_name,
                xsec=xsec_map[process_name],
                ngen=ngen_map[process_name],
                frac_events=1.0
            ))

        print(f"  Extracted {len(extracted_data[process_name])} subsets for {process_name}")
        # Concatenate all subsets for this process
        if extracted_data[process_name]:
            feature_names = extracted_data[process_name][0].keys()
            combined_features = {}
            for feature in feature_names:
                combined_features[feature] = np.concatenate([
                    subset[feature] for subset in extracted_data[process_name]
                ])
            extracted_data[process_name] = combined_features
        else:
            extracted_data[process_name] = {}


    # Combine ttbar processes
    if "ttbar_had" in extracted_data and "ttbar_semilep" in extracted_data and "ttbar_lep" in extracted_data:
        print("Combining ttbar processes...")
        ttbar_combined = {}
        feature_names = extracted_data["ttbar_had"].keys()
        for feature in feature_names:
            ttbar_combined[feature] = np.concatenate([
                extracted_data["ttbar_had"][feature],
                extracted_data["ttbar_semilep"][feature],
                extracted_data["ttbar_lep"][feature]
            ])
        extracted_data["ttbar"] = ttbar_combined

    # Save each process separately
    for process_name, features in extracted_data.items():
        # Skip individual ttbar components (we save combined version)
        if process_name in ["ttbar_had", "ttbar_semilep"]:
            continue

        print(f"\nSaving {process_name}...")
        # Save as .npz (compressed numpy format)
        output_file = output_dir / f"{process_name}.npz"
        np.savez_compressed(output_file, **features)
        print(f"  Saved to {output_file}")
        print(f"  Features: {list(features.keys())}")
        print(f"  Shape: {features['n_jet'].shape}")

        # Note for real data
        if process_name == "data":
            print(f"  ⭐ Real CMS collision data from 2016!")

    print("\n" + "="*60)
    print("Data extraction complete!")
    print(f"Output directory: {output_dir}")
    print("="*60)

    # Print summary statistics
    print("\nDataset summary:")
    for process_name in ["data", "wjets", "ttbar", "signal"]:
        if process_name in extracted_data:
            n_events = len(extracted_data[process_name]["n_jet"])
            data_type = "Real CMS Data" if process_name == "data" else "MC Simulation"
            print(f"  {process_name}: {n_events} events ({data_type})")


if __name__ == "__main__":
    main()
