"""
Extract a small subset of the GRAEP data for the tutorial.
This creates lightweight numpy files suitable for a Kaggle tutorial.
"""

import uproot
import numpy as np
import awkward as ak
from pathlib import Path
import warnings
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")

def extract_and_process_subset(file_path, process_name, frac_events=0.1):
    """
    Extract a subset of events and compute physics features.

    Parameters:
    -----------
    file_path : str
        Path to the ROOT file
    process_name : str
        Name of the physics process (wjets, ttbar, signal)
    frac_events : float
        Fraction of events to extract (between 0 and 1)

    Returns:
    --------
    dict : Dictionary with features and labels
    """
    print(f"Processing {process_name}...")

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

    # Event selection: at least 2 jets and exactly 1 muon
    mask = (ak.num(jets, axis=1) >= 2) & (ak.num(muons, axis=1) == 1)

    # Apply mask
    muons = muons[mask]
    jets = jets[mask]
    met = events.PuppiMET[mask]

    # Compute arrays to materialize them (this triggers the lazy evaluation)
    jets = jets.compute() if hasattr(jets, 'compute') else jets
    muons = muons.compute() if hasattr(muons, 'compute') else muons
    met = met.compute() if hasattr(met, 'compute') else met

    # Now we can get the length
    n_selected = len(met)
    n_events = int(n_selected * frac_events)

    print(f"  Events passing selection: {n_selected}, taking {n_events} ({frac_events*100:.0f}%)")

    muons = muons[:n_events]
    jets = jets[:n_events]
    met = met[:n_events]

    # Extract features
    features = {}

    # Event weights (for MC normalization)
    if process_name == "data":
        # Real data has no generator weight, use 1.0
        features["weight"] = np.ones(n_events, dtype=np.float32)
    else:
        # MC has genWeight that needs to be applied for proper normalization
        # Extract genWeight with same mask and slicing as other features
        genWeight_masked = events.genWeight[mask]
        genWeight_masked = genWeight_masked.compute() if hasattr(genWeight_masked, 'compute') else genWeight_masked
        features["weight"] = genWeight_masked[:n_events].to_numpy().astype(np.float32)

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

    return features


def main():
    """Extract tutorial data from preprocessed ROOT files."""

    # Define file paths (adjust based on where script is run from)
    script_dir = Path(__file__).parent
    # Check if we're in GRAEP directory or parent
    if (Path.cwd() / "preproc_uproot").exists():
        base_dir = Path("preproc_uproot/z-prime-ttbar-data/")
    else:
        base_dir = Path("GRAEP/preproc_uproot/z-prime-ttbar-data/")

    output_dir = script_dir / "tutorial_data"
    output_dir.mkdir(exist_ok=True)

    print(f"Looking for data in: {base_dir.absolute()}")
    print(f"Output directory: {output_dir.absolute()}")

    files = {
        "data": base_dir / "data__nominal/file__0/part0.root",  # Real CMS data
        "wjets": base_dir / "wjets__nominal/file__0/part0.root",
        "ttbar_had": base_dir / "ttbar_had__nominal/file__0/part0.root",
        "ttbar_semilep": base_dir / "ttbar_semilep__nominal/file__0/part0.root",
        "signal": base_dir / "signal__nominal/file__0/part0.root",
    }

    # Extract data for each process
    extracted_data = {}
    for process_name, file_path in files.items():
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping...")
            continue

        extracted_data[process_name] = extract_and_process_subset(
            str(file_path),
            process_name,
            frac_events=0.1
        )

    # Combine ttbar processes
    if "ttbar_had" in extracted_data and "ttbar_semilep" in extracted_data:
        print("Combining ttbar processes...")
        ttbar_combined = {}
        feature_names = extracted_data["ttbar_had"].keys()
        for feature in feature_names:
            ttbar_combined[feature] = np.concatenate([
                extracted_data["ttbar_had"][feature],
                extracted_data["ttbar_semilep"][feature]
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
            print(f"  {process_name:12s}: {n_events:6d} events ({data_type})")


if __name__ == "__main__":
    main()
