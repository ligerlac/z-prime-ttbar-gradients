import glob
import argparse
from tqdm import tqdm

import numpy as np

# scikit-hep
import awkward as ak
import uproot


def process_with_selection(
    input_file, output_file, tree_name, branches, cut_str="", chunk_size=100_000
):
    """
    Process a ROOT file by applying a selection function on chunks of data
    and saving the filtered results to a new file, with a progress bar.
    """

    # First, get the total number of entries for the progress bar (takes ~3min for 170M events)
    total_entries = len(
        uproot.concatenate(
            f"{input_file}:{tree_name}", ["run"], library="np", how=tuple
        )[0]
    )
    print(f"Found {total_entries} entries")

    iterable = uproot.iterate(
        f"{input_file}:{tree_name}",
        branches,
        step_size=chunk_size,
        cut=cut_str,
        library="ak",
        num_workers=1,  # for some reason, more workers are slower
    )

    n_chunks = (total_entries + chunk_size - 1) // chunk_size  # Ceiling division
    pbar = tqdm(iterable, total=n_chunks, desc="Processing events")

    # Initialize output file and tree
    output = None
    output_tree = None
    branch_types = {}

    for arrays in pbar:
        branches = arrays.fields

        # For the first chunk, create the output file
        if output is None:
            output = uproot.recreate(output_file)

            # Remember the branch structure from the first successful chunk
            for branch in arrays.fields:
                if isinstance(arrays[branch], ak.Array):
                    branch_types[branch] = arrays[branch].type
                else:
                    branch_types[branch] = np.dtype(arrays[branch].dtype)

            # Create the output tree with proper types
            output_tree = output.mktree(tree_name, branch_types)

        # Make sure we only write available branches that match the output tree
        # This handles the case where some branches might be missing in later chunks
        available_branches = set(branches) & set(branch_types.keys())
        filtered_data_to_write = {
            branch: arrays[branch] for branch in available_branches
        }

        # Write the filtered data for available branches only
        output_tree.extend(filtered_data_to_write)

    # Close the progress bar
    pbar.close()

    # Close the output file if it was created
    if output is not None:
        output.close()
        print(f"Saved filtered data to {output_file}")
        return True
    else:
        print(f"No events passed selection for {input_file}")
        return False


def main(args):

    branches = [
        "Muon_pt",
        "Muon_eta",
        "Muon_phi",
        "Muon_mass",
        "Muon_miniIsoId",
        "Muon_tightId",
        "FatJet_particleNet_TvsQCD",
        "FatJet_pt",
        "FatJet_eta",
        "FatJet_phi",
        "FatJet_mass",
        "Jet_btagDeepB",
        "Jet_jetId",
        "Jet_pt",
        "Jet_eta",
        "Jet_phi",
        "Jet_mass",
        "PuppiMET_pt",
        "PuppiMET_pt",
        "PuppiMET_phi",
    ]

    if args.is_mc:
        branches += ["genWeight", "Pileup_nTrueInt"]

    process_with_selection(
        input_file=args.input,
        output_file=args.output,
        tree_name="Events",
        branches=branches,
        cut_str="HLT_TkMu50*(PuppiMET_pt>50)",
        chunk_size=100_000,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Apply preselection to ROOT files")
    argparser.add_argument(
        "--input",
        type=str,
        # default="root://eospublic.cern.ch//eos/opendata/cms/Run2016H/SingleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/280000/F302A865-17B0-064B-8154-41526BB38244.root",
        default="root://eospublic.cern.ch//eos/opendata/cms/Run2016H/SingleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/*/*.root",
        help="Input ROOT file to process (can be single file or glob pattern)",
    )
    argparser.add_argument(
        "--output",
        type=str,
        # default="/eos/user/l/ligerlac/z-prime-ttbar-data/filtered_data.root",
        default="filtered_data.root",
        help="Output file to save the filtered data",
    )
    argparser.add_argument(
        "--branch-file",
        type=str,
        default="branches.txt",
        help="File containing the branches to be used",
    )
    argparser.add_argument(
        "--is-mc",
        action="store_true",
        help="Flag to indicate if the input file is from MC (default: False)",
    )
    main(argparser.parse_args())
