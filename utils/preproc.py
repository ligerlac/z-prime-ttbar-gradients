
import glob
import argparse
from tqdm import tqdm

import uproot
import awkward as ak
import numpy as np
import dask_awkward as dak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

# -----------------------------
# Branch Selection
# -----------------------------
def build_branches_to_keep(configuration, mode="uproot", is_mc=False):
    """
    Define the branches to retain during pre-processing.

    Returns
    -------
    dict
        Dictionary mapping object names to lists of branch names.
    """
    if mode == "dak":
        branches = configuration.preprocess.branches
        filtered_branches  = {}
        if not is_mc:
            for obj, obj_branches in configuration.preprocess.mc_branches.items():
                filtered_obj_branches = []
                for br in obj_branches:
                    if br in configuration.preprocess.mc_branches[obj]:
                        continue
                    else:
                        filtered_obj_branches.append(br)

                filtered_branches[obj] = obj_branches
        else:
            filtered_branches = branches
        return filtered_branches

    elif mode == "uproot":
        branches_list = []
        branches = configuration.preprocess.branches
        filtered_branches = []

        for obj, obj_branches in branches.items():
            filtered_obj_branches = []
            if not is_mc:
                for br in obj_branches:
                    if br in configuration.preprocess.mc_branches[obj]:
                        continue
                    else:
                        filtered_obj_branches.append(br)
            else:
                filtered_obj_branches = obj_branches

            if obj == "event":
                branches_list.extend(filtered_obj_branches)
            else:
                branches_list.extend([f"{obj}_{br}" for br in filtered_obj_branches])

        return branches_list
    else:
        raise ValueError("Invalid mode for constructing branches to keep. Use 'dak' or 'uproot'.")

# -----------------------------
# Preprocessing Logic with dak
# -----------------------------
def pre_process_dak(input_path, tree, output_dir, configuration, step_size=100_000, logger=None, is_mc=True):
    """
    Preprocess input ROOT file by applying basic filtering and reducing branches.

    Parameters
    ----------
    input_path : str
        Path to the input ROOT file.
    tree : str
        Name of the TTree inside the file.
    output_path : str
        Destination directory for filtered output.
    step_size : int
        Chunk size to load events incrementally.

    Returns
    -------
    int
        Total number of input events before filtering.
    """
    with uproot.open(f"{input_path}:{tree}") as f:
        total_events = f.num_entries

    logger.info("========================================")
    logger.info(f"📂 Preprocessing file: {input_path} with {total_events:,} events")

    branches = build_branches_to_keep(configuration, mode="dak", is_mc=is_mc)
    selected = None

    for start in range(0, total_events, step_size):
        stop = min(start + step_size, total_events)

        events = NanoEventsFactory.from_root(
            {input_path: tree},
            schemaclass=NanoAODSchema,
            entry_start=start,
            entry_stop=stop,
            delayed=True,
            #xrootd_handler= uproot.source.xrootd.MultithreadedXRootDSource,
        ).events()

        mu_sel = (
            (events.Muon.pt > 55) &
            (abs(events.Muon.eta) < 2.4) &
            events.Muon.tightId &
            (events.Muon.miniIsoId > 1)
        )
        muon_count = ak.sum(mu_sel, axis=1)
        mask = (
            events.HLT.TkMu50 &
            (muon_count == 1) &
            (events.PuppiMET.pt > 50)
        )

        filtered = events[mask]

        subset = {}
        for obj, obj_branches in branches.items():
            if obj == "event":
                subset.update({br: filtered[br] for br in obj_branches if br in filtered.fields})
            elif obj in filtered.fields:
                subset.update({f"{obj}_{br}": filtered[obj][br] for br in obj_branches if br in filtered[obj].fields})

        compact = dak.zip(subset, depth_limit=1)
        selected = compact if selected is None else ak.concatenate([selected, compact])

    logger.info(f"💾 Writing skimmed output to: {output_dir}")
    uproot.dask_write(selected, destination=output_dir, compute=True, tree_name=tree)
    return total_events

def pre_process_uproot(input_path, tree, output_path, configuration, step_size=100_000, logger=None, is_mc=True):
    """
    Process a ROOT file by applying a selection function on chunks of data
    and saving the filtered results to a new file, with a progress bar.
    """

    cut_str = "HLT_TkMu50*(PuppiMET_pt>50)"
    branches = build_branches_to_keep(configuration, mode="uproot", is_mc=is_mc)

    # First, get the total number of entries for the progress bar (takes ~3min for 170M events)
    total_events = len(uproot.concatenate(f"{input_path}:{tree}", ["run"], library="np", how=tuple)[0])
    logger.info("========================================")
    logger.info(f"📂 Preprocessing file: {input_path} with {total_events:,} events")

    iterable = uproot.iterate(
        f"{input_path}:{tree}",
        branches,
        step_size=step_size, cut=cut_str, library="ak", num_workers=1,  # for some reason, more workers are slower
    )

    n_chunks = (total_events + step_size - 1) // step_size  # Ceiling division
    pbar = tqdm(iterable, total=n_chunks, desc="Processing events")

    # Initialize output file and tree
    output = None
    output_tree = None
    branch_types = {}

    for arrays in pbar:
        branches = arrays.fields

        # For the first chunk, create the output file
        if output is None:
            output = uproot.recreate(output_path)

            # Remember the branch structure from the first successful chunk
            for branch in arrays.fields:
                if isinstance(arrays[branch], ak.Array):
                    branch_types[branch] = arrays[branch].type
                else:
                    branch_types[branch] = np.dtype(arrays[branch].dtype)

            # Create the output tree with proper types
            output_tree = output.mktree(tree, branch_types)

        # Make sure we only write available branches that match the output tree
        # This handles the case where some branches might be missing in later chunks
        available_branches = set(branches) & set(branch_types.keys())
        filtered_data_to_write = {branch: arrays[branch] for branch in available_branches}

        # Write the filtered data for available branches only
        output_tree.extend(filtered_data_to_write)

    # Close the progress bar
    pbar.close()

    # Close the output file if it was created
    if output is not None:
        output.close()
        logger.info(f"💾 Writing skimmed output to: {output_path}")
        return True
    else:
        logger.info(f"💾 No events passed selection for {input_path}")
        return False