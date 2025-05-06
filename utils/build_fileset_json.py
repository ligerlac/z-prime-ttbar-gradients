"""
Utility to generate JSON metadata for NanoAOD ROOT files used in
ZprimeTtbar analysis.

For each process, this script reads ROOT file paths from `.txt` files in a
specified location, extracts the number of events per file using uproot,
and stores the data in a JSON dictionary for downstream usage.

Based on original code from AGC repo:
https://github.com/iris-hep/analysis-grand-challenge/blob/main/datasets/
cms-open-data-2015/build_ntuple_json.py
"""

from collections import defaultdict
from glob import glob
import json
import logging
import os
import time

import awkward as ak
import uproot


logger = logging.getLogger(__name__)

# Mapping of process names to folder paths containing .txt files listing ROOT files
PROCESS_TO_DATASETS_FOLDER = {
    "signal": "datasets/signal/m2000_w20/",
    "ttbar_semilep": "datasets/ttbar_semilep/",
    "ttbar_had": "datasets/ttbar_had/",
    "ttbar_lep": "datasets/ttbar_lep/",
    "wjets": "datasets/wjets/",
    "data": "datasets/data/",
}


def get_paths(folder: str, recids=None) -> list[str]:
    """
    Get a list of ROOT file paths based on .txt file(s) in the given folder.

    If `recids` is provided, it will only read .txt files with those identifiers.
    Otherwise, all .txt files in the folder are used.

    Args:
        folder (str): Path to the folder containing .txt files.
        recids (Optional[int or list[int]]): Optional list of IDs for specific text
        files.

    Returns:
        list[str]: List of ROOT file paths.
    """
    if recids is None:
        text_files = glob(f"{folder}/*.txt")
    else:
        if not isinstance(recids, list):
            recids = [recids]
        text_files = [f"{folder}/{str(recid)}.txt" for recid in recids]

    all_files = []
    for text_file in text_files:
        with open(text_file) as f:
            root_files = f.readlines()
        all_files.extend(f.strip() for f in root_files)

    return all_files


def num_events_list(files: list[str]) -> list[int]:
    """
    Retrieve the number of events from each ROOT file.

    Args:
        files (list[str]): List of ROOT file paths.

    Returns:
        list[int]: List of event counts for each file.
    """
    num_events = []
    num_weight_events = []
    t0 = time.time()

    for i, filename in enumerate(files):
        if i % 10 == 0 or i == len(files) - 1:
            logger.info(f"{i+1} / {len(files)} in {time.time() - t0:.0f} s")
        try:
            with uproot.open(filename) as f:
                num_events.append(f["Events"].num_entries)
                num_weight_events.append(
                    float(ak.sum(f["Events"]["genWeight"].array()))
                )
        except Exception as e:
            logger.warning(f"Could not read {filename}: {e}")
            num_events.append(0)
            num_weight_events.append(0)

    return num_events, num_weight_events


def update_dict(
    file_dict: dict, process: str, folder: str, variation: str, recid
) -> dict:
    """
    Update the file_dict with information about a given process and variation.

    Args:
        file_dict (dict): Dictionary to be updated.
        process (str): Name of the physics process.
        folder (str): Folder containing .txt file(s) listing ROOT files.
        variation (str): Variation name (e.g. "nominal", "JESUp").
        recid: Optional specific .txt file ID(s).

    Returns:
        dict: Updated file_dict.
    """
    logger.info(f"Processing: {process}")
    files = get_paths(folder, recid)
    nevts_list, nevts_wt_list = num_events_list(files)

    file_dict[process].update(
        {
            variation: {
                "files": [
                    {"path": f, "nevts": n, "nevts_wt": n_wt}
                    for f, n, n_wt in zip(files, nevts_list, nevts_wt_list)
                ],
                "nevts_total": sum(nevts_list),
                "nevts_wt_total": sum(nevts_wt_list),
            }
        }
    )

    os.makedirs("datasets/nanoaods_jsons_per_proces", exist_ok=True)
    # Create partial backup JSON file
    write_to_file(
        file_dict,
        f"datasets/nanoaods_jsons_per_proces/nanoaods_{process}_{variation}.json",
    )

    return file_dict


def write_to_file(file_dict: dict, path: str):
    """
    Write the file_dict to a JSON file at the given path.

    Args:
        file_dict (dict): The dictionary to write.
        path (str): Destination path for the output JSON file.
    """
    with open(path, "w") as f:
        json.dump(file_dict, f, indent=4)
        f.write("\n")


# Main execution block
if __name__ == "__main__":
    file_dict = defaultdict(dict)

    for process, folder in PROCESS_TO_DATASETS_FOLDER.items():
        update_dict(
            file_dict, process, folder, variation="nominal", recid=None
        )

    # Final master JSON
    write_to_file(file_dict, "datasets/nanoaods.json")
