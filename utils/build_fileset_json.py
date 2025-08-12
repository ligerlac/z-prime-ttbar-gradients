"""
Utility module to generate JSON metadata for NanoAOD ROOT datasets
used in ZprimeTtbar analysis.

This script provides:
  - `get_root_file_paths`: Read .txt listings to gather ROOT file paths.
  - `count_events_in_files`: Query each ROOT file for total and weighted event counts.
  - `NanoAODMetadataGenerator`: Class-based API to build and export metadata.
  - CLI entrypoint for standalone execution.

Original inspiration from:
https://github.com/iris-hep/analysis-grand-challenge/blob/main/datasets/
cms-open-data-2015/build_ntuple_json.py
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import awkward as ak
import uproot


# Configure module-level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Default directory mapping for each physics process
DEFAULT_DATASET_DIRECTORIES: Dict[str, Path] = {
    "signal": Path("datasets/signal/m400_w40/"),
    "ttbar_semilep": Path("datasets/ttbar_semilep/"),
    "ttbar_had": Path("datasets/ttbar_had/"),
    "ttbar_lep": Path("datasets/ttbar_lep/"),
    "wjets": Path("datasets/wjets/"),
    "data": Path("datasets/data/"),
}


def get_root_file_paths(
    directory: Union[str, Path],
    identifiers: Optional[Union[int, List[int]]] = None,
) -> List[Path]:
    """
    Collect ROOT file paths from text listings in a directory.

    Searches for `*.txt` files in the specified directory (or specific
    `<id>.txt` files if `identifiers` is given) and reads each line as a
    ROOT file path.

    Parameters
    ----------
    directory : str or Path
        Path to the folder containing text listing files.
    identifiers : int or list of ints, optional
        Specific listing file IDs (without `.txt`) to process. If `None`, all
        `.txt` files in the folder are used.

    Returns
    -------
    List[Path]
        Resolved list of ROOT file paths.

    Raises
    ------
    FileNotFoundError
        If no listing files are found or any specified file is missing.
    """
    dir_path = Path(directory)
    # Determine which text files to parse
    if identifiers is None:
        listing_files = list(dir_path.glob("*.txt"))
    else:
        ids = [identifiers] if isinstance(identifiers, int) else identifiers
        listing_files = [dir_path / f"{i}.txt" for i in ids]

    if not listing_files:
        raise FileNotFoundError(f"No listing files found in {dir_path}")

    root_paths: List[Path] = []
    for txt_file in listing_files:
        if not txt_file.is_file():
            raise FileNotFoundError(f"Missing listing file: {txt_file}")
        # Read all non-empty lines as file paths
        for line in txt_file.read_text().splitlines():
            path_str = line.strip()
            if path_str:
                root_paths.append(Path(path_str))

    return root_paths


def count_events_in_files(files: List[Path]) -> Tuple[List[int], List[float]]:
    """
    Query ROOT files for event counts and sum of generator weights.

    Opens each file with uproot, reads the "Events" TTree,
    and accumulates the number of entries and sum of "genWeight".

    Parameters
    ----------
    files : list of Path
        Paths to ROOT files to inspect.

    Returns
    -------
    num_entries : list of int
        Number of events in each file's "Events" tree.
    sum_weights : list of float
        Total of `genWeight` values per file.
    """
    num_entries: List[int] = []
    sum_weights: List[float] = []
    start_time = time.time()

    for idx, file_path in enumerate(files):
        # Log progress every 10 files and at completion
        if idx % 10 == 0 or idx == len(files) - 1:
            elapsed = int(time.time() - start_time)
            logger.info(
                f"Reading file {idx+1}/{len(files)} ({elapsed}s elapsed)"
            )
        try:
            with uproot.open(file_path) as root_file:
                tree = root_file["Events"]
                num_entries.append(tree.num_entries)
                weights = tree["genWeight"].array(library="ak")
                sum_weights.append(float(ak.sum(weights)))
        except Exception as err:
            logger.warning(f"Error reading {file_path}: {err}")
            num_entries.append(0)
            sum_weights.append(0.0)

    return num_entries, sum_weights


class NanoAODMetadataGenerator:
    """
    Class-based API to build and export metadata for NanoAOD datasets.

    Attributes
    ----------
    process_directories : Dict[str, Path]
        Map from process name to directory of `.txt` listings.
    output_directory : Path
        Directory where individual JSON files and master index will be written.

    Methods
    -------
    get_metadata(identifiers=None)
        Build metadata dict without writing to disk.
    run(identifiers=None)
        Generate metadata dict and write JSON files.
    """

    def __init__(
        self,
        process_directories: Optional[Dict[str, Union[str, Path]]] = None,
        output_directory: Union[
            str, Path
        ] = "datasets/nanoaods_jsons_per_process",
    ):
        # Initialize mapping from process to directory path
        raw_map = process_directories or DEFAULT_DATASET_DIRECTORIES
        self.process_directories: Dict[str, Path] = {
            name: Path(path) for name, path in raw_map.items()
        }
        # Ensure output directory exists
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def get_metadata(
        self, identifiers: Optional[Union[int, List[int]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Assemble metadata for each process/variation without file I/O.

        Parameters
        ----------
        identifiers : int or list of ints, optional
            Specific listing IDs to process. If None, all listings are used.

        Returns
        -------
        metadata : dict
            Nested structure: metadata[process]["nominal"] = {
              "files": [ {"path": str, "nevts": int, "nevts_wt": float}, ... ],
              "nevts_total": int,
              "nevts_wt_total": float
            }
        """
        results: Dict[str, Dict[str, Any]] = defaultdict(dict)

        for process_name, listing_dir in self.process_directories.items():
            logger.info(f"Processing process: {process_name}")
            try:
                file_paths = get_root_file_paths(listing_dir, identifiers)
            except FileNotFoundError as fnf:
                logger.error(fnf)
                continue

            entries_count, weight_sums = count_events_in_files(file_paths)
            variation_label = "nominal"
            file_records = [
                {"path": str(fp), "nevts": cnt, "nevts_wt": wt}
                for fp, cnt, wt in zip(file_paths, entries_count, weight_sums)
            ]

            results[process_name][variation_label] = {
                "files": file_records,
                "nevts_total": sum(entries_count),
                "nevts_wt_total": sum(weight_sums),
            }

        return results

    def run(self, identifiers: Optional[Union[int, List[int]]] = None) -> None:
        """
        Generate metadata and write individual JSON files and a master index.

        Parameters
        ----------
        identifiers : int or list of ints, optional
            Specific listing IDs to process. If None, all listings are used.
        """
        metadata = self.get_metadata(identifiers)

        # Write per-process JSON files
        for process_name, variations in metadata.items():
            for variation_label, data in variations.items():
                output_file = (
                    self.output_directory
                    / f"nanoaods_{process_name}_{variation_label}.json"
                )
                with output_file.open("w") as json_f:
                    json.dump(
                        {process_name: {variation_label: data}},
                        json_f,
                        indent=4,
                    )
                logger.debug(f"Wrote file: {output_file}")

        # Write master metadata index
        master_file = Path("datasets/nanoaods.json")
        master_file.parent.mkdir(parents=True, exist_ok=True)
        with master_file.open("w") as mfile:
            json.dump(metadata, mfile, indent=4)
        logger.info(f"Master metadata written to {master_file}")


# CLI entrypoint for standalone usage
def main() -> None:
    """
    Command-line interface: instantiate the generator and run.
    """
    logging.basicConfig(level=logging.INFO)
    generator = NanoAODMetadataGenerator()
    generator.run()


if __name__ == "__main__":
    main()
