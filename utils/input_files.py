import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

from tabulate import tabulate

defaul_dataset_json = Path("datasets/nanoaods.json")

# Configure module-level logger
logger = logging.getLogger(__name__)


def construct_fileset(
    max_files_per_sample: int,
    preprocessor: str = "uproot",
    json_path: Union[str, Path] = defaul_dataset_json,
) -> Dict[str, Dict[str, Any]]:
    """
    Build a structured fileset mapping for physics analyses, including file paths and metadata.

    This function reads dataset definitions from a JSON file and constructs a nested
    dictionary where each key is "<process>__<variation>" and values contain:
      - files: a mapping of file or glob patterns to ROOT TTrees
      - metadata: information on event counts, cross-sections, etc.

    Parameters
    ----------
    max_files_per_sample : int
        Maximum number of files to include for each sample. Use -1 to include all files.
    preprocessor : str, optional
        Type of file access to prepare. Supported values:
        - "uproot": use glob patterns for directory-level access
        - other: list each file individually
        Default is "uproot".
    json_path : str or Path, optional
        Path to the JSON configuration file specifying samples, variations, and file lists.
        Defaults to 'datasets/nanoaods.json'.

    Returns
    -------
    fileset : dict
        Nested mapping where each key "<process>__<variation>" maps to:
        - files (dict): {file_path_or_pattern: "Events"}
        - metadata (dict): {
            "process": str,
            "variation": str,
            "nevts": int,
            "nevts_wt": float,
            "xsec": float,
        }

    Raises
    ------
    FileNotFoundError
        If the JSON configuration file does not exist.
    ValueError
        If `max_files_per_sample` is less than -1.
    JSONDecodeError
        If the JSON file cannot be parsed.
    """
    # Validate inputs
    if max_files_per_sample < -1:
        raise ValueError(
            f"max_files_per_sample must be -1 or non-negative; got {max_files_per_sample}"
        )

    json_file = Path(json_path)
    if not json_file.is_file():
        raise FileNotFoundError(f"Dataset JSON file not found: {json_file}")

    # Load dataset definitions
    with json_file.open("r") as f:
        dataset_info = json.load(f)

    # Cross-section lookup (in picobarns)
    cross_section_map: Dict[str, float] = {
        "signal": 1.0,
        "ttbar_semilep": 831.76 * 0.438,
        "ttbar_had": 831.76 * 0.457,
        "ttbar_lep": 831.76 * 0.105,
        "wjets": 61526.7,
        "data": 1.0,
    }

    fileset: Dict[str, Dict[str, Any]] = {}

    # Iterate over each process and its systematic variations
    for process_name, variations in dataset_info.items():
        for variation_name, info in variations.items():
            # Extract raw file entries
            raw_entries = info.get("files", [])

            # Limit number of files if requested
            if max_files_per_sample != -1:
                raw_entries = raw_entries[:max_files_per_sample]

            # Compute total event counts
            total_events = sum(entry.get("nevts", 0) for entry in raw_entries)
            total_weighted = sum(entry.get("nevts_wt", 0.0) for entry in raw_entries)

            # Prepare metadata dict
            metadata = {
                "process": process_name,
                "variation": variation_name,
                "nevts": total_events,
                "nevts_wt": total_weighted,
                "xsec": cross_section_map.get(process_name, 0.0),
            }

            # Determine file path patterns or explicit paths
            if preprocessor == "uproot":
                # Use glob pattern for directory-based access
                if process_name == "data":
                    # CMS public EOS path for collision data
                    base_pattern = (
                        "root://eospublic.cern.ch//eos/opendata/cms/"
                        "Run2016*/SingleMuon/NANOAOD/"
                        "UL2016_MiniAODv2_NanoAODv9-v1"
                    )
                else:
                    # Deduce directory from first file path
                    first_path = raw_entries[0].get("path", "")
                    base_pattern = str(Path(first_path).parents[1])

                file_map = {f"{base_pattern}/*/*.root": "Events"}
            else:
                # Explicit file listings for other preprocessors
                file_map = {entry.get("path", ""): "Events" for entry in raw_entries}

            key = f"{process_name}__{variation_name}"
            fileset[key] = {"files": file_map, "metadata": metadata}

            logger.debug(f"Added fileset entry: {key} with {len(file_map)} files")

    logger.info(f"Constructed fileset with {len(fileset)} entries.")

    # --- Add summary table ---
    summary_data = []
    headers = ["Key", "Process", "Variation", "# Files"]
    for key, content in fileset.items():
        process = content["metadata"]["process"]
        variation = content["metadata"]["variation"]
        num_files = len(content["files"])
        summary_data.append([key, process, variation, num_files])

    # Sort by key for consistent output
    summary_data.sort(key=lambda x: x[0])

    logger.info("Fileset Summary:\n" + tabulate(summary_data, headers=headers, tablefmt="grid"))

    return fileset

# End of module
