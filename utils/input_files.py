"""
Utilities for downloading and managing input files.

Taken from the AGC project:
https://github.com/MoAly98/analysis-grand-challenge/blob/main/analyses/cms-open-data-ttbar/utils/file_input.py
"""

import json


def construct_fileset(n_files_max_per_sample, preprocessor="uproot"):

    # x-secs are in pb
    xsec_info = {
        "signal": 1.0,
        "ttbar_semilep": 831.76 * 0.438,
        "ttbar_had": 831.76 * 0.457,
        "ttbar_lep": 831.76 * 0.105,
        "wjets": 61526.7,
        "data": 1.0,
    }

    # list of files
    with open("datasets/nanoaods.json") as f:
        file_info = json.load(f)

    # process into "fileset" summarizing all info
    fileset = {}
    for process in file_info.keys():
        # if process == "data":   continue  # skip data

        for variation in file_info[process].keys():
            file_list = file_info[process][variation]["files"]
            if n_files_max_per_sample != -1:
                file_list = file_list[
                    :n_files_max_per_sample
                ]  # use partial set of samples

            file_paths = {f["path"]: "Events" for f in file_list}
            nevts_total = sum([f["nevts"] for f in file_list])
            nevts_wt_total = sum([f["nevts_wt"] for f in file_list])

            metadata = {
                "process": process,
                "variation": variation,
                "nevts": nevts_total,
                "nevts_wt": nevts_wt_total,
                "xsec": xsec_info[process],
            }
            fileset.update(
                {
                    f"{process}__{variation}": {
                        "files": file_paths,
                        "metadata": metadata,
                    }
                }
            )
        if preprocessor == "uproot":
            file_list = file_info[process][variation]["files"]
            a_file = file_list[0]
            process_root_path = "/".join(a_file["path"].split("/")[:-2])
            process_all_files = f"{process_root_path}/*/*.root"
            nevts_total = sum([f["nevts"] for f in file_list])
            nevts_wt_total = sum([f["nevts_wt"] for f in file_list])
            metadata = {
                "process": process,
                "variation": variation,
                "nevts": nevts_total,
                "nevts_wt": nevts_wt_total,
                "xsec": xsec_info[process],
            }
            file_paths = {process_all_files: "Events"}
            fileset.update(
                {
                    f"{process}__{variation}": {
                        "files": file_paths,
                        "metadata": metadata,
                    }
                }
            )

    return fileset
