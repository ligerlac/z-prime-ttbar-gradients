import logging

from . import build_fileset_json as build_fileset_json
from . import configuration as configuration
from . import cuts as cuts
from . import input_files as input_files
from . import observables as observables
from . import output_files as output_files
from . import preproc as preproc
from . import schema as schema
from . import systematics as systematics


__all__ = [
    "build_fileset_json",
    "configuration",
    "cuts",
    "input_files",
    "observables",
    "output_files",
    "preproc",
    "schema",
    "systematics",
]


def __dir__():
    return __all__


def set_logging() -> None:
    """Sets up customized and verbose logging output.

    Logging can be alternatively customized with the Python ``logging`` module directly.
    """
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(message)s"
    )
    logging.getLogger("ZprimeAnalysis").setLevel(logging.DEBUG)
    # logging.getLogger("pyhf").setLevel(logging.INFO)
