import logging

from . import build_fileset_json as build_fileset_json
from . import configuration as configuration
from . import input_files as input_files
from . import output_files as output_files
from . import prepproc as prepproc
from . import schema as schema
from . import systematics as systematics


__all__ = [
    "configuration",
    "output_files",
    "input_files",
    "prepproc",
    "schema",
    "systematics",
    "build_fileset_json",
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
