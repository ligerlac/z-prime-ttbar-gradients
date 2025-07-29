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
from . import jax_stats as jax_stats
#from . import mva as mva


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
    "jax_stats",
    "mva",
]


def __dir__():
    return __all__


def set_logging(name) -> None:
    """Sets up customized and verbose logging output.

    Logging can be alternatively customized with the Python ``logging`` module directly.
    """
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(message)s"
    )
    logging.getLogger(f"utils::{name}").setLevel(logging.DEBUG)
