from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import awkward as ak


def nested_defaultdict_to_dict(nested_structure: defaultdict) -> dict:
    """
    Recursively convert any nested defaultdicts into standard Python dictionaries.

    Parameters
    ----------
    nested_structure : Any
        A nested structure possibly containing defaultdicts.

    Returns
    -------
    dict
        Fully converted structure using built-in dict.
    """
    if isinstance(nested_structure, (defaultdict, dict)):
        return {key: nested_defaultdict_to_dict(value) for key, value in nested_structure.items()}
    return nested_structure


def recursive_to_backend(data_structure: Any, backend: str = "jax") -> Any:
    """
    Recursively convert all Awkward Arrays in a data structure to the specified backend.

    Parameters
    ----------
    data_structure : Any
        Input data structure possibly containing Awkward Arrays.
    backend : str
        Target backend to convert arrays to (e.g. 'jax', 'cpu').

    Returns
    -------
    Any
        Data structure with Awkward Arrays converted to the desired backend.
    """
    if isinstance(data_structure, ak.Array):
        # Convert only if not already on the target backend
        return ak.to_backend(data_structure, backend) if ak.backend(data_structure) != backend else data_structure

    if isinstance(data_structure, Mapping):
        # Recurse into dictionary values
        return {key: recursive_to_backend(value, backend) for key, value in data_structure.items()}

    if isinstance(data_structure, Sequence) and not isinstance(data_structure, (str, bytes)):
        # Recurse into list or tuple elements
        return [recursive_to_backend(value, backend) for value in data_structure]

    # Leave unchanged if not an Awkward structure
    return data_structure
