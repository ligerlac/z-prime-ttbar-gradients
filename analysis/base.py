import gzip
import logging
from typing import Any, Callable, Literal, Optional, Union
import warnings

import awkward as ak
from coffea.nanoevents import NanoAODSchema
from correctionlib import CorrectionSet
import vector

from utils.schema import GoodObjectMasksConfig

# -----------------------------
# Register backends
# -----------------------------
ak.jax.register_and_check()
vector.register_awkward()

# -----------------------------
# Logging Configuration
# -----------------------------

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ZprimeAnalysis")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")


def is_jagged(arraylike) -> bool:
    try:
        return ak.num(arraylike, axis=1) is not None
    except Exception:
        return False

# -----------------------------
# Base class
# -----------------------------
class Analysis:
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize ZprimeAnalysis with configuration for systematics, corrections,
        and channels.

        Parameters
        ----------
        config : dict
            Configuration dictionary with 'systematics', 'corrections', 'channels',
            and 'general'.
        """
        self.config = config
        self.channels = config.channels
        self.systematics = config.systematics
        self.corrections = config.corrections
        self.corrlib_evaluators = self._load_correctionlib()


    def _load_correctionlib(self) -> dict[str, CorrectionSet]:
        """
        Load correctionlib JSON files into evaluators.

        Returns
        -------
        dict
            Dictionary of correction name to CorrectionSet evaluator.
        """
        evaluators = {}
        for systematic in self.corrections:
            if not systematic.get("use_correctionlib"):
                continue
            name = systematic["name"]
            path = systematic["file"]

            if path.endswith(".json.gz"):
                with gzip.open(path, "rt") as f:
                    evaluators[name] = CorrectionSet.from_string(
                        f.read().strip()
                    )
            elif path.endswith(".json"):
                evaluators[name] = CorrectionSet.from_file(path)
            else:
                raise ValueError(f"Unsupported correctionlib format: {path}")

        return evaluators

    def get_object_copies(self, events: ak.Array) -> dict[str, ak.Array]:
        """
        Extract a dictionary of objects from the NanoEvents array.

        Parameters
        ----------
        events : ak.Array
            Input events.

        Returns
        -------
        dict
            Dictionary of field name to awkward array.
        """
        return {field: events[field] for field in events.fields}

    def get_good_objects(
        self, object_copies: dict[str, ak.Array],
        masks: list[GoodObjectMasksConfig] = []
    ) -> dict[str, ak.Array]:

        good_objects = {}
        for obj_mask in masks:
            mask_args = self._get_function_arguments(
                obj_mask.use, object_copies
            )
            mask = obj_mask.function(*mask_args)
            if not isinstance(mask, ak.Array):
                raise TypeError(
                    f"Expected mask to be an awkward array, got {type(mask)}"
                )

            good_objects[obj_mask.object] = object_copies[obj_mask.object][mask]

        return good_objects

    def apply_correctionlib(
        self,
        name: str,
        key: str,
        direction: Literal["up", "down", "nominal"],
        correction_arguments: list[ak.Array],
        target: Optional[Union[ak.Array, list[ak.Array]]] = None,
        op: Optional[str] = None,
        transform: Optional[Callable[..., Any]] = None,
    ) -> Union[ak.Array, list[ak.Array]]:
        """
        Apply a correction using correctionlib.
        """
        logger.info(
            f"Applying correctionlib correction: name={name}, "
            f"key={key}, direction={direction}"
        )
        if transform is not None:
            correction_arguments = transform(*correction_arguments)

        flat_args, counts_to_unflatten = [], []
        for arg in correction_arguments:
            if is_jagged(arg):
                flat_args.append(ak.flatten(arg))
                counts_to_unflatten.append(ak.num(arg))
            else:
                flat_args.append(arg)

        correction = self.corrlib_evaluators[name][key].evaluate(
            *flat_args, direction
        )

        if counts_to_unflatten:
            correction = ak.unflatten(correction, counts_to_unflatten[0])

        if target is not None and op is not None:
            if isinstance(target, list):
                correction = ak.to_backend(correction, ak.backend(target[0]))
                return [self.apply_op(op, t, correction) for t in target]
            else:
                correction = ak.to_backend(correction, ak.backend(target))
                return self.apply_op(op, target, correction)

        return correction

    def apply_syst_fn(
        self,
        name: str,
        fn: Callable[..., ak.Array],
        args: list[ak.Array],
        affects: Union[ak.Array, list[ak.Array]],
        op: str,
    ) -> Union[ak.Array, list[ak.Array]]:
        """
        Apply function-based systematic variation.
        """
        logger.debug(f"Applying function-based systematic: {name}")
        correction = fn(*args)
        if isinstance(affects, list):
            return [self.apply_op(op, a, correction) for a in affects]
        else:
            return self.apply_op(op, affects, correction)

    def apply_op(self, op: str, lhs: ak.Array, rhs: ak.Array) -> ak.Array:
        """
        Apply a binary operation.
        """
        if op == "add":
            return lhs + rhs
        elif op == "mult":
            return lhs * rhs
        else:
            raise ValueError(f"Unsupported operation: {op}")

    def _get_function_arguments(
        self,
        use: list[tuple[str, Optional[str]]],
        object_copies: dict[str, ak.Array],
    ) -> list[ak.Array]:
        """
        Extract correction arguments from object_copies.
        """
        return [
            object_copies[obj][var] if var is not None else object_copies[obj]
            for obj, var in use
        ]

    def _get_targets(
        self,
        target: Union[tuple[str, str], list[tuple[str, str]]],
        object_copies: dict[str, ak.Array],
    ) -> list[ak.Array]:
        """
        Extract one or more target arrays from object_copies.
        """
        targets = target if isinstance(target, list) else [target]
        return [object_copies[obj][var] for obj, var in targets]

    def _set_targets(
        self,
        target: Union[tuple[str, str], list[tuple[str, str]]],
        object_copies: dict[str, ak.Array],
        new_values: Union[ak.Array, list[ak.Array]],
    ) -> None:
        """
        Set corrected values in object_copies.
        """
        targets = target if isinstance(target, list) else [target]
        for (obj, var), val in zip(targets, new_values):
            object_copies[obj][var] = val

    def apply_object_corrections(
        self,
        object_copies: dict[str, ak.Array],
        corrections: list[dict[str, Any]],
        direction: Literal["up", "down", "nominal"] = "nominal",
    ) -> dict[str, ak.Array]:
        """
        Apply object-level corrections to input object copies.
        """
        for corr in corrections:
            if corr["type"] != "object":
                continue
            args = self._get_function_arguments(corr["use"], object_copies)
            targets = self._get_targets(corr["target"], object_copies)
            op = corr["op"]
            key = corr.get("key")
            transform = corr.get("transform", lambda *x: x)
            dir_map = corr.get("up_and_down_idx", ["up", "down"])
            corr_dir = (
                dir_map[0 if direction == "up" else 1]
                if direction in ["up", "down"]
                else "nominal"
            )

            if corr.get("use_correctionlib", False):
                corrected = self.apply_correctionlib(
                    corr["name"], key, corr_dir, args, targets, op, transform
                )
            else:
                fn = corr.get(f"{direction}_function")
                corrected = (
                    self.apply_syst_fn(corr["name"], fn, args, targets, op)
                    if fn
                    else targets
                )

            self._set_targets(corr["target"], object_copies, corrected)

        return object_copies

    def apply_event_weight_correction(
        self,
        weights: ak.Array,
        systematic: dict[str, Any],
        direction: Literal["up", "down"],
        object_copies: dict[str, ak.Array],
    ) -> ak.Array:
        """
        Apply event-level correction to weights.
        """
        if systematic["type"] != "event":
            return weights

        args = self._get_function_arguments(systematic["use"], object_copies)
        op = systematic["op"]
        key = systematic.get("key")
        transform = systematic.get("transform", lambda *x: x)
        dir_map = systematic.get("up_and_down_idx", ["up", "down"])
        corr_dir = dir_map[0 if direction == "up" else 1]

        if systematic.get("use_correctionlib", False):
            return self.apply_correctionlib(
                systematic["name"], key, corr_dir, args, weights, op, transform
            )
        else:
            fn = systematic.get(f"{direction}_function")
            return (
                self.apply_syst_fn(systematic["name"], fn, args, weights, op)
                if fn
                else weights
            )


    def compute_ghost_observables(
        self, obj_copies: dict[str, ak.Array], jax=False,
    ) -> dict[str, ak.Array]:
        for ghost in self.config.ghost_observables:
            if not ghost.works_with_jax and jax:
                logger.warning(
                    f"Ghost observable {ghost.names} does not work with JAX, skipping."
                )
                continue

            logger.info(f"Computing ghost observables {ghost.names}")
            ghost_args = self._get_function_arguments(ghost["use"], obj_copies)
            ghost_outputs = ghost["function"](*ghost_args)

            if not isinstance(ghost_outputs, (list, tuple)):
                ghost_outputs = [ghost_outputs]

            names = (
                ghost.names if isinstance(ghost.names, list) else [ghost.names]
            )
            colls = (
                ghost.collections
                if isinstance(ghost.collections, list)
                else [ghost.collections] * len(names)
            )
            # update object_copies with ghost outputs
            for out, name, coll in zip(ghost_outputs, names, colls):
                if (
                    isinstance(out, ak.Array)
                    and len(ak.fields(out)) == 1
                    and name in out.fields
                ):
                    out = out[name]
                if coll in obj_copies:
                    try:
                        # add new field to existing awkward array
                        obj_copies[coll][name] = out
                    # happens if we are adding a field to an awkward
                    # array with no fields (scalar fields)
                    except ValueError as e:
                        raise e
                else:
                    # create new awkward array with single field
                    obj_copies[coll] = ak.Array({name: out})
        return obj_copies