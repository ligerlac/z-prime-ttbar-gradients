"""Central output routing.

:class:`OutputManager` owns a root directory and a small set of named
subdirectories (``histograms``, ``plots``, ``fits``, ``fileset``).
Indexing it returns the path of the named category, ready to be used
as the parent of a write target. New categories can be added or
existing ones renamed at runtime via item assignment; the
corresponding directory is created on the spot.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graep.config.output import OutputSpec

DEFAULT_CATEGORIES: dict[str, str] = {
    "histograms": "histograms",
    "plots": "plots",
    "fits": "fits",
    "fileset": "datasets",
}


class OutputManager:
    """Tracks a root output directory and the per-category subdirs under it.

    Parameters
    ----------
    root
        Directory under which all category subdirectories live.
        Created if it doesn't exist.
    categories
        Optional mapping from category name to its subdirectory name
        (relative to ``root``). When ``None``, the framework default
        :data:`DEFAULT_CATEGORIES` is used. Pass an explicit mapping
        to rename or add categories at construction time.

    Notes
    -----
    The subdirectories are created on construction so callers can
    write files under them without extra ``mkdir`` boilerplate.
    Adding a new category at runtime via ``mgr["foo"] = "bar"`` also
    creates the directory immediately.
    """

    def __init__(
        self,
        root: Path,
        *,
        categories: Mapping[str, str] | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self._categories: dict[str, str] = (
            dict(categories) if categories is not None else dict(DEFAULT_CATEGORIES)
        )
        self.root.mkdir(parents=True, exist_ok=True)
        for subdir in self._categories.values():
            (self.root / subdir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_spec(cls, spec: OutputSpec) -> OutputManager:
        """Construct a manager from an :class:`OutputSpec`.

        Lets the user keep the spec as pure data in their config and
        instantiate the runtime object from the notebook.
        """
        return cls(spec.root, categories=spec.categories)

    def __getitem__(self, category: str) -> Path:
        if category not in self._categories:
            msg = (
                f"unknown output category {category!r}; "
                f"known: {sorted(self._categories)}"
            )
            raise KeyError(msg)
        return self.root / self._categories[category]

    def __setitem__(self, category: str, subdir: str) -> None:
        """Add or rename a category at runtime, creating its directory."""
        self._categories[category] = subdir
        (self.root / subdir).mkdir(parents=True, exist_ok=True)

    def __contains__(self, category: object) -> bool:
        return category in self._categories

    def __iter__(self) -> Iterator[str]:
        return iter(self._categories)

    def __len__(self) -> int:
        return len(self._categories)

    def __repr__(self) -> str:
        return (
            f"OutputManager(root={self.root!s}, "
            f"categories={sorted(self._categories)})"
        )
