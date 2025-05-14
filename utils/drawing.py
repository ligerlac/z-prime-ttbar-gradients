from typing import List, Callable, Union
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import numpy.typing as npt

from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# from utils import get_fractions_above_threshold


class Draw:
    def __init__(self, output_dir: Path = Path("plots"), interactive: bool = False):
        self.output_dir = output_dir
        self.interactive = interactive
        self.process_color_dict = dict()
        self.model_color_dict = dict()
        hep.style.use("CMS")

    def _parse_name(self, name: str) -> str:
        return name.replace(" ", "-").lower()
    
    def _get_process_color(self, label: str) -> str:
        return self.process_color_dict.get(label, next(plt.gca()._get_lines.prop_cycler)['color'])

    def _get_model_color(self, label: str) -> str:
        return self.model_color_dict.get(label, next(plt.gca()._get_lines.prop_cycler)['color'])

    def _save_fig(self, name: str) -> None:
        plt.savefig(
            # f"{self.output_dir}/{self._parse_name(name)}.png", bbox_inches="tight"
            f"{self.output_dir}/{self._parse_name(name)}.pdf", bbox_inches="tight"
        )
        if self.interactive:
            plt.show()
        plt.close()

    def plot_loss_history(
        self, training_loss: npt.NDArray, validation_loss: npt.NDArray, name: str
    ):
        plt.plot(np.arange(1, len(training_loss) + 1), training_loss, label="Training")
        plt.plot(
            np.arange(1, len(validation_loss) + 1), validation_loss, label="Validation"
        )
        plt.legend(loc="upper right")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        self._save_fig(name)

    def plot_loss_histories(
        self, loss_dict: dict[str, (npt.NDArray, npt.NDArray)], name: str
    ):
        for model_name, (train_loss, val_loss) in loss_dict.items():
            c = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.plot(np.arange(1, len(train_loss) + 1), train_loss, color=c, label=f"{model_name} (Training)")
            plt.plot(np.arange(1, len(val_loss) + 1), val_loss, color=c, ls=":", label=f"{model_name} (Validation)")
        plt.legend(loc="upper right")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        self._save_fig(name)

    def plot_simple_hist(
        self,
        data: dict[str, npt.NDArray],
        bins: Union[int, List[float]] = 50,
        x_label: str = "x",
        name: str = "histogram",
    ):
        plt.figure(figsize=(8, 4))
        print(f"Plotting {x_label}")
        for label, values in data.items():
            print(f"label: {label}, shape: {values.shape}, min: {values.min()}, max: {values.max()}")
            plt.hist(
                values,
                bins=bins,
                color=self.process_color_dict[label],
                alpha=0.5,
                label=label,
                density=True,
            )
        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel("a.u.")
        # plt.title(name)
        self._save_fig(name)