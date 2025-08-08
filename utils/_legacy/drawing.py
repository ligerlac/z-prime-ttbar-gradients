from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import numpy.typing as npt


class Draw:
    def __init__(
        self, output_dir: Optional[str] = "plots", interactive: bool = False
    ):
        output_dir = Path("plots")
        self.output_dir = output_dir
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        self.interactive = interactive
        self.process_color_dict = dict()
        self.model_color_dict = dict()
        hep.style.use("CMS")

    def _parse_name(self, name: str) -> str:
        return name.replace(" ", "-").lower()

    def _get_process_color(self, label: str) -> str:
        return self.process_color_dict.get(
            label, next(plt.gca()._get_lines.prop_cycler)["color"]
        )

    def _get_model_color(self, label: str) -> str:
        return self.model_color_dict.get(
            label, next(plt.gca()._get_lines.prop_cycler)["color"]
        )

    def _save_fig(self, name: str) -> None:
        plt.savefig(
            # f"{self.output_dir}/{self._parse_name(name)}.png", bbox_inches="tight"
            f"{self.output_dir}/{self._parse_name(name)}.pdf",
            bbox_inches="tight",
        )
        if self.interactive:
            plt.show()
        plt.close()

    def plot_loss_history(
        self,
        training_loss: npt.NDArray,
        validation_loss: npt.NDArray,
        name: str,
    ):
        plt.plot(
            np.arange(1, len(training_loss) + 1),
            training_loss,
            label="Training",
        )
        plt.plot(
            np.arange(1, len(validation_loss) + 1),
            validation_loss,
            label="Validation",
        )
        plt.legend(loc="upper right")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        self._save_fig(name)

    def plot_loss_histories(
        self, loss_dict: dict[str, (npt.NDArray, npt.NDArray)], name: str
    ):
        for model_name, (train_loss, val_loss) in loss_dict.items():
            c = next(plt.gca()._get_lines.prop_cycler)["color"]
            plt.plot(
                np.arange(1, len(train_loss) + 1),
                train_loss,
                color=c,
                label=f"{model_name} (Training)",
            )
            plt.plot(
                np.arange(1, len(val_loss) + 1),
                val_loss,
                color=c,
                ls=":",
                label=f"{model_name} (Validation)",
            )
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
            print(
                f"label: {label}, shape: {values.shape}, "
                + f"min: {values.min()}, max: {values.max()}"
            )
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
        self._save_fig(name)
