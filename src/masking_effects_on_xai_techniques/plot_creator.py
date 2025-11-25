import os
import masking_effects_on_xai_techniques.utils as utils
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Explanation:
    def __init__(
        self, name: str, accuracy: float, importance: list[tuple[str, float]]
    ) -> None:
        self.name = name
        self.accuracy = accuracy
        self._ranking = [x[0] for x in importance]
        self._ranking_values = [x[1] for x in importance]

    def get_ranking(self) -> list[str]:
        return self._ranking


class Comparer:
    def __init__(self, name: str, path: str) -> None:
        self.name = name
        paths = self.get_abs_paths(path)
        self.explanations = self.load_explanations(paths)

    def load_explanations(self, paths: list[str]) -> list[Explanation]:
        explanations = []
        for path in paths:
            array = np.load(path)
            name = os.path.basename(path).split(".csv", 1)[0]
            accuracy = array[0][0]
            importance = array[1:]
            explanations.append(Explanation(name, accuracy, importance))
        return explanations

    def get_abs_paths(self, path: str) -> list[str]:
        paths = sorted([path for path in utils.get_absolute_file_paths(path)])
        paths.append(os.path.abspath(f"{Path(path).parent.absolute()}/clean.csv.npy"))
        paths = list(reversed(paths))
        return paths

    def plot_line(self) -> None:
        fig, ax = plt.subplots(figsize=(12, 8))
        model_names = [e.name for e in self.explanations]
        y_locs = range(len(model_names))

        all_items = set(item for e in self.explanations for item in e.get_ranking())

        for item in all_items:
            ranks = []

            y_vals = []
            for i, e in enumerate(self.explanations):
                ranking = e.get_ranking()
                if item in ranking:
                    ranks.append(ranking.index(item) + 1)

                    y_vals.append(i)
                else:
                    ranks.append(None)
                    y_vals.append(i)

            p = ax.plot(ranks, y_vals, marker="o", linewidth=2, alpha=0.8)
            color = p[0].get_color()

            if ranks[0] is not None:
                ax.text(
                    ranks[0],
                    -0.2,
                    item,
                    ha="left",
                    va="bottom",
                    color=color,
                    fontweight="bold",
                    rotation=45,
                    fontsize=9,
                )

            if ranks[-1] is not None:
                ax.text(
                    ranks[-1],
                    len(model_names) - 0.8,
                    item,
                    ha="right",
                    va="top",
                    color=color,
                    fontweight="bold",
                    rotation=45,
                    fontsize=9,
                )

        ax.set_yticks(y_locs)
        ax.set_yticklabels(model_names)
        ax.invert_yaxis()

        # Powerpoint magic happening here. Making room between labels and x-axis
        ax.set_ylim(12, -1)
        ax.set_xlim(0, 12)

        ax.set_xticks(range(1, len(all_items) + 1))
        ax.set_xlabel("Rank")

        ax.grid(axis="x", linestyle="--", alpha=0.5)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)

        plt.title(f"Feature Rank Comparison: {self.name}")
        plt.tight_layout()
        plt.show()

    def plot_boxes(self) -> None:
        if not self.explanations:
            print("No explanations to plot.")
            return

        all_features = set()
        for exp in self.explanations:
            all_features.update(exp.get_ranking())
        sorted_features = sorted(list(all_features))

        cmap = plt.get_cmap("Set3")

        if len(sorted_features) > 12:
            cmap = plt.get_cmap("tab20")

        feature_to_color = {
            f: cmap(i / max(1, len(sorted_features) - 1))
            for i, f in enumerate(sorted_features)
        }

        num_rows = len(self.explanations)
        max_rank = max(len(exp.get_ranking()) for exp in self.explanations)

        fig, ax = plt.subplots(figsize=(12, 0.8 * num_rows + 1))

        for row_idx, exp in enumerate(self.explanations):
            ranking = exp.get_ranking()
            y_coord = num_rows - 1 - row_idx

            for rank_idx, feature_name in enumerate(ranking):
                color = feature_to_color.get(feature_name, (0.5, 0.5, 0.5, 1.0))

                rect = patches.Rectangle(
                    (rank_idx, y_coord),
                    width=1,
                    height=1,
                    linewidth=0.5,
                    edgecolor="grey",
                    facecolor=color,
                )
                ax.add_patch(rect)

        ax.set_xlim(0, max_rank)
        ax.set_ylim(0, num_rows)

        ax.axhline(y=num_rows - 1, color="black", linewidth=3)

        ax.set_xticks(np.arange(max_rank) + 0.5)
        ax.set_xticklabels(np.arange(1, max_rank + 1))
        ax.set_xlabel("Rank (Importance Order)")

        row_names = [e.name for e in self.explanations]
        ax.set_yticks(np.arange(num_rows) + 0.5)
        ax.set_yticklabels(row_names[::-1])

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)

        legend_handles = [
            patches.Patch(color=feature_to_color[f], label=f) for f in sorted_features
        ]
        ax.legend(
            handles=legend_handles,
            title="Features",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

        plt.title(f"Feature Rank Comparison: {self.name}")
        plt.tight_layout()
        plt.show()
