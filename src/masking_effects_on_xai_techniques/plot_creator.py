import os
import masking_effects_on_xai_techniques.utils as utils
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from copy import copy


class Explanation:
    def __init__(
        self, name: str, accuracy: float, importance: list[tuple[str, float]]
    ) -> None:
        self.name = name
        self.accuracy = accuracy
        self._ranking_values = [float(x[1]) for x in importance if float(x[1]) > 0]
        self._ranking = [
            x[0] for i, x in enumerate(importance) if i < len(self._ranking_values)
        ]

    def get_ranking(self) -> list[str]:
        return self._ranking


class Comparer:
    def __init__(self, name: str, path: str, ascending=False) -> None:
        self.name = name
        paths = self.get_abs_paths(path, ascending)
        self.explanations = self.load_explanations(paths)
        self.kendaltau = self.calcualte_kendaltau()

    def load_explanations(self, paths: list[str]) -> list[Explanation]:
        explanations = []
        for path in paths:
            array = np.load(path)
            name = os.path.basename(path).split(".csv", 1)[0]
            accuracy = array[0][0]
            importance = array[1:]
            explanations.append(Explanation(name, accuracy, importance))
        return explanations

    def get_abs_paths(self, path: str, ascending: bool) -> list[str]:
        def get_number_from_filename(file_path):
            filename = os.path.basename(file_path)
            number_str = filename.split(".csv.npy")[0]
            return float(number_str)

        paths = sorted(
            [path for path in utils.get_absolute_file_paths(path)],
            key=get_number_from_filename,
        )
        if ascending:
            paths = list(reversed(paths))
        paths.append(os.path.abspath(f"{Path(path).parent.absolute()}/clean.csv.npy"))

        paths = list(reversed(paths))
        return paths

    def calcualte_kendaltau(self):
        kendaltau = []
        model_ranking = self.explanations[0].get_ranking()
        for e in self.explanations[1:]:
            m_rnk, rnk = self.make_compatible(model_ranking, e.get_ranking())
            kendaltau.append(stats.kendalltau(m_rnk, rnk))
        return kendaltau

    def add_missing_categories(self, ref: list[str], rnk: list[str]) -> list[str]:
        out = copy(rnk)
        for feat in ref:
            if feat not in rnk:
                out.append(feat)
        return out

    def make_compatible(
        self, ref: list[str], rnk: list[str]
    ) -> tuple[list[str], list[str]]:
        rnk = self.add_missing_categories(ref, rnk)
        ref = self.add_missing_categories(rnk, ref)
        return (ref, rnk)

    def plot_kendaltau(self) -> None:
        statistics = [n.correlation for n in self.kendaltau]
        pvalues = [n.pvalue for n in self.kendaltau]
        run_labels = [e.name for e in self.explanations[1:]]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        ax1.bar(run_labels, statistics, color="skyblue")
        ax1.set_ylabel("Kendall's Tau Statistic ($\tau$)")
        ax1.set_title(
            f"Kendall's Tau Correlation and Significance (vs. Baseline Run) {self.name}"
        )
        ax1.grid(axis="y", linestyle="--", alpha=0.6)
        ax1.set_ylim(-1.1, 1.1)

        ax2.plot(
            run_labels, pvalues, color="red", marker="o", linestyle="-", linewidth=2
        )
        ax2.set_ylabel("P-value")
        ax2.set_xlabel("anonymity-value")
        ax2.grid(axis="y", linestyle="--", alpha=0.6)
        ax2.set_ylim(-0.05, 1.05)

        ax2.axhline(
            0.05,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Significance Threshold (0.05)",
        )
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_line(self) -> None:
        # TODO: Multiple features with same color. Fix this if relevant
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
        ax.set_ylim(len(model_names) + 1, -2)
        ax.set_xlim(0, 12)

        ax.set_xticks(range(1, len(all_items) + 1))
        ax.set_xlabel("Rank")

        ax.grid(axis="x", linestyle="--", alpha=0.5)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)

        plt.title(f"Feature Rank Comparison: {self.name}")
        plt.tight_layout()
        plt.show()
        # return ax, plt

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


def plot_comparison(
    comparisors: list[Comparer],
    series_names: list[str],
    title: str,
) -> None:
    run_labels = [e.name for e in comparisors[0].explanations[1:]]
    statistics_list = [
        [n.correlation for n in comparisor.kendaltau] for comparisor in comparisors
    ]
    pvalues_list = [
        [n.pvalue for n in comparisor.kendaltau] for comparisor in comparisors
    ]

    if len(statistics_list) != len(pvalues_list) or len(statistics_list) != len(
        series_names
    ):
        raise ValueError(
            "Input lists (statistics, pvalues, names) must have the same length."
        )

    n_series = len(statistics_list)
    x = np.arange(len(run_labels))
    width = 0.8 / n_series  # Dynamically calculate bar width

    # Define distinct colors/markers for plotting
    colors = ["skyblue", "orange", "lightgreen", "salmon"]
    markers = ["o", "s", "^", "D"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Subplot 1: Grouped Bar Chart ---
    for i, (kendall_tau, name) in enumerate(zip(statistics_list, series_names)):
        # Calculate offset to center the group of bars
        offset = (i - (n_series - 1) / 2) * width
        ax1.bar(
            x + offset, kendall_tau, width, label=name, color=colors[i % len(colors)]
        )

    ax1.set_ylabel("Kendall's Tau Statistic ($\\tau$)")
    ax1.set_title(f"Kendall's Tau Correlation Comparison for {title}")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.set_ylim(-1.1, 1.1)

    # --- Subplot 2: Line Chart ---
    for i, (pvals, name) in enumerate(zip(pvalues_list, series_names)):
        ax2.plot(
            x,
            pvals,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            linestyle="--" if i > 0 else "-",  # Dashed for secondary lines
            linewidth=2,
            label=f"{name} p-value",
        )

    ax2.axhline(
        0.05, color="black", linestyle="--", linewidth=1.5, label="Significance (0.05)"
    )

    ax2.set_ylabel("P-value")
    ax2.set_xlabel("anonymity-value")
    ax2.set_xticks(x)
    ax2.set_xticklabels(run_labels)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()

    plt.tight_layout()
    plt.show()
