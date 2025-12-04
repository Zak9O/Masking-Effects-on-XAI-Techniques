import os
import masking_effects_on_xai_techniques.utils as utils
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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
    ANONYMIZATION_MODELS = [
        ("t", "t_closeness"),
        ("l", "l_diversity"),
        ("k", "k_anonymity"),
        ("a", "alpha_k_anonymity"),
    ]

    EXPLANATION_METHODS = ["shap", "lime"]

    def __init__(self, dataset: str, root: str, skip: list[str] | None = None) -> None:
        self.dataset = dataset
        self.explanations: dict[str, dict[str, list[Explanation]]] = {}
        self.kendaltau: dict[str, dict[str, list[object]]] = {}
        for method in self.EXPLANATION_METHODS:
            self.explanations[method] = {}
            self.kendaltau[method] = {}

        if skip is None:
            skip = []

        self.models = [
            (id, model) for id, model in self.ANONYMIZATION_MODELS if id not in skip
        ]
        for method in self.EXPLANATION_METHODS:
            for id, model in self.models:
                paths = self.get_abs_paths(
                    f"{root}/data/{self.dataset}/{method}/{model}", id in ["l", "k"]
                )
                rankings = self.load_explanations(paths)
                self.explanations[method][id] = rankings
                self.kendaltau[method][id] = self.calcualte_kendaltau(rankings)

    def load_explanations(self, paths: list[str]) -> list[Explanation]:
        explanations = []
        for path in paths:
            array = np.load(path)
            name = os.path.basename(path).split(".csv", 1)[0]
            accuracy = float(array[0][1])
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

    def calcualte_kendaltau(self, rankings: list[Explanation]):
        kendaltau = []
        model_ranking = rankings[0].get_ranking()
        for e in rankings[1:]:
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

    def plot_kendaltau(self, method: str) -> None:
        for id, model in self.models:
            statistics = [n.correlation for n in self.kendaltau[method][id]]  # pyright: ignore[reportAttributeAccessIssue]
            pvalues = [n.pvalue for n in self.kendaltau[method][id]]  # pyright: ignore[reportAttributeAccessIssue]
            run_labels = [e.name for e in self.explanations[method][id][1:]]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

            ax1.bar(run_labels, statistics, color="skyblue")
            ax1.set_ylabel("Kendall's Tau Statistic ($\tau$)")
            ax1.set_title(
                f"Kendall's Tau Correlation and Significance (vs. Baseline Run) for {self.dataset} and {model}"
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

    def plot_accuracy(self, method: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        max_n = 0

        for id, model_name in self.models:
            explanations = self.explanations[method][id]

            accuracies = [e.accuracy for e in explanations]
            n_values = range(1, len(accuracies) + 1)

            ax.plot(n_values, accuracies, marker="o", label=model_name)

            max_n = max(max_n, len(accuracies))

        ax.set_xlabel("N")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy vs. N for {self.dataset} using {method}")
        ax.set_xticks(range(1, max_n + 1))
        ax.set_xlim(0.5, max_n + 0.5)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend(title="Anonymization Model")
        plt.tight_layout()
        plt.show()

    def plot_accuracy2(self, method: str) -> None:
        for id, model in self.models:
            explanations = self.explanations[method][id]

            names = [e.name for e in explanations]
            accuracies = [e.accuracy for e in explanations]

            plt.figure(figsize=(10, 6))
            plt.plot(names, accuracies, marker="o")
            plt.xlabel("Explanation Name")
            plt.ylabel("Accuracy")
            plt.title(
                f"Accuracy vs. Explanation Name for {self.dataset} and {model} using {method}"
            )
            plt.xticks(rotation=45, ha="right")
            plt.ylim(0, 1)  # Accuracy is between 0 and 1
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.show()

    def plot_line(self, method: str) -> None:
        for id, model in self.models:
            explanations = self.explanations[method][id]

            # Create a figure and an axes.
            # TODO: Multiple features with same color. Fix this if relevant
            fig, ax = plt.subplots(figsize=(12, 8))
            model_names = [e.name for e in explanations]
            y_locs = range(len(model_names))

            all_items = set(item for e in explanations for item in e.get_ranking())

            for item in all_items:
                ranks = []

                y_vals = []
                for i, e in enumerate(explanations):
                    ranking = e.get_ranking()
                    if item in ranking:
                        ranks.append(ranking.index(item) + 1)

                        y_vals.append(i)
                    else:
                        ranks.append(None)
                        y_vals.append(i)
                ax.plot(ranks, y_locs, marker="o", linestyle="-", label=item)

            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
            ax.set_yticks(y_locs)
            ax.set_yticklabels(model_names)
            ax.invert_yaxis()

            ax.set_xlim(0, 12)

            ax.set_xticks(range(1, len(all_items) + 1))
            ax.set_xlabel("Rank")

            ax.grid(axis="x", linestyle="--", alpha=0.5)
            for s in ["top", "right", "left"]:
                ax.spines[s].set_visible(False)
            ax2 = ax.twinx()
            accuracies = [e.accuracy for e in explanations]
            ax2.set_yticks(y_locs)  # Align ticks with the primary y-axis
            ax2.set_yticklabels(
                [f"{acc:.2f}" for acc in accuracies]
            )  # Display formatted accuracies
            ax2.set_ylabel("Model Accuracy")  # Label the new axis
            ax2.invert_yaxis()  # Keep y-axis inversion consistent
            ax2.set_ylim(ax.get_ylim())  # Crucial: Match Y-limits of the primary axis

            plt.title(f"Feature Rank Comparison: {self.dataset} for {model}")
            plt.tight_layout()
            plt.show()
