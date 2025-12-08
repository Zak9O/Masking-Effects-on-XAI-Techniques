import os
import masking_effects_on_xai_techniques.utils as utils
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from copy import copy


class Explanation:
    def __init__(
        self,
        name: str,
        accuracy: float,
        importance: list[tuple[str, float]],
        transform_n: int,
        transform_n_max: int,
    ) -> None:
        self.name = name
        self.accuracy = accuracy
        self._ranking_values = [float(x[1]) for x in importance if float(x[1]) > 0]
        self._ranking = [
            x[0] for i, x in enumerate(importance) if i < len(self._ranking_values)
        ]
        self.transform_n = transform_n
        self.transform_n_max = transform_n_max

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

    def __init__(
        self, dataset: str, root: str, classifier: str, skip: list[str] | None = None
    ) -> None:
        self.dataset = dataset
        self.explanations: dict[str, dict[str, list[Explanation]]] = {}
        self.kendaltau: dict[str, dict[str, list[object]]] = {}
        self.classifier = classifier
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
                    f"{root}/data/{self.classifier}/{method}/{self.dataset}/{model}",
                    id in ["l", "k"],
                )
                rankings = self.load_explanations(paths)
                self.explanations[method][id] = rankings
                self.kendaltau[method][id] = self.calcualte_kendaltau(rankings)

    def load_explanations(self, paths: list[str]) -> list[Explanation]:
        explanations = []
        for path in paths:
            try:
                array = np.load(path)
            except FileNotFoundError:
                print(f"File not found: {path}")
                continue
            if array[1][0] != "transform_n":
                # delete the path
                print(f"Deleting old file at {path}")
                os.remove(path)
                continue
            name = os.path.basename(path).split(".csv", 1)[0]
            accuracy = float(array[0][1])
            transform_n = int(float(array[1][1]))
            transform_n_max = int(float(array[2][1]))
            importance = array[3:]
            explanations.append(
                Explanation(name, accuracy, importance, transform_n, transform_n_max)
            )
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

    def plot_accuracy(self, method: str, baseline: float | None = None) -> None:
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

        # Default y-limits for accuracy (0..1). Expand if baseline is outside.
        default_low, default_high = 0.0, 1.05
        low, high = default_low, default_high
        if baseline is not None:
            # Ensure baseline is numeric
            try:
                b = float(baseline)
            except Exception:
                b = None
            else:
                if b < low:
                    low = max(b - 0.05, b - abs(0.05))
                if b > high:
                    high = b + 0.05

        ax.set_ylim(low, high)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Draw baseline horizontal dashed line if provided
        if baseline is not None and isinstance(baseline, (int, float)):
            ax.axhline(
                baseline,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=f"Share of most common target class ({baseline:.2f})",
            )

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

    def plot_line(self, method: str, only: list[str] | None = None) -> None:
        # Create a single figure with 2x2 subplots
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
        axes = axes.flatten()  # Flatten the 2x2 array for easy iteration

        fig.suptitle(
            f"Feature Rank Comparison for {self.dataset} using {method}", fontsize=16
        )

        for i, (id, model) in enumerate(self.models):
            if only is not None and model not in only:
                continue
            ax = axes[i]  # Get the current subplot axis
            explanations = self.explanations[method][id]

            model_names = [e.name for e in explanations]
            y_locs = range(len(model_names))

            all_items = set(item for e in explanations for item in e.get_ranking())
            # Use a deterministic ordering for colors/legend and exactly 20 colors
            all_items_list = sorted(all_items)
            cmap = plt.get_cmap("tab20")  # tab20 provides 20 distinct colors

            for idx, item in enumerate(all_items_list):
                ranks = []
                y_vals = []
                for j, e in enumerate(explanations):
                    ranking = e.get_ranking()
                    if item in ranking:
                        ranks.append(ranking.index(item) + 1)
                        y_vals.append(j)
                    else:
                        ranks.append(None)
                        y_vals.append(j)
                color = cmap(idx % 20)
                ax.plot(
                    ranks, y_locs, marker="o", linestyle="-", label=item, color=color
                )

            ax.legend(
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                borderaxespad=0.0,
                fontsize="small",
            )

            ax.set_yticks(y_locs)
            ax.set_yticklabels(model_names)
            ax.invert_yaxis()

            # Calculate the maximum rank for the current subplot
            max_rank_for_subplot = 0
            for e in explanations:
                max_rank_for_subplot = max(max_rank_for_subplot, len(e.get_ranking()))

            ax.set_xlim(0, max_rank_for_subplot + 1)
            ax.set_xticks(range(1, max_rank_for_subplot + 1))
            ax.set_xlabel("Rank")

            ax.grid(axis="x", linestyle="--", alpha=0.5)
            for s in ["top", "right", "left"]:
                ax.spines[s].set_visible(False)

            ax2 = ax.twinx()
            accuracies = [e.accuracy for e in explanations]
            generalization_levels = [
                round(100 * e.transform_n / e.transform_n_max) for e in explanations
            ]
            ax2.set_yticks(y_locs)
            # Generate new yticklabels with both accuracy and generalization level
            formatted_labels = []
            for j in range(len(accuracies)):
                label = f"{accuracies[j]:.2f} ({generalization_levels[j]}%)"
                formatted_labels.append(label)
            ax2.set_yticklabels(formatted_labels)

            ax2.set_ylabel("Model Accuracy (Generalization Level)")
            ax2.invert_yaxis()
            ax2.set_ylim(ax.get_ylim())

            ax.set_title(f"Model: {model}")

        # Create a single legend for the entire figure
        handles, labels = [], []
        for ax in axes:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
            ax.get_legend().remove()  # Remove individual subplot legends
        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(1.0, 0.95),
            title="Features",
            fontsize="small",
        )

        fig.tight_layout(
            rect=[0, 0, 0.9, 0.96]  # pyright: ignore[reportArgumentType]
        )  # Adjust rect to make space for the global legend
        plt.show()

    def plot_unique_counts(self, method: str) -> None:
        """
        Plot the number of unique elements in each feature for k_anonymity model.
        Features are arranged according to their ranking in the explanations.
        X-axis shows feature rankings, and unique counts are displayed as text at each point.
        """
        import pandas as pd

        # Only plot for k_anonymity model
        id = "k"
        model = "k_anonymity"

        # Check if k_anonymity exists in the models
        if not any(m_id == id for m_id, _ in self.models):
            print(f"Model '{model}' not found in the available models.")
            return

        explanations = self.explanations[method][id]

        # Filter out explanations with name="clean"
        explanations = [e for e in explanations if e.name != "clean"]

        if not explanations:
            print("No explanations available after filtering out 'clean'.")
            return

        # Create figure with single subplot
        fig, ax = plt.subplots(figsize=(15, 10))
        fig.suptitle(
            f"Unique Value Counts per Feature for {self.dataset} using {method} and {model}",
            fontsize=16,
        )

        model_names = [e.name for e in explanations]
        y_locs = range(len(model_names))

        # Get all unique features across all explanations
        all_items = set(item for e in explanations for item in e.get_ranking())
        all_items_list = sorted(all_items)
        cmap = plt.get_cmap("tab20")

        # Calculate maximum rank for x-axis limits
        max_rank = max(len(e.get_ranking()) for e in explanations)

        # For each explanation, load the corresponding dataframe and count unique values
        for idx, item in enumerate(all_items_list):
            ranks = []
            y_vals = []
            unique_counts_for_annotation = []

            for j, e in enumerate(explanations):
                ranking = e.get_ranking()

                # Check if item is in this ranking
                if item not in ranking:
                    continue

                # Get the rank (1-indexed)
                rank = ranking.index(item) + 1

                # Determine the path to load the dataframe
                if e.name == "clean":
                    df_path = f"./data/{self.dataset}/clean.csv"
                else:
                    # e.name is a number
                    df_path = f"./data/{self.dataset}/{model}/{e.name}.csv"

                # Convert to absolute path
                base_dir = Path(__file__).parent.parent.parent
                abs_df_path = (base_dir / df_path).resolve()

                # Load the dataframe
                try:
                    df = pd.read_csv(abs_df_path)

                    # Check if the feature exists in the dataframe
                    if item in df.columns:
                        # Count unique values for this feature
                        n_unique = df[item].nunique()
                        ranks.append(rank)
                        y_vals.append(j)
                        unique_counts_for_annotation.append(n_unique)
                    else:
                        print(f"Feature '{item}' not found in {abs_df_path}")
                except Exception as e_error:
                    print(f"Error loading {abs_df_path}: {e_error}")

            if not ranks:
                continue

            color = cmap(idx % 20)
            ax.plot(ranks, y_vals, marker="o", linestyle="-", label=item, color=color)

            # Annotate each point with the unique count
            for rank, y_val, count in zip(ranks, y_vals, unique_counts_for_annotation):
                ax.text(
                    rank,
                    y_val,
                    str(count),
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor=color,
                        alpha=0.3,
                        edgecolor="none",
                    ),
                )

        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize="small",
            title="Features",
        )

        ax.set_yticks(y_locs)
        ax.set_yticklabels(model_names)
        ax.invert_yaxis()

        ax.set_xlim(0, max_rank + 1)
        ax.set_xticks(range(1, max_rank + 1))
        ax.set_xlabel("Feature Rank")
        ax.set_ylabel("Anonymization Level")

        ax.grid(axis="x", linestyle="--", alpha=0.5)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)

        # Add accuracy on the right y-axis
        ax2 = ax.twinx()
        accuracies = [e.accuracy for e in explanations]
        ax2.set_yticks(y_locs)
        formatted_accuracies = [f"{acc:.2f}" for acc in accuracies]
        ax2.set_yticklabels(formatted_accuracies)

        # Highlight significant accuracy changes
        for j, label_obj in enumerate(ax2.get_yticklabels()):
            if j > 0 and abs(accuracies[j] - accuracies[j - 1]) > 0.02:
                label_obj.set_color("red")
                label_obj.set_fontweight("bold")
                label_obj.set_fontsize("large")

        ax2.set_ylabel("Model Accuracy")
        ax2.invert_yaxis()
        ax2.set_ylim(ax.get_ylim())

        plt.tight_layout()
        plt.show()


# adult_knn = Comparer("adult", ".", "knn")
# adult_knn.plot_accuracy("lime", baseline=0.76)
