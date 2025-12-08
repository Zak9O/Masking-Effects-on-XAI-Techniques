import numpy as np
from copy import copy
from scipy import stats
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


class Explanation:
    def __init__(self, path: str) -> None:
        array = np.load(path)
        self.name = os.path.basename(path).split(".csv", 1)[0]
        self.accuracy = float(array[0][1])
        self.transform_n = int(float(array[1][1]))
        self.transform_n_max = int(float(array[2][1]))
        importance = array[3:]
        self._ranking_values = [float(x[1]) for x in importance if float(x[1]) > 0]
        self._ranking = [
            x[0] for i, x in enumerate(importance) if i < len(self._ranking_values)
        ]

    def get_ranking(self) -> list[str]:
        return self._ranking


class Dataset:
    def __init__(self, path: str, dataset_name: str) -> None:
        self.explanations: dict[str, dict[str, list[Explanation]]] = {}

        for method in PlotCreator.EXPLANATION_METHODS:
            method_dir = os.path.join(path, method, dataset_name)
            self.explanations[method] = self._load_method_dir(method_dir)

    def _load_method_dir(self, path: str) -> dict[str, list[Explanation]]:
        method_dict = {}

        file_path_clean = os.path.join(path, "clean.csv.npy")
        method_dict["clean"] = [Explanation(file_path_clean)]

        for _, anon_model_dir_name in PlotCreator.ANONYMIZATION_MODELS:
            anon_model_dir = os.path.join(path, anon_model_dir_name)

            explanations = []
            files_sorted = self._sort_files(anon_model_dir_name, anon_model_dir)
            for file_name in files_sorted:
                file_path = os.path.join(anon_model_dir, file_name)
                explanations.append(Explanation(file_path))
            method_dict[anon_model_dir_name] = explanations

        return method_dict

    def _sort_files(self, name: str, path: str) -> list[str]:
        files = [f for f in os.listdir(path) if f.endswith(".npy")]

        def _num_key(fname: str) -> float:
            return float(fname.rstrip(".csv.npy"))

        ascending_models = ("l_diversity", "k_anonymity")
        reverse = False if name in ascending_models else True
        files = sorted(files, key=_num_key, reverse=reverse)
        return files

    def get_kendal_taus(self) -> dict[str, dict[str, list[object]]]:
        kendal_taus = {}

        for method, anon_models in self.explanations.items():
            kendal_taus[method] = {}

            clean_ranking = anon_models["clean"][0].get_ranking()

            for anon_model_name, explanations in anon_models.items():
                if anon_model_name == "clean":
                    continue

                kendal_taus[method][anon_model_name] = self._compute_kendal_tau(
                    clean_ranking, explanations
                )
        return kendal_taus

    def _compute_kendal_tau(
        self, model_ranking: list[str], rankings: list[Explanation]
    ) -> list[object]:
        kendaltau = []
        for e in rankings:
            if len(e.get_ranking()) == 1:
                continue
            m_rnk, rnk = self._make_compatible(e.get_ranking(), model_ranking)
            kendaltau.append(stats.kendalltau(m_rnk, rnk, alternative="greater"))
        return kendaltau

    def _add_missing_categories(self, ref: list[str], rnk: list[str]) -> list[str]:
        out = copy(rnk)
        for feat in ref:
            if feat not in rnk:
                out.append(feat)
        return out

    def _make_compatible(
        self, ref: list[str], rnk: list[str]
    ) -> tuple[list[str], list[str]]:
        return (ref, rnk[: len(ref)])


class Model:
    MODEL_TYPES = ["knn", "forest", "MLP"]

    def __init__(self, name: str, path: str, datasets: list[str]) -> None:
        self.name = name
        self.path = path
        self.datasets: dict[str, Dataset] = {}
        for dataset_name in datasets:
            self.datasets[dataset_name] = Dataset(path, dataset_name)


class PlotCreator:
    ANONYMIZATION_MODELS = [
        ("t", "t_closeness"),
        ("l", "l_diversity"),
        ("k", "k_anonymity"),
        ("a", "alpha_k_anonymity"),
    ]

    EXPLANATION_METHODS = ["shap", "lime"]

    def __init__(
        self, models: list[str], datasets: list[str], base_path: str = "../data/"
    ) -> None:
        self.models: dict[str, Model] = {}

        for model in models:
            model_path = os.path.join(base_path, f"{model}/")

            if not os.path.isdir(model_path):
                raise FileNotFoundError(f"Directory not found: {model_path}")

            self.models[model] = Model(model, model_path, datasets)

    def plot_heatmap(
        self,
        datasets: Optional[list[str]] = None,
        methods: Optional[list[str]] = None,
        explanation_methods: Optional[list[str]] = None,
        classifiers: Optional[list[str]] = None,
    ) -> None:
        # Collect Kendall-tau series across datasets/methods
        # If a parameter is None or empty, include all values for that filter.
        # methods: filters anonymization models (e.g., 't_closeness', 'l_diversity', etc.)
        # explanation_methods: filters explanation methods ('lime' or 'shap')
        # classifiers: includes only specified classifier types (e.g., 'knn', 'forest', 'MLP')
        rows = []
        labels = []
        max_len = 0

        for model_name, model in self.models.items():
            # apply classifier filter if provided (non-empty)
            if classifiers and model_name not in classifiers:
                continue

            for dataset_name, dataset in model.datasets.items():
                # apply dataset filter if provided (non-empty)
                if datasets and dataset_name not in datasets:
                    continue

                kendal_taus = dataset.get_kendal_taus()
                for method, anon_model_map in kendal_taus.items():
                    # apply explanation method filter if provided (non-empty)
                    if explanation_methods and method not in explanation_methods:
                        continue

                    for anon_model_name, kt_list in anon_model_map.items():
                        # apply anonymization method filter if provided (non-empty)
                        if methods and anon_model_name not in methods:
                            continue

                        # extract p-values from Kendalltau results (0..1)
                        vals = []
                        for kt in kt_list:
                            pval = getattr(kt, "pvalue", None)
                            if pval is not None:
                                try:
                                    vals.append(float(pval))
                                except Exception:
                                    vals.append(np.nan)

                        labels.append(
                            f"{model_name}-{dataset_name}-{method}-{anon_model_name}"
                        )
                        rows.append(vals)
                        if len(vals) > max_len:
                            max_len = len(vals)

        if not rows:
            raise RuntimeError("No Kendall tau data available to plot")

        # Build 2D array, pad missing values with NaN
        data = np.full((len(rows), max_len), np.nan)
        for i, r in enumerate(rows):
            data[i, : len(r)] = r

        # Plot heatmap
        plt.figure(figsize=(max(8, max_len * 0.4), max(6, len(rows) * 0.4)))
        mask = np.isnan(data)
        ax = sns.heatmap(
            data,
            mask=mask,
            cmap="viridis_r",
            vmin=0,
            vmax=1,
            xticklabels=[str(x) for x in range(1, max_len + 1)],
            yticklabels=labels,
            cbar_kws={"label": "p-value"},
            linewidths=0.5,
            linecolor="lightgrey",
        )
        ax.set_xlabel("Anonymization Level")
        ax.set_ylabel("Model-Dataset-Method-AnonModel")
        if explanation_methods and len(explanation_methods) == 1:
            plt.title(f"Kendall Tau p-values for {explanation_methods[0]}")
        else:
            plt.title("Kendall Tau p-values")
        plt.tight_layout()
        plt.show()

    def plot_line(
        self, classifier: str, dataset: str, method: str, only: list[str] | None = None
    ) -> None:
        # Create a single figure with 2x2 subplots
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
        axes = axes.flatten()  # Flatten the 2x2 array for easy iteration
        explanationss = self.models[classifier].datasets[dataset].explanations[method]

        fig.suptitle(
            f"Feature Rank Comparison for {f'{classifier}-{dataset}'} using {method}",
            fontsize=16,
        )

        for i, (_, model) in enumerate(PlotCreator.ANONYMIZATION_MODELS):
            if only is not None and model not in only:
                continue
            ax = axes[i]  # Get the current subplot axis
            explanations = explanationss["clean"] + explanationss[model]

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


pl = PlotCreator(
    [
        "forest",
        "MLP",
    ],
    ["usa_house", "cervic_cancer", "adult"],
    "./data/",
)
pl.plot_line(
    classifier="forest",
    method="shap",
    dataset="adult",
)
# Example usage of plot_line:
# pl.plot_line(classifier="forest", method="shap", dataset="adult")
