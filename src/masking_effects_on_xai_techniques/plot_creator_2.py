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
        importance = array[1:]
        self._ranking_values = [float(x[1]) for x in importance if float(x[1]) > 0]
        self._ranking = [
            x[0] for i, x in enumerate(importance) if i < len(self._ranking_values)
        ]

    def get_ranking(self) -> list[str]:
        return self._ranking


class Dataset:
    def __init__(self, path: str) -> None:
        self.explanations: dict[str, dict[str, list[Explanation]]] = {}

        for method in PlotCreator.EXPLANATION_METHODS:
            method_dir = os.path.join(path, method)
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


class PlotCreator:
    ANONYMIZATION_MODELS = [
        ("t", "t_closeness"),
        ("l", "l_diversity"),
        ("k", "k_anonymity"),
        ("a", "alpha_k_anonymity"),
    ]

    EXPLANATION_METHODS = ["shap", "lime"]

    def __init__(self, paths: list[str], base_path: str = "../data/") -> None:
        self.paths = paths
        self.datasets: dict[str, Dataset] = {}

        for dataset in self.paths:
            dataset_path = os.path.join(base_path, f"{dataset}/")

            if not os.path.isdir(dataset_path):
                raise FileNotFoundError(f"Directory not found: {dataset_path}")

            self.datasets[dataset] = Dataset(dataset_path)

    def plot_heatmap(
        self,
        datasets: Optional[list[str]] = None,
        methods: Optional[list[str]] = None,
        models: Optional[list[str]] = None,
    ) -> None:
        # Collect Kendall-tau series across datasets/methods/models
        # If a parameter is None or empty, include all values for that filter.
        rows = []
        labels = []
        max_len = 0

        for dataset_name, dataset in self.datasets.items():
            # apply dataset filter if provided (non-empty)
            if datasets:
                if dataset_name not in datasets:
                    continue

            kendal_taus = dataset.get_kendal_taus()
            for method, model_map in kendal_taus.items():
                # apply method filter if provided (non-empty)
                if methods:
                    if method not in methods:
                        continue

                for model_name, kt_list in model_map.items():
                    # apply model filter if provided (non-empty)
                    if models:
                        if model_name not in models:
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

                    labels.append(f"{dataset_name}-{method}-{model_name}")
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
        ax.set_ylabel("Dataset-Method-Model")
        if methods and len(methods) == 1:
            plt.title(f"Kendall Tau p-values for {methods[0]}")
        else:
            plt.title("Kendall Tau p-values")
        plt.tight_layout()
        plt.show()


# pl = PlotCreator(["usa_house","cervic_cancer", "adult"], "./data/")
# pl.plot_heatmap(methods=['shap'])
