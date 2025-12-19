from collections.abc import Iterable
from itertools import groupby, zip_longest
import numpy as np
from functools import reduce
from scipy import stats
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from typing import Any, Callable, Optional


class Explanation:
    def __init__(self, path: str) -> None:
        try:
            array = np.load(path)
        except FileNotFoundError:
            print(f"File not found: {path}!")
            return
        self.name = os.path.basename(path).split(".csv", 1)[0]
        if "l_diversity" in path:
            self.name = str(int(float(self.name)))
        self.accuracy = float(array[0][1])
        self.transform_n = int(float(array[1][1]))
        self.transform_n_max = int(float(array[2][1]))
        importance = array[3:]
        self._ranking_values = [float(x[1]) for x in importance if float(x[1]) > 0]
        self._ranking = [
            x[0] for i, x in enumerate(importance) if i < len(self._ranking_values)
        ]

    def get_transform_level(self) -> float:
        return self.transform_n / self.transform_n_max

    def get_ranking(self) -> list[str]:
        return self._ranking

    def compute_kendal_tau(self, ref_ranking: list[str]):
        ref_ranking_enc = [i for i, _ in enumerate(ref_ranking)]
        encoder = {feat: idx for idx, feat in enumerate(self.get_ranking())}
        ranking_enc = [encoder.get(feat, len(encoder)) for feat in ref_ranking]
        return stats.kendalltau(ref_ranking_enc, ranking_enc, alternative="greater")


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
            # if not os.path.exists(anon_model_dir):
            #     method_dict[anon_model_dir_name] = []
            #     continue

            explanations = []
            files_sorted = self._sort_files(anon_model_dir_name, anon_model_dir)
            for file_name in files_sorted:
                file_path = os.path.join(anon_model_dir, file_name)
                if "alpha" in anon_model_dir_name and "1" in file_name:
                    continue
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

                kendal_taus[method][anon_model_name] = [
                    e.compute_kendal_tau(clean_ranking) for e in explanations
                ]
        return kendal_taus


class Model:
    MODEL_TYPES = ["knn", "forest", "MLP"]

    def __init__(self, name: str, path: str, datasets: list[str]) -> None:
        self.name = name
        self.path = path
        self.datasets: dict[str, Dataset] = {}
        for dataset_name in datasets:
            self.datasets[dataset_name] = Dataset(path, dataset_name)


class _PlotResult:
    def __init__(
        self,
        value: dict[str, Any],
        classifier: str,
        anonymization_method: str,
        explanation_method: str,
        dataset: str,
    ) -> None:
        self.value = value
        self.classifier = classifier
        self.anonymization_method = anonymization_method
        self.explanation_method = explanation_method
        self.dataset = dataset


class PlotCreator:
    ANONYMIZATION_MODELS = [
        ("t", "t_closeness"),
        ("l", "l_diversity"),
        ("k", "k_anonymity"),
        ("a", "alpha_k_anonymity"),
    ]

    ANONYMIZATION_COLORS = {
        "t_closeness": "tab:orange",
        "l_diversity": "tab:brown",
        "k_anonymity": "tab:purple",
        "alpha_k_anonymity": "tab:red",
    }

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

    def _filter_data(
        self, f: Callable[[list[Explanation], Explanation], dict[str, Any]]
    ) -> list[_PlotResult]:
        result = []
        for method in ["shap", "lime"]:
            for classifier in ["MLP", "forest", "knn"]:
                for dataset in self.models[classifier].datasets.keys():
                    explanations = (
                        self.models[classifier].datasets[dataset].explanations[method]
                    )
                    for anon_model_name, expl_list in explanations.items():
                        if anon_model_name == "clean":
                            continue

                        result.append(
                            _PlotResult(
                                value=f(expl_list, explanations["clean"][0]),
                                classifier=classifier,
                                anonymization_method=anon_model_name,
                                explanation_method=method,
                                dataset=dataset,
                            )
                        )
        return result

    def plot_utility(
        self,
        dataset: str,
        classifiers: Optional[list[str]] = None,
        average_models=False,
        only_look_at_k=False,
        color_num_feature_left=False,
        show_feature_counts=False,
    ) -> None:
        if classifiers is None:
            classifiers = ["knn", "forest", "MLP"]

        def _filter(x: list[Explanation], clean: Explanation):
            transform_levels = [0] + [e.get_transform_level() * 100 for e in x]
            accuracies = [clean.accuracy] + [e.accuracy for e in x]
            features_left = [len(clean.get_ranking())] + [
                len(e.get_ranking()) for e in x
            ]
            return {
                "accuracies": accuracies,
                "generalization levels": transform_levels,
                "features left": features_left,
            }

        result = self._filter_data(_filter)
        result = list(
            filter(
                lambda x: x.classifier in classifiers
                and x.dataset == dataset
                and x.explanation_method == "lime",
                result,
            )
        )
        if average_models:

            def aggregate_by_iterator(
                results: list[_PlotResult],
            ) -> Iterable[_PlotResult]:
                sorted_results = sorted(results, key=lambda x: x.anonymization_method)

                for method, group in groupby(
                    sorted_results, key=lambda x: x.anonymization_method
                ):
                    items = list(group)
                    accuracies = [item.value["accuracies"] for item in items]
                    features_left = [item.value["features left"] for item in items]

                    avg_val = np.nanmean(np.array(accuracies), axis=0).tolist()
                    features_left = [
                        max(items)
                        for items in zip_longest(
                            *features_left, fillvalue=-float("inf")
                        )
                    ]

                    yield _PlotResult(
                        value={
                            "accuracies": avg_val,
                            "features left": features_left,
                            "generalization levels": items[0].value[
                                "generalization levels"
                            ],
                        },
                        classifier=items[0].classifier,
                        anonymization_method=method,
                        explanation_method=items[0].explanation_method,
                        dataset=items[0].dataset,
                    )

            result = list(aggregate_by_iterator(result))

        def f(x: dict[int, float], y: _PlotResult) -> dict[int, float]:
            if y.anonymization_method != "k_anonymity" and only_look_at_k:
                return x
            gen_levels = y.value.get("generalization levels", [])
            feat_left = y.value.get("features left", [])
            for gen_level, feat_count in zip(gen_levels, feat_left):
                if int(feat_count) == 0:
                    continue
                existing_gen_level = x.get(int(feat_count), 999999)
                x[int(feat_count)] = min(existing_gen_level, gen_level)
            return x

        min_feature_left_y = reduce(f, result, {})

        # Create subplots, one for each classifier (or single plot if averaging)
        if average_models:
            n_classifiers = 1
            plot_labels = ["Averaged Models"]
            fig, axes = plt.subplots(
                1, 1, figsize=(7, 7), sharey=True, constrained_layout=True
            )
            axes = [axes]
        else:
            n_classifiers = len(classifiers)
            plot_labels = classifiers
            fig, axes = plt.subplots(
                1,
                n_classifiers,
                figsize=(6 * n_classifiers + 1.5, 7),
                sharey=True,
                constrained_layout=True,
            )
            # Handle case where there's only one classifier
            if n_classifiers == 1:
                axes = [axes]

        # Define marker styles for different anonymization methods
        marker_styles = {
            "k_anonymity": "o",
            "l_diversity": "s",
            "t_closeness": "^",
            "alpha_k_anonymity": "D",
        }

        # Color map for accuracy values or features left
        cmap = plt.get_cmap("viridis")

        # Get global min/max accuracy or features left for consistent color mapping across all series
        from matplotlib.colors import Normalize
        from matplotlib.ticker import MaxNLocator

        if color_num_feature_left:
            all_features_left = [f for p in result for f in p.value["features left"]]
            global_min_val = min(all_features_left)
            global_max_val = max(all_features_left)
            colorbar_label = "Features Left"
        else:
            global_min_val = min([min(p.value["accuracies"]) for p in result])
            global_max_val = max([max(p.value["accuracies"]) for p in result])
            colorbar_label = "Accuracy"
        norm = Normalize(vmin=global_min_val, vmax=global_max_val)

        # Plot for each classifier
        for ax, label in zip(axes, plot_labels):
            # Filter results for this classifier (or use all results if averaging)
            if average_models:
                classifier_result = result
            else:
                classifier_result = [p for p in result if p.classifier == label]

            if not classifier_result:
                continue

            # Plot each series for this classifier
            for plot_result in classifier_result:
                gen_levels = plot_result.value["generalization levels"]
                accuracies = plot_result.value["accuracies"]
                features_left = plot_result.value["features left"]

                # Create indices for x-axis
                indices = list(range(len(gen_levels)))

                # Get marker style for this anonymization method
                marker = marker_styles.get(plot_result.anonymization_method, "o")

                # Plot lines connecting the points
                ax.plot(indices, gen_levels, "k-", alpha=0.3, linewidth=1)

                # Plot points with colors based on accuracy or features left
                for i, (idx, gen_level, acc, feat) in enumerate(
                    zip(indices, gen_levels, accuracies, features_left)
                ):
                    # Get normalized value for this specific point
                    if color_num_feature_left:
                        normalized_val = norm(feat)
                    else:
                        normalized_val = norm(acc)
                    color = cmap(normalized_val)
                    ax.scatter(
                        idx,
                        gen_level,
                        marker=marker,
                        s=100 if show_feature_counts else 150,
                        color=color,
                        label=f"{plot_result.anonymization_method}" if i == 0 else "",
                    )
                    if show_feature_counts:
                        ax.text(
                            idx,
                            gen_level,
                            str(feat),
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="red",
                            weight="bold",
                            zorder=5,
                        )

            # Set x-tick labels with 'clean' as the first label
            all_indices = list(
                range(
                    max(
                        [
                            len(p.value["generalization levels"])
                            for p in classifier_result
                        ]
                    )
                )
            )
            tick_labels = ["clean"] + [str(i) for i in range(1, len(all_indices))]
            ax.set_xticks(all_indices[: len(tick_labels)])
            ax.set_xticklabels(tick_labels[: len(all_indices)])

            ax.set_xlabel("Anonymization Level")
            if not average_models:
                ax.set_title(f"{self.string_beautify(label)}")
            ax.invert_yaxis()
            # Only whole numbers on the primary y-axis
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, alpha=0.3)

        # Add colorbar for accuracy or features left values on the rightmost subplot
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, pad=0.02)
        cbar.set_label(colorbar_label)

        # Add legend for marker styles on the first subplot
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor="gray",
                markersize=8,
                label=self.string_beautify(method),
            )
            for method, marker in marker_styles.items()
        ]
        axes[0].legend(
            handles=legend_elements, loc="upper right", title="Anonymization Method"
        )

        axes[0].set_ylabel("Generalization Level (%)")

        # Add twin y-axis with feature labels only on the rightmost plot
        # Skip if color_num_feature_left is True
        if (
            min_feature_left_y
            and len(axes) > 0
            and not color_num_feature_left
            and not show_feature_counts
        ):
            ax_r = axes[-1].twinx()
            ax_r.set_ylim(axes[-1].get_ylim())
            # ax_r.invert_yaxis()
            items_sorted = sorted(min_feature_left_y.items(), key=lambda kv: kv[1])
            y_ticks = [y for _, y in items_sorted]
            y_labels = [str(k) for k, _ in items_sorted]
            ax_r.set_yticks(y_ticks)
            ax_r.set_yticklabels(y_labels)
            ax_r.set_ylabel("Features left")

        # Use a figure-level title so it shows above all subplots
        fig.suptitle(self.string_beautify(dataset), fontsize=14)

        plt.show()

    def plot_histogram(
        self,
        dataset: str,
        anon_models: list[str] | None = None,
        threshold: float = 0.05,
        start_from: int | None = None,
        x_labels: tuple[int, int] | None = None,
    ) -> None:
        def _filter(x: list[Explanation], clean: Explanation):
            if start_from is None:
                ran = [e.compute_kendal_tau(clean.get_ranking()) for e in x]
            else:
                try:
                    clean = x[start_from]
                    x = x[start_from + 1 :]
                except IndexError:
                    return {"value": []}
                if len(clean.get_ranking()) < 2:
                    ran = []
                else:
                    ran = [e.compute_kendal_tau(clean.get_ranking()) for e in x]
            return {
                "p-value": [kt.pvalue for kt in ran],
                "value": [kt.statistic for kt in ran],
            }

        result = self._filter_data(_filter)

        def f(x: _PlotResult) -> _PlotResult:
            diffs = [abs(a - b) for a, b in zip(x.value["value"], x.value["value"][1:])]
            count = sum([1 for diff in diffs if diff > threshold])
            x.value["value"] = count
            return x

        lime = list(
            map(
                f,
                filter(
                    lambda x: x.explanation_method == "lime"
                    and x.dataset == dataset
                    and (
                        x.anonymization_method in anon_models if anon_models else True
                    ),
                    result,
                ),
            )
        )
        shap = list(
            map(
                f,
                filter(
                    lambda x: x.explanation_method == "shap"
                    and x.dataset == dataset
                    and (
                        x.anonymization_method in anon_models if anon_models else True
                    ),
                    result,
                ),
            )
        )

        # Get unique classifiers and anonymization methods
        classifiers = sorted(set([r.classifier for r in lime]))
        anon_methods = sorted(set([r.anonymization_method for r in lime]))

        # Create data structure for plotting
        # Structure: {anon_method: {classifier: std_value}}
        lime_data = {}
        shap_data = {}

        for anon_method in anon_methods:
            lime_data[anon_method] = {}
            shap_data[anon_method] = {}
            for classifier in classifiers:
                # Find the corresponding result
                lime_result = next(
                    (
                        r
                        for r in lime
                        if r.classifier == classifier
                        and r.anonymization_method == anon_method
                    ),
                    None,
                )
                shap_result = next(
                    (
                        r
                        for r in shap
                        if r.classifier == classifier
                        and r.anonymization_method == anon_method
                    ),
                    None,
                )

                lime_data[anon_method][classifier] = (
                    lime_result.value["value"] if lime_result else 0
                )
                shap_data[anon_method][classifier] = (
                    shap_result.value["value"] if shap_result else 0
                )

        # Create a single plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))

        # Set up bar positions - three groups (one per classifier)
        # Each group has 8 bars in pairs (LIME + SHAP for each method)
        x = np.arange(len(classifiers))
        width = 0.1  # width of each bar
        n_methods = len(anon_methods)
        total_bars = n_methods * 2  # LIME + SHAP
        offset = width * (total_bars - 1) / 2

        # Define colors for anonymization methods using class constant
        colors = {
            method: PlotCreator.ANONYMIZATION_COLORS[method] for method in anon_methods
        }

        # Plot bars in pairs (LIME and SHAP together for each anonymization method)
        for i, anon_method in enumerate(anon_methods):
            # LIME bar (filled) - only add to legend on first iteration
            lime_values = [
                lime_data[anon_method][classifier] for classifier in classifiers
            ]
            lime_positions = x - offset + (i * 2) * width
            ax.bar(
                lime_positions,
                lime_values,
                width,
                label=anon_method if i == 0 else "",
                color=colors[anon_method],
                alpha=0.8,
            )

            # SHAP bar (striped pattern) - right next to LIME
            shap_values = [
                shap_data[anon_method][classifier] for classifier in classifiers
            ]
            shap_positions = x - offset + (i * 2 + 1) * width
            ax.bar(
                shap_positions,
                shap_values,
                width,
                label="" if i < len(anon_methods) - 1 else "",
                color="none",
                alpha=1.0,
                edgecolor=colors[anon_method],
                # linewidth=2.5,
                hatch="///",
            )

        # Create custom legend combining both color and shading information

        # Color legend (anonymization methods)
        color_handles = [
            Patch(facecolor=colors[method], label=self.string_beautify(method))
            for method in anon_methods
        ]

        # Shading legend (explanation methods)
        shading_handles = [
            Patch(facecolor="gray", alpha=0.8, label="LIME"),
            Patch(
                facecolor="none",
                alpha=1.0,
                edgecolor="gray",
                linewidth=1.5,
                hatch="///",
                label="SHAP",
            ),
        ]

        # Combine all handles in one legend with a separator
        all_handles = color_handles + shading_handles
        ax.legend(handles=all_handles, loc="upper center", ncol=len(anon_methods) + 2)

        ax.set_xlabel("Classifier")
        ax.set_ylabel(f"# of times |$\\Delta \\tau$|> {threshold}")
        ax.set_title(f"Stability LIME vs SHAP for {self.string_beautify(dataset)}")
        ax.set_xticks(x)
        ax.set_xticklabels([self.string_beautify(c) for c in classifiers])
        ax.grid(axis="y", alpha=0.3)

        # Set y-axis to only show whole numbers
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Set y-axis limits if x_labels provided
        if x_labels is not None:
            ax.set_ylim(x_labels[0], x_labels[1] + 0.2)

        plt.tight_layout()
        plt.show()

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
        generalization_levels = []  # Track generalization levels for annotations
        accuracies = []  # Track accuracies for annotations

        for model_name, model in self.models.items():
            # apply classifier filter if provided (non-empty)
            if classifiers and model_name not in classifiers:
                continue

            for dataset_name, dataset in model.datasets.items():
                # apply dataset filter if provided (non-empty)
                if datasets and dataset_name not in datasets:
                    continue

                # Access the actual explanations to get generalization levels and accuracy
                explanations_dict = dataset.explanations

                kendal_taus = dataset.get_kendal_taus()
                for method, anon_model_map in kendal_taus.items():
                    # apply explanation method filter if provided (non-empty)
                    if explanation_methods and method not in explanation_methods:
                        continue

                    for anon_model_name, kt_list in anon_model_map.items():
                        # apply anonymization method filter if provided (non-empty)
                        if methods and anon_model_name not in methods:
                            continue

                        # Get the actual explanation list to extract generalization levels and accuracy
                        expl_list = explanations_dict.get(method, {}).get(
                            anon_model_name, []
                        )

                        # extract p-values from Kendalltau results (0..1)
                        vals = []
                        gen_levels = []
                        acc_values = []
                        for idx, kt in enumerate(kt_list):
                            pval = getattr(kt, "pvalue", None)
                            if pval is not None:
                                try:
                                    vals.append(float(pval))
                                except Exception:
                                    vals.append(np.nan)

                            # Get generalization level and accuracy for this explanation
                            if idx < len(expl_list):
                                e = expl_list[idx]
                                gen_level = round(
                                    100 * e.transform_n / e.transform_n_max
                                )
                                gen_levels.append(gen_level)
                                acc_values.append(e.accuracy)
                            else:
                                gen_levels.append(None)
                                acc_values.append(None)

                        labels.append(
                            f"{model_name}-{dataset_name}-{method}-{anon_model_name}"
                        )
                        rows.append(vals)
                        generalization_levels.append(gen_levels)
                        accuracies.append(acc_values)
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
            annot=False,
        )

        # Add accuracy and generalization level annotations in black small text
        for row_idx, (acc_list, gen_levels) in enumerate(
            zip(accuracies, generalization_levels)
        ):
            for col_idx, (acc, gen_level) in enumerate(zip(acc_list, gen_levels)):
                if (
                    gen_level is not None
                    and acc is not None
                    and not np.isnan(data[row_idx, col_idx])
                ):
                    ax.text(
                        col_idx + 0.15,
                        row_idx + 0.15,
                        f"{round(acc * 100, 1)}%\n({gen_level}%)",
                        ha="left",
                        va="top",
                        color="red",
                        fontsize=6,
                        weight="bold",
                    )

        ax.set_xlabel("Anonymization Level")
        ax.set_ylabel("Model-Dataset-Method-AnonModel")
        if explanation_methods and len(explanation_methods) == 1:
            plt.title(
                f"Kendall Tau p-values for {self.string_beautify(explanation_methods[0])}"
            )
        else:
            plt.title("Kendall Tau p-values")
        plt.tight_layout()
        plt.show()

    def string_beautify(self, s: str) -> str:
        if s == "t_closeness":
            return "$t$-closeness"
        elif s == "k_anonymity":
            return "$k$-anonymity"
        elif s == "l_diversity":
            return "$\\ell$-diversity"
        elif s == "alpha_k_anonymity":
            return "$(\\alpha,k$)-anonymity"
        elif s == "knn":
            return "k-NN"
        elif s == "forest":
            return "Random Forest"
        elif s == "usa_house":
            return "USA House Equal-Size"
        elif s == "usa_house_old":
            return "USA House Equal-Width"
        elif s == "adult":
            return "Adult"
        elif s == "adult_imbalanced":
            return "Adult Imbalanced"
        elif s == "cervic_cancer":
            return "Cervical Cancer"
        elif s == "cervic_cancer_original":
            return "Cervical Cancer Reversed"
        return s

    def plot_line(
        self, classifier: str, dataset: str, method: str, only: list[str] | None = None
    ) -> None:
        # Create a single figure with 2x2 subplots
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
        axes = axes.flatten()  # Flatten the 2x2 array for easy iteration
        explanationss = self.models[classifier].datasets[dataset].explanations[method]

        fig.suptitle(
            f"Rank Topology of {self.string_beautify(classifier)} {self.string_beautify(dataset)} using {self.string_beautify(method).upper()}",
            fontsize=16,
        )
        fig.supxlabel("Dataset")

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

            if model == "t_closeness":
                ylabel = "Anonymization Level ($t$)"
            elif model == "k_anonymity":
                ylabel = "Anonymization Level ($k$)"
            elif model == "l_diversity":
                ylabel = "Anonymization Level ($\\ell$)"
            elif model == "alpha_k_anonymity":
                ylabel = "Anonymization Level ($\\alpha$)"
            else:
                ylabel = "Anonymization Level"

            ax.set_ylabel(ylabel)
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

            ax.set_title(
                f"{self.string_beautify(model)}",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor=PlotCreator.ANONYMIZATION_COLORS[model],
                    alpha=0.3,
                    edgecolor=PlotCreator.ANONYMIZATION_COLORS[model],
                ),
            )

        # Create a single legend for the entire figure
        handles, labels = [], []
        for ax in axes:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
            ax.get_legend().remove()  # Remove individual subplot legends

        # Place legend in the lower left subplot (axes[2]) at lower left
        axes[3].legend(
            handles,
            labels,
            loc="lower right",
            title="Features",
            fontsize="small",
        )

        fig.tight_layout(
            rect=[0, 0, 0.9, 0.96]  # pyright: ignore[reportArgumentType]
        )  # Adjust rect to make space for the global legend
        plt.show()

    def plot_consistency_with_accuracy(
        self, dataset: str, methods: list[str] | None = None
    ) -> None:
        def _filter(x: list[Explanation], clean: Explanation):
            ran = [e.compute_kendal_tau(clean.get_ranking()).pvalue for e in x]
            return {
                "value": ran,
                "generalization_levels": [
                    round(100 * e.transform_n / e.transform_n_max) for e in x
                ],
            }

        result = self._filter_data(_filter)

        def f(x: list[_PlotResult], y: _PlotResult) -> list[_PlotResult]:
            existing_item = next(
                (
                    item
                    for item in x
                    if item.anonymization_method == y.anonymization_method
                ),
                None,
            )
            if existing_item is None:
                return [y] + x

            existing_item.value["value"] = [
                (np.nan_to_num(e, r) + np.nan_to_num(e, r)) / 2
                for e, r in zip(existing_item.value["value"], y.value["value"])
            ]
            return x

        lime = list(
            reduce(
                f,
                filter(
                    lambda x: x.explanation_method == "lime" and x.dataset == dataset,
                    result,
                ),
                [],
            )
        )
        shap = list(
            reduce(
                f,
                filter(
                    lambda x: x.explanation_method == "shap" and x.dataset == dataset,
                    result,
                ),
                [],
            )
        )
        # Add plotting code below
        data = {"shap": {}, "lime": {}}
        for item in lime:
            data["lime"][item.anonymization_method] = item.value
        for item in shap:
            data["shap"][item.anonymization_method] = item.value

        above = {}
        for method in data["lime"].keys():
            above[method] = sum(
                1
                for s, la in zip(
                    data["shap"][method]["value"], data["lime"][method]["value"]
                )
                if la > s
            )
            above[method] /= len(data["lime"][method]["value"])
        avg_above = sum(above.values()) / len(above)

        # Plotting
        plt.figure(figsize=(10, 6))
        colors = {
            "t_closeness": "tab:orange",
            "k_anonymity": "tab:purple",
            "alpha_k_anonymity": "tab:red",
            "l_diversity": "tab:brown",
        }
        linestyles = {
            "shap": "-",
            "lime": "--",
        }
        markers = {
            "shap": "o",
            "lime": "s",
        }
        max_gen_level = 0
        for method, method_data in data.items():
            for anon_model_name, values_dict in method_data.items():
                x_vals = np.array(values_dict["generalization_levels"], dtype=float)
                max_gen_level = max(
                    max_gen_level, max(values_dict["generalization_levels"])
                )
                plt.plot(
                    x_vals,
                    values_dict["value"],
                    color=colors.get(anon_model_name, None),
                    linestyle=linestyles.get(method, "-"),
                    marker=markers.get(method, "o"),
                    markersize=6,
                    linewidth=2.0,
                    alpha=0.85,
                )

        plt.xlabel("Generalization Level (%)")
        plt.ylabel("Kendall Tau p-value")
        plt.title(
            f"{self.string_beautify(dataset)} Kendall Tau p-value Comparison: LIME vs SHAP"
        )

        # Legends: colors for anonymization models, linestyles for methods
        ax = plt.gca()
        color_handles = [
            Line2D(
                [0],
                [0],
                color=col,
                lw=2,
                label=f"{self.string_beautify(anon)} ({above.get(anon, 0) * 100:.1f}%)",
            )
            for anon, col in colors.items()
            if anon in above
        ]

        linestyle_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                lw=2,
                linestyle=ls,
                label=f"{self.string_beautify(method).upper()} ({avg_above * 100:.1f}%)"
                if method == "lime"
                else f"{self.string_beautify(method).upper()}",
            )
            for method, ls in list(linestyles.items())[::-1]
        ]

        anon_legend = ax.legend(
            handles=color_handles,
            title="Anonymization (# LIME > SHAP %)",
            loc="lower right",
        )
        ax.add_artist(anon_legend)
        ax.legend(
            handles=linestyle_handles,
            title="Method",
            loc="upper right",
        )
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_consistency(
        self,
        dataset: str,
        methods: list[str] | None = None,
        legend_placement: str = "upper right",
        show_tau: bool = True,
    ) -> None:
        def _filter(x: list[Explanation], clean: Explanation):
            ran = [e.compute_kendal_tau(clean.get_ranking()) for e in x]
            return {
                "value": [e.pvalue for e in ran],
                "tau": [e.statistic for e in ran],
                "generalization_levels": [
                    round(100 * e.transform_n / e.transform_n_max) for e in x
                ],
            }

        result = self._filter_data(_filter)

        def aggregate_by_iterator(results: list[_PlotResult]) -> Iterable[_PlotResult]:
            sorted_results = sorted(results, key=lambda x: x.anonymization_method)

            for method, group in groupby(
                sorted_results, key=lambda x: x.anonymization_method
            ):
                items = list(group)
                values = [item.value["value"] for item in items]
                taus = [item.value["tau"] for item in items]

                avg_val = np.nanmean(np.array(values), axis=0).tolist()
                avg_tau = np.nanmean(np.array(taus), axis=0).tolist()

                yield _PlotResult(
                    value={"value": avg_val, "tau": avg_tau},
                    classifier=items[0].classifier,
                    anonymization_method=method,
                    explanation_method=items[0].explanation_method,
                    dataset=items[0].dataset,
                )

        lime = list(
            filter(
                lambda x: x.explanation_method == "lime" and x.dataset == dataset,
                result,
            ),
        )
        shap = list(
            filter(
                lambda x: x.explanation_method == "shap" and x.dataset == dataset,
                result,
            ),
        )
        lime = list(aggregate_by_iterator(lime))
        shap = list(aggregate_by_iterator(shap))

        data = {"shap": {}, "lime": {}}
        pval_data = {"shap": {}, "lime": {}}
        # Populate data dict from aggregated lime/shap results
        # Map anonymization method -> list of p-values ("value") or tau values
        data_key = "tau" if show_tau else "value"
        for item in lime:
            data["lime"][item.anonymization_method] = item.value.get(data_key, [])
            pval_data["lime"][item.anonymization_method] = item.value.get("value", [])
        for item in shap:
            data["shap"][item.anonymization_method] = item.value.get(data_key, [])
            pval_data["shap"][item.anonymization_method] = item.value.get("value", [])

        # If specific anonymization methods are provided, filter to only those
        if methods:
            allowed = set(methods)
            for expl_method in ["lime", "shap"]:
                data[expl_method] = {
                    k: v for k, v in data[expl_method].items() if k in allowed
                }
                pval_data[expl_method] = {
                    k: v for k, v in pval_data[expl_method].items() if k in allowed
                }

        above = {}
        # Only compute for methods present in both lime and shap
        common_methods = set(data["lime"].keys()).intersection(set(data["shap"].keys()))
        for method in common_methods:
            pairs = list(zip(data["shap"][method], data["lime"][method]))
            if not pairs:
                continue
            if show_tau:
                # When showing tau, count when LIME < SHAP
                above[method] = sum(1 for s, la in pairs if la < s) / len(pairs)
            else:
                # When showing p-value, count when LIME > SHAP
                above[method] = sum(1 for s, la in pairs if la > s) / len(pairs)
        avg_above = sum(above.values()) / len(above)

        # Plotting
        plt.figure(figsize=(10, 6))
        colors = {
            "t_closeness": "tab:orange",
            "k_anonymity": "tab:purple",
            "alpha_k_anonymity": "tab:red",
            "l_diversity": "tab:brown",
        }
        linestyles = {
            "shap": "-",
            "lime": "--",
        }
        max_len = 0
        for method, method_data in data.items():
            for anon_model_name, values in method_data.items():
                x_vals = np.arange(1, len(values) + 1, dtype=float)
                max_len = max(max_len, len(x_vals))
                color = colors.get(anon_model_name, None)
                plt.plot(
                    x_vals,
                    values,
                    color=color,
                    linestyle=linestyles.get(method, "-"),
                    linewidth=2.0,
                    alpha=0.85,
                )

                # When plotting tau, overlay markers to indicate p-value significance
                if show_tau:
                    pvals = pval_data.get(method, {}).get(anon_model_name, [])
                    sig_x = [x for x, p in zip(x_vals, pvals) if p < 0.05]
                    sig_y = [y for y, p in zip(values, pvals) if p < 0.05]
                    nonsig_x = [x for x, p in zip(x_vals, pvals) if p >= 0.05]
                    nonsig_y = [y for y, p in zip(values, pvals) if p >= 0.05]

                    # Filled markers for significant points
                    plt.scatter(
                        sig_x,
                        sig_y,
                        color=color,
                        edgecolors=color,
                        s=50,
                        zorder=5,
                    )

                    # Hollow markers for non-significant points
                    if nonsig_x:
                        plt.scatter(
                            nonsig_x,
                            nonsig_y,
                            facecolors="none",
                            edgecolors=color,
                            s=50,
                            zorder=5,
                        )

        plt.xlabel("Anonymization Level")
        plt.xticks(range(1, max_len + 1))
        if show_tau:
            plt.ylabel("Kendall Tau Statistic")
            plt.title(
                f"{self.string_beautify(dataset)} Kendall Tau Statistic Comparison: LIME vs SHAP"
            )
        else:
            plt.ylabel("Kendall Tau p-value")
            plt.title(
                f"{self.string_beautify(dataset)} Kendall Tau p-value Comparison: LIME vs SHAP"
            )

        # Legends: colors for anonymization models, linestyles for methods
        ax = plt.gca()
        color_handles = [
            Line2D(
                [0],
                [0],
                color=col,
                lw=2,
                label=f"{self.string_beautify(anon)} ({above.get(anon, 0) * 100:.1f}%)",
            )
            for anon, col in colors.items()
            if anon in above
        ]

        linestyle_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                lw=2,
                linestyle=ls,
                label=f"{self.string_beautify(method).upper()} ({avg_above * 100:.1f}%)"
                if method == "lime"
                else f"{self.string_beautify(method).upper()}",
            )
            for method, ls in list(linestyles.items())[::-1]
        ]

        # Combine both legends into one
        combined_handles = color_handles + linestyle_handles
        ax.legend(
            handles=combined_handles,
            title="Anonymization (# LIME > SHAP %)"
            if not show_tau
            else "Anonymization (# LIME < SHAP %)",
            loc=legend_placement,
        )
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_line_compare(self, dataset: str, filters: list[str]) -> None:
        # Filters must have form "classifier-method-anonmodel"]"
        filters_list = []
        for f in filters:
            parts = f.split("-")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid filter format: {f}. Example filter: 'knn-shap-t_closeness'"
                )
            classifier, method, anon_model = parts
            value = {
                "classifier": classifier,
                "method": method,
                "anon_model": anon_model,
            }
            filters_list.append(value)

        results = self._filter_data(
            lambda x, clean: {
                "rankings": [clean.get_ranking()] + [e.get_ranking() for e in x],
                "accuracy": [clean.accuracy] + [e.accuracy for e in x],
                "generalization_levels": [0]
                + [round(100 * e.transform_n / e.transform_n_max) for e in x],
                "names": [clean.name] + [e.name for e in x],
            }
        )
        result = list(
            filter(
                lambda x: any(
                    x.classifier == f["classifier"]
                    and x.explanation_method == f["method"]
                    and x.anonymization_method == f["anon_model"]
                    and x.dataset == dataset
                    for f in filters_list
                ),
                results,
            )
        )

        num_plots = len(result)

        if num_plots == 0:
            raise ValueError("No data to plot with the given filters")

        # Determine grid layout based on number of plots
        if num_plots == 1:
            nrows, ncols = 1, 1
        elif num_plots == 2:
            nrows, ncols = 1, 2
        elif num_plots == 3:
            nrows, ncols = 1, 3
        elif num_plots == 4:
            nrows, ncols = 2, 2
        else:
            # For 5+, arrange as 2xN layout
            nrows = 2
            ncols = (num_plots + 1) // 2

        # Create figure with appropriate size
        fig_width = 10 * ncols
        fig_height = 8 * nrows
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height)
        )

        # Ensure axes is always a 2D array for consistent indexing
        if num_plots == 1:
            axes = np.array([[axes]])
        elif num_plots <= 2:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(nrows, ncols) if axes.ndim == 1 else axes

        fig.suptitle(
            f"Feature Rank Comparison for {dataset}",
            fontsize=16,
        )

        # Track all items for consistent coloring across all subplots
        all_items_set = set()
        for plot_result in result:
            for ranking in plot_result.value["rankings"]:
                all_items_set.update(ranking)

        all_items_list = sorted(all_items_set)
        cmap = plt.get_cmap("tab20")

        # Plot each result
        plot_idx = 0
        for i in range(nrows):
            for j in range(ncols):
                if plot_idx >= num_plots:
                    # Hide unused subplots
                    axes[i, j].set_visible(False)
                    plot_idx += 1
                    continue

                ax = axes[i, j]
                plot_result = result[plot_idx]

                rankings = plot_result.value["rankings"]
                accuracies = plot_result.value["accuracy"]
                generalization_levels = plot_result.value["generalization_levels"]
                model_names = plot_result.value["names"]

                y_locs = range(len(model_names))

                # Plot each feature as a line
                for idx, item in enumerate(all_items_list):
                    ranks = []
                    y_vals = []
                    for j_inner, ranking in enumerate(rankings):
                        if item in ranking:
                            ranks.append(ranking.index(item) + 1)
                            y_vals.append(j_inner)
                        else:
                            ranks.append(None)
                            y_vals.append(j_inner)

                    color = cmap(idx % 20)
                    ax.plot(
                        ranks,
                        y_locs,
                        marker="o",
                        linestyle="-",
                        label=item,
                        color=color,
                    )

                ax.set_yticks(y_locs)
                ax.set_yticklabels(model_names)
                ax.invert_yaxis()

                # Calculate the maximum rank for the current subplot
                max_rank_for_subplot = 0
                for ranking in rankings:
                    max_rank_for_subplot = max(max_rank_for_subplot, len(ranking))

                ax.set_xlim(0, max_rank_for_subplot + 1)
                ax.set_xticks(range(1, max_rank_for_subplot + 1))
                ax.set_xlabel("Rank")

                ax.grid(axis="x", linestyle="--", alpha=0.5)
                for s in ["top", "right", "left"]:
                    ax.spines[s].set_visible(False)

                # Add secondary y-axis with accuracy and generalization level
                ax2 = ax.twinx()
                ax2.set_yticks(y_locs)

                # Generate new yticklabels with both accuracy and generalization level
                formatted_labels = []
                for k in range(len(accuracies)):
                    label = f"{accuracies[k]:.2f} ({generalization_levels[k]}%)"
                    formatted_labels.append(label)
                ax2.set_yticklabels(formatted_labels)

                ax2.set_ylabel("Model Accuracy (Generalization Level)")
                ax2.invert_yaxis()
                ax2.set_ylim(ax.get_ylim())

                ax.set_title(
                    f"{self.string_beautify(plot_result.classifier)}-{self.string_beautify(plot_result.explanation_method).upper()}-{self.string_beautify(plot_result.anonymization_method)}"
                )

                plot_idx += 1

        # Create a single legend in the lower left corner of the first plot
        handles, labels = [], []
        for i in range(nrows):
            for j in range(ncols):
                ax = axes[i, j]
                if ax.get_visible():
                    for handle, label in zip(*ax.get_legend_handles_labels()):
                        if label not in labels:
                            handles.append(handle)
                            labels.append(label)

        # Remove individual subplot legends
        for i in range(nrows):
            for j in range(ncols):
                ax = axes[i, j]
                if ax.get_visible() and ax.get_legend():
                    ax.get_legend().remove()

        # Add legend to the first subplot (lower left corner)
        axes[0, 0].legend(
            handles,
            labels,
            loc="lower right",
            title="Features",
            fontsize="small",
        )

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    pl = PlotCreator(
        ["MLP", "forest", "knn"],
        [
            # "old_adult",
            "usa_house",
            "usa_house_old",
            "cervic_cancer",
            "adult",
            "cervic_cancer_reversed",
        ],
        "./data/",
    )
    # pl.plot_utility("adult", ["MLP", "forest", "knn"], True)
    pl.plot_utility("cervic_cancer", average_models=True, show_feature_counts=True)
    # pl.plot_line("forest", "adult", "shap")
    # pl.plot_line("MLP", "usa_house", "shap")
    # pl.plot_consistency(
    #     "usa_house",
    #     show_tau=True,
    #     legend_placement="upper right",
    # )
    # pl.plot_histogram("usa_house", threshold=0.05)
    # pl.plot_line_compare(
    #     "usa_house",
    #     [
    #         "knn-lime-k_anonymity",
    #         "forest-lime-k_anonymity",
    #         "MLP-lime-k_anonymity",
    #         "MLP-shap-k_anonymity",
    #     ],
    # )
