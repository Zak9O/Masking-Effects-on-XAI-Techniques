import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
        anonymizatino_method: str,
        explanation_method: str,
        dataset: str,
    ) -> None:
        self.value = value
        self.classifier = classifier
        self.anonymizatino_method = anonymizatino_method
        self.explanation_method = explanation_method
        self.dataset = dataset


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
                                anonymizatino_method=anon_model_name,
                                explanation_method=method,
                                dataset=dataset,
                            )
                        )
        return result

    def plot_utility(
        self, dataset: str, classifiers: Optional[list[str]] = None
    ) -> None:
        if classifiers is None:
            classifiers = ["knn", "forest", "MLP"]

        def _filter(x: list[Explanation], clean: Explanation):
            transform_levels = [0] + [e.get_transform_level() for e in x]
            accuracies = [clean.accuracy] + [e.accuracy for e in x]
            return {"accuracies": accuracies, "generalization levels": transform_levels}

        result = self._filter_data(_filter)
        result = list(
            filter(
                lambda x: x.classifier in classifiers
                and x.dataset == dataset
                and x.explanation_method == "lime",
                result,
            )
        )

        # Create subplots, one for each classifier
        n_classifiers = len(classifiers)
        fig, axes = plt.subplots(
            1, n_classifiers, figsize=(6 * n_classifiers, 7), sharey=True
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

        # Color map for accuracy values
        cmap = plt.get_cmap("viridis")

        # Get global min/max accuracy for consistent color mapping across all series
        from matplotlib.colors import Normalize

        global_min_acc = min([min(p.value["accuracies"]) for p in result])
        global_max_acc = max([max(p.value["accuracies"]) for p in result])
        norm = Normalize(vmin=global_min_acc, vmax=global_max_acc)

        # Plot for each classifier
        for ax, classifier in zip(axes, classifiers):
            # Filter results for this classifier
            classifier_result = [p for p in result if p.classifier == classifier]

            if not classifier_result:
                continue

            # Plot each series for this classifier
            for plot_result in classifier_result:
                gen_levels = plot_result.value["generalization levels"]
                accuracies = plot_result.value["accuracies"]

                # Create indices for x-axis
                indices = list(range(len(gen_levels)))

                # Get marker style for this anonymization method
                marker = marker_styles.get(plot_result.anonymizatino_method, "o")

                # Plot lines connecting the points
                ax.plot(indices, gen_levels, "k-", alpha=0.3, linewidth=1)

                # Plot points with colors based on accuracy
                for i, (idx, gen_level, acc) in enumerate(
                    zip(indices, gen_levels, accuracies)
                ):
                    # Get normalized accuracy for this specific point
                    normalized_acc = norm(acc)
                    color = cmap(normalized_acc)
                    ax.scatter(
                        idx,
                        gen_level,
                        marker=marker,
                        s=100,
                        color=color,
                        label=f"{plot_result.anonymizatino_method}" if i == 0 else "",
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

            ax.set_xlabel("Generalization Level Index")
            ax.set_title(f"{classifier.upper()} - {dataset}")
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)

        # Add colorbar for accuracy values on the rightmost subplot
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes[-1])
        cbar.set_label("Accuracy")

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
                label=method,
            )
            for method, marker in marker_styles.items()
        ]
        axes[0].legend(
            handles=legend_elements, loc="upper right", title="Anonymization Method"
        )

        axes[0].set_ylabel("Generalization Level (%)")

        plt.tight_layout()
        plt.show()

    def plot_histogram(self, dataset: str, threshold: float = 0.05) -> None:
        def _filter(x: list[Explanation], clean: Explanation):
            ran = [e.compute_kendal_tau(clean.get_ranking()).pvalue for e in x]
            return {"value": ran}

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
                    lambda x: x.explanation_method == "lime" and x.dataset == dataset,
                    result,
                ),
            )
        )
        shap = list(
            map(
                f,
                filter(
                    lambda x: x.explanation_method == "shap" and x.dataset == dataset,
                    result,
                ),
            )
        )

        # Normalize LIME values per anonymization method so entries sharing the same
        # anonymization_method are divided by their group sum.
        # for _, anon_method in self.ANONYMIZATION_MODELS:
        #     for classifier in set(r.classifier for r in lime):
        #         lime_element = next(filter(lambda x: x.classifier == classifier and x.anonymizatino_method == anon_method, lime))
        #         shap_element = next(filter(lambda x: x.classifier == classifier and x.anonymizatino_method == anon_method, shap))
        #         total = lime_element.value + shap_element.value
        #         lime_element.value = lime_element.value / total
        #         shap_element.value = shap_element.value / total

        # Get unique classifiers and anonymization methods
        classifiers = sorted(set([r.classifier for r in lime]))
        anon_methods = sorted(set([r.anonymizatino_method for r in lime]))

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
                        and r.anonymizatino_method == anon_method
                    ),
                    None,
                )
                shap_result = next(
                    (
                        r
                        for r in shap
                        if r.classifier == classifier
                        and r.anonymizatino_method == anon_method
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

        # Define colors for anonymization methods
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(n_methods)]

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
                color=colors[i],
                alpha=0.8,
            )

            # SHAP bar (semi-transparent with solid outline) - right next to LIME
            shap_values = [
                shap_data[anon_method][classifier] for classifier in classifiers
            ]
            shap_positions = x - offset + (i * 2 + 1) * width
            ax.bar(
                shap_positions,
                shap_values,
                width,
                label="" if i < len(anon_methods) - 1 else "",
                color=colors[i],
                alpha=0.3,
                edgecolor=colors[i],
                linewidth=1.5,
            )

        # Create custom legend combining both color and shading information

        # Color legend (anonymization methods)
        color_handles = [
            Patch(facecolor=colors[i], label=method)
            for i, method in enumerate(anon_methods)
        ]

        # Shading legend (explanation methods)
        shading_handles = [
            Patch(facecolor="gray", alpha=0.8, label="LIME"),
            Patch(
                facecolor="gray",
                alpha=0.3,
                edgecolor="gray",
                linewidth=1.5,
                label="SHAP",
            ),
        ]

        # Combine all handles in one legend with a separator
        all_handles = color_handles + shading_handles
        ax.legend(handles=all_handles, bbox_to_anchor=(1.05, 1), loc="upper left")

        ax.set_xlabel("Classifier")
        ax.set_ylabel(f"# of times |$\\Delta$p-value|> {threshold}")
        ax.set_title(f"Stability LIME vs SHAP for {dataset}")
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers)
        ax.grid(axis="y", alpha=0.3)

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

    def plot_consistency(self, dataset: str, methods: list[str] | None = None) -> None:
        data = {"shap": {}, "lime": {}}
        for classifier in ["MLP", "forest", "knn"]:
            for method in data.keys():
                explanations = (
                    self.models[classifier].datasets[dataset].explanations[method]
                )
                for anon_model_name, expl_list in explanations.items():
                    if anon_model_name == "clean" or (
                        methods and anon_model_name not in methods
                    ):
                        continue

                    ran = [
                        e.compute_kendal_tau(
                            explanations["clean"][0].get_ranking()
                        ).pvalue
                        for e in expl_list
                    ]
                    # tmp = explanations['clean'] + expl_list
                    # for e1, e2 in zip(tmp[:-1], tmp[1:]):
                    #     kt = e2.compute_kendal_tau(e1.get_ranking())
                    #     ran.append(kt.pvalue)

                    existing = data[method].get(anon_model_name, [])
                    if existing:
                        data[method][anon_model_name] = [
                            (np.nan_to_num(e, r) + np.nan_to_num(r, e)) / 2
                            for e, r in zip(existing, ran)
                        ]
                    else:
                        data[method][anon_model_name] = ran

        above = {}
        for method in data["lime"].keys():
            above[method] = sum(
                1 for s, la in zip(data["shap"][method], data["lime"][method]) if la > s
            )
            above[method] /= len(data["lime"][method])
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
        x_offsets = {
            "shap": -0.05,
            "lime": 0.05,
        }
        max_len = 0
        for method, method_data in data.items():
            for anon_model_name, values in method_data.items():
                x_vals = np.arange(1, len(values) + 1, dtype=float)
                x_vals += x_offsets.get(method, 0.0)
                max_len = max(max_len, len(x_vals))
                plt.plot(
                    x_vals,
                    values,
                    color=colors.get(anon_model_name, None),
                    linestyle=linestyles.get(method, "-"),
                    linewidth=2.0,
                    alpha=0.85,
                )

        plt.xlabel("Anonymization Level")
        plt.xticks(range(1, max_len + 1))
        plt.ylabel("Kendall Tau p-value")
        plt.title(f"{dataset} Kendall Tau p-value Comparison: LIME vs SHAP")

        # Legends: colors for anonymization models, linestyles for methods
        ax = plt.gca()
        color_handles = [
            Line2D(
                [0],
                [0],
                color=col,
                lw=2,
                label=f"{anon} ({above.get(anon, 0) * 100:.1f}%)",
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
                label=f"{method} ({avg_above * 100:.1f}%)"
                if method == "lime"
                else f"{method}",
            )
            for method, ls in list(linestyles.items())[::-1]
        ]

        anon_legend = ax.legend(
            handles=color_handles,
            title="Anonymization (# LIME > SHAP %)",
            loc="upper left",
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

    def plot_line_compare(
        self,
        classifier1: str,
        classifier2: str,
        dataset: str,
        method: str,
        model_type: str,
    ) -> None:
        # Create two plots side by side using plot_line for two different classifiers
        # classifier1: first classifier type
        # classifier2: second classifier type
        # dataset: dataset name
        # method: explanation method

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
        fig.suptitle(
            f"Feature Rank Comparison: {classifier1} vs {classifier2} on {dataset} using {method}",
            fontsize=16,
        )

        classifiers_to_plot = [classifier1, classifier2]

        for ax_idx, classifier in enumerate(classifiers_to_plot):
            ax = axes[ax_idx]
            plt.sca(ax)

            # Get the explanations for this classifier
            if classifier not in self.models:
                ax.text(
                    0.5,
                    0.5,
                    f"Classifier '{classifier}' not found",
                    ha="center",
                    va="center",
                )
                ax.set_title(f"{classifier}")
                continue

            model = self.models[classifier]

            if dataset not in model.datasets:
                ax.text(
                    0.5, 0.5, f"Dataset '{dataset}' not found", ha="center", va="center"
                )
                ax.set_title(f"{classifier}")
                continue

            dataset_obj = model.datasets[dataset]

            if method not in dataset_obj.explanations:
                ax.text(
                    0.5, 0.5, f"Method '{method}' not found", ha="center", va="center"
                )
                ax.set_title(f"{classifier}")
                continue

            explanations = dataset_obj.explanations[method]

            # Collect anonymization models
            anon_models_list = []
            for anon_model_name, expl_list in explanations.items():
                if anon_model_name == "clean":
                    continue
                if anon_model_name == model_type:
                    anon_models_list.append((anon_model_name, expl_list))

            # Only use first anonymization method for this simplified version
            if anon_models_list:
                anon_model_name, expl_list = anon_models_list[0]

                # Get clean explanation
                clean_expl = explanations.get("clean", [None])[0]
                expl_list_with_clean = (
                    [clean_expl] + list(expl_list) if clean_expl else list(expl_list)
                )

                # Y-axis positions
                y_locs = range(len(expl_list_with_clean))

                # Collect y-axis labels
                y_labels = []
                expl_names = []
                for e in expl_list_with_clean:
                    accuracy = e.accuracy
                    generalization_level = round(
                        100 * e.transform_n / e.transform_n_max
                    )
                    label = f"{accuracy:.2f} ({generalization_level}%)"
                    y_labels.append(label)
                    expl_names.append(e.name)

                # Get all unique features
                all_items = set()
                for e in expl_list_with_clean:
                    all_items.update(e.get_ranking())

                all_items_list = sorted(all_items)
                cmap = plt.get_cmap("tab20")

                # Plot lines for each feature
                for idx, item in enumerate(all_items_list):
                    ranks = []
                    y_vals = []
                    for i, e in enumerate(expl_list_with_clean):
                        ranking = e.get_ranking()
                        if item in ranking:
                            ranks.append(ranking.index(item) + 1)
                            y_vals.append(i)

                    # Only plot if there's at least one valid rank
                    if ranks:
                        color = cmap(idx % 20)
                        ax.plot(
                            ranks,
                            y_vals,
                            marker="o",
                            linestyle="-",
                            label=item,
                            color=color,
                        )

                ax.set_yticks(y_locs)
                ax.set_yticklabels(y_labels)
                ax.invert_yaxis()

                # Create a second y-axis showing explanation names
                ax2 = ax.twinx()
                ax2.set_yticks(y_locs)
                ax2.set_yticklabels(expl_names[::-1])
                ax2.set_ylim(ax.get_ylim())
                ax2.invert_yaxis()
                ax2.set_ylabel("Explanation Name")

                # Calculate the maximum rank
                max_rank = 0
                for e in expl_list_with_clean:
                    max_rank = max(max_rank, len(e.get_ranking()))

                ax.set_xlim(0, max_rank + 1)
                ax.set_xticks(range(1, max_rank + 1))
                ax.set_xlabel("Rank")
                ax.set_ylabel("Model Accuracy (Generalization %)")

                ax.grid(axis="x", linestyle="--", alpha=0.5)
                for s in ["top", "right"]:
                    ax.spines[s].set_visible(False)

                ax.set_title(f"{classifier}-{anon_model_name}")

        # Create a single legend for the entire figure
        handles, labels = [], []
        for ax in axes:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
            # ax.get_legend().remove()  # Remove individual subplot legends
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


if __name__ == "__main__":
    pl = PlotCreator(
        ["MLP", "forest", "knn"],
        ["adult", "old_adult", "usa_house", "cervic_cancer"],
        "./data/",
    )
    pl.plot_utility("usa_house")
