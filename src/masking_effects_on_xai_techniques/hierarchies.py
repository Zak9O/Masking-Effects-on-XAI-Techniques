import pandas as pd
import numpy as np


def save_hierarchy(hierarchy: dict[int, pd.Series], path: str, sort=True):  # pyright: ignore[reportRedeclaration]
    hierarchy: pd.DataFrame = pd.DataFrame(hierarchy)
    if sort:
        hierarchy = hierarchy.sort_values(by=0)  # pyright: ignore[reportArgumentType]
        hierarchy = hierarchy.drop_duplicates(subset=[0])
    hierarchy.to_csv(path, index=False, header=False)


def generate_qcut_hierarchy(values: pd.Series, levels: int) -> dict[int, pd.Series]:
    unique = values.copy().drop_duplicates().sort_values()

    hierarchy: dict[int, pd.Series] = {}

    hierarchy[0] = unique

    for i in range(1, levels - 1):
        bins = _generate_bins(unique, levels - i)
        current_level_hierarchy = []

        for j in unique.index:
            current_level_hierarchy.append(bins[j])

        hierarchy[i] = pd.Series(current_level_hierarchy, index=unique.index)
        pass

    hierarchy[levels - 1] = pd.Series("*", index=unique.index)

    return hierarchy


def _generate_bins(values: pd.Series, levels: int) -> pd.Series:
    n = levels
    n_old = levels
    while True:
        bins = pd.qcut(values, n, duplicates="drop")
        bins_len = bins.nunique()
        if bins_len == levels:
            break
        elif bins_len < levels:
            n_old = n
            n *= 2
        elif bins_len > levels:
            n = n - int((n - n_old) / 2)
    return bins


def generate_hierarchy(values: pd.Series, levels: int) -> dict[int, pd.Series]:
    hierarchy: dict[int, pd.Series] = {}

    hierarchy[0] = values

    for i in range(1, levels):
        hierarchy[i] = generalize_series(values, levels - i + 1)

    hierarchy[levels] = pd.Series("*", index=values.index)

    return hierarchy


def generalize_series(data_series: pd.Series, n: int) -> pd.Series:
    epsilon = 1e-9
    min_val = data_series.min() - epsilon
    max_val = data_series.max() + epsilon

    if min_val < 0:
        min_val = 0

    bins = np.linspace(min_val, max_val, n + 1)

    labels = []
    for i in range(n):
        lower = int(np.floor(bins[i]))
        upper = int(np.ceil(bins[i + 1]))
        label = f"[{lower}, {upper}["
        labels.append(label)

    binned_series = pd.cut(data_series, bins=bins, labels=labels, include_lowest=True)

    # Below is added to adhere to anjana API
    binned_series.name = n
    binned_series = binned_series.astype(object)

    return binned_series
