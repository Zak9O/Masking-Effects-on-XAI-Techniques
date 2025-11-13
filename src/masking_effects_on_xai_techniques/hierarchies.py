import pandas as pd
import numpy as np


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
 
