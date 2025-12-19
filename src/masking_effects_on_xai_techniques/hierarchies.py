import pandas as pd


def save_hierarchy(hierarchy: dict[int, pd.Series], path: str):  # pyright: ignore[reportRedeclaration]
    hierarchy: pd.DataFrame = pd.DataFrame(hierarchy)
    hierarchy.to_csv(path, index=False, header=False)


def generate_qcut_hierarchy(
    values: pd.Series, levels: int, search_for_n=False, prepend_level=False
) -> dict[int, pd.Series]:
    unique = values.copy().drop_duplicates().sort_values()

    hierarchy: dict[int, pd.Series] = {}

    hierarchy[0] = unique

    for i in range(0, levels - 1):
        if search_for_n:
            bins = _search_for_bins(values, levels - i)
        else:
            bins = pd.qcut(values, levels - i)
        current_level_hierarchy = []

        for j in unique.index:
            if prepend_level:
                level_val = str(i + 1) + str(bins[j])
            else:
                level_val = bins[j]

            current_level_hierarchy.append(level_val)

        hierarchy[i + 1] = pd.Series(current_level_hierarchy, index=unique.index)
        pass

    hierarchy[levels] = pd.Series("*", index=unique.index)

    return hierarchy


def _search_for_bins(values: pd.Series, levels: int) -> pd.Series:
    iterations = 0
    max_iterations = 10
    n = levels
    n_old = levels
    while True:
        if iterations >= max_iterations:
            raise RuntimeError(f"Iterations exceeded {max_iterations}")
        bins = pd.qcut(values, n, duplicates="drop")
        bins_len = bins.nunique()
        if bins_len == levels:
            break
        elif bins_len < levels:
            n_old = n
            n *= 2
        elif bins_len > levels:
            n = n - int((n - n_old) / 2)
        iterations += 1
    return bins
