#
# This code has been copied from the anjana github library
# https://github.com/IFCA-Advanced-Computing/anjana
# The library cannot be used because it is incompatible with the current versions of other libraries
import pandas as pd
import numpy as np


def _check_gen_level(
    data: pd.DataFrame,
    quasi_ident: np.ndarray,
    hierarchies,
):
    gen_level = {}

    for qi in quasi_ident:
        if qi not in hierarchies:
            continue

        current_data = data[qi].values
        unique_data = np.unique(current_data)

        is_float = np.issubdtype(unique_data.dtype, np.floating)

        for level in hierarchies[qi].keys():
            hier_vals = np.array(hierarchies[qi][level])

            if is_float:
                matches_matrix = np.isclose(unique_data[:, None], hier_vals[None, :])
                point_exists = np.any(matches_matrix, axis=1)
                is_subset = np.all(point_exists)

            else:
                is_subset = set(unique_data).issubset(hier_vals)

            if is_subset:
                gen_level[qi] = level
                break

    return gen_level


def get_transformation(
    data_anon: pd.DataFrame, quasi_ident: list[str], hierarchies
) -> tuple[int, int]:
    max_transformation_level = 0
    for feat in quasi_ident:
        max_transformation_level += len(hierarchies[feat].keys()) - 1

    gen_level = _check_gen_level(data_anon, quasi_ident, hierarchies)
    transformation = []
    for qi in quasi_ident:
        if qi in gen_level.keys():
            transformation.append(gen_level[qi])
        else:
            transformation.append(0)

    return sum(transformation), max_transformation_level
