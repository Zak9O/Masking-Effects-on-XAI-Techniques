# Ordinal encode everything that has been anonymized
# Use the hiearachy as a the ordinal encoding order
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from pandas import DataFrame


def encode(
    df: DataFrame, hierarchy_path="./hierarchies/", skip: None | list[str] = None
) -> DataFrame:
    df, _ = _encode_inner(df, hierarchy_path, skip)
    return df


def _encode_inner(
    df: DataFrame, hierarchy_path="./hierarchies/", skip: None | list[str] = None
) -> tuple[DataFrame, dict[int, np.ndarray]]:
    if skip is None:
        skip = []
    encoder_mappings: dict[int, np.ndarray] = {}

    for column in df.columns:
        if column == "index" or column in skip:
            continue
        hierarachy = dict(pd.read_csv(f"{hierarchy_path}{column}.csv", header=None))

        first_item = df[column].iloc[0]

        if isinstance(first_item, (float, int)):
            continue

        for i, values in enumerate(hierarachy.values()):
            if first_item in list(values):
                try:
                    encoder = OrdinalEncoder(categories=[values.unique().tolist()])
                    df[column] = encoder.fit_transform(df[[column]])
                    column_index: int = df.columns.get_loc(column)  # pyright: ignore[reportAssignmentType]
                    encoder_mappings[column_index] = np.array(encoder.categories_[0])
                    break
                except ValueError:
                    continue
            if i + 1 == len(hierarachy.values()):
                raise ValueError(
                    f"Value: {first_item} not found in hierarchy for {column}"
                )
    return df, encoder_mappings
