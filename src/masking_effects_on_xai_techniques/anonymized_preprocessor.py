# Ordinal encode everything that has been anonymized
# Use the hiearachy as a the ordinal encoding order
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from pandas import DataFrame


def encode(df: DataFrame, hierarchy_path="./hierarchies/") -> DataFrame:
    encoding_order = {}

    for column in df.columns:
        if column == "index":
            continue
        hierarachy = dict(pd.read_csv(f"{hierarchy_path}{column}.csv", header=None))

        # We assume that each item only occurs once in a hierarchy file across different hieracrhies
        first_item = df[column].iloc[0]
        for values in hierarachy.values():
            if first_item in list(values):
                encoding_order[column] = values.unique().tolist()
                break
    return _encode_by_order(df, encoding_order)


def _encode_by_order(df: DataFrame, encoding_order: dict[str, list[str]]) -> DataFrame:
    df = df.copy()
    for column, order in encoding_order.items():
        encoder = OrdinalEncoder(categories=[order])
        df[column] = encoder.fit_transform(df[[column]])
    return df
