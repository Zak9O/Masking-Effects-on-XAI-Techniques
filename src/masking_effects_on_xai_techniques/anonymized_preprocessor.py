# Ordinal encode everything that has been anonymized
# Use the hiearachy as a the ordinal encoding order
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from pandas import DataFrame


def encode(
    df: DataFrame, hierarchy_path="./hierarchies/", skip: None | list[str] = None
) -> DataFrame:
    df = df.copy()
    if skip is None:
        skip = []

    for column in df.columns:
        if column == "index" or column in skip:
            continue
        hierarachy = dict(pd.read_csv(f"{hierarchy_path}{column}.csv", header=None))

        first_item = df[column].iloc[0]

        if isinstance(first_item, (float, int)):
            continue

        # We assume that each item only occurs once in a hierarchy file across different hieracrhies

        for values in hierarachy.values():
            if first_item in list(values):
                try:
                    encoder = OrdinalEncoder(categories=[values.unique().tolist()])
                    df[column] = encoder.fit_transform(df[[column]])
                    break
                except ValueError:
                    continue
    return df
