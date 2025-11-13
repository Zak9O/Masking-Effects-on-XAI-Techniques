# Ordinal encode everything that has been anonymized
# Use the hiearachy as a the ordinal encoding order
import pandas as pd
from pandas import DataFrame




def encode(df: DataFrame, encoding_order: dict[str, dict[str, int]]) -> DataFrame:
    df = df.copy()
    for column, order in encoding_order.items():
        df[column] = df[column].map(order)
    return df
