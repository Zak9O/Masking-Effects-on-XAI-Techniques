# Ordinal encode everything that has been anonymized
# Use the hiearachy as a the ordinal encoding order
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from pandas import DataFrame


def normalize_feature(train: DataFrame, test: DataFrame, feature: str) -> None:
    scaler = StandardScaler()
    _ = scaler.fit(train[[feature]])
    train[feature] = scaler.transform(train[[feature]])
    test[feature] = scaler.transform(test[[feature]])


def encode_by_hierarchy(
    df: DataFrame, hierarchy_path="./hierarchies/", skip_columns=[]
) -> DataFrame:
    return _encode_by_hierarchy_inner(df, hierarchy_path, skip_columns)[0]


def create_one_hot_encoder(df: pd.DataFrame, feature: str) -> OneHotEncoder:
    return OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
        df[[feature]]
    )


def one_hot_encoding(df: DataFrame, feature: str, encoder) -> DataFrame:
    encoded_array = encoder.transform(df[[feature]])

    feature_names = encoder.get_feature_names_out([feature])
    df_encoded_part = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
    return df.drop([feature], axis=1).join(df_encoded_part)


def _encode_by_hierarchy_inner(
    df: DataFrame, hierarchy_path="./hierarchies/", skip_columns=[]
) -> tuple[DataFrame, dict[int, list[str]]]:
    encoding_order = {}

    for column in df.columns:
        if column == "index" or column in skip_columns:
            continue
        hierarachy = dict(pd.read_csv(f"{hierarchy_path}{column}.csv", header=None))

        # We assume that each item only occurs once in a hierarchy file across different hieracrhies
        first_item = df[column].iloc[0]
        for values in hierarachy.values():
            if first_item in list(values):
                encoding_order[column] = values.unique().tolist()
                break
    return _encode_by_order(df, encoding_order), {
        df.columns.tolist().index(column): order
        for column, order in encoding_order.items()
    }


def _encode_by_order(df: DataFrame, encoding_order: dict[str, list[str]]) -> DataFrame:
    df = df.copy()
    for column, order in encoding_order.items():
        encoder = OrdinalEncoder(categories=[order], dtype=int)
        df[column] = encoder.fit_transform(df[[column]])
    return df
