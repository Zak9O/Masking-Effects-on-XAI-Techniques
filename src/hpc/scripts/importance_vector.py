import argparse
import logging
import pathlib

import numpy as np
import pandas as pd
import shap
import sklearn
from lime import submodular_pick
from lime import lime_tabular
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def _clean_feature_names(features):
    names = []
    for key in features:
        if 1 == (key.count("<") + key.count(">")):
            key = key.split(" ")[0]
        elif 2 <= key.count("<"):
            key = key.split(" ")[2]
        else:
            key = key.split("=")[0]
        names.append(key)
    return names


def _name_weights(relation, features):
    names = _clean_feature_names(features)
    return [(names[i], weight) for i, weight in relation]


def _importance_vector_of(exp):
    relations = exp.as_map()[1]
    features = exp.domain_mapper.feature_names
    relations = _name_weights(relations, features)
    return [(_, np.abs(w)) for _, w in relations]


def importance_vector_sum(submoduler_exp) -> list[tuple[str, float]]:
    sp_exp = [exp for exp in submoduler_exp.sp_explanations]

    importance_vectors = [_importance_vector_of(exp) for exp in sp_exp]

    feature_sums = {}
    for explanation in importance_vectors:
        for feature_name, importance_score in explanation:
            feature_sums[feature_name] = (
                feature_sums.get(feature_name, 0.0) + importance_score
            )

    summed_list_sorted = sorted(
        list(feature_sums.items()),
        key=lambda item: item[1],  # pyright: ignore[reportUnknownLambdaType]
        reverse=True,
    )
    return summed_list_sorted


def normalize_feature(train: DataFrame, test: DataFrame, feature: str) -> None:
    scaler = StandardScaler()
    _ = scaler.fit(train[[feature]])
    train[feature] = scaler.transform(train[[feature]])
    test[feature] = scaler.transform(test[[feature]])


def one_hot_encoding(df: DataFrame, feature: str, encoder) -> DataFrame:
    encoded_array = encoder.transform(df[[feature]])

    feature_names = encoder.get_feature_names_out([feature])
    df_encoded_part = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
    return df.drop([feature], axis=1).join(df_encoded_part)


def _encode_by_order(df: DataFrame, encoding_order: dict[str, list[str]]) -> DataFrame:
    df = df.copy()
    logging.info("Applying ordinal encoding.")
    for column, order in encoding_order.items():
        logging.debug(f"Encoding column '{column}' with order: {order}")
        encoder = OrdinalEncoder(categories=[order], dtype=int)
        df[column] = encoder.fit_transform(df[[column]])
    logging.info("Ordinal encoding applied successfully.")
    return df


def _encode_by_hierarchy_inner(
    df: DataFrame,
    hierarchy_path="./hierarchies/",
    skip_columns=[],  # pyright: ignore[reportCallInDefaultInitializer]
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


def encode_by_hierarchy(
    df: DataFrame,
    hierarchy_path="./hierarchies/",
    skip_columns=[],  # pyright: ignore[reportCallInDefaultInitializer]
) -> DataFrame:
    return _encode_by_hierarchy_inner(df, hierarchy_path, skip_columns)[0]


def importance_values_to_str(features: list[str], importance) -> list[tuple[str, int]]:
    return [(features[i], importance[i]) for i in np.argsort(-importance)]


def create_one_hot_encoder(df: pd.DataFrame, feature: str) -> OneHotEncoder:
    return OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
        df[[feature]]
    )


def shap_importance(df: pd.DataFrame) -> tuple[float, list[tuple[str, int]]]:
    logging.info("Calculating SHAP importance values.")
    numeric_features = ["age", "capital-gain", "capital-loss", "hours-per-week"]
    skip_columns = []
    logging.info("Identifying numeric features to skip from encoding.")
    for feature in numeric_features:
        # if feature is numeric in df then add to skip_columns
        d_type = type(df[feature])
        if d_type is not str:  # pyright: ignore[reportUnnecessaryComparison]
            skip_columns.append(feature)
    logging.info(f"Skipping encoding for numeric features: {skip_columns}")

    df = encode_by_hierarchy(df, skip_columns=skip_columns)

    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(  # pyright: ignore[reportAttributeAccessIssue]
        df.drop(["income"], axis=1), df["income"], test_size=0.4, random_state=0
    )

    for feature in numeric_features:
        if feature not in skip_columns:
            continue
        logging.info(f"Normalizing feature '{feature}'")
        normalize_feature(X_train, X_test, feature)

    logging.info(
        f"Data split into training and testing sets. Training set size: {len(X_train)}, Testing set size: {len(X_test)}"
    )

    clf = sklearn.neural_network.MLPClassifier(  # pyright: ignore[reportAttributeAccessIssue]
        solver="sgd", alpha=1e-5, hidden_layer_sizes=(10), random_state=1
    )
    logging.info("Training MLPClassifier.")
    clf.fit(np.array(X_train), y_train)
    score = clf.score(np.array(X_train), y_train)
    logging.info(f"Model training finished. Score: {score}")

    def f(x):
        return clf.predict_proba(x)[:, 1]

    med = X_train.median().values.reshape((1, X_train.shape[1]))
    explainer = shap.Explainer(f, med)

    # TODO: change this
    logging.info("Calculating SHAP values.")
    logging.warning(
        "Using only the first 10 instances of the test set for SHAP value calculation."
    )
    shap_values = explainer(X_test)

    importance = np.mean(np.abs(shap_values.values), axis=0)
    logging.info("SHAP importance calculation finished.")

    return score, importance_values_to_str(X_train.columns, importance)


def lime_importance(df: pd.DataFrame):
    logging.info("Calculating LIME importance values.")
    numeric_features = ["age", "capital-gain", "capital-loss", "hours-per-week"]

    skip_columns = []
    logging.info("Identifying numeric features to skip from encoding.")
    for feature in numeric_features:
        # if feature is numeric in df then add to skip_columns
        d_type = type(df[feature].iloc[0])
        if d_type is not str:
            skip_columns.append(feature)
    logging.info(f"Skipping encoding for numeric features: {skip_columns}")

    df, encoding_mappings = _encode_by_hierarchy_inner(df, skip_columns=skip_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["income"]), df["income"], test_size=0.4, random_state=0
    )
    logging.info(
        f"Data split into training and testing sets. Training set size: {len(X_train)}, Testing set size: {len(X_test)}"
    )

    encoders = {}
    logging.info("Creating one-hot encoders for categorical features.")
    for feature in X_train.columns:
        if feature in skip_columns:
            continue
        encoder = create_one_hot_encoder(X_train, feature)
        encoders[feature] = encoder

    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    logging.info("Applying one-hot encoding to training and testing sets.")
    for feature, encoder in encoders.items():
        X_train_enc = one_hot_encoding(X_train_enc, feature, encoder)
        X_test_enc = one_hot_encoding(X_test_enc, feature, encoder)

    # Scale numeric values and one hot encode categorical features
    logging.info("Normalizing numeric features.")
    for feature in numeric_features:
        if feature not in skip_columns:
            continue
        logging.debug(f"Normalizing feature '{feature}'.")
        normalize_feature(X_train_enc, X_test_enc, feature)

    clf = MLPClassifier(
        solver="sgd", alpha=1e-5, hidden_layer_sizes=(10), random_state=1
    )
    logging.info("Training MLPClassifier.")
    _ = clf.fit(X_train_enc, y_train)
    score = clf.score(X_test_enc, y_test)
    logging.info(f"Model training finished. Score: {score}")

    def f(x):
        x = pd.DataFrame(x, columns=X_train.columns.tolist())
        for feature, encoder in encoders.items():
            x = one_hot_encoding(x, feature, encoder)
        return clf.predict_proba(x).astype(float)

    cat_features = list(X_train.columns)
    for numeric_feat in numeric_features:
        cat_features.remove(numeric_feat)
    cat_features = [X_train.columns.tolist().index(i) for i in cat_features]

    X_train_np = np.array(X_train)

    logging.info("Creating LIME Tabular Explainer.")
    explainer = lime_tabular.LimeTabularExplainer(
        X_train_np,
        feature_names=X_train.columns,
        class_names=list(encoding_mappings.values())[-1],
        categorical_features=cat_features,
        categorical_names=encoding_mappings,
        kernel_width=3,
    )

    logging.info("Running Submodular Pick to get explanations.")
    sb_pick = submodular_pick.SubmodularPick(
        explainer,
        X_train_np,
        f,
        sample_size=15000,
        num_features=5,
        num_exps_desired=1000,
    )
    logging.info("LIME importance calculation finished.")
    return score, importance_vector_sum(sb_pick)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Starting importance vector calculation script.")
    parser = argparse.ArgumentParser(
        description="Calculate importance vecotr for models"
    )
    _ = parser.add_argument(
        "data_path",
        type=pathlib.Path,
    )
    _ = parser.add_argument(
        "data_out",
        type=pathlib.Path,
    )
    _ = parser.add_argument(
        # Can be "shap", "lime", "integrated_gradients"
        "explainer_type",
        type=str,
    )
    args = parser.parse_args()
    logging.info(
        f"Arguments parsed: data_path={args.data_path}, data_out={args.data_out}, explainer_type={args.explainer_type}"
    )

    logging.info(f"Reading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    if "index" in df.columns:
        df = df.drop("index", axis=1)

    if args.explainer_type == "shap":
        score, importance = shap_importance(df)
    elif args.explainer_type == "lime":
        score, importance = lime_importance(df)
    elif args.explainer_type == "integrated_gradients":
        logging.warning("Integrated Gradients explainer is not yet implemented.")
        pass
    else:
        logging.error(f"Explainer type '{args.explainer_type}' is not supported.")
        raise NotImplementedError

    output_path = pathlib.Path(args.data_out)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving importance vector to {args.data_out}")
    np.save(args.data_out, [(score, "accuracy")] + importance)
    logging.info("Script finished successfully.")
