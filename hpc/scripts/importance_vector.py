import argparse
import logging
import pathlib

import masking_effects_on_xai_techniques.anjana_utils as utils
import masking_effects_on_xai_techniques.anonymized_preprocessor as anon_prep
from masking_effects_on_xai_techniques.datasets import Dataset

import os
import numpy as np
import pandas as pd
import shap
import sklearn
from lime import submodular_pick
from lime import lime_tabular
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def standardize_sensitive_attr(y_train, y_test):
    scaler = StandardScaler()
    _ = scaler.fit(y_train.values.reshape(-1, 1))
    y_train = scaler.transform(y_train.values.reshape(-1, 1))
    y_test = scaler.transform(y_test.values.reshape(-1, 1))
    return y_train, y_test


def _create_MLP(MPL_type: str) -> MLPClassifier | MLPRegressor:
    if MPL_type == "classifier":
        return MLPClassifier(
            solver="sgd",
            alpha=1e-5,
            hidden_layer_sizes=(10),
            random_state=1,
            max_iter=2000,
        )
    else:
        return MLPRegressor(
            solver="sgd", alpha=1e-5, hidden_layer_sizes=(10), random_state=1
        )


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


def normalize_feature(train: DataFrame, feature: str) -> StandardScaler:
    scaler = StandardScaler()
    _ = scaler.fit(train[[feature]])
    return scaler


def one_hot_encoding(df: DataFrame, feature: str, encoder) -> DataFrame:
    encoded_array = encoder.transform(df[[feature]])

    feature_names = encoder.get_feature_names_out([feature])
    df_encoded_part = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
    return df.drop([feature], axis=1).join(df_encoded_part)


def importance_values_to_str(features: list[str], importance) -> list[tuple[str, int]]:
    return [(features[i], importance[i]) for i in np.argsort(-importance)]


def create_one_hot_encoder(df: pd.DataFrame, feature: str) -> OneHotEncoder:
    return OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
        df[[feature]]
    )


def create_hierachy(quasi_identifiers: list[str], hierarchy_path: str):
    hierarachies = {}
    for feat in quasi_identifiers:
        hierarachy = dict(pd.read_csv(f"{hierarchy_path}{feat}.csv", header=None))
        hierarachies[feat] = hierarachy

    return hierarachies


def shap_importance(
    df: pd.DataFrame, dataset: Dataset
) -> tuple[float, list[tuple[str, int]]]:
    logging.info("Calculating SHAP importance values.")
    numeric_features = dataset.numeric_features
    skip_columns = []
    logging.info("Identifying numeric features to skip from encoding.")
    for feature in numeric_features:
        # if feature is numeric in df then add to skip_columns
        d_type = type(df[feature].iloc[0])
        if d_type is not str:
            skip_columns.append(feature)
    logging.info(f"Skipping encoding for numeric features: {skip_columns}")

    df = anon_prep.encode(df, hierarchy_path=dataset.hierarchy_path, skip=skip_columns)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(  # pyright: ignore[reportAttributeAccessIssue]
        df.drop([dataset.sensitive_attr], axis=1),
        df[dataset.sensitive_attr],
        test_size=0.4,
        random_state=0,
    )

    numeric_encoders = {}

    for feature in numeric_features:
        if feature not in skip_columns:
            continue
        logging.info(f"Normalizing feature '{feature}'")
        numeric_encoders[feature] = normalize_feature(X_train, feature)
        X_train[feature] = numeric_encoders[feature].transform(X_train[[feature]])
        X_test[feature] = numeric_encoders[feature].transform(X_test[[feature]])

    logging.info(
        f"Data split into training and testing sets. Training set size: {len(X_train)}, Testing set size: {len(X_test)}"
    )

    clf = _create_MLP(dataset.classifier_type)
    logging.info("Training Model.")

    if dataset.classifier_type == "regressor":
        y_train, y_test = standardize_sensitive_attr(y_train, y_test)

    _ = clf.fit(np.array(X_train), y_train)
    score = clf.score(np.array(X_train), y_train)
    logging.info(f"Model training finished. Score: {score}")

    if dataset.classifier_type == "classifier":

        def f(x):  # pyright: ignore[reportRedeclaration]
            return clf.predict_proba(x)[:, 1]  # pyright: ignore[reportAttributeAccessIssue]
    else:

        def f(x):
            return clf.predict(x)

    med = X_train.median().values.reshape((1, X_train.shape[1]))
    explainer = shap.KernelExplainer(f, med)

    logging.info("Calculating SHAP values.")
    shap_values = explainer(X_test)

    importance = np.mean(np.abs(shap_values.values), axis=0)
    logging.info("SHAP importance calculation finished.")

    return score, importance_values_to_str(X_train.columns, importance)  # pyright: ignore[reportReturnType]


def lime_importance(df: pd.DataFrame, dataset: Dataset):
    logging.info("Calculating LIME importance values.")
    numeric_features = dataset.numeric_features

    skip_columns = []
    logging.info("Identifying numeric features to skip from encoding.")
    for feature in numeric_features:
        # if feature is numeric in df then add to skip_columns
        d_type = type(df[feature].iloc[0])
        if d_type is not str:
            skip_columns.append(feature)
    logging.info(f"Skipping encoding for numeric features: {skip_columns}")

    df, encoding_mappings = anon_prep._encode_inner(  # pyright: ignore[reportPrivateUsage]
        df, hierarchy_path=dataset.hierarchy_path, skip=skip_columns
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[dataset.sensitive_attr]),
        df[dataset.sensitive_attr],
        test_size=0.4,
        random_state=0,
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
    numeric_encoders = {}
    logging.info("Normalizing numeric features.")
    for feature in numeric_features:
        if feature not in skip_columns:
            continue
        logging.debug(f"Normalizing feature '{feature}'.")
        numeric_encoders[feature] = normalize_feature(X_train_enc, feature)
        X_train_enc[feature] = numeric_encoders[feature].transform(
            X_train_enc[[feature]]
        )
        X_test_enc[feature] = numeric_encoders[feature].transform(X_test_enc[[feature]])

    if dataset.classifier_type == "regressor":
        y_train, y_test = standardize_sensitive_attr(y_train, y_test)

    clf = _create_MLP(dataset.classifier_type)

    logging.info("Training MLPClassifier.")
    _ = clf.fit(X_train_enc, y_train)
    score = clf.score(X_test_enc, y_test)
    logging.info(f"Model training finished. Score: {score}")

    if dataset.classifier_type == "classifier":

        def f(x):  # pyright: ignore[reportRedeclaration]
            x = pd.DataFrame(x, columns=X_train.columns.tolist())
            for feature, encoder in encoders.items():
                x = one_hot_encoding(x, feature, encoder)
            for feature in numeric_features:
                if feature not in skip_columns:
                    continue
                x[feature] = numeric_encoders[feature].transform(x[[feature]])
            return clf.predict_proba(x).astype(float)  # pyright: ignore[reportAttributeAccessIssue]
    else:

        def f(x):
            x = pd.DataFrame(x, columns=X_train.columns.tolist())
            for feature, encoder in encoders.items():
                x = one_hot_encoding(x, feature, encoder)
            return clf.predict(x).astype(float)

    cat_features = list(X_train.columns)
    for numeric_feat in numeric_features:
        cat_features.remove(numeric_feat)
    cat_features = [X_train.columns.tolist().index(i) for i in cat_features]

    X_train_np = np.array(X_train)

    logging.info("Creating LIME Tabular Explainer.")
    explainer = lime_tabular.LimeTabularExplainer(
        X_train_np,
        feature_names=X_train.columns,
        class_names=list(encoding_mappings.values()),
        categorical_features=cat_features,
        categorical_names=encoding_mappings,
        kernel_width=3,
        mode="regression"
        if dataset.classifier_type != "classifier"
        else "classification",
    )

    logging.info("Running Submodular Pick to get explanations.")
    sb_pick = submodular_pick.SubmodularPick(
        explainer,
        X_train_np,
        f,
        sample_size=15000,
        num_features=len(X_train.columns),  # We want to consider all features allways
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
    _ = parser.add_argument(
        # Can be adult, usa_house
        "dataset",
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

    if args.dataset == "adult":
        qi = list(df.columns)
        sensitive_attr = "income"
        qi.remove(sensitive_attr)

        numeric_features = ["age", "capital-gain", "capital-loss", "hours-per-week"]
        dataset = Dataset(
            args.dataset,
            numeric_features,
            "./hierarchies/adult/",
            sensitive_attr,
            "classifier",
            qi,
        )
    elif args.dataset == "usa_house":
        qi = list(df.columns)
        sensitive_attr = "Price"
        qi.remove(sensitive_attr)
        numeric_features = list(df.columns)
        numeric_features.remove(sensitive_attr)
        dataset = Dataset(
            args.dataset,
            numeric_features,
            "./hierarchies/usa_house/",
            sensitive_attr,
            "classifier",
            qi,
        )
    else:
        qi = list(df.columns)
        sensitive_attr = "disease"
        qi.remove(sensitive_attr)
        num_i = [0, 5, 6, 8]
        numeric_features = [df.columns[i] for i in num_i]
        dataset = Dataset(
            args.dataset,
            numeric_features,
            "./hierarchies/cervic_cancer/",
            sensitive_attr,
            "classifier",
            qi,
        )

    transform_n, transform_n_max = utils.get_transformation(
        df,
        dataset.quasi_identifiers,
        create_hierachy(dataset.quasi_identifiers, dataset.hierarchy_path),
    )
    logging.info(
        f"Considering dataset that has been generalized {round(transform_n / transform_n_max, 1) * 100}%"
    )

    if args.explainer_type == "shap":
        score, importance = shap_importance(df, dataset)
    elif args.explainer_type == "lime":
        score, importance = lime_importance(df, dataset)
    else:
        logging.error(f"Explainer type '{args.explainer_type}' is not supported.")
        raise NotImplementedError

    logging.info(f"Saving importance vector to {args.data_out}")
    output_dir = os.path.dirname(args.data_out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.save(
        args.data_out,
        [
            ("accuracy", float(score)),
            ("transform_n", float(transform_n)),
            ("transform_n_max", float(transform_n_max)),
        ]
        + importance,
    )
    logging.info("Script finished successfully.")
