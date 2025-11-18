import argparse
import pathlib
import pandas as pd
import numpy as np
import sklearn
import shap
import logging
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from pandas import DataFrame


def normalize_feature(train: DataFrame, test: DataFrame, feature: str) -> None:
    scaler = StandardScaler()
    _ = scaler.fit(train[[feature]])
    train[feature] = scaler.transform(train[[feature]])
    test[feature] = scaler.transform(test[[feature]])


def encode(
    df: DataFrame, hierarchy_path="./hierarchies/", skip_columns=[]
) -> DataFrame:
    encoding_order = {}
    logging.info(f"Starting encoding process. Skipping columns: {skip_columns}")

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
    return _encode_by_order(df, encoding_order)


def _encode_by_order(df: DataFrame, encoding_order: dict[str, list[str]]) -> DataFrame:
    df = df.copy()
    logging.info("Applying ordinal encoding.")
    for column, order in encoding_order.items():
        logging.debug(f"Encoding column '{column}' with order: {order}")
        encoder = OrdinalEncoder(categories=[order], dtype=int)
        df[column] = encoder.fit_transform(df[[column]])
    logging.info("Ordinal encoding applied successfully.")
    return df


def importance_values_to_str(features: list[str], importance) -> list[tuple[str, int]]:
    return [(features[i], importance[i]) for i in np.argsort(-importance)]


def shap_importance(df: pd.DataFrame) -> tuple[float, list[tuple[str, int]]]:
    logging.info("Calculating SHAP importance values.")
    numeric_features = ["age", "capital-gain", "capital-loss", "hours-per-week"]
    skip_columns = []
    for feature in numeric_features:
        # if feature is numeric in df then add to skip_columns
        d_type = type(df[feature].iloc[0])
        if d_type is not str:
            skip_columns.append(feature)

    df = encode(df, skip_columns=skip_columns)

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
    shap_values = explainer(X_test.iloc[:10])

    importance = np.mean(np.abs(shap_values.values), axis=0)
    logging.info("SHAP importance calculation finished.")

    return score, importance_values_to_str(X_train.columns, importance)


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
        logging.warning("LIME explainer is not yet implemented.")
        pass
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
