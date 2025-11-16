import argparse
import pathlib
import pandas as pd
import numpy as np
import masking_effects_on_xai_techniques.anonymized_preprocessor as anonymized_preprocessor  # TODO: remove this
import sklearn
import shap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate SHAP values for models")
    _ = parser.add_argument(
        "data_path",
        type=pathlib.Path,
    )
    _ = parser.add_argument(
        "data_out",
        type=pathlib.Path,
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    df = anonymized_preprocessor.encode(df)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        df.drop(["income"], axis=1), df["income"], test_size=0.4, random_state=0
    )

    clf = sklearn.neural_network.MLPClassifier(
        solver="sgd", alpha=1e-5, hidden_layer_sizes=(10), random_state=1
    )
    clf.fit(np.array(X_train), y_train)
    clf.score(np.array(X_train), y_train)

    def f(x):
        return clf.predict_proba(x)[:, 1]

    med = X_train.median().values.reshape((1, X_train.shape[1]))

    explainer = shap.Explainer(f, med)
    shap_values = explainer(X_test.iloc[:1000, :])

    np.save(args.data_out, shap_values)
