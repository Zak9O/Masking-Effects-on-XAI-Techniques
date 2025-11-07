import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    import lime

    return (
        BaseEstimator,
        ColumnTransformer,
        MLPClassifier,
        OneHotEncoder,
        OrdinalEncoder,
        Pipeline,
        StandardScaler,
        TransformerMixin,
        pd,
        train_test_split,
    )


@app.cell
def _(BaseEstimator, TransformerMixin):
    class IncomeCleaner(BaseEstimator, TransformerMixin):
        """Cleans the 'income' column by stripping a trailing period."""

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X["income"] = X["income"].str.rstrip(".")
            return X

    return (IncomeCleaner,)


@app.cell
def _(BaseEstimator, TransformerMixin):
    class RowDropper(BaseEstimator, TransformerMixin):
        """Removes specified columns and rows containing null values."""

        def __init__(self, columns_to_drop=["education-num", "fnlwgt"]):
            self.columns_to_drop = columns_to_drop

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X_copy = X.copy()

            X_copy = X_copy.drop(columns=self.columns_to_drop)

            X_copy = X_copy.dropna()

            return X_copy

    return (RowDropper,)


@app.cell
def _(ColumnTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler):
    education_order = [
        "Preschool",
        "1st-4th",
        "5th-6th",
        "7th-8th",
        "9th",
        "10th",
        "11th",
        "12th",
        "HS-grad",
        "Some-college",
        "Assoc-voc",
        "Assoc-acdm",
        "Bachelors",
        "Masters",
        "Prof-school",
        "Doctorate",
    ]

    NUM_COLS = ["age", "capital-gain", "capital-loss", "hours-per-week"]
    CAT_COLS = [
        "workclass",
        "marital-status",
        "sex",
        "race",
        "relationship",
        "native-country",
        "occupation",
    ]
    ORDINAL_COL = ["education"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            (
                "ordinal",
                OrdinalEncoder(categories=[education_order], dtype="int"),
                ORDINAL_COL,
            ),
            ("cat", OneHotEncoder(dtype="int", handle_unknown="ignore"), CAT_COLS),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
        sparse_threshold=0,
    )
    return (preprocessor,)


@app.cell
def _(IncomeCleaner, Pipeline, RowDropper, preprocessor):
    full_pipeline = Pipeline(
        steps=[
            ("clean_income", IncomeCleaner()),
            ("drop_rows_and_cols", RowDropper()),
            ("preprocessor", preprocessor),
        ]
    )
    return (full_pipeline,)


@app.cell
def _(full_pipeline, pd):
    data = pd.read_csv("./data/data.csv")
    ppd_array = full_pipeline.fit_transform(data)

    ppd = pd.DataFrame(
        ppd_array, columns=full_pipeline["preprocessor"].get_feature_names_out()
    )
    y = ppd["income"]
    X = ppd.drop(columns=["income"])
    return X, y


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(MLPClassifier, X_test, X_train, y_test, y_train):
    clf = MLPClassifier(
        solver="sgd", alpha=1e-5, hidden_layer_sizes=(10), random_state=1
    )
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    return


if __name__ == "__main__":
    app.run()
