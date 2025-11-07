import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import marimo as mo
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    return (
        ColumnTransformer,
        MLPClassifier,
        OneHotEncoder,
        OrdinalEncoder,
        StandardScaler,
        mo,
        pd,
        train_test_split,
    )


@app.cell
def _(pd):
    data = pd.read_csv("./data/data.csv")
    len_data = len(data)
    data
    return data, len_data


@app.cell
def _(mo):
    mo.md(r"""# Preprocessing Data""")
    return


@app.cell
def _(data, len_data):
    # Cleaning the data
    data["income"] = data["income"].str.rstrip(".")

    # Remove all rows containing null values
    data.dropna(inplace=True)

    # Drop rows
    data.drop(columns=["education-num", "fnlwgt"], inplace=True)

    len_clean_data = len(data)
    print(f"Dropped {len_data - len_clean_data} rows")
    return


@app.cell
def _(mo):
    mo.md(r"""## Standarddize data""")
    return


@app.cell
def _(data):
    cols_need_standard = ["age", "capital-gain", "capital-loss", "hours-per-week"]
    data[cols_need_standard] = data[cols_need_standard].astype("float64")
    tmp_data = data[cols_need_standard]
    return cols_need_standard, tmp_data


@app.cell
def _(StandardScaler, cols_need_standard, pd, tmp_data):
    scaler = StandardScaler().fit(tmp_data)
    cols_scale = scaler.transform(tmp_data)
    cols_scale = pd.DataFrame(
        data=cols_scale, columns=cols_need_standard, index=tmp_data.index
    )
    return (cols_scale,)


@app.cell
def _(cols_scale, data):
    data.update(cols_scale)
    return


@app.cell
def _(mo):
    mo.md(r"""## Encode categories""")
    return


@app.cell
def _(ColumnTransformer, OneHotEncoder, OrdinalEncoder, data):
    education_order = [
        [
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
    ]

    column_trans = ColumnTransformer(
        [
            ("workclass", OneHotEncoder(dtype="int"), ["workclass"]),
            (
                "education",
                OrdinalEncoder(dtype="int", categories=education_order),
                ["education"],
            ),
            ("marital-status", OneHotEncoder(dtype="int"), ["marital-status"]),
            ("sex", OneHotEncoder(dtype="int"), ["sex"]),
            ("race", OneHotEncoder(dtype="int"), ["race"]),
            ("relationship", OneHotEncoder(dtype="int"), ["relationship"]),
            ("native-country", OneHotEncoder(dtype="int"), ["native-country"]),
            ("occupation", OneHotEncoder(dtype="int"), ["occupation"]),
        ],
        sparse_threshold=0,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    _ = column_trans.fit(data)
    return (column_trans,)


@app.cell
def _(column_trans, data, pd):
    ppd = pd.DataFrame(
        data=column_trans.transform(data), columns=column_trans.get_feature_names_out()
    )
    return (ppd,)


@app.cell
def _(mo):
    mo.md(r"""# Training the Model""")
    return


@app.cell
def _(ppd):
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
