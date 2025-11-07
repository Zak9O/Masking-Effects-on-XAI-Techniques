import marimo

__generated_with = "0.17.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    # from sklearn import set_config
    # from sklearn.base import BaseEstimator, TransformerMixin
    # from sklearn.pipeline import Pipeline
    # from sklearn.compose import ColumnTransformer
    # from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    import lime
    import lime.lime_tabular
    from preprocessing import AdultPrep

    return AdultPrep, MLPClassifier, lime, mo, np, pd, train_test_split


@app.cell
def _(mo):
    mo.md(r"""# Model training""")
    return


@app.cell
def _(AdultPrep, pd):
    data = pd.read_csv("./data/data.csv")
    adult_prep = AdultPrep(data)
    adult_prep
    return (adult_prep,)


@app.cell
def _(adult_prep, np, train_test_split):
    X_train, X_test, y_train, y_test = train_test_split(
        adult_prep.X, adult_prep.y, test_size=0.4, random_state=0
    )
    X_train_encoded = adult_prep.one_hot_encoder.transform(np.array(X_train))
    return X_test, X_train, X_train_encoded, y_test, y_train


@app.cell
def _(MLPClassifier, X_test, X_train_encoded, adult_prep, y_test, y_train):
    clf = MLPClassifier(
        solver="sgd", alpha=1e-5, hidden_layer_sizes=(10), random_state=1
    )
    clf.fit(X_train_encoded, y_train)
    clf.score(adult_prep.one_hot_encoder.transform(X_test), y_test)
    return (clf,)


@app.cell
def _(X_train_np, adult_prep, clf):
    predict_fn = lambda x: clf.predict_proba(
        adult_prep.one_hot_encoder.transform(x)
    ).astype(float)
    # X_train.iloc[0,:].columns
    predict_fn(X_train_np[[4], :])
    return (predict_fn,)


@app.cell
def _(X_train, np):
    X_train_np = np.array(X_train)
    return (X_train_np,)


@app.cell
def _(X_train_np, adult_prep, lime, np):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        np.array(X_train_np),
        feature_names=adult_prep.feature_names,
        class_names=adult_prep.labels,
        categorical_features=adult_prep.categories_i(),
        categorical_names=adult_prep.encoder_mappings,
        kernel_width=3,
    )
    return (explainer,)


@app.cell
def _(X_train_np, explainer, mo, np, predict_fn):
    np.random.seed(1)
    i_2 = 4
    exp_2 = explainer.explain_instance(X_train_np[i_2], predict_fn, num_features=5)
    html_out = exp_2.as_html(show_all=False)
    mo.iframe(
        html=exp_2.as_html(show_all=False),
        height="200px",  # Adjust to fit the LIME output
        width="100%",
    )
    return (exp_2,)


@app.cell
def _(exp_2):
    exp_2.as_list()
    return


if __name__ == "__main__":
    app.run()
