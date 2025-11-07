import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn import preprocessing
    from sklearn import tree
    from sklearn.model_selection import cross_val_score

    return KaggleDatasetAdapter, kagglehub, pd, plt, preprocessing, sns, tree


@app.cell
def _(sns):
    sns.set_theme()
    return


@app.cell
def _(KaggleDatasetAdapter, kagglehub):
    path = kagglehub.dataset_download(
        "pablomgomez21/drugs-a-b-c-x-y-for-decision-trees"
    )
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "pablomgomez21/drugs-a-b-c-x-y-for-decision-trees",
        "drug200.csv",
    )
    df
    return (df,)


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df, plt):
    df["Drug"].value_counts().plot(kind="bar")
    plt.title("Distribution of Drug Classes")
    plt.xlabel("Drug")
    plt.ylabel("Count")
    plt.show()
    return


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(df, plt, sns):
    categorical_features = ["Sex", "BP", "Cholesterol"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, feature in enumerate(categorical_features):
        sns.countplot(x="Drug", hue=feature, data=df, ax=axes[i])
        axes[i].set_title(f"{feature} vs Drug")
        axes[i].set_xlabel("Drug")
        axes[i].set_ylabel("Count")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, sns):
    sns.scatterplot(data=df, x="Age", y="Na_to_K", hue="Drug")
    return


@app.cell
def _(df, pd, preprocessing):
    # Transforming data to remove categorial data
    data = pd.get_dummies(df, columns=["Sex", "BP", "Cholesterol"])

    le = preprocessing.LabelEncoder()
    le.fit(df["Drug"].unique().tolist())
    data["Drug"] = le.transform(data["Drug"])
    return (data,)


@app.cell
def _(data):
    X = data.drop("Drug", axis=1)
    y = data["Drug"]
    sep = 150
    X_train, y_train = X[0:sep], y[0:sep]
    X_test, y_test = X[sep:], y[sep:]
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_test, X_train, tree, y_test, y_train):
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    return


app._unparsable_cell(
    r"""
    clf2 = tree.DecisionTreeClassifier()
    scores = cross_val_score(clf2, X, y, cv=10)
    print(scores.mean())
    print(scores.std()
    """,
    name="_",
)


if __name__ == "__main__":
    app.run()
