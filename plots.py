import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    return pd, sns


@app.cell
def _():
    from ucimlrepo import fetch_ucirepo 
  
    # fetch dataset 
    adult = fetch_ucirepo(id=2) 
  
    # data (as pandas dataframes) 
    X = adult.data.features 
    y = adult.data.targets 
  
    # metadata 
    print(adult.metadata) 
  
    # variable information 
    print(adult.variables) 
    print(y)
    return X, y


@app.cell
def _(y):
    y_clean = y['income'].str.rstrip('.')
    y_clean.nunique()
    return (y_clean,)


@app.cell
def _(X, pd, sns, y_clean):
    df = pd.concat([X, y_clean], axis=1)
    sns.pairplot(data=df, hue='income')
    return


if __name__ == "__main__":
    app.run()
