from pandas import DataFrame
from anjana.anonymity import k_anonymity
from masking_effects_on_xai_techniques import anonymized_preprocessor as anon_prep
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


def generate_k_anonymity_data(
    df: DataFrame,
    supp_level: int,
    ident,
    quasi_ident,
    hierarchies,
    k_list: None | list[int] = None,
) -> list[DataFrame]:
    if not k_list:
        k_list = [2**n for n in range(1, 9)]
    anon_data = []
    for k in k_list:
        print(f"Anonymizing for k={k}")
        anon_df = k_anonymity(df, ident, quasi_ident, k, supp_level, hierarchies)
        try:
            anon_df.drop("index", axis=1, inplace=True)
        except KeyError:
            pass
        anon_data.append(anon_df)
    print("Done anonymizing")
    return anon_data


def train_models(anon_data: list[DataFrame], target: str, hierarchies_path: str, clf):
    models = []
    scores = []
    for df in anon_data:
        df = anon_prep.encode(df, hierarchy_path=hierarchies_path + "/")

        y = df[target]
        X = df.drop(columns=[target])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=0
        )
        _ = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)
        models.append(clf)
        print(score)
    return models, scores


def plot_k_vs_score(scores, k_list: None | list[int] = None):
    if not k_list:
        k_list = [2**n for n in range(1, 9)]
    df = pd.DataFrame({"k_value": k_list, "Score": scores})

    plt.figure(figsize=(10, 6))

    plt.plot(df["k_value"], df["Score"], marker="o", linestyle="-", color="blue")

    plt.xscale("log", base=2)

    plt.xlabel("k Value")
    plt.ylabel("Score")
    plt.title("Score vs. k Value Progression")

    plt.grid(True, linestyle="--", alpha=0.6)

    plt.xticks(k_list)

    min_score = min(scores) - 0.005
    max_score = max(scores) + 0.005
    plt.ylim(min_score, max_score)
    plt.show()
