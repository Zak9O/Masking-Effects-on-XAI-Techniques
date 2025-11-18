import preprocessing
import pandas as pd
import numpy as np

from lime import submodular_pick  # pyright: ignore[reportMissingTypeStubs]
import lime.lime_tabular  # pyright: ignore[reportMissingTypeStubs]

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# path = "./clean.csv"
path = "/zhome/05/f/187481/hpc/clean.csv"
adult_prep = preprocessing.AdultPrep(pd.read_csv(path), is_anonymized=False)

# Training the model
X_train, X_test, y_train, y_test = train_test_split(
    adult_prep.X, adult_prep.y, test_size=0.4, random_state=0
)
clf = MLPClassifier(solver="sgd", alpha=1e-5, hidden_layer_sizes=(10), random_state=1)
_ = clf.fit(adult_prep.one_hot_encoder.transform(np.array(X_train)), y_train)
print(
    clf.score(adult_prep.one_hot_encoder.transform(np.array(X_test)), np.array(y_test))
)

explainer = lime.lime_tabular.LimeTabularExplainer(
    np.array(X_train),
    feature_names=adult_prep.feature_names,
    class_names=adult_prep.labels,
    categorical_features=adult_prep.categories_i(),
    categorical_names=adult_prep.encoder_mappings,
    kernel_width=3,
)


def predict_fn(x):
    return clf.predict_proba(adult_prep.one_hot_encoder.transform(x)).astype(float)


sp_exp = submodular_pick.SubmodularPick(
    explainer,
    np.array(X_train),
    predict_fn,
    num_features=5,
    num_exps_desired=1000,  # 5, #np.floor(len(X_train) / 100),  # pyright: ignore[reportAny]
    sample_size=10000,
    # method='full'
)


def clean_feature_names(features):
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


def name_weights(relation, features):
    names = clean_feature_names(features)
    return [(names[i], weight) for i, weight in relation]


def importance_vector_of(exp):
    relations = list(exp.as_map().values())[0]
    features = exp.domain_mapper.feature_names
    relations = name_weights(relations, features)
    return [(_, np.abs(w)) for _, w in relations]


sp_exp = [exp for exp in sp_exp.sp_explanations]

importance_vectors = [importance_vector_of(exp) for exp in sp_exp]

feature_sums = {}
for explanation in importance_vectors:
    for feature_name, importance_score in explanation:  # pyright: ignore[reportAny]
        feature_sums[feature_name] = (
            feature_sums.get(feature_name, 0.0) + importance_score
        )

summed_list_sorted = sorted(
    list(feature_sums.items()),
    key=lambda item: item[1],
    reverse=True,  # pyright: ignore[reportUnknownLambdaType]
)
print(summed_list_sorted)

with open("/zhome/05/f/187481/hpc/out.txt", "w") as f:
    for item in summed_list_sorted:
        _ = f.write(f"{item}\n")
