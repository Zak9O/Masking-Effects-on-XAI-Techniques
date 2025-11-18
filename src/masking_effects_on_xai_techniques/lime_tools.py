import numpy as np


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
        list(feature_sums.items()), key=lambda item: item[1], reverse=True
    )
    return summed_list_sorted
