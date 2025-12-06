class Dataset:
    def __init__(
        self,
        name: str,
        numeric_features: list[str],
        hierarchy_path: str,
        sensitive_attr: str,
        classifier_type: str,
        quasi_identifiers: list[str],
    ):
        self.name = name
        self.numeric_features = numeric_features
        self.hierarchy_path = hierarchy_path
        self.sensitive_attr = sensitive_attr
        self.classifier_type = classifier_type
        self.quasi_identifiers = quasi_identifiers
