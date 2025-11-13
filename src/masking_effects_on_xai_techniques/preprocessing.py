from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


class AdultPrep:
    _labels = ["<=50K", ">50K"]
    _num_cols = ["age", "capital-gain", "capital-loss", "hours-per-week"]
    _nominal_cols = [
        "workclass",
        "marital-status",
        "sex",
        "race",
        "relationship",
        "native-country",
        "occupation",
    ]
    _ordinal_col = ["education", "income"]

    def __init__(
        self,
        df: pd.DataFrame,
        is_anonymized=False,
    ) -> None:
        self.num_cols = self._num_cols.copy()
        self.nominal_cols = self._nominal_cols.copy()
        self.ordinal_col = self._ordinal_col.copy()
        self.labels = self._labels.copy()
        self.is_anonymized = is_anonymized
        self.create_education_order(df["education"].unique())
        self.ordinal_encoding_levels = [self.education_order, self.labels]

        self.consider_age(df["age"][0])

        self.pipeline = Pipeline(
            steps=[
                ("clean_income", self.IncomeCleaner()),
                ("drop_rows_and_cols", self.RowDropper(self.is_anonymized)),
                ("ordinal_encoding", self.create_ordinal_encoder()),
            ]
        )
        if self.is_anonymized:
            self.index = df["index"]

        prep_data = self.pipeline.fit_transform(df)
        ppd = pd.DataFrame(
            prep_data, columns=self.pipeline["ordinal_encoding"].get_feature_names_out()
        )

        self.y = ppd["income"]
        self.X = ppd.drop(columns=["income"])
        self.feature_names = self.X.columns.to_list()
        self.one_hot_encoder = self.create_encoder()
        self.one_hot_encoder.fit(np.array(self.X))
        self.encoder_mappings = self.create_encoder_mappings()

    def consider_age(self, age_val):
        hiearchy = [
            [
                "[15, 20[",
                "[20, 25[",
                "[25, 30[",
                "[30, 35[",
                "[35, 40[",
                "[40, 45[",
                "[45, 50[",
                "[50, 55[",
                "[55, 60[",
                "[60, 65[",
                "[65, 70[",
                "[70, 75[",
                "[75, 80[",
                ">=80",
            ],
            [
                "[10, 20[",
                "[20, 30[",
                "[30, 40[",
                "[40, 50[",
                "[50, 60[",
                "[60, 70[",
                "[70, 80[",
                ">=80",
            ],
            ["[0, 20[", "[20, 40[", "[40, 60[", "[60, 80[", ">=80"],
            ["[0, 40[", "[40, 80[", ">=80"],
            ["[0, 80[", ">=80"],
            ["*"],
        ]
        for level in hiearchy:
            if age_val in level:
                self.num_cols.remove("age")
                self.ordinal_col.append("age")
                self.ordinal_encoding_levels.append(level)
                return

    def create_education_order(self, education):
        if "Undergraduate" in education:
            order = [
                "Primary School",
                "High School",
                "Undergraduate",
                "Professional Education",
                "Graduate",
            ]
        elif "Primary education" in education:
            order = [
                "Primary education",
                "Secondary education",
                "Higher education",
            ]
        elif "*" in education:
            order = ["*"]
        else:
            order = [
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
        self.education_order = order

    def categories_i(self) -> list[int]:
        output = []
        for category in self.nominal_cols:
            output.append(self.feature_names.index(category))
        output.append(self.feature_names.index("education"))
        return output

    def create_encoder_mappings(self) -> dict[int, np.ndarray]:
        nominal_mappings: list[np.ndarray] = (
            self.pipeline["ordinal_encoding"].named_transformers_["nominal"].categories_
        )
        encoder_mappings = {}
        for feature_name, categories in zip(self.nominal_cols, nominal_mappings):
            encoder_mappings[self.feature_names.index(feature_name)] = categories
        encoder_mappings[self.feature_names.index("education")] = np.array(
            self.education_order
        )
        return encoder_mappings

    def create_encoder(self) -> ColumnTransformer:
        one_hot_indices = [self.feature_names.index(cat) for cat in self.nominal_cols]
        num_indices = [self.feature_names.index(col) for col in self.num_cols]
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_indices),
                ("ohe", OneHotEncoder(sparse_output=False), one_hot_indices),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

    class IncomeCleaner(BaseEstimator, TransformerMixin):
        """Cleans the 'income' column by stripping a trailing period."""

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X["income"] = X["income"].str.rstrip(".")
            return X

    class RowDropper(BaseEstimator, TransformerMixin):
        """Removes specified columns and rows containing null values."""

        def __init__(self, is_anonymized):
            columns_to_drop = []
            if is_anonymized:
                columns_to_drop.append("index")

            self.columns_to_drop = columns_to_drop

        def fit(self, X, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            X_copy = X.copy()

            X_copy = X_copy.drop(columns=self.columns_to_drop)

            X_copy = X_copy.dropna()
            cols_dropped = len(X) - len(X_copy)
            if cols_dropped > 0:
                print(f"Dropped {cols_dropped} rows")

            return X_copy

    def create_ordinal_encoder(self):
        return ColumnTransformer(
            transformers=[
                ("nominal", OrdinalEncoder(dtype=np.float64), self.nominal_cols),
                (
                    "ordinal",
                    OrdinalEncoder(
                        categories=self.ordinal_encoding_levels, dtype=np.float64
                    ),
                    self.ordinal_col,
                ),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
