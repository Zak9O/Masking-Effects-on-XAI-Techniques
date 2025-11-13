import argparse
import pathlib
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from anjana.anonymity import (
    alpha_k_anonymity,
    k_anonymity,
    l_diversity,
    t_closeness,
)
from loguru import logger
from pandas import DataFrame


class Anonymizer(ABC):
    def __init__(
        self,
        data: DataFrame,
        hierarchies_path: str,
        name: str,
    ) -> None:
        self.df = data
        self.hierarchies = self.create_hierarchy(hierarchies_path)
        self.name = name

    def create_hierarchy(self, hierarchies_path) -> dict[str, dict[int, pd.Series]]:
        hierarchies = {}
        for col in self.df.columns:
            hierarchies[col] = dict(
                pd.read_csv(f"{hierarchies_path}/{col}.csv", header=None)
            )
        return hierarchies

    def anonymize(
        self,
        k: int,
        supp_level: int,
        values: list[float],
        quasi_identifiers: list[str],
        identifiers: list[str],
        sensitive_attribute: str,
        save_dir_path: str,
    ):
        logger.info("Begnining anonymization")
        for val in values:
            logger.info(f"Anonymizing with {self.name}={round(val, 1)}")
            df = self._anonymize_df(
                k, supp_level, val, identifiers, quasi_identifiers, sensitive_attribute
            )
            if df.empty:
                logger.info(f"{self.name}={round(val, 1)} produced an empty dataframe")
                continue

            df.to_csv(f"{save_dir_path}/{round(val, 2)}.csv", index=False)

    @abstractmethod
    def _anonymize_df(
        self,
        k: int,
        supp_level: int,
        val: float,
        identifiers: list[str],
        quasi_identifiers: list[str],
        sensitive_attribute: str,
    ):
        raise NotImplementedError


class TCloseness(Anonymizer):
    def __init__(
        self,
        data: DataFrame,
        hierarchies_path: str,
        save_dir_path: str,
    ) -> None:
        super().__init__(
            data,
            hierarchies_path,
            save_dir_path,
        )

    def _anonymize_df(
        self,
        k: int,
        supp_level: int,
        val: float,
        identifiers: list[str],
        quasi_identifiers: list[str],
        sensitive_attribute: str,
    ):
        return t_closeness(
            self.df,
            identifiers,
            quasi_identifiers,
            sensitive_attribute,
            k,
            val,
            supp_level,
            self.hierarchies,
        )


class AlphaKAnonymity(Anonymizer):
    def __init__(
        self,
        data: DataFrame,
        hierarchies_path: str,
        save_dir_path: str,
    ) -> None:
        super().__init__(
            data,
            hierarchies_path,
            save_dir_path,
        )

    def _anonymize_df(
        self,
        k: int,
        supp_level: int,
        val: float,
        identifiers: list[str],
        quasi_identifiers: list[str],
        sensitive_attribute: str,
    ):
        return alpha_k_anonymity(
            self.df,
            identifiers,
            quasi_identifiers,
            sensitive_attribute,
            k,
            val,
            supp_level,
            self.hierarchies,
        )


class LDiversity(Anonymizer):
    def __init__(
        self,
        data: DataFrame,
        hierarchies_path: str,
        save_dir_path: str,
    ) -> None:
        super().__init__(
            data,
            hierarchies_path,
            save_dir_path,
        )

    def _anonymize_df(
        self,
        k: int,
        supp_level: int,
        val: float,
        identifiers: list[str],
        quasi_identifiers: list[str],
        sensitive_attribute: str,
    ):
        if len(self.df[sensitive_attribute].unique()) < int(val):
            return pd.DataFrame()
        return l_diversity(
            self.df,
            identifiers,
            quasi_identifiers,
            sensitive_attribute,
            k,
            int(val),
            supp_level,
            self.hierarchies,
        )


class KAnonymity(Anonymizer):
    def __init__(
        self,
        data: DataFrame,
        hierarchies_path: str,
        save_dir_path: str,
    ) -> None:
        super().__init__(
            data,
            hierarchies_path,
            save_dir_path,
        )

    def _anonymize_df(
        self,
        k: int,
        supp_level: int,
        val: float,
        identifiers: list[str],
        quasi_identifiers: list[str],
        sensitive_attribute: str,
    ):
        return k_anonymity(
            self.df,
            identifiers,
            quasi_identifiers,
            int(val),
            supp_level,
            self.hierarchies,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anonymize a dataframe")
    _ = parser.add_argument(
        "data_path",
        type=pathlib.Path,
    )
    _ = parser.add_argument(
        "hierarchies_path",
        type=pathlib.Path,
    )
    _ = parser.add_argument(
        "save_dir_path",
        type=pathlib.Path,
    )
    _ = parser.add_argument(
        "anonymization_method",
        type=str,
    )

    args = parser.parse_args()

    df = pd.read_csv(args.data_path)

    identifiers = ["race"]
    sensitive_attribute = "income"

    quasi_identifiers = df.columns.to_list()
    for identifier in identifiers:
        quasi_identifiers.remove(identifier)
    quasi_identifiers.remove(sensitive_attribute)

    method = args.anonymization_method
    if method == "t_closeness":
        anonymizer = TCloseness(df, args.hierarchies_path, args.save_dir_path)
        values = list(np.linspace(0.1, 1.0, 10))
    elif method == "alpha_k_anonymity":
        anonymizer = AlphaKAnonymity(df, args.hierarchies_path, args.save_dir_path)
        values = list(np.linspace(0.1, 1.0, 10))
    elif method == "l_diversity":
        anonymizer = LDiversity(df, args.hierarchies_path, args.save_dir_path)
        values = list(np.linspace(1, 10, 10))
    elif method == "k_anonymity":
        anonymizer = KAnonymity(df, args.hierarchies_path, args.save_dir_path)
        values = [2**i for i in range(1, 9)]
    else:
        raise ValueError(f"Unknown anonymization method: {method}")

    anonymizer.anonymize(
        16,
        20,
        values,
        quasi_identifiers,
        identifiers,
        sensitive_attribute,
        args.save_dir_path,
    )
