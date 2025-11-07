import pandas as pd
from pathlib import Path


def common_elements(dfs: list[list[int]]) -> list[int]:
    common_indices = None
    for df in dfs:
        current_indices = set(df)

        if common_indices is None:
            common_indices = current_indices
            continue

        common_indices = common_indices.intersection(current_indices)

        if not common_indices:
            break
    return list(common_indices)  # pyright: ignore[reportArgumentType]


def common_indices_of_df(dfs) -> list[int]:
    if not dfs:
        return []

    indices = []
    for file_path in dfs:
        try:
            df = pd.read_csv(file_path, usecols=["index"])
            indices.append(df["index"].values)
        except Exception as e:
            print(f"Error loading and processing file {file_path}: {e}")

    return common_elements(indices)


def get_absolute_file_paths(folder_path: str) -> list[str]:
    folder = Path(folder_path)
    return [str(item.resolve()) for item in folder.iterdir() if item.is_file()]
