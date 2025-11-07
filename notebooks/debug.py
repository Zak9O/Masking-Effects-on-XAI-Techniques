import pandas as pd
from preprocessing import AdultPrep
import utils

paths = [
    path
    for path in utils.get_absolute_file_paths("./data/anonymized_data/t-closeness")
    if not path.endswith(".txt")
]
common_indices = utils.get_common_indices(paths)
data = []
for path in paths:
    data.append(AdultPrep(pd.read_csv(path).iloc[:, 1:], is_anonymized=True))
