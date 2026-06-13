import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[0] = str(Path(__file__).parent.parent.parent.parent.resolve())

from third_party.data_science.utils import get_df_stats


def test_get_df_stats() -> int:
    df = pd.DataFrame(np.random.rand(1100, 10), columns=[list("abcdefghij")])
    columns_subset = ["f", "i", "a"]
    summary = get_df_stats(df=df, columns_subset=columns_subset)
    print(f"Using: {columns_subset}\n", summary, end="\n\n")
    summary = get_df_stats(df=df, columns_subset=None)
    print("Using None:\n", summary, end="\n\n")
    return 0


if __name__ == '__main__':
    test_get_df_stats()
