import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "model"))
from train import get_csvs_df, split_data


def test_csvs_no_files():
    with pytest.raises(RuntimeError):
        get_csvs_df("non_existent_path")


def test_csvs_no_files_invalid_path():
    with pytest.raises(RuntimeError):
        get_csvs_df("tests/datasets/empty")


def test_csvs_creates_dataframe():
    df = get_csvs_df("tests/datasets")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_split_data_shapes():
    df = get_csvs_df("tests/datasets")
    X_train, X_test, y_train, y_test = split_data(df)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == 8
