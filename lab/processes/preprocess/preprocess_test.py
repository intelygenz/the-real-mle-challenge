"""
This file contains all test for preprocess.py
"""

import numpy as np
import pandas as pd
import pytest

from processes.preprocess.preprocess import prepare_bathrooms_column, rename_columns


@pytest.mark.parametrize(
    "text_input, expected",
    [("3 bathrooms", 3), ("bathrooms", np.NaN), ("no bathrooms", np.NaN), ("", np.NaN), (np.NaN, np.NaN)],
)
def test_prepare_bathrooms_column(text_input, expected):
    # GIVEN an input text and result expected
    # WHEN executed
    result = prepare_bathrooms_column(text=text_input)

    # THEN the result has been like expected
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == expected


def test_rename_columns_with_column_name():
    # GIVEN a dataframe with column to rename
    data_raw = [["one"], ["two"], ["three"]]
    df_test = pd.DataFrame(data_raw, columns=["neighbourhood_group_cleansed"])
    expected_columns = ["neighbourhood"]

    # WHEN executed rename columns
    rename_columns(df=df_test)

    # THEN the columns name has to be like expected
    assert expected_columns == df_test.columns


def test_rename_columns_without_column_name():
    # GIVEN a dataframe without column to rename
    data_raw = [["one"], ["two"], ["three"]]
    df_test = pd.DataFrame(data_raw, columns=["other_column"])
    expected_columns = ["other_column"]

    # WHEN executed rename columns
    rename_columns(df=df_test)

    # THEN the columns name has to be like expected
    assert expected_columns == df_test.columns
