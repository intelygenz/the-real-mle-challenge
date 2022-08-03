"""
This file contains all test for preprocess.py
"""

import numpy as np
import pandas as pd
import pytest

from processes.preprocess.preprocess import (
    create_new_column,
    prepare_bathrooms_column,
    preprocess_categorical_column,
    preprocess_mapping_columns,
    preprocess_nan,
    rename_columns,
)


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
    df_test = rename_columns(df=df_test)

    # THEN the columns name has to be like expected
    assert expected_columns == df_test.columns


def test_rename_columns_without_column_name():
    # GIVEN a dataframe without column to rename
    data_raw = [["one"], ["two"], ["three"]]
    df_test = pd.DataFrame(data_raw, columns=["other_column"])
    expected_columns = ["other_column"]

    # WHEN executed rename columns
    df_test = rename_columns(df=df_test)

    # THEN the columns name has to be like expected
    assert expected_columns == df_test.columns


def test_preprocess_nan_values():
    # GIVEN a dataframe without column to rename
    data_raw = [["one", np.NaN], ["two", "a"], ["three", "b"], [np.NaN, "b"]]
    df_test = pd.DataFrame(data_raw, columns=["column_a", "column_b"])
    expected_size = 2
    expected_column_a = ["two", "three"]
    expected_column_b = ["a", "b"]

    # WHEN executed preprocess nan
    df_test = preprocess_nan(df=df_test)

    # THEN the dataframe have new len
    assert expected_size == len(df_test), "Wrong size of dataframe"
    # AND the values in column a are like expected
    assert expected_column_a == list(df_test["column_a"].values), "Error in column a"
    # AND the values in column b are like expected
    assert expected_column_b == list(df_test["column_b"].values), "Error in column b"


def test_preprocess_categorical_column():
    # GIVEN a dataframe with information about price
    data_raw = [["$150.00"], ["$8.00"], ["$500.00"], ["$200.00"], ["$30.00"], ["$80.00"]]
    df_test = pd.DataFrame(data_raw, columns=["price"])
    expected_len = 5
    expected_count = [2, 1, 1, 1]

    # WHEN preprocess the categorical column
    df_test = preprocess_categorical_column(df=df_test)

    # THEN the dataframe have a new len
    assert expected_len == len(df_test), "Wrong size of dataframe"
    # AND must have expected counts
    for i, value in enumerate(df_test["category"].value_counts()):
        assert value == expected_count[i], "Wrong count for category " + str(i)


def test_create_new_column():
    # GIVEN a dataframe with a simple column
    data_raw = [["the one"], ["the search_value two"], ["the three"], ["four search_value"], ["five"]]
    df_test = pd.DataFrame(data_raw, columns=["description"])
    expected_new_column = [0, 1, 0, 1, 0]

    # WHEN execute create_new_column
    df_test = create_new_column(df_test, "description", "search_value")

    # THEN the dataframe have two columns
    assert len(df_test.columns) == 2, "Wrong number of columns"
    assert list(df_test.columns) == ["description", "search_value"], "Wrong name of columns"
    assert list(df_test["search_value"].values) == expected_new_column, "Wrong values in columns"


@pytest.mark.parametrize(
    "neighborhood, room, expected_neighborhood, expected_room",
    [
        ("Shared room", "Queens", 1, 2),
        ("Private room", "Bronx", 2, 1),
        ("Hotel room", "Brooklyn", 4, 4),
        ("Shared room", "Bronx", 1, 1),
        ("Hotel room", "Manhattan", 4, 5),
        ("Entire home/apt", "Staten Island", 3, 3),
    ],
)
def test_preprocess_mapping_columns(neighborhood, room, expected_neighborhood, expected_room):
    # GIVEN  dataframe with columns to map
    data_raw = [[neighborhood, room]]
    df_test = pd.DataFrame(data_raw, columns=["neighbourhood", "room_type"])

    # WHEN executed it
    df_test = preprocess_mapping_columns(df_test)
    print(df_test.head())

    # THEN the results must be like expected
    assert len(df_test) == 1, "Wrong size of dataframe"
    assert df_test.loc[0, "neighbourhood"] == expected_neighborhood, "Wrong category for neighborhood"
    assert df_test.loc[0, "room_type"] == expected_room, "Wrong category for room type"
