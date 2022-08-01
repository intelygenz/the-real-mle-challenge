"""
This file contains all functions for preprocessing the dataset
"""
import logging

import numpy as np
import pandas as pd

from processes.preprocess.config import DataPreprocessColumns, DataRawColumns

logger = logging.getLogger(__name__)


def prepare_bathrooms_column(text: str) -> float:
    """
    Extract number of bathtrooms from text

    Args:
        text (str): _description_

    Returns:
        float: _description_
    """
    try:
        return float(text.split(" ")[0]) if isinstance(text, str) else np.NaN
    except ValueError:
        return np.NaN


def rename_columns(df: pd.DataFrame) -> None:
    """
    Rename the column in the dataframe

    Args:
        df (pd.DataFrame): preprocess dataframe
    """

    df.rename(columns={DataRawColumns.NEIGHBOURHOOD_GROUP_CLEANSED: DataPreprocessColumns.NEIGHBOURHOOD}, inplace=True)


def preprocess_nan(df: pd.DataFrame) -> None:
    """
    This function deal with nan values in the dataframe

    Args:
        df (pd.DataFrame): preprocess dataframe
    """
    df.dropna(axis=0, inplace=True)


def preprocess_low_prices():
    pass


def create_categorical_column():
    pass


def create_new_columns():
    pass


def preprocess(df: pd.DataFrame, preprocess_path: str) -> None:
    """
    Preprocess dataframe and save the info in a new dataframe

    Args:
        df (pd.DataFrame): dataframe to preprocess
        preprocess_path (str): path to save the solution
    """
    # Create a copy of df
    df_preprocess = df.copy()

    # Create bathrooms column from bathrooms text
    df_preprocess[DataRawColumns.BATHROOMS] = df_preprocess[DataRawColumns.BATHROOMS_TEXT].apply(
        prepare_bathrooms_column
    )

    # Get columns of interest
    df_preprocess = df_preprocess[DataRawColumns.SUBSET_TRAINING]

    # Rename columns
    rename_columns(df_preprocess)
    print(len(df_preprocess))

    # Deal with nan values
    preprocess_nan(df_preprocess)
    print(len(df_preprocess))
