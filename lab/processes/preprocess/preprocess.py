"""
This file contains all functions for preprocessing the dataset
"""
import logging

import numpy as np
import pandas as pd

from processes.config import ConfigPreprocess, DataPreprocessColumns, DataRawColumns

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


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename the column in the dataframe

    Args:
        df (pd.DataFrame): preprocess dataframe

    Returns:
        pd.DataFrame: the dataframe updated
    """

    return df.rename(columns={DataRawColumns.NEIGHBOURHOOD_GROUP_CLEANSED: DataPreprocessColumns.NEIGHBOURHOOD})


def preprocess_nan(df: pd.DataFrame) -> None:
    """
    This function deal with nan values in the dataframe

    Args:
        df (pd.DataFrame): preprocess dataframe

    Returns:
        pd.DataFrame: the dataframe updated
    """
    return df.dropna(axis=0)


def preprocess_categorical_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the categorical column

    Args:
        df (pd.DataFrame): preprocess dataframe

    Returns:
        pd.DataFrame: the dataframe updated
    """
    # Convert price to value
    df[DataPreprocessColumns.PRICE] = df[DataPreprocessColumns.PRICE].str.extract(r"(\d+).")
    df[DataPreprocessColumns.PRICE] = df[DataPreprocessColumns.PRICE].astype(int)

    # Remove values below configured value
    df = df[df[DataPreprocessColumns.PRICE] >= ConfigPreprocess.MIN_PRICE].copy()

    # Categorize values
    df[DataPreprocessColumns.CATEGORY] = pd.cut(
        df[DataPreprocessColumns.PRICE], bins=ConfigPreprocess.BINS_PRICE, labels=ConfigPreprocess.LABELS_PRICE
    )

    return df


def create_new_column(df: pd.DataFrame, column_search: str, new_column_name: str) -> pd.DataFrame:
    """
    Create a new column if the text contains a specific text

    Args:
        df (pd.DataFrame): dataframe for search and create new column
        column_search (str): column where search the text
        new_column_name (str): new column name and text to search in original column

    Returns:
        pd.DataFrame: the dataframe updated
    """
    df[new_column_name] = df[column_search].str.contains(new_column_name)
    df[new_column_name] = df[new_column_name].astype(int)
    return df


def preprocess_amenities_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new columns in from amenities column

    Args:
        df (pd.DataFrame): preprocess dataframe

    Returns:
        pd.DataFrame: the dataframe updated
    """
    columns_to_add = [
        DataPreprocessColumns.TV,
        DataPreprocessColumns.INTERNET,
        DataPreprocessColumns.AIR_CONDITIONING,
        DataPreprocessColumns.KITCHEN,
        DataPreprocessColumns.HEATING,
        DataPreprocessColumns.WIFI,
        DataPreprocessColumns.ELEVATOR,
        DataPreprocessColumns.BREAKFAST,
    ]

    for new_column in columns_to_add:
        df = create_new_column(df=df, column_search=DataRawColumns.AMENITIES, new_column_name=new_column)

    return df.drop(DataRawColumns.AMENITIES, axis=1)


def preprocess_mapping_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert in categorical with map some columns

    Args:
        df (pd.DataFrame): dataframe to transform

    Returns:
        pd.DataFrame: dataframe updated
    """
    for column, mapping in ConfigPreprocess.MAPING_COLUMNS.items():
        df[column] = df[column].map(mapping)

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the original dataframe

    Args:
        df (pd.DataFrame): dataframe to preprocess

    Returns:
        pd.DataFrame: the dataframe updated
    """

    # Create a copy of df
    df_preprocess = df.copy()

    # Create bathrooms column from bathrooms text
    df_preprocess[DataRawColumns.BATHROOMS] = df_preprocess[DataRawColumns.BATHROOMS_TEXT].map(
        lambda text: prepare_bathrooms_column(text=text)
    )

    # Get columns of interest
    df_preprocess = df_preprocess[DataRawColumns.SUBSET_TRAINING]

    # Deal with nan values
    df_preprocess = preprocess_nan(df_preprocess)

    # Rename columns
    df_preprocess = rename_columns(df_preprocess)

    # Prepare categorical column
    df_preprocess = preprocess_categorical_column(df_preprocess)

    # Prepare new columns
    df_preprocess = preprocess_amenities_column(df_preprocess)

    # Prepare mapping columns
    df_preprocess = preprocess_mapping_columns(df_preprocess)

    return df_preprocess
