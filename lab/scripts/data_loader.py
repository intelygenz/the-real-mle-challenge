"""
This module provides functions for loading raw and preprocessed data used in the ML pipeline.
It includes functions to load the raw dataset directly from a CSV file and to load the preprocessed
data with the necessary feature transformations for modeling.

Functions:
    - load_raw_data(filepath: str) -> pd.DataFrame:
        Loads the raw data from a CSV file into a pandas DataFrame.

    - load_preprocessed_data(filepath: str) -> (pd.DataFrame, pd.Series):
        Loads the preprocessed data from a CSV file, applies categorical mappings, and returns
        the feature matrix (X) and target vector (y) for training and evaluation.
"""

import pandas as pd
from pathlib import Path

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads the raw data from a CSV file into a pandas DataFrame.

    Parameters:
    - filepath (str): The file path of the raw CSV file to load.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the raw data.
    """
    df = pd.read_csv(filepath)
    return df

def load_preprocessed_data(filepath: str) -> pd.DataFrame:
    """
    Loads the preprocessed data from a CSV file, applies categorical mappings to the necessary columns, 
    and returns the feature matrix (X) and target vector (y) ready for training and evaluation.

    Parameters:
    - filepath (str): The file path of the preprocessed CSV file to load.

    Returns:
    - (pd.DataFrame, pd.Series): 
        A tuple containing:
        - X (pd.DataFrame): The feature matrix with selected columns and categorical mappings applied.
        - y (pd.Series): The target vector representing the price category.
    """
    df = pd.read_csv(filepath)

    # Categorical variable mapping dictionaries
    MAP_ROOM_TYPE = {"Shared room": 1, "Private room": 2, "Entire home/apt": 3, "Hotel room": 4}
    MAP_NEIGHB = {"Bronx": 1, "Queens": 2, "Staten Island": 3, "Brooklyn": 4, "Manhattan": 5}

    # Map categorical features
    df["neighbourhood"] = df["neighbourhood"].map(MAP_NEIGHB)
    df["room_type"] = df["room_type"].map(MAP_ROOM_TYPE)

    FEATURE_NAMES = ['neighbourhood', 'room_type', 'accommodates', 'bathrooms', 'bedrooms']

    X = df[FEATURE_NAMES]
    y = df['category']

    return X, y