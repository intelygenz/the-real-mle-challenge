import pandas as pd
from pathlib import Path

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Loads the CSV file into a DataFrame."""
    df = pd.read_csv(filepath)
    return df

def load_preprocessed_data(filepath: str) -> pd.DataFrame:
    """Loads the CSV file into a DataFrame."""
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