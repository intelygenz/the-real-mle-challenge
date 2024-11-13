import pandas as pd

# Preprocess the data to match the model's expected features
def preprocess_listing_input(input_listing: dict) -> pd.DataFrame:
    """
    Preprocess input data to match the features expected by the trained model.
    
    Parameters:
    - input_listing (dict): Dictionary containing the raw input listing data.
    
    Returns:
    - pd.DataFrame: Preprocessed DataFrame with selected and transformed features.
    """

    # Convert input data to a DataFrame
    df = pd.DataFrame([input_listing])

    # Map categorical features
    MAP_ROOM_TYPE = {"Shared room": 1, "Private room": 2, "Entire home/apt": 3, "Hotel room": 4}
    MAP_NEIGHB = {"Bronx": 1, "Queens": 2, "Staten Island": 3, "Brooklyn": 4, "Manhattan": 5}
    df["neighbourhood"] = df["neighbourhood"].map(MAP_NEIGHB)
    df["room_type"] = df["room_type"].map(MAP_ROOM_TYPE)

    # Define the expected columns in the order required by the model
    FEATURE_NAMES = ['neighbourhood', 'room_type', 'accommodates', 'bathrooms', 'bedrooms']
    df = df[FEATURE_NAMES]
    return df