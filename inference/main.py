import pickle

import pandas as pd

from configuration.config import FILEPATH_MODEL

ALLOWED_COLUMNS = ['neighbourhood', 'room_type', 'accommodates', 'bathrooms', 'bedrooms']

# Define the mapping dictionaries
MAP_CATEGORY = {0: "Low", 1: "Mid", 2: "High", 3: "Luxury"}

# Load the trained model
with open(FILEPATH_MODEL, 'rb') as f:
    model = pickle.load(f)


def predict_price_category(input_data: dict):
    df_input = pd.DataFrame(input_data)
    df_input = df_input[ALLOWED_COLUMNS]

    # Make prediction
    category = model.predict(df_input)[0]
    price_category = MAP_CATEGORY[category]
    return price_category