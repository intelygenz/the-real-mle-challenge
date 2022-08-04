"""
This file has the class definition for app settings
"""
from pydantic import BaseSettings


class AppSettings(BaseSettings):
    app_name: str = "Inference API"
    model_path: str = "models/random_forest_classifier_2022-08-04 08:31:07.734769.pkl"
    mapping_columns = {
        "room_type": {
            "Shared room": 1,
            "Private room": 2,
            "Entire home/apt": 3,
            "Hotel room": 4,
        },
        "neighbourhood": {
            "Bronx": 1,
            "Queens": 2,
            "Staten Island": 3,
            "Brooklyn": 4,
            "Manhattan": 5,
        },
        "price_category": {0: "Low", 1: "Mid", 2: "High", 3: "Lux"},
    }
