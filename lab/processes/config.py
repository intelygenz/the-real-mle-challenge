"""
This file contain the class that encapsulate config for preprocess
"""

import numpy as np


class DataRawColumns:
    """
    This class has all names of columns for the raw dataframe
    """

    ID = "id"
    BATHROOMS = "bathrooms"
    BATHROOMS_TEXT = "bathrooms_text"
    NEIGHBOURHOOD_GROUP_CLEANSED = "neighbourhood_group_cleansed"
    PROPERTY_TYPE = "property_type"
    ROOM_TYPE = "room_type"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    ACCOMMODATES = "accommodates"
    BEDROOMS = "bedrooms"
    BEDS = "beds"
    AMENITIES = "amenities"
    PRICE = "price"

    SUBSET_TRAINING = [
        ID,
        BATHROOMS,
        NEIGHBOURHOOD_GROUP_CLEANSED,
        PROPERTY_TYPE,
        ROOM_TYPE,
        LATITUDE,
        LONGITUDE,
        ACCOMMODATES,
        BEDROOMS,
        BEDS,
        AMENITIES,
        PRICE,
    ]


class DataPreprocessColumns:
    """
    This class has all names of columns for the preprocess dataframe
    """

    ID = "id"
    NEIGHBOURHOOD = "neighbourhood"
    PROPERTY_TYPE = "property_type"
    ROOM_TYPE = "room_type"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    ACCOMMODATES = "accommodates"
    BATHROOMS = "bathrooms"
    BEDROOMS = "bedrooms"
    BEDS = "beds"
    PRICE = "price"
    CATEGORY = "category"
    TV = "TV"
    INTERNET = "Internet"
    AIR_CONDITIONING = "Air_conditioning"
    KITCHEN = "Kitchen"
    HEATING = "Heating"
    WIFI = "Wifi"
    ELEVATOR = "Elevator"
    BREAKFAST = "Breakfast"


class ConfigPreprocess:
    """
    This class encapsulate the config for preprocess
    """

    # Paths
    RAW_FILE = "data/raw/listings.csv"
    PREPROCESS_FILE = "data/processed/new_processed_listings.csv"

    # Preprocess config
    MIN_PRICE = 10
    BINS_PRICE = [10, 90, 180, 400, np.inf]
    LABELS_PRICE = [0, 1, 2, 3]
    MAPING_COLUMNS = {
        DataPreprocessColumns.ROOM_TYPE: {
            "Shared room": 1,
            "Private room": 2,
            "Entire home/apt": 3,
            "Hotel room": 4,
        },
        DataPreprocessColumns.NEIGHBOURHOOD: {
            "Bronx": 1,
            "Queens": 2,
            "Staten Island": 3,
            "Brooklyn": 4,
            "Manhattan": 5,
        },
    }


class ConfigTrain:
    """
    This class encapsulate the config for train process
    """

    FEATURE_NAMES = [
        DataPreprocessColumns.NEIGHBOURHOOD,
        DataPreprocessColumns.ROOM_TYPE,
        DataPreprocessColumns.ACCOMMODATES,
        DataPreprocessColumns.BATHROOMS,
        DataPreprocessColumns.BEDROOMS,
    ]
    FEATURE_CATEGORY = DataPreprocessColumns.CATEGORY

    # Split parameters
    TEST_SIZE = 1
    RANDOM_STATE_SPLIT = 1

    # Train parameters
    N_ESTIMATORS = 500
    RANDOM_STATE_TRAIN = 0
    CLASS_WEIGHT = "balanced"
    N_JOBS = 4
