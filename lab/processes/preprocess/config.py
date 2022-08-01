"""
This file contain the class that encapsulate config for preprocess
"""


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


class ConfigPreprocess:
    """
    This class encapsulate the config
    """

    # PATHS
    RAW_FILE = "data/raw/listings.csv"
    PREPROCESS_FILE = "data/processed/new_processed_listings.csv"
