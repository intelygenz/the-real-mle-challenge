from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

DIR_REPO = Path.cwd()

DIR_DATA_RAW = DIR_REPO / "data" / "raw"
FILEPATH_RAW = DIR_DATA_RAW / "listings.csv"

DIR_DATA_PROCESSED = DIR_REPO / "data" / "processed"
FILEPATH_PROCESSED = DIR_DATA_PROCESSED / "preprocessed_listings.csv"

SOURCE_COLUMNS = [
    "id",
    "neighbourhood_group_cleansed",
    "property_type",
    "room_type",
    "latitude",
    "longitude",
    "accommodates",
    "bathrooms_text",
    "bedrooms",
    "beds",
    "amenities",
    "price",
]
MAP_NEIGHBOURHOOD = {
    "Bronx": 1,
    "Queens": 2,
    "Staten Island": 3,
    "Brooklyn": 4,
    "Manhattan": 5,
}
MAP_ROOM_TYPE = {
    "Shared room": 1,
    "Private room": 2,
    "Entire home/apt": 3,
    "Hotel room": 4,
}
FEATURE_NAMES = [
    "neighbourhood",
    "room_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
]

MIN_PRICE = 10
PRICE_CATEGORIES_CUTS = [10, 90, 180, 400, np.inf]
TEST_SIZE = 0.15
SEED = 0


def num_bathroom_from_text(text):
    try:
        if isinstance(text, str):
            bath_num = text.split(" ")[0]
            return float(bath_num)
        else:
            return np.nan
    except ValueError:
        return np.nan


if __name__ == "__main__":
    data = pd.read_csv(FILEPATH_RAW, usecols=SOURCE_COLUMNS)

    data.rename(
        columns={
            "neighbourhood_group_cleansed": "neighbourhood",
            "bathrooms_text": "bathrooms",
        },
        inplace=True,
    )

    # TODO Improve
    data["bathrooms"] = data["bathrooms"].apply(num_bathroom_from_text)

    data.dropna(axis=0, inplace=True)

    # BUG (?) The regex should be r"(\d+)\."
    data["price"] = data["price"].str.extract(r"(\d+).").astype(int)
    data = data[data["price"] >= MIN_PRICE]

    data["neighbourhood"] = data["neighbourhood"].map(MAP_NEIGHBOURHOOD)

    data["room_type"] = data["room_type"].map(MAP_ROOM_TYPE)

    data["category"] = pd.cut(
        x=data["price"],
        bins=PRICE_CATEGORIES_CUTS,
        labels=range(len(PRICE_CATEGORIES_CUTS) - 1),
    )
    data.drop(columns=["price"], inplace=True)
    data.dropna(axis=0, inplace=True)

    data_x = data[FEATURE_NAMES]
    data_y = data[["category"]]

    data_x.to_parquet(DIR_DATA_PROCESSED / "ny_listings_x.parquet")
    data_y.to_parquet(DIR_DATA_PROCESSED / "ny_listings_y.parquet")
