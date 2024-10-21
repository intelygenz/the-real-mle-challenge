import pickle

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from api.schemas.listing import ListingSchema

MAP_NEIGHBOURHOOD = {
    "Bronx": 1,
    "Queens": 2,
    "Staten Island": 3,
    "Brooklyn": 4,
    "Manhattan": 5
}
MAP_ROOM_TYPE = {
    "Shared room": 1,
    "Private room": 2,
    "Entire home/apt": 3,
    "Hotel room": 4
}
FEATURE_NAMES = [
    "neighbourhood",
    "room_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
]
MAP_PREDICTION = {0.0: "Low", 1.0: "Mid", 2.0: "High", 3.0: "Lux"}


class NYClassifier():
    """
    New York price classifier.

    This classifier predicts the price category of a listing based on its features.
    """

    def __init__(self, model: BaseEstimator):
        if not isinstance(model, ClassifierMixin):
            raise ValueError("Loaded model is not a classifier")

        self.model = model

    @staticmethod
    def from_pickle(model_path: str):
        with open(model_path, "rb") as model_file:
            model: BaseEstimator = pickle.load(model_file)

        return NYClassifier(model)

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data["neighbourhood"] = data["neighbourhood"].map(MAP_NEIGHBOURHOOD)
        data["room_type"] = data["room_type"].map(MAP_ROOM_TYPE)

        return data

    def predict(self, listing: ListingSchema):
        x = pd.DataFrame(
            [
                [
                    listing.neighbourhood,
                    listing.room_type,
                    listing.accommodates,
                    listing.bathrooms,
                    listing.bedrooms,
                ],
            ],
            columns=FEATURE_NAMES,
        )
        x = self._preprocess_data(x)

        return MAP_PREDICTION[self.model.predict(x)[0]]
