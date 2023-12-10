import logging
from typing import List, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Preprocess:
    def __init__(self, columns: List[str], amenities: List[str], min_price_to_consider: int, price_bins: List[float],
                 map_neighb: Dict[str, int], map_room_type: Dict[str, int]):
        self.columns = columns
        self.amenities = amenities
        self.min_price_to_consider = min_price_to_consider
        self.price_bins = price_bins
        self.map_neighb = map_neighb
        self.map_room_type = map_room_type

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df['bathrooms'] = df['bathrooms_text'].apply(self._num_bathroom_from_text)
        df = df[self.columns]
        df = df.rename(columns={'neighbourhood_group_cleansed': 'neighbourhood'})
        df = df.dropna(axis=0)
        df = self._preprocess_amenities(df)
        df = self._preprocess_price(df)
        df = self._map_to_categories(df)
        return df.dropna(axis=0)

    def _num_bathroom_from_text(self, text: str) -> float:
        try:
            if isinstance(text, str):
                bath_num = text.split(" ")[0]
                return float(bath_num)
            else:
                return np.NaN
        except ValueError:
            return np.NaN

    def _price_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        df['price'] = df['price'].str.extract(r"(\d+).")
        df['price'] = df['price'].astype(int)
        return df

    def _price_to_category(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[:, 'category'] = pd.cut(df['price'],
                                       bins=self.price_bins,
                                       labels=np.arange(len(self.price_bins) - 1).tolist())
        return df

    def _preprocess_price(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._price_to_int(df)
        df = df[df['price'] >= self.min_price_to_consider]
        return self._price_to_category(df)

    def _add_amenity(self, df: pd.DataFrame, amenity: str) -> pd.DataFrame:
        df[amenity] = df['amenities'].str.contains(amenity).astype(int)
        return df

    def _preprocess_amenities(self, df: pd.DataFrame) -> pd.DataFrame:
        for amenity in self.amenities:
            df = self._add_amenity(df=df, amenity=amenity)

        return df.drop('amenities', axis=1)

    def _map_to_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        df["neighbourhood"] = df["neighbourhood"].map(self.map_neighb)
        df["room_type"] = df["room_type"].map(self.map_room_type)
        return df
