import unittest

import pandas as pd
from pyspark.sql import SparkSession

from configuration.config import (FILEPATH_TEST_EXPECTED_DATA,
                                  FILEPATH_TEST_INPUT_DATA)
from data_preprocessing.main import load_data, preprocess_data


class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.appName("TestPipeline").getOrCreate()
        cls.test_data_path = FILEPATH_TEST_INPUT_DATA
        cls.expected_data_path = FILEPATH_TEST_EXPECTED_DATA

    def test_load_data(self):
        df = load_data(self.test_data_path)
        self.assertEqual(df.count(), 38277)  # Number of rows
        self.assertEqual(len(df.columns), 74)  # Number of columns
        expected_columns = ['id', 'listing_url', 'scrape_id', 'last_scraped', 'name', 'description',
       'neighborhood_overview', 'picture_url', 'host_id', 'host_url',
       'host_name', 'host_since', 'host_location', 'host_about',
       'host_response_time', 'host_response_rate', 'host_acceptance_rate',
       'host_is_superhost', 'host_thumbnail_url', 'host_picture_url',
       'host_neighbourhood', 'host_listings_count',
       'host_total_listings_count', 'host_verifications',
       'host_has_profile_pic', 'host_identity_verified', 'neighbourhood',
       'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'latitude',
       'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms',
       'bathrooms_text', 'bedrooms', 'beds', 'amenities', 'price',
       'minimum_nights', 'maximum_nights', 'minimum_minimum_nights',
       'maximum_minimum_nights', 'minimum_maximum_nights',
       'maximum_maximum_nights', 'minimum_nights_avg_ntm',
       'maximum_nights_avg_ntm', 'calendar_updated', 'has_availability',
       'availability_30', 'availability_60', 'availability_90',
       'availability_365', 'calendar_last_scraped', 'number_of_reviews',
       'number_of_reviews_ltm', 'number_of_reviews_l30d', 'first_review',
       'last_review', 'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value', 'license', 'instant_bookable',
       'calculated_host_listings_count',
       'calculated_host_listings_count_entire_homes',
       'calculated_host_listings_count_private_rooms',
       'calculated_host_listings_count_shared_rooms', 'reviews_per_month']
        self.assertListEqual(df.columns, expected_columns)  # Column names

    def test_preprocess_data(self):
        df = load_data(self.test_data_path)
        df_preprocessed = preprocess_data(df)
        expected_df = pd.read_csv(self.expected_data_path)
        pd.testing.assert_frame_equal(df_preprocessed.toPandas().reset_index(drop=True).sort_index(axis=1).sort_values(by='id'),
                  expected_df.drop(columns=[expected_df.columns[0]]).reset_index(drop=True).sort_index(axis=1).sort_values(by='id'),
                  check_dtype=False)
if __name__ == "__main__":
    unittest.main()