import os
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import re
from code.src.transformer import (
    ArrayOneHotEncoder,
    ColumnRenamer,
    ColumnSelector,
    CustomOrdinalEncoder,
    DropColumns,
    DropNan,
    DiscretizerTransformer,
    QueryFilter,
    StringToFloatTransformer,
    StringToInt
)

DIR_REPO = Path(__file__).parent.parent.parent.parent
log_dir = os.path.join(DIR_REPO, "solution", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "unittests.log")


class TestStringToFloatTransformer(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame({
            'bathrooms_text': ['1 private bath', '1 bath', 'NaN', '1.5 baths'],
        })

    def test_transform_with_dict_column(self):
        transformer = StringToFloatTransformer(columns={"bathrooms_text": "bathrooms"})
        transformed_data = transformer.transform(self.data)
        expected_data = self.data.copy()
        expected_data["bathrooms"] = pd.Series([1, 1, np.nan, 1.5])
        pd.testing.assert_frame_equal(transformed_data, expected_data)

    def test_transform_with_list_column(self):
        transformer = StringToFloatTransformer(columns=["bathrooms_text"])
        transformed_data = transformer.transform(self.data)
        expected_data = self.data.copy()
        expected_data["bathrooms_text"] = pd.Series([1, 1, np.nan, 1.5])
        pd.testing.assert_frame_equal(transformed_data, expected_data)

    def test_get_feature_names_out(self):
        transformer = StringToFloatTransformer(columns={"bathrooms_text": "bathrooms"})
        transformer.transform(self.data)
        self.assertListEqual(transformer.get_feature_names_out(None), list(transformer.out_cols))

# Test for ColumnSelector
class TestColumnSelector(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'price': [10, 20, 30],
            'quantity': [1, 2, 3],
            'description': ['A', 'B', 'C']
        })


    def test_transform_single_column(self):
        transformer = ColumnSelector(columns="price")
        transformed_data = transformer.transform(self.data)
        expected_output = pd.DataFrame({'price': [10, 20, 30]})
        pd.testing.assert_frame_equal(transformed_data, expected_output)

    def test_transform_multiple_columns(self):
        transformer = ColumnSelector(columns=["price", "quantity"])
        transformed_data = transformer.transform(self.data)
        expected_output = self.data[["price", "quantity"]]
        pd.testing.assert_frame_equal(transformed_data, expected_output)

    def test_get_feature_names_out(self):
        transformer = ColumnSelector(columns=["price", "quantity"])
        transformer.transform(self.data)
        self.assertListEqual(list(transformer.get_feature_names_out(None)), transformer.columns)


class TestColumnRenamer(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'old_price': [10, 20, 30],
            'quantity': [1, 2, 3]
        })

    def test_transform_column_renaming(self):
        transformer = ColumnRenamer(columns={"old_price": "price", "old_description": "description"})
        transformed_data = transformer.transform(self.data)
        expected_output = self.data.rename(columns={"old_price": "price", "old_description": "description"})
        pd.testing.assert_frame_equal(transformed_data, expected_output)

    def test_transform_invalid_data_type(self):
        transformer = ColumnRenamer(columns={"old_price": "price"})
        with self.assertRaises(ValueError):
            transformer.transform(["Not a DataFrame"])

    def test_get_feature_names_out(self):
        transformer = ColumnRenamer(columns={"old_price": "price", "old_description": "description"})
        transformer.transform(self.data)
        self.assertListEqual(list(transformer.get_feature_names_out(None)), list(transformer.out_cols))


class TestDropNan(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'price': [10, np.nan, 30, 40, 50],
            'quantity': [1, 2, 3, 4, 5]
        },
        dtype=np.float32
        )

    def test_transform_rows(self):
        transformer = DropNan(axis=0)
        transformed_data = transformer.transform(self.data)
        expected_output = pd.DataFrame({
            'price': [10.0, 30.0, 40.0, 50.0],
            'quantity': [1.0, 3.0, 4.0, 5.0]
        },
        dtype= np.float32
        )
        pd.testing.assert_frame_equal(transformed_data.reset_index(drop=True), expected_output)

    def test_transform_columns(self):
        transformer = DropNan(axis=1)
        transformed_data = transformer.transform(self.data)
        expected_output = pd.DataFrame({
            'quantity': [1.0, 2.0, 3.0, 4.0, 5.0]
        },
        dtype= np.float32
        )
        pd.testing.assert_frame_equal(transformed_data, expected_output)

    def test_transform_invalid_data_type(self):
        transformer = DropNan(axis=0)
        with self.assertRaises(ValueError):
            transformer.transform("This is not valid")  

    def test_get_feature_names_out(self):
        transformer = DropNan(axis=0)
        transformed_data = transformer.transform(self.data)
        feature_names = transformer.get_feature_names_out(transformed_data.columns)
        self.assertListEqual(list(feature_names), list(self.data.columns))


class TestDropColumns(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'price': [10, 20, 30, 40, 50],
            'quantity': [1, 2, 3, 4, 5],
            'description': ['A', 'B', 'C', 'D', 'E']
        })

    def test_transform_single_column(self):
        transformer = DropColumns(columns="price")
        transformed_data = transformer.transform(self.data)
        expected_output = pd.DataFrame({
            'quantity': [1, 2, 3, 4, 5],
            'description': ['A', 'B', 'C', 'D', 'E']
        })
        pd.testing.assert_frame_equal(transformed_data, expected_output)

    def test_transform_multiple_columns(self):
        transformer = DropColumns(columns=["price", "quantity"])
        transformed_data = transformer.transform(self.data)
        expected_output = pd.DataFrame({
            'description': ['A', 'B', 'C', 'D', 'E']
        })
        pd.testing.assert_frame_equal(transformed_data, expected_output)

    def test_transform_invalid_data_type(self):
        transformer = DropColumns(columns="price")
        with self.assertRaises(ValueError):
            transformer.transform("This is not valid")  

    def test_get_feature_names_out(self):
        transformer = DropColumns(columns="price")
        transformed_data = transformer.transform(self.data)
        feature_names = transformer.get_feature_names_out(transformed_data.columns)
        self.assertListEqual(list(feature_names), list(transformed_data.columns))


class TestStringToInt(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame({
            'price': ["$10", "$20", "$30", "$40", "$50"],
            'quantity': ["1 unit", "2 units", "3 units", "4 units", "5 units"]
        })

        self.columns = ['price', 'quantity']
        self.patterns = [r'\d+', r'\d+']

    def test_initialization_mismatched_types(self):
        with self.assertRaises(ValueError):
            StringToInt(columns=['price', 'quantity'], patterns=r'\d+')

    def test_initialization_mismatched_list_lengths(self):
        with self.assertRaises(ValueError):
            StringToInt(columns=['price', 'quantity'], patterns=[r'\d+'])

    def test_apply_regex(self):
        transformer = StringToInt(columns=self.columns, patterns=self.patterns)
        match = transformer._apply_regex("$10", re.compile(r'\d+'))
        self.assertEqual(match, "10")

    def test_transform_single_column(self):
        transformer = StringToInt(columns=self.columns[0], patterns=self.patterns[0])
        transformed_data = transformer.transform(self.data.drop(columns='quantity'))
        expected_output = pd.DataFrame({'price': [10, 20, 30, 40, 50]})
        pd.testing.assert_frame_equal(transformed_data, expected_output)

    def test_transform_multiple_columns(self):
        transformer = StringToInt(columns=self.columns, patterns=self.patterns)
        transformed_data = transformer.transform(self.data)
        expected_output = pd.DataFrame({
            'price': [10, 20, 30, 40, 50],
            'quantity': [1, 2, 3, 4, 5]
        })
        pd.testing.assert_frame_equal(transformed_data, expected_output)


    def test_get_feature_names_out(self):
        transformer = StringToInt(columns=self.columns[0], patterns=self.patterns[0])
        transformer.transform(self.data.drop(columns='quantity'))
        feature_names = transformer.get_feature_names_out(self.data.columns[0])
        self.assertListEqual(list(feature_names), [self.data.columns[0]])


class TestQueryFilter(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame({
            'price': [10, 20, 30, 40, 50],
            'quantity': [1, 2, 3, 4, 5]
        })

    def test_transform_valid_query(self):
        filter_transformer = QueryFilter(query_string="price > 20")
        transformed_data = filter_transformer.transform(self.data)
        expected_output = pd.DataFrame({
            'price': [30, 40, 50],
            'quantity': [3, 4, 5]
        })
        pd.testing.assert_frame_equal(transformed_data.reset_index(drop=True), expected_output)


    def test_get_feature_names_out(self):
        filter_transformer = QueryFilter(query_string="price > 20")
        filter_transformer.transform(self.data)
        feature_names = filter_transformer.get_feature_names_out(self.data.columns)
        self.assertListEqual(list(feature_names), list(self.data.columns))


class TestDiscretizerTransformer(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'price': [10, 30, 50, 70, 100],
            'age': [20, 40, 60, 80, 100]
        })

        self.bins = [[0, 25, 50, 75, np.inf], [0, 35, 70, np.inf]]
        self.labels = [[0, 1, 2, 3], [0, 1, 2]]
        self.columns = ['price', 'age']
        self.new_colnames = ['price_category', 'age_category']

    def test_initialization_invalid_params(self):
        with self.assertRaises(ValueError):
            DiscretizerTransformer(
                columns=['price', 'age'],
                new_colnames=['price_category'],
                bins=[self.bins[0]],
                labels=[self.labels]
            )

        with self.assertRaises(ValueError):
            DiscretizerTransformer(
                columns='price',
                new_colnames='price_category',
                bins=self.bins,  
                labels=[0, 1]
            )

    def test_transform_single_column(self):
        transformer = DiscretizerTransformer(
            columns='price',
            new_colnames='price_category',
            bins=self.bins[0],
            labels=self.labels[0]
        )

        transformed_data = transformer.fit_transform(self.data.drop(columns=['age']))
        expected_output = pd.DataFrame({
            'price': [10, 30, 50, 70, 100],
            'price_category': pd.Categorical([0, 1, 1, 2, 3], categories=self.labels[0], ordered=True)
        })
        pd.testing.assert_frame_equal(transformed_data, expected_output)

    def test_transform_multiple_columns(self):
        transformer = DiscretizerTransformer(
            columns=self.columns,
            new_colnames=self.new_colnames,
            bins=self.bins,
            labels=self.labels
        )
        transformed_data = transformer.transform(self.data)
        expected_output = pd.DataFrame({
            'price': [10, 30, 50, 70, 100],
            'age': [20, 40, 60, 80, 100],
            'price_category': pd.Categorical([0, 1, 1, 2, 3], categories=self.labels[0], ordered=True),
            'age_category': pd.Categorical([0, 1, 1, 2, 2], categories=self.labels[1], ordered=True)
        })
        pd.testing.assert_frame_equal(transformed_data, expected_output)

    def test_get_feature_names_out(self):
        transformer = DiscretizerTransformer(
            columns=self.columns,
            new_colnames=self.new_colnames,
            bins=self.bins,
            labels=self.labels
        )

        transformer.fit_transform(self.data)
        feature_names = transformer.get_feature_names_out(self.columns)
        expected_output = list(self.data.columns) + self.new_colnames
        self.assertListEqual(list(feature_names), expected_output)


class TestArrayOneHotEncoder(unittest.TestCase):

    def setUp(self):
        self.data_df = pd.DataFrame({
            'amenities': [
                '["Extra pillows and blankets", "Baking sheet", "Wifi", "Heating", "Dishes and silverware", "Essentials", ]',
                '["Extra pillows and blankets", "Luggage dropoff allowed", "Free parking on premises", "Wifi", "Heating"]',
                '["Kitchen", "Long term stays allowed", "Heating", "Air conditioning", "Pool"]'
                ]
        })
        self.categories = ['Wifi', 'Parking', 'Pool', 'Heating']
        self.data_ndarray = self.data_df.amenities.to_numpy()
        self.encoder = ArrayOneHotEncoder(column='amenities', categories=self.categories)

    def test_initialization(self):
        self.assertEqual(self.encoder.column, 'amenities')
        self.assertEqual(self.encoder.categories, self.categories)


    def test_transform_with_dataframe(self):
        transformed_df = self.encoder.transform(self.data_df)
        expected_columns = list(self.data_df.columns) + self.categories
        self.assertTrue(all(col in transformed_df.columns for col in expected_columns))
        expected_output = pd.DataFrame({
            'amenities': self.data_df.amenities.tolist(),
            'Wifi': [1, 1, 0],
            'Parking': [0, 0, 0],
            'Pool': [0, 0, 1],
            'Heating': [1, 1, 1]
        })
        pd.testing.assert_frame_equal(transformed_df, expected_output)

    def test_transform_with_ndarray(self):
        transformed_array = self.encoder.transform(self.data_ndarray)
        expected_output = [[1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1]]
        np.testing.assert_array_equal(transformed_array, expected_output)

    def test_get_feature_names_out(self):
        self.encoder.transform(self.data_df)
        feature_names = self.encoder.get_feature_names_out(['amenities'])
        self.assertListEqual(list(feature_names), list(self.data_df.columns) + self.categories)


class TestCustomOrdinalEncoder(unittest.TestCase):

    def setUp(self):
        self.categories = [['low', 'medium', 'high']]
        self.data = pd.DataFrame({'quality': ['low', 'medium', 'high', 'low']})

    def test_initialization_invalid_categories_type(self):
        with self.assertRaises(ValueError) as context:
            CustomOrdinalEncoder(categories="not list")
        self.assertIn("The categories must be passed as a list os list", str(context.exception))

    def test_initialization_invalid_categories_format(self):
        with self.assertRaises(ValueError) as context:
            CustomOrdinalEncoder(categories=["not nested list"])
        self.assertIn("The categories must be passed as a list os list", str(context.exception))

    def test_initialization_invalid_start_category(self):
        with self.assertRaises(ValueError) as context:
            CustomOrdinalEncoder(categories=self.categories, start_category="not int")
        self.assertIn("The starting category must be a integer", str(context.exception))

    def test_fit_transform_with_start_category(self):
        encoder = CustomOrdinalEncoder(categories=self.categories, start_category=1)
        transformed_data = encoder.fit_transform(self.data)
        expected_output = np.array([[1], [2], [3], [1]])
        np.testing.assert_array_equal(transformed_data, expected_output)

    def test_fit_transform_no_start_category(self):
        encoder = CustomOrdinalEncoder(categories=self.categories)
        transformed_data = encoder.fit_transform(self.data)
        expected_output = np.array([[0], [1], [2], [0]])
        np.testing.assert_array_equal(transformed_data, expected_output)

    def test_get_feature_names_out(self):
        encoder = CustomOrdinalEncoder(categories=self.categories)
        encoder.fit(self.data)
        feature_names = encoder.get_feature_names_out(['quality'])
        self.assertEqual(feature_names, ['quality'])





if __name__ == '__main__':
    # Open the log file in write mode and run tests with custom TextTestRunner
    with open(log_file, "w") as f:
        runner = unittest.TextTestRunner(stream=f)
        unittest.main(testRunner=runner, exit=False)
