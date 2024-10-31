import os
import sys
import logging
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

DIR_REPO = Path(__file__).parent.parent.parent.parent.parent
os.chdir(DIR_REPO)


# Import custom functions
from code.src.transformer import (
    ArrayOneHotEncoder,
    ColumnRenamer,
    ColumnSelector,
    DropColumns,
    DropNan,
    DiscretizerTransformer,
    QueryFilter,
    StringToFloatTransformer,
    StringToInt,
)


LOG_DIR =  DIR_REPO / 'solution' / 'logs' 
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, "test_eda.log")

# Configure logging
logging.basicConfig(
    filename=log_file, 
    level=logging.DEBUG,
    filemode='w+'
)

logger = logging.getLogger(__name__)


pd.set_option('display.max_columns', 150)

DIR_DATA_RAW = Path(DIR_REPO) / "data" / "raw"
FILEPATH_DATA = DIR_DATA_RAW / "listings.csv"

COLUMNS = ['id', 'neighbourhood_group_cleansed', 'property_type', 'room_type', 'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 'beds','amenities', 'price']
COLUMNS_PIPE = ['id', 'neighbourhood_group_cleansed', 'property_type', 'room_type', 'latitude', 'longitude', 'accommodates', 'bathrooms_text', 'bedrooms', 'beds','amenities', 'price']
CAT_COLS = ['TV', 'Internet', 'Air conditioning', 'Kitchen', 'Heating', 'Wifi', 'Elevator', 'Breakfast'] 

MAP_ROOM_TYPE = {"Shared room": 1, "Private room": 2, "Entire home/apt": 3, "Hotel room": 4}
MAP_NEIGHB = {"Bronx": 1, "Queens": 2, "Staten Island": 3, "Brooklyn": 4, "Manhattan": 5}


df_raw = pd.read_csv(FILEPATH_DATA, low_memory=False)

logger.info("Aplying old code")
df_raw_old = df_raw.copy()
t1_old_code = datetime.datetime.now()

df_raw_old_code = df_raw_old.drop(columns=['bathrooms'])

# Get number of bathrooms from `bathrooms_text`
def num_bathroom_from_text(text):
    try:
        if isinstance(text, str):
            bath_num = text.split(" ")[0]
            return float(bath_num)
        else:
            return np.nan
    except ValueError:
        return np.nan
df_raw_old_code['bathrooms'] = df_raw_old_code['bathrooms_text'].apply(num_bathroom_from_text)
df = df_raw_old_code[COLUMNS].copy()
df.rename(columns={'neighbourhood_group_cleansed': 'neighbourhood'}, inplace=True)
df = df.dropna(axis=0)

# Convert string to numeric
df['price'] = df['price'].str.extract(r"(\d+).")
df['price'] = df['price'].astype(int)

df = df[df['price'] >= 10]

df['category'] = pd.cut(df['price'], bins=[10, 90, 180, 400, np.inf], labels=[0, 1, 2, 3])


def preprocess_amenities_column(df: DataFrame) -> DataFrame:
    
    df['TV'] = df['amenities'].str.contains('TV')
    df['TV'] = df['TV'].astype(int)
    df['Internet'] = df['amenities'].str.contains('Internet')
    df['Internet'] = df['Internet'].astype(int)
    df['Air_conditioning'] = df['amenities'].str.contains('Air conditioning')
    df['Air_conditioning'] = df['Air_conditioning'].astype(int)
    df['Kitchen'] = df['amenities'].str.contains('Kitchen')
    df['Kitchen'] = df['Kitchen'].astype(int)
    df['Heating'] = df['amenities'].str.contains('Heating')
    df['Heating'] = df['Heating'].astype(int)
    df['Wifi'] = df['amenities'].str.contains('Wifi')
    df['Wifi'] = df['Wifi'].astype(int)
    df['Elevator'] = df['amenities'].str.contains('Elevator')
    df['Elevator'] = df['Elevator'].astype(int)
    df['Breakfast'] = df['amenities'].str.contains('Breakfast')
    df['Breakfast'] = df['Breakfast'].astype(int)

    df.drop('amenities', axis=1, inplace=True)
    
    return df


df = preprocess_amenities_column(df)

t2_old_code = datetime.datetime.now()


logger.info("Aplying new code")
t1_new_code = datetime.datetime.now()

# ct = ColumnTransformer(
#     transformers=[
#         ('bathroom_processing', StringToFloatTransformer({'bathrooms_text': 'bathrooms'}), ['bathrooms_text']),
#         ('array_to_cat', ArrayOneHotEncoder('amenities', CAT_COLS), 'amenities')
#     ],
#     remainder='passthrough',
#     n_jobs= -1,
#     verbose_feature_names_out=False,
# )

# ct.set_output(transform='pandas')

# preprocessing_pipeline = Pipeline(steps=[
#     ('col_selector', ColumnSelector(COLUMNS_PIPE)),
#     ('column_transformer', ct),
#     ('drop_na', DropNan(axis=0)),
#     ('cast_price', StringToInt('price', r"(\d+)")),
#     ('bin_price', DiscretizerTransformer('price', 'category', bins=[10, 90, 180, 400, np.inf], labels=[0, 1, 2, 3])),
#     ('filter_rows', QueryFilter("price >= 10")),
#     ('col_renamer_conditioning', ColumnRenamer(columns={'Air conditioning': 'Air_conditioning', 'neighbourhood_group_cleansed': 'neighbourhood'})),
#     ('drop_cols', DropColumns('bathrooms_text'))
#     ]
# )

preprocessing_pipeline = Pipeline(steps=[
    ('col_selector', ColumnSelector(COLUMNS_PIPE)),
    ('bathroom_processing', StringToFloatTransformer({'bathrooms_text': 'bathrooms'})),
    ('cast_price', StringToInt('price', r"(\d+)")),
    ('filter_rows', QueryFilter("price >= 10")),
    ('drop_na', DropNan(axis=0)),
    ('bin_price', DiscretizerTransformer('price', 'category', bins=[10, 90, 180, 400, np.inf], labels=[0, 1, 2, 3])),
    ('array_to_cat', ArrayOneHotEncoder('amenities', CAT_COLS)),
    ('col_renamer_conditioning', ColumnRenamer(columns={'Air conditioning': 'Air_conditioning', 'neighbourhood_group_cleansed': 'neighbourhood'})),
    ('drop_cols', DropColumns('amenities'))
])

preprocessing_pipeline.set_output(transform='pandas')

df_processed = preprocessing_pipeline.fit_transform(df_raw)

t2_new_code = datetime.datetime.now()

logger.info(f"""
Old code time: {t2_old_code - t1_old_code}
New code time: {t2_new_code - t1_new_code}
Same result: {all(df == df_processed[df.columns])}
""")