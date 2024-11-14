"""
This module provides functions for preprocessing raw data, including cleaning, feature engineering, 
and transformations necessary for model training.

Functions:
    - preprocess_data: Cleans and transforms the raw DataFrame, extracts relevant features, 
      and applies categorical transformations. Saves the preprocessed data to a specified path.
"""

import numpy as np
import pandas as pd
from utils import num_bathroom_from_text, preprocess_amenities_column

def preprocess_data(df_raw: pd.DataFrame, output_path: str) -> None:
    """
    Preprocesses the raw DataFrame and saves the cleaned data to the specified output path.
    
    Parameters:
    - df_raw: Raw input DataFrame to preprocess.
    - output_path: Path to save the preprocessed data as a CSV file.
    
    Returns:
    - None
    """
    
    # Replace the 'bathrooms' column by extracting bathroom counts from 'bathrooms_text'
    df_raw.drop(columns=['bathrooms'], inplace=True)
    df_raw['bathrooms'] = df_raw['bathrooms_text'].apply(num_bathroom_from_text)

    # Select specific columns for analysis and create a copy
    COLUMNS = ['id', 'neighbourhood_group_cleansed', 'property_type', 'room_type', 
               'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 
               'beds', 'amenities', 'price']
    df = df_raw[COLUMNS].copy()
    df.rename(columns={'neighbourhood_group_cleansed': 'neighbourhood'}, inplace=True)

    # Drop rows with any missing values
    df = df.dropna(axis=0)

    # Convert the 'price' column from string to integer
    df['price'] = df['price'].str.extract(r"(\d+).")  # Extract numeric part
    df['price'] = df['price'].astype(int)  # Convert to integer type

    # Filter out listings where the price is less than $10
    df = df[df['price'] >= 10]

    # Create a categorical 'category' column based on price ranges:
    # 0 for Low ($10-$90), 1 for Mid ($90-$180), 2 for High ($180-$400), 3 for Luxury ($400+)
    df['category'] = pd.cut(df['price'], bins=[10, 90, 180, 400, np.inf], labels=[0, 1, 2, 3])

    # Extract information from the 'amenities' column
    df = preprocess_amenities_column(df)

    # Drop any remaining rows with missing values
    df = df.dropna(axis=0)

    # Save the preprocessed DataFrame to the specified output path
    df.to_csv(output_path, index=False)