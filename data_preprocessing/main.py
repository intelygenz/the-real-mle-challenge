from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, when

from configuration.config import FILEPATH_DATA, FILEPATH_PROCESSED
from data_preprocessing.auxiliary_functions import (
    num_bathroom_udf, preprocess_amenities_column)
from logger import logger


def load_data(filepath: Path):
    """
    Load data from a CSV file using Spark.
    
    Args:
        filepath (Path): Path to the CSV file.
        
    Returns:
        DataFrame: Loaded data as a Spark DataFrame.
    """
    try:
        spark = SparkSession.builder.appName("DataPreprocessing").getOrCreate()
        df = spark.read.csv(str(filepath), header=True, inferSchema=True, multiLine=True, escape='"')
        logger.info(f"Data loaded successfully from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        raise

def preprocess_data(df):
    """
    Preprocess the data by performing various transformations.
    
    Args:
        df (DataFrame): Raw data.
        
    Returns:
        DataFrame: Preprocessed data.
    """
    try:
        logger.info(f"Initial shape: {df.count()} rows, {len(df.columns)} columns")
        logger.info(f"Initial columns: {df.columns}")
        
        # Drop unnecessary columns
        df = df.drop('bathrooms')
        
        # Extract number of bathrooms from 'bathrooms_text'
        df = df.withColumn('bathrooms', num_bathroom_udf(col('bathrooms_text')))
        
        # Select specific columns and rename 'neighbourhood_group_cleansed' to 'neighbourhood'
        columns = ['id', 'neighbourhood_group_cleansed', 'property_type', 'room_type', 'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'amenities', 'price']
        df = df.select(columns)
        df = df.withColumnRenamed('neighbourhood_group_cleansed', 'neighbourhood')
        df.show(10, truncate=False)
        # Drop rows with missing values
        df = df.dropna()
        
        # Convert 'price' from string to numeric
        df = df.withColumn('price', regexp_extract(col('price'), r'(\d+).', 1).cast('int'))
        
        # Filter rows where 'price' is greater than or equal to 10
        df = df.filter(col('price') >= 10)
        
        # Preprocess the 'amenities' column
        df = preprocess_amenities_column(df)
        
        # Create a categorical price column
        df = df.withColumn('category', when(col('price') <= 90, 0)
                                      .when((col('price') > 90) & (col('price') <= 180), 1)
                                      .when((col('price') > 180) & (col('price') <= 400), 2)
                                      .otherwise(3))
        
        # Summary statistics
        df.describe(['price']).show()
        
        logger.info(f"Number of rows after preprocessing: {df.count()}")
        logger.info(f"Columns after preprocessing: {df.columns}")
        return df
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

def save_data(df, filepath: Path):
    """
    Save the preprocessed data to a CSV file.
    
    Args:
        df (DataFrame): Preprocessed data.
        filepath (Path): Path to save the CSV file.
    """
    try:
        df.write.mode("overwrite").csv(str(filepath), header=True)
        logger.info(f"Data saved successfully to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {e}")
        raise

def main():
    """
    Main function to execute the data loading and preprocessing steps.
    """
    try:
        df_raw = load_data(FILEPATH_DATA)
        df_processed = preprocess_data(df_raw)
        save_data(df_processed, FILEPATH_PROCESSED)
    finally:
        SparkSession.builder.getOrCreate().stop()

if __name__ == "__main__":
    main()