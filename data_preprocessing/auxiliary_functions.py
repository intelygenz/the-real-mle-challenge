import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType


def num_bathroom_from_text(text):
    try:
        if isinstance(text, str):
            bath_num = text.split(" ")[0]
            return float(bath_num)
        else:
            return np.NaN
    except ValueError:
        return np.NaN
    
num_bathroom_udf = udf(num_bathroom_from_text, DoubleType())



def preprocess_amenities_column(df: DataFrame) -> DataFrame:
    """
    Preprocess the 'amenities' column by creating separate columns for each amenity.
    
    Args:
        df (DataFrame): Input DataFrame with an 'amenities' column.
        
    Returns:
        DataFrame: DataFrame with separate columns for each amenity.
    """
    amenities_list = ['TV', 'Internet', 'Air conditioning', 'Kitchen', 'Heating', 'Wifi', 'Elevator', 'Breakfast']
    
    for amenity in amenities_list:
        df = df.withColumn(amenity.replace(" ", "_"), col('amenities').contains(amenity).cast('int'))
    
    df = df.drop('amenities')
    
    return df