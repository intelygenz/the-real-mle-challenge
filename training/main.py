import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

from configuration.config import FILEPATH_MODEL, FILEPATH_PROCESSED
from data_preprocessing.main import main as preprocess_raw_data
from logger import logger


def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        filepath (Path): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    try:
        spark = SparkSession.builder.appName("ModelTraining").getOrCreate()
        df_spark = spark.read.csv(str(filepath), header=True, inferSchema=True)
        df_pandas = df_spark.toPandas()
        logger.info(f"First ten rows of the dataframe:\n{df_pandas.head(10).to_string(index=False)}")
        df_pandas.to_csv('/app/data/processed_data_visualize.csv', index=False)
        logger.info(f"Data loaded successfully from {filepath}")
        logger.info(f"First ten rows of the dataframe:\n{df_pandas.head(10).to_string(index=False)}")
        logger.info(f"Columns of the dataframe: {df_pandas.columns.tolist()}")
        return df_pandas
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by dropping rows with missing values.
    
    Args:
        df (pd.DataFrame): Raw data.
        
    Returns:
        pd.DataFrame: Preprocessed data.
    """
    try:
        df = df.dropna(axis=0)
        logger.info(f"Data preprocessed: {df.shape[0]} rows, {df.shape[1]} columns")
        # Categorical variable mapping dictionaries
        MAP_ROOM_TYPE = {"Shared room": 1, "Private room": 2, "Entire home/apt": 3, "Hotel room": 4}
        MAP_NEIGHB = {"Bronx": 1, "Queens": 2, "Staten Island": 3, "Brooklyn": 4, "Manhattan": 5}

        # Map categorical features
        df["neighbourhood"] = df["neighbourhood"].map(MAP_NEIGHB)
        df["room_type"] = df["room_type"].map(MAP_ROOM_TYPE)
        return df
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

def evaluate_model(model, X_test):
    """
    Evaluate the model and return predictions and feature importances.
    
    Args:
        model (RandomForestClassifier): Trained model.
        X_test (pd.DataFrame): Test features.
        
    Returns:
        tuple: Predictions, predicted probabilities, feature importances, feature names.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X_test.columns[indices]
    importances = importances[indices]
    
    return y_pred, y_pred_proba, features, importances

def log_metrics(y_test, y_pred, y_pred_proba, features, importances):
    """
    Log the evaluation metrics and feature importances.
    
    Args:
        y_test (pd.Series): Test labels.
        y_pred (np.ndarray): Predicted labels.
        y_pred_proba (np.ndarray): Predicted probabilities.
        features (pd.Index): Feature names.
        importances (np.ndarray): Feature importances.
    """
    for feature, importance in zip(features, importances):
        logger.info(f"Feature: {feature}, Importance: {importance}")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    logger.info(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba, multi_class='ovr')}")
    
    maps = {'0.0': 'low', '1.0': 'mid', '2.0': 'high', '3.0': 'lux'}
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info("Classification report generated successfully.")
    #BUG: Some issue formatting the report. Not urgent.
    # df_report = pd.DataFrame.from_dict(report).T[:-3]
    # df_report.index = [maps[i] for i in df_report.index]
    logger.info(f"Classification Report:\n{report}")

def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier model.
    
    Args:
        df (pd.DataFrame): Preprocessed data.
        
    Returns:
        RandomForestClassifier: Trained model.
    """
    try:
        FEATURE_NAMES = ['neighbourhood', 'room_type', 'accommodates', 'bathrooms', 'bedrooms']

        X = df[FEATURE_NAMES]
        logger.info(f"Feature columns: {X.columns.tolist()}")
        ten_values_df = X.head(10)
        logger.info(f"First ten rows of features:\n{ten_values_df.to_string(index=False)}")
        logger.info(f"Data types of the feature columns:\n{X.dtypes}")
        for row_idx, row in X.iterrows():
            for col_idx, value in enumerate(row):
                if "Essentials" in str(value):
                    logger.info(f"Found 'Essentials' at row {row_idx}, column {X.columns[col_idx]}")
        y = df['category']
        logger.info(f"First ten values of y: {y.head(10).tolist()}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
        
        model = RandomForestClassifier(n_estimators=500, random_state=0, class_weight='balanced', n_jobs=4)
        model.fit(X_train, y_train)
        
        y_pred, y_pred_proba, features, importances = evaluate_model(model, X_test)
        log_metrics(y_test, y_pred, y_pred_proba, features, importances)
        logger.info("Model training completed successfully.")
        return model
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise Exception from e

def save_model(model: RandomForestClassifier, filepath: Path):
    """
    Save the trained model to a file.
    
    Args:
        model (RandomForestClassifier): Trained model.
        filepath (Path): Path to save the model file.
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved successfully to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def main():
    """
    Main function to execute the data loading, preprocessing, model training, and saving steps.
    """
    try:
        df = load_data(FILEPATH_PROCESSED)
        df = preprocess_data(df)
        model = train_model(df)
        save_model(model, FILEPATH_MODEL)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    preprocess_raw_data()
    main()