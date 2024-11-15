"""
Main script to execute the ML pipeline, including data loading, preprocessing, 
training, evaluation, and model saving.

Usage:
  python main.py
"""

from pathlib import Path
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data_loader import load_raw_data, load_preprocessed_data
from preprocessing import preprocess_data
from model import train_model, evaluate_model, save_model

# Get current date and time as a string
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Directory configuration
DIR_REPO = Path.cwd().parent.parent  # Assuming this script is located two levels deep; verify if this is correct.
DIR_DATA_RAW = DIR_REPO / "data" / "raw"
DIR_DATA_PROCESSED = DIR_REPO / "data" / "processed"
FILEPATH_PROCESSED = DIR_DATA_PROCESSED / "preprocessed_listings.csv"
FILEPATH_DATA = DIR_DATA_RAW / "listings.csv"

DIR_MODELS = DIR_REPO / "models"
DIR_NEW_MODEL = DIR_MODELS / f"model_{timestamp}"
DIR_NEW_MODEL_PLOTS = DIR_NEW_MODEL / "plots"

# Ensure necessary directories exist
os.makedirs(DIR_DATA_PROCESSED, exist_ok=True)
os.makedirs(DIR_NEW_MODEL_PLOTS, exist_ok=True)


def main():
    print("Starting the ML pipeline...")

    try:
        # Load raw data
        print("Loading raw data...")
        df = load_raw_data(FILEPATH_DATA)

        # Preprocess data and save
        print("Preprocessing data...")
        df = preprocess_data(df, FILEPATH_PROCESSED)

        # Load model data
        print("Loading preprocessed data...")
        X, y = load_preprocessed_data(FILEPATH_PROCESSED)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)

        # Train the model
        print("Training model...")
        model = train_model(X_train, y_train)

        # Evaluate the model and save evaluation plots
        print("Evaluating model...")
        evaluate_model(model, X_test, y_test, DIR_NEW_MODEL_PLOTS)

        # Save the trained model
        print("Saving model...")
        save_model(model, DIR_NEW_MODEL / "model.pkl")

    except Exception as e:
        print("An error occurred during the pipeline execution: %s", e)

if __name__ == "__main__":
    main()