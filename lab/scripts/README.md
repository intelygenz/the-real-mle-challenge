# Property Price Prediction Pipeline

This project provides a pipeline to predict property price categories using machine learning. The main components are located in the `scripts` folder.

## `scripts/` Folder Overview

- **`data_loader.py`**: Loads raw and preprocessed data.
- **`main.py`**: Runs the end-to-end pipeline, including data loading, preprocessing, model training, evaluation, and saving.
- **`model.py`**: Handles model training, evaluation, saving, and loading.
- **`preprocessing.py`**: Cleans and transforms raw data for modeling.
- **`utils.py`**: Utility functions for processing amenities, extracting features, and generating evaluation plots.

## How to Run

1. **Install Dependencies**:

    ```bash
    pip install -r requirements-dev.txt
    ```

2. **Run the Pipeline**:

    ```bash
    python lab/scripts/main.py
    ```

The trained model and evaluation results will be saved in a timestamped folder under `models`.