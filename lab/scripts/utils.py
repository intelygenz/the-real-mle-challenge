import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def preprocess_amenities_column(df: pd.DataFrame) -> pd.DataFrame:

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

def num_bathroom_from_text(text):
    # Get number of bathrooms from `bathrooms_text`
    try:
        if isinstance(text, str):
            bath_num = text.split(" ")[0]
            return float(bath_num)
        else:
            return np.nan
    except ValueError:
        return np.nan
    
def compute_feature_importance(model, X_test, output_path):
    """
    Creates and saves a bar plot of feature importances for a given classifier.
    
    Parameters:
    - model: A trained classifier with a `feature_importances_` attribute (e.g., RandomForestClassifier).
    - X_test: DataFrame containing the test features. The column names will be used as feature names.
    - output_path: Path to save the plot image.
    
    Returns:
    - None
    """
    
    # Extract and sort feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X_test.columns[indices] # Feature names
    sorted_importances = importances[indices]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.barh(range(len(sorted_importances)), sorted_importances)
    plt.yticks(range(len(sorted_importances)), features, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance", fontsize=12)
    
    # Ensure directory exists
    os.makedirs(output_path, exist_ok=True)

    # Save the plot
    plt.savefig(Path(output_path) / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)  # Close the plot to free memory

def compute_confusion_matrix(y_test, y_pred, output_path):
    """
    Creates and saves a normalized confusion matrix plot.
    
    Parameters:
    - y_test: Array of true labels.
    - y_pred: Array of predicted labels.
    - output_path: Path to save the plot image (default is "confusion_matrix.png").
    
    Returns:
    - None
    """

    classes = [0, 1, 2, 3]
    labels = ['low', 'mid', 'high', 'lux']

    # Compute confusion matrix and normalize
    c = confusion_matrix(y_test, y_pred)
    c = c / c.sum(axis=1).reshape(len(classes), 1)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(c, annot=True, cmap='BuGn', square=True, fmt='.2f', annot_kws={'size': 10}, cbar=False)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('Real', fontsize=16)
    plt.xticks(ticks=np.arange(.5, len(classes)), labels=labels, rotation=0, fontsize=12)
    plt.yticks(ticks=np.arange(.5, len(classes)), labels=labels, rotation=0, fontsize=12)
    plt.title("Simple model", fontsize=18)
    
    # Ensure directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Save the plot
    plt.savefig(Path(output_path) / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)  # Close the plot to free memory

def compute_classification_report(y_test, y_pred, output_path):
    """
    Creates a DataFrame from the classification report, applies label mappings, and saves it to a CSV file.
    Also saves the a classification report plot.
    
    Parameters:
    - y_test: Array of true labels.
    - y_pred: Array of predicted labels.
    - maps: Dictionary mapping numerical labels to descriptive labels.
    - output_path: Path to save the CSV file (default is "classification_report.csv").
    
    Returns:
    - None
    """

    maps = {'0.0': 'low', '1.0': 'mid', '2.0': 'high', '3.0': 'lux'}

    report = classification_report(y_test, y_pred, output_dict=True)    
    df_report = pd.DataFrame.from_dict(report).T[:-3]
    df_report.index = [maps.get(str(i), i) for i in df_report.index]
    df_report.to_csv(Path(output_path)/"classification_report.csv")

    # Compute and save the classification report plot
    metrics = ['precision', 'recall', 'support']

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 7))

    for i, ax in enumerate(axes):

        ax.barh(df_report.index, df_report[metrics[i]], alpha=0.9)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlabel(metrics[i], fontsize=12)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle("Simple model", fontsize=14)
    plt.savefig(Path(output_path) / "classification_report.png", bbox_inches="tight")
