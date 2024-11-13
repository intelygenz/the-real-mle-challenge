import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from utils import compute_feature_importance, compute_confusion_matrix, compute_classification_report

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Trains a Random Forest model on the provided training data.
    
    Parameters:
    - X_train: Features for training.
    - y_train: Labels for training.
    
    Returns:
    - model: Trained RandomForestClassifier model.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, output_path: str) -> None:
    """
    Evaluates the model on the test data and generates evaluation plots.
    
    Parameters:
    - model: Trained model to evaluate.
    - X_test: Features for testing.
    - y_test: Labels for testing.
    - output_dir: Directory to save evaluation plots.
    """
    y_pred = model.predict(X_test)

    # Compute overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("  - Accuracy:", format(accuracy, ".2f"))

    # Compute overall one-versus-rest area under the ROC
    y_proba = model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    print("  - ROC AUC:", format(roc_auc, ".2f"))

    # Compute Feature importance
    compute_feature_importance(model, X_test, output_path)

    # Compute confusion matrix
    compute_confusion_matrix(y_test, y_pred, output_path)

    # Compute classification report
    compute_classification_report(y_test, y_pred, output_path)

def save_model(model, filepath: str) -> None:
    """
    Saves the model to the specified filepath using pickle.
    
    Parameters:
    - model: The trained model to save.
    - filepath: Path to save the model (e.g., "models/model.pkl").
    
    Returns:
    - None
    """
    
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

def load_model(filepath: str) -> RandomForestClassifier:
    """
    Loads a model from the specified filepath using pickle.
    
    Parameters:
    - filepath: Path from which to load the model.
    
    Returns:
    - model: The loaded model.
    """
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model