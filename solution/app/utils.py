import os
from pathlib import Path
import mlflow


mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

def load_model():

    try:
        return mlflow.sklearn.load_model("models:/price_category_clf@prod")
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None
    
def load_pipeline():

    try:
        return mlflow.pyfunc.load_model("models:/processing_pipeline@prod")
    except Exception as e:
        print(f"Error loading the pipeline: {e}")
        return None


def load_transformer():

    try:
        return mlflow.pyfunc.load_model("models:/mapping_transformer@prod")
    except Exception as e:
        print(f"Error loading the transformer: {e}")
        return None