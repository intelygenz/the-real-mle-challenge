from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
from utils import preprocess_listing_input
from pydantic_models import ListingInput, ListingOutput
import yaml

# Load configuration from YAML file
def load_config():
    with open("./api/config.yaml", "r") as f:
        return yaml.safe_load(f)
    
config = load_config()
print("Loaded configuration", config)

# Load the trained model
with open(config['model']['path'], "rb") as f:
    model = pickle.load(f)
    print("Loaded model:", model)

# Create FastAPI app instance
app = FastAPI()

# Endpoint to make predictions
@app.post("/predict")
def predict_price_category(listing: ListingInput):
    """
    Predict the price category of a property listing.

    Parameters:
    - listing (ListingInput): Input data containing property details for prediction.

    Returns:
    - ListingOutput: Response with property 'id' and predicted 'price_category'.
    """

    # Preprocess input data for the model
    # model_input_data = preprocess_listing_input(listing.dict())
    model_input_data = preprocess_listing_input(listing.model_dump())

    try:
        # Make the prediction
        category_num = model.predict(model_input_data)[0]  # Get the first prediction
        
        # Map numeric categories to text labels
        category_map = {0: "Low", 1: "Mid", 2: "High", 3: "Luxury"}
        price_category = category_map.get(category_num, "Unknown")

        # Return the result in the expected format
        return ListingOutput(id=listing.id, price_category=price_category)
    
    except Exception as e:
        # Raise an HTTP exception if an error occurs during prediction
        raise HTTPException(status_code=500, detail=str(e))