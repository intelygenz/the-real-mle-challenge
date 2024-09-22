from pathlib import Path
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel

from inference.main import predict_price_category
from logger import logger

# Log the current directory
current_directory = Path.cwd()
logger.info(f"Current directory: {current_directory}")



# Define the input data model
class ListingInput(BaseModel):
    id: int
    accommodates: int
    room_type: str
    beds: int
    bedrooms: int
    bathrooms: int
    neighbourhood: str
    tv: int
    elevator: int
    internet: int
    latitude: float
    longitude: float

# Define the output data model
class ListingOutput(BaseModel):
    id: int
    price_category: str

# Create the FastAPI application
app = FastAPI()

MAP_ROOM_TYPE = {"Shared room": 1, "Private room": 2, "Entire home/apt": 3, "Hotel room": 4}
MAP_NEIGHB = {"Bronx": 1, "Queens": 2, "Staten Island": 3, "Brooklyn": 4, "Manhattan": 5}

# Define the prediction endpoint
@app.post("/predict", response_model=ListingOutput)
def do_prediction(listing: ListingInput):
    """
    Predicts the price category for a given listing.
    Args:
        listing (ListingInput): An instance of ListingInput containing the details of the listing.
    Returns:
        ListingOutput: An instance of ListingOutput containing the listing ID and the predicted price category.
    Raises:
        KeyError: If the neighbourhood or room type in the listing is not found in the respective mapping dictionaries.
    """

    # Convert input data to DataFrame
    data = {
        "neighbourhood": [MAP_NEIGHB[listing.neighbourhood]],
        "room_type": [MAP_ROOM_TYPE[listing.room_type]],
        "accommodates": [listing.accommodates],
        "bathrooms": [listing.bathrooms],
        "bedrooms": [listing.bedrooms],
        "beds": [listing.beds],
        "tv": [listing.tv],
        "elevator": [listing.elevator],
        "internet": [listing.internet],
        "latitude": [listing.latitude],
        "longitude": [listing.longitude]
    }
    

    # Make prediction
    
    price_category = predict_price_category(data)

    # Return the result
    return ListingOutput(id=listing.id, price_category=price_category)