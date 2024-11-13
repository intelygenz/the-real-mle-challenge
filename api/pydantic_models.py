from pydantic import BaseModel
from typing import Literal

# Define input model with Pydantic
class ListingInput(BaseModel):
    """Pydantic model to validate input data for a property listing."""
    id: int
    accommodates: int
    room_type: Literal["Shared room", "Private room", "Entire home/apt", "Hotel room"]
    beds: int
    bedrooms: int
    bathrooms: int
    neighbourhood: Literal["Bronx", "Queens", "Staten Island", "Brooklyn", "Manhattan"]
    tv: int
    elevator: int
    internet: int
    latitude: float
    longitude: float

# Define output model with Pydantic
class ListingOutput(BaseModel):
    """Pydantic model to structure the output response for price category prediction."""
    id: int
    price_category: Literal["Low", "Mid", "High", "Luxury"]