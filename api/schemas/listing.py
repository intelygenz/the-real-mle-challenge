from pydantic import BaseModel


class ListingSchema(BaseModel):
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
