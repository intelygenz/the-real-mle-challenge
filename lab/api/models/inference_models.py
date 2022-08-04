"""
This file contains all modesl used on inference petition
"""

from pydantic import BaseModel


class InferenceRequest(BaseModel):
    """
    Request received for inference endpoint
    """

    id: int
    accommodates: int
    room_type: str
    beds: int
    bedrooms: int
    bathrooms: float
    neighbourhood: str
    tv: int
    elevator: int
    internet: int
    latitude: float
    longitude: float


class InferenceResponse(BaseModel):
    """
    Response returned for inference endpoint
    """

    id: int
    price_category: str
