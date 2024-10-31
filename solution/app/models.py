from pydantic import BaseModel, field_validator, conint, confloat, Field
from pydantic.functional_validators import AfterValidator
from enum import Enum
from typing import List, Union
from typing_extensions import Annotated


def validate_one_hot(self, value: Union[int, List[int]])-> Union[int, List[int]]:

    if isinstance(value, int):
        if value not in [0, 1]:
            raise ValueError("The input should be wither 1 or 0")
    if isinstance(value, list):
        if not all(map(lambda x: x in [0, 1], value)):
            raise ValueError("All inputs in the list should be wither 1 or 0")
    return value

OneZero = Annotated[Union[int, List[int]], AfterValidator(validate_one_hot)]


class RoomTypeEnum(str, Enum):
    shared_room = "Shared room"
    private_room = "Private room"
    entire_home_apt = "Entire home/apt"
    hotel_room = "Hotel room"

class NeighbourhoodEnum(str, Enum):
    bronx = "Bronx"
    queens = "Queens"
    staten_island = "Staten Island"
    brooklyn = "Brooklyn"
    manhattan = "Manhattan"


class ModelInput(BaseModel):
    id: Union[int, List[int]]
    accommodates: Union[conint(ge=0), List[conint(ge=0)]]
    room_type: Union[RoomTypeEnum, list[RoomTypeEnum]] 
    beds: Union[conint(ge=0), List[conint(ge=0)]] 
    bedrooms: Union[conint(ge=0), List[conint(ge=0)]] 
    bathrooms: Union[conint(ge=0), List[conint(ge=0)], confloat(ge=0), List[confloat(ge=0)]] 
    neighbourhood: Union[NeighbourhoodEnum, list[NeighbourhoodEnum]] 
    tv: OneZero 
    elevator: OneZero 
    internet: OneZero 
    latitude: Union[float, List[float]]
    longitude: Union[float, List[float]] 

    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "id": 1001,
                    "accommodates": 4,
                    "room_type": "Entire home/apt",
                    "beds": 2,
                    "bedrooms": 1,
                    "bathrooms": 2,
                    "neighbourhood": "Brooklyn",
                    "tv": 1,
                    "elevator": 1,
                    "internet": 0,
                    "latitude": 40.71383,
                    "longitude": -73.9658
                }
            ]
        }


class ModelOutput(BaseModel):
    id: Union[int, List[int]]
    price_category: Union[str, List[str]]
