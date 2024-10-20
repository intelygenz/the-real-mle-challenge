from typing import Literal

from pydantic import BaseModel


class PredictionOutputSchema(BaseModel):
    id: int
    price_category: Literal["Low", "Mid", "High", "Lux"]
