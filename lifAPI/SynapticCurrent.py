
from typing import List

from pydantic import BaseModel

class Current(BaseModel):
    mean: float
    variance: float


class NeuronalResponse(BaseModel):
    grid: List[float]
    probabilities: List[float]
    fluxes: List[float]
