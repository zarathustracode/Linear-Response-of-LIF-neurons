
import string
from typing import List

from pydantic import BaseModel, Field

class Current(BaseModel):
    mean: float = Field(gt=-2, le=30, default=20)
    variance: float = Field(ge=1, le=5, default=5)
