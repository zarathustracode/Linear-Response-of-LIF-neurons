
import string
from typing import List

from pydantic import BaseModel, Field

class Current(BaseModel):
    mean: float = Field(gt=-10, le=10, default=1)
    variance: float = Field(ge=1, le=5, default=1)
