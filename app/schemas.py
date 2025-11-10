from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

# Схема для вхідних даних (використовується в /predict)
class InferenceInputSchema(BaseModel):
    Gender: str = Field(..., example="Female")
    Age: float = Field(..., gt=0, example=21.0)
    Height: float = Field(..., gt=0, example=1.62)
    Weight: float = Field(..., gt=0, example=64.0)
    FAVC: str = Field(..., example="yes")

# Схема для відповіді
class PredictionResponseSchema(BaseModel):
    prediction: str
    log_id: int
    source: str