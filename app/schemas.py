from uuid import UUID
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class ModelVersionSchema(BaseModel):
    """
    ModelVersion schema class to keep track of model versions.
    """
    id: Optional[UUID] = None
    version: str
    is_active: Optional[bool] = True
    model_metrics: Dict[str, Any]

class PredictionInput(BaseModel):
    customer_id: UUID
    tenure_months: int
    num_logins_last_30d: int
    num_tickets_last_90d: int
    plan_type: str
    country: str

class PredictionResult(BaseModel):
    customer_id: UUID
    prediction: float
    model_version: str

class BatchPredictionRequest(BaseModel):
    inputs: List[PredictionInput]

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResult]

class CreatePrediction(BaseModel):
    """
    Prediction schema class to keep track of predictions.
    """
    customer_id: int
    model_version_id: int
    prediction: float