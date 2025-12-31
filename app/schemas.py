from pydantic import BaseModel

class ModelVersion(BaseModel):
    version: int
    model_metrics: dict
