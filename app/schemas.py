from pydantic import BaseModel

class ModelVersion(BaseModel):
    version: str
    model_metrics: dict
