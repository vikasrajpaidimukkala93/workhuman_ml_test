from pydantic import BaseModel

class ModelVersion(BaseModel):
    """
    ModelVersion schema class to keep track of model versions.
    """
    version: int
    model_metrics: dict
