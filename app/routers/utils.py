from requests import post, get
from fastapi import HTTPException
from app.config import settings
from app.schemas import ModelVersion

def log_model_to_db(model_version: int, model_metrics: object) -> ModelVersion:
    try:
        response = post(settings.CHURN_PRED_URL + "/models/create", json={"version": model_version, "model_metrics": model_metrics})
        return ModelVersion(**response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_model_version() -> int:
    try:
        response = get(settings.CHURN_PRED_URL + "/models/latest")
        model_version = ModelVersion(**response.json()).version
        return model_version
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))