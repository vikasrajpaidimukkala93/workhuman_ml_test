from fastapi import APIRouter, Depends, HTTPException 
from app.schemas import ModelVersionSchema
from app.models import ModelVersion
from app.database import get_db
from sqlalchemy.orm import Session
from app.config import setup_logger
from app.routers.utils import model_version_cache
import json

logger = setup_logger("model_versions")

router = APIRouter(prefix="/models", tags=["Model Versions"])

@router.get("/latest", response_model=ModelVersionSchema)
async def get_latest_model_version(db: Session = Depends(get_db)):
    try:
        model = model_version_cache.get("latest_model_version")
        if model:
            return model
        model = db.query(ModelVersion).filter(ModelVersion.is_active == True).order_by(ModelVersion.created_at.desc()).first()
        if not model:
            logger.info("No model found")
            return ModelVersionSchema(version='', model_metrics={})
        model_version_cache.set("latest_model_version", model)
        return model    
    except HTTPException:
        logger.error("UNKNOWN: Failed to get latest model version")
        raise
    except Exception as e:
        logger.error(f"Failed to get latest model version: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create", response_model=ModelVersionSchema)
async def create_model_version(model_version: ModelVersionSchema, db: Session = Depends(get_db)):
    try:
        db_model_version = ModelVersion(**model_version.dict())
        db.add(db_model_version)
        db.commit()
        db.refresh(db_model_version)
        logger.info(f"Model version {db_model_version.id} created successfully")
        model_version_cache.delete("latest_model_version")
        return db_model_version
    except Exception as e:
        logger.error(f"Failed to create model version: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
