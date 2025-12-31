from fastapi import APIRouter, Depends, HTTPException 
from app.schemas import ModelVersion
from app.models import ModelVersion as ModelVersionModel
from app.database import get_db
from sqlalchemy.orm import Session
from app.config import setup_logger

logger = setup_logger("model_versions")

router = APIRouter(prefix="/models", tags=["Model Versions"])

@router.get("/latest", response_model=ModelVersion)
async def get_latest_model_version(db: Session = Depends(get_db)):
    try:
        model = db.query(ModelVersionModel).order_by(ModelVersionModel.id.desc()).first()
        if not model:
            logger.info("No model found")
            return ModelVersion(version=0, model_metrics={})
        return model
    except HTTPException:
        logger.error("UNKNOWN: Failed to get latest model version")
        raise
    except Exception as e:
        logger.error(f"Failed to get latest model version: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create", response_model=ModelVersion)
async def create_model_version(model_version: ModelVersion, db: Session = Depends(get_db)):
    try:
        db_model_version = ModelVersionModel(**model_version.dict())
        db.add(db_model_version)
        db.commit()
        db.refresh(db_model_version)
        logger.info(f"Model version {db_model_version.id} created successfully")
        return db_model_version
    except Exception as e:
        logger.error(f"Failed to create model version: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
