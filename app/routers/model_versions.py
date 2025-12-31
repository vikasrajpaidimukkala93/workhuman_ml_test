from fastapi import APIRouter, Depends, HTTPException 
from app.schemas import ModelVersion
from app.database import get_db
from sqlalchemy.orm import Session

router = APIRouter(prefix="/models", tags=["Model Versions"])

@router.get("/latest", response_model=ModelVersion)
async def get_latest_model_version(db: Session = Depends(get_db)):
    try:
        return db.query(ModelVersion).order_by(ModelVersion.id.desc()).first()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create", response_model=ModelVersion)
async def create_model_version(model_version: ModelVersion, db: Session = Depends(get_db)):
    try:
        db_model_version = ModelVersion(**model_version.dict())
        db.add(db_model_version)
        db.commit()
        db.refresh(db_model_version)
        return db_model_version
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
