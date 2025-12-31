from fastapi import APIRouter, HTTPException, Depends
from app.models import Predictions, ModelVersion as ModelVersionModel
from app.config import setup_logger
from app.database import get_db
from sqlalchemy.orm import Session
from app.schemas import (
    PredictionInput, 
    PredictionResult, 
    BatchPredictionRequest, 
    BatchPredictionResponse,
    CreatePrediction    
)
import pickle
import json
import pandas as pd
from app.routers.utils import get_model_version, get_model, get_encoders
import os

logger = setup_logger(
    name="inferences",
)

router = APIRouter(prefix='/inferences', tags=['Inferences'])

@router.post("/infer", response_model=PredictionResult) 
def predict_churn(input_data: PredictionInput, db: Session = Depends(get_db)):
        """
        Perform churn prediction for a single customer.
        """
        try:
            # Load model and version
            model_info = get_model_version()

            # Ensure model_info has attributes we need
            version_id = model_info.version if hasattr(model_info, 'version') else model_info['version']
            logger.info(f"Version ID: {version_id}")

            try:
                model = get_model(model_version_id=version_id)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            
            try:
                encoders = get_encoders(model_version_id=version_id)
                logger.info("Encoders loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load encoders: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            
            # Prepare input features
            input_dict = input_data.model_dump()
            X_df = pd.DataFrame([input_dict]).drop('customer_id', axis=1)
            
            # Apply encoding
            for col, encoder in encoders.items():
                if col in X_df.columns:
                    # Handle unseen data by mapping to the first class in the encoder
                    X_df[col] = X_df[col].map(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
                    X_df[col] = encoder.transform(X_df[col])
            
            # Prediction
            prediction_prob = float(model['model'].predict_proba(X_df)[:, 1][0])
            
            # Log to database
            db_prediction = Predictions(
                customer_id=input_data.customer_id,
                model_version_id=version_id,
                prediction=prediction_prob
            )
            db.add(db_prediction)
            db.commit()
            db.refresh(db_prediction)
            
            logger.info(f"Prediction for customer {input_data.customer_id}: {prediction_prob:.4f}")
            
            return PredictionResult(
                customer_id=input_data.customer_id,
                prediction=prediction_prob,
                model_version=str(version_id)
            )
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
   