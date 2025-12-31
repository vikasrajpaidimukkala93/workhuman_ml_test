import joblib
from PIL.Image import logger
import json
from typing import Dict, Any, Optional
from requests import post, get
from fastapi import HTTPException
from app.config import settings
from app.schemas import ModelVersionSchema
import time
import redis
from app.config import settings
import boto3
import pickle

def log_model_to_db(model_version: str, model_metrics: Any) -> ModelVersionSchema:
    """
    Log model to database

    args:
        model_version: str
        model_metrics: json

    returns:
        ModelVersionSchema
    """
    try:
        response = post(settings.CHURN_PRED_URL + "/models/create", json={"version": model_version, "model_metrics": model_metrics})
        print(response.json())
        return ModelVersionSchema(**response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_model_version() -> ModelVersionSchema:
    """
    Get latest model version from database

    returns:
        ModelVersionSchema
    """
    try:
        response = get(settings.CHURN_PRED_URL + "/models/latest")
        model_version = ModelVersionSchema(**response.json())
        return model_version
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SimpleTTLCache:
    """Simple in-memory TTL cache for DEV environment."""
    def __init__(self, ttl=300):
        self.cache = {}
        self.ttl = ttl

    def get(self, key):
        if key in self.cache:
            val, expiry = self.cache[key]
            if expiry > time.time():
                return val
            del self.cache[key]
        return None

    def set(self, key, value):
        self.cache[key] = (value, time.time() + self.ttl)
    
    def delete(self, key):
        if key in self.cache:
            del self.cache[key]

class ValkeyCache:
    """Valkey-backed cache for PRD environment."""
    def __init__(self, host, port, ttl=300):
        # decode_responses=False to allow handling bytes (models/encoders)
        self.client = redis.Redis(host=host, port=port, db=0, decode_responses=False)
        self.ttl = ttl

    def get(self, key):
        try:
            val = self.client.get(key)
            if val is None:
                return None
            # Try parsing as JSON (for metadata)
            try:
                return json.loads(val)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # If not JSON, return unprocessed bytes (for binary artifacts)
                return val
        except Exception:
            return None

    def set(self, key, value):
        try:
            # If bytes/bytearray, store directly
            if isinstance(value, (bytes, bytearray)):
                self.client.set(key, value, ex=self.ttl)
            # Handle Pydantic models
            elif hasattr(value, "model_dump"):
                self.client.set(key, json.dumps(value.model_dump()), ex=self.ttl)
            elif hasattr(value, "dict"):
                self.client.set(key, json.dumps(value.dict()), ex=self.ttl)
            else:
                # Default to JSON for dicts, lists, strings
                self.client.set(key, json.dumps(value), ex=self.ttl)
        except Exception:
            pass
            
    def delete(self, key):
        try:
            self.client.delete(key)
        except Exception:
            pass

# Initialize cache based on environment
if settings.ENV == "PRD":
    model_version_cache = ValkeyCache(settings.VALKEY_HOST, settings.VALKEY_PORT)
else:
    model_version_cache = SimpleTTLCache()

def load_local_model(model_version_id):
    try:
        with open(f"{settings.MODEL_ARTIFACTS_DIR}/{model_version_id}/model.pkl", "rb") as f:
            logger.info(f"Loading model from {settings.MODEL_ARTIFACTS_DIR}/{model_version_id}/model.pkl")
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

def load_local_encoders(model_version_id):
    try:
        with open(f"{settings.MODEL_ARTIFACTS_DIR}/{model_version_id}/encoders.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load encoders: {e}")

def get_model(model_version_id: Optional[str] = None):
    try:
        if settings.ENV == "PRD":
            s3 = boto3.client("s3", aws_access_key_id=settings.AWS_ACCESS_KEY_ID, aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY, region_name=settings.AWS_REGION)
            model_info = get_model_version()
            response = s3.get_object(Bucket=settings.S3_BUCKET, Key=f"models/{model_info.version}/model.pkl")
            model_bytes = response["Body"].read()
            # Store raw bytes in cache
            model_version_cache.set("latest_model", model_bytes)
            # Return deserialized object
            return pickle.loads(model_bytes)
        else:
            model = load_local_model(model_version_id)
            # # Store serialized bytes in cache
            # model_version_cache.set("latest_model", pickle.dumps(model))
            return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_encoders(model_version_id: str):
    try:
        encoders_bytes = model_version_cache.get("latest_encoders")
        if encoders_bytes:
            return pickle.loads(encoders_bytes)
        else:
            if settings.ENV == "PRD":
                s3 = boto3.client("s3", aws_access_key_id=settings.AWS_ACCESS_KEY_ID, aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY, region_name=settings.AWS_REGION)
                model_info = get_model_version()
                response = s3.get_object(Bucket=settings.S3_BUCKET, Key=f"models/{model_info.version}/encoders.pkl")
                encoders_bytes = response["Body"].read()
                model_version_cache.set("latest_encoders", encoders_bytes)
                return pickle.loads(encoders_bytes)    
            else:
                encoders = load_local_encoders(model_version_id)
                # model_version_cache.set("latest_encoders", pickle.dumps(encoders))
                return encoders
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        