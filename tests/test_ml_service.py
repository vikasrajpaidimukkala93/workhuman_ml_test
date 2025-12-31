import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from app.main import app
from app.database import Base, get_db

from app.config import settings

# Use local PostgreSQL database
SQLALCHEMY_DATABASE_URL = settings.database_url

engine = create_engine(SQLALCHEMY_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    # Clean up potentially existing test data
    db = TestingSessionLocal()
    from app.models import ModelVersion, Predictions
    db.query(Predictions).delete()
    db.query(ModelVersion).delete()
    db.commit()
    db.close()
    yield

def test_get_latest_model_empty():
    response = client.get("/models/latest")
    assert response.status_code == 200
    assert response.json()["version"] == ""

def test_create_and_get_latest_model():
    # 1. Create a model version
    model_data = {
        "version": "v1.0.0",
        "model_metrics": {"accuracy": 0.95}
    }
    response = client.post("/models/create", json=model_data)
    assert response.status_code == 200
    
    # 2. Get latest
    response = client.get("/models/latest")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "v1.0.0"
    assert data["model_metrics"] == {"accuracy": 0.95}
    
    # 3. Create a newer version
    model_data_2 = {
        "version": "v1.1.0",
        "model_metrics": {"accuracy": 0.97}
    }
    client.post("/models/create", json=model_data_2)
    
    # 4. Get latest again
    response = client.get("/models/latest")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "v1.1.0"
    assert data["model_metrics"] == {"accuracy": 0.97}
