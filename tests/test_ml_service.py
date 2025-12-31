import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from app.main import app
from app.database import Base, get_db

# Use a separate test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_api.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
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
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test_api.db"):
        os.remove("./test_api.db")

def test_get_latest_model_empty():
    response = client.get("/models/latest")
    assert response.status_code == 200

def test_create_and_get_latest_model():
    # 1. Create a model version
    model_data = {
        "version": 1,
        "model_metrics": {"accuracy": 0.95}
    }
    response = client.post("/models/create", json=model_data)
    assert response.status_code == 200
    
    # 2. Get latest
    response = client.get("/models/latest")
    assert response.status_code == 200
    assert response.json() == {"version": 1, "model_metrics": {"accuracy": 0.95}}
    
    # 3. Create a newer version
    model_data_2 = {
        "version": 2,
        "model_metrics": {"accuracy": 0.97}
    }
    response = client.post("/models/create", json=model_data_2)
    assert response.status_code == 200
    
    # 4. Get latest again
    response = client.get("/models/latest")
    assert response.status_code == 200
    assert response.json() == {"version": 2, "model_metrics": {"accuracy": 0.97}}
