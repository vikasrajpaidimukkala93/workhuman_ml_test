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
    assert response.status_code == 404
    assert response.json()["detail"] == "No models found"

def test_create_and_get_latest_model():
    # 1. Create a model version
    model_data = {
        "version_id": "v1.0.0",
        "metadata_json": {"accuracy": 0.95}
    }
    response = client.post("/models/", json=model_data)
    assert response.status_code == 200
    
    # 2. Get latest
    response = client.get("/models/latest")
    assert response.status_code == 200
    assert response.json() == {"version_id": "v1.0.0"}
    
    # 3. Create a newer version
    model_data_2 = {
        "version_id": "v1.1.0",
        "metadata_json": {"accuracy": 0.97}
    }
    client.post("/models/", json=model_data_2)
    
    # 4. Get latest again
    response = client.get("/models/latest")
    assert response.status_code == 200
    assert response.json() == {"version_id": "v1.1.0"}
