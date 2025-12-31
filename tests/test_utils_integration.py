import pytest
import os
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db
from app.routers.utils import log_model_to_db, get_model_version
from app.config import settings

# Integration test using the local database on port 5431
# We override the DATABASE_URL to ensure it's pointing to 5431 (already done in config.py)

engine = create_engine(settings.database_url)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_database():
    # Ensure tables exist
    Base.metadata.create_all(bind=engine)
    yield

def mock_requests_post(url, json=None, **kwargs):
    """Mock requests.post to call TestClient instead."""
    path = url.replace(settings.CHURN_PRED_URL, "")
    response = client.post(path, json=json)
    # Wrap TestClient response into something that looks like requests response
    class MockResponse:
        def __init__(self, res):
            self.res = res
            self.status_code = res.status_code
        def json(self):
            return self.res.json()
    return MockResponse(response)

def mock_requests_get(url, **kwargs):
    """Mock requests.get to call TestClient instead."""
    path = url.replace(settings.CHURN_PRED_URL, "")
    response = client.get(path)
    class MockResponse:
        def __init__(self, res):
            self.res = res
            self.status_code = res.status_code
        def json(self):
            return self.res.json()
    return MockResponse(response)

@patch("app.routers.utils.post", side_effect=mock_requests_post)
@patch("app.routers.utils.get", side_effect=mock_requests_get)
def test_utils_with_db(mock_get, mock_post):
    # 1. Clear existing model versions to have a clean state for this test if needed
    # (Actually, we'll just use what's there and verify increment)
    
    # 2. Try to get current version
    try:
        initial_version = get_model_version()
    except Exception:
        initial_version = 0

    # 3. Log a new model
    new_version = initial_version + 1
    metrics = {"accuracy": 0.85, "f1": 0.82}
    result = log_model_to_db(new_version, metrics)
    
    assert result.version == new_version
    assert result.model_metrics == metrics

    # 4. Verify getting the latest version now returns the new one
    latest_version = get_model_version()
    assert latest_version == new_version
