import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.database import get_db
import pickle
import uuid
import numpy as np

client = TestClient(app)

# Mock data
mock_version = MagicMock()
mock_version.version = "v1.0.0"
mock_version.id = 1

mock_model = MagicMock()
mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

mock_encoder = MagicMock()
mock_encoder.transform.side_effect = lambda x: x

@patch("app.routers.inferences.get_model_version")
@patch("app.routers.inferences.get_model")
@patch("app.routers.inferences.get_encoders")
def test_predict_churn_success(mock_get_encoders, mock_get_model, mock_get_mv):
    # Setup mocks
    mock_get_mv.return_value = mock_version
    mock_get_model.return_value = {'model': mock_model}
    mock_get_encoders.return_value = {"plan_type": mock_encoder, "country": mock_encoder}
    
    # Mock DB
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    
    payload = {
        "customer_id": str(uuid.uuid4()),
        "tenure_months": 12,
        "num_logins_last_30d": 5,
        "num_tickets_last_90d": 1,
        "plan_type": "Gold",
        "country": "USA"
    }
    
    response = client.post("/inferences/infer", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0.8
    assert data["model_version"] == "v1.0.0"
    
    # Verify DB logging
    assert mock_db.add.called
    assert mock_db.commit.called
    
    # Cleanup
    app.dependency_overrides.clear()

@patch("app.routers.inferences.get_model_version")
@patch("app.routers.inferences.get_model")
def test_predict_churn_no_model(mock_get_model, mock_get_mv):
    mock_get_mv.return_value = mock_version
    mock_get_model.side_effect = Exception("Model not found")
    
    payload = {
        "customer_id": str(uuid.uuid4()),
        "tenure_months": 12,
        "num_logins_last_30d": 5,
        "num_tickets_last_90d": 1,
        "plan_type": "Gold",
        "country": "USA"
    }
    
    response = client.post("/inferences/infer", json=payload)
    assert response.status_code == 500
    assert "Model not found" in response.json()["detail"]
