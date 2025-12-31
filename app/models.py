import uuid
from datetime import datetime
from sqlalchemy import Column, Integer, DateTime, JSON, Float, ForeignKey, Index, String, Boolean
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base       


class ModelVersion(Base):
    """
    ModelVersion to keep track of model versions.
    """
    __tablename__ = "model_versions"
    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid.uuid4)
    version = Column(String, unique=True, index=True)
    model_metrics = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Predictions(Base):
    """
    Predictions to keep track of predictions.
    """
    __tablename__ = "predictions"
    id = Column(UUID(as_uuid=True), primary_key=True, index=True, default=uuid.uuid4)
    model_version_id = Column(String, ForeignKey("model_versions.version"), nullable=False)
    prediction = Column(Float, nullable=False)
    customer_id = Column(UUID(as_uuid=True), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    idx_customer_id = Index("idx_customer_id", customer_id)
    idx_model_version_id = Index("idx_model_version_id", model_version_id)
    idx_created_at = Index("idx_created_at", created_at)