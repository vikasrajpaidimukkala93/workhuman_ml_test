from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, JSON
from app.database import Base       


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True, index=True)
    model_metrics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)