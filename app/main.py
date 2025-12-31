"""Main FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import model_versions, inferences
from app.routers.utils import get_model_version, get_model, get_encoders

app = FastAPI(
    title="Workhuman Churn Prediction API",
    description="Workhuman Churn Prediction API",
    version="1.0.0",
    debug=settings.debug,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(model_versions.router)
app.include_router(inferences.router)


@app.get("/")
def read_root():
    """Root endpoint."""
    return {
        "message": "Welcome to Workhuman Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}