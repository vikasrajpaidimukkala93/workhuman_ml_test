# Workhuman Churn Prediction API

This project provides a FastAPI-based service for tracking machine learning model versions and their metrics. It also includes synthetic data generation and model training scripts.

## Core Components

- **FastAPI Application**: REST API to log and retrieve model versions.
- **SQLAlchemy/Alembic**: Database ORM and automated migrations.
- **ML Scripts**: Tools for synthetic data generation (`generate_data.py`) and model training (`train.py`).
- **PostgreSQL**: Local database storage for model metadata.
- **uv**: Next-generation Python package manager for fast builds.

---

## ⚡ Quick Start with Docker

The easiest way to get the application running is using Docker and Docker Compose. This path automatically handles the database setup, environment configuration, and database migrations.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Deployment Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd workhuman_ml_test
   ```

2. **Build and Start the services**:
   ```bash
   docker compose up --build
   ```

   This command will:
   - Pull the latest PostgreSQL image.
   - Build the FastAPI application image using `uv`.
   - Start the database on port `5431`.
   - Wait for the database to be ready.
   - **Automatically run Alembic migrations** to set up the schema.
   - Start the API server on `http://localhost:8000`.

3. **Verify the installation**:
   - API Health Check: [http://localhost:8000/health](http://localhost:8000/health)
   - Interactive API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🛠 Usage & Development

### Working with the ML Scripts

The project includes scripts for the full ML lifecycle: data generation, training, and inference.

#### 1. Setup local environment
```bash
uv sync
```

#### 2. Generate Synthetic Data
```bash
python -m app.scripts.generate_data --samples 1000
```
*Saves a parquet file to the `data/` directory.*

#### 3. Train Model
```bash
python -m app.scripts.train --data-path data/churn_data.parquet
```
*Trains a RandomForest model, saves artifacts to `ml_models/`, and logs version metadata to the API.*

#### 4. Batch Inference (Local)
Run inference on a batch of data using the latest local model artifacts:
```bash
python -m app.scripts.batch_infer
```
*Reads input from `data/churn_data.parquet` and saves predictions to `data/churn_predictions.parquet`.*

#### 5. API Inference Client
Test the API's prediction endpoint using sample data:
```bash
python -m app.scripts.api_infer
```
*Sends requests to `http://localhost:8000/inferences/infer` and logs the responses.*

### Database Migrations

Migrations are automated in Docker. If you need to run them manually:
```bash
alembic upgrade head
```

---

## 🏗 Architecture

- **FastAPI**: Serves the REST API.
- **PostgreSQL**: Stores model metadata (`model_versions`) and inference logs (`predictions`).
- **Valkey (Redis)**: Available for caching, though currently disabled for direct model loading in inference.

---

## 🧪 Testing

Run the test suite using `pytest`:

```bash
# Unit tests (Services & Inference Logic)
uv run pytest tests/test_ml_service.py
uv run pytest tests/test_inferences.py

# Integration tests
uv run pytest tests/test_utils_integration.py

# Cache tests
uv run pytest tests/test_cache.py
```