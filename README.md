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

The project includes scripts to generate data and train models. To run these *inside* the project context using your local environment (assuming you have `uv` installed):

#### 1. Setup local environment
```bash
uv sync
```

#### 2. Generate Synthetic Data
```bash
python -m app.scripts.generate_data --samples 1000
```
*This saves a parquet file to the `data/` directory.*

#### 3. Train Model and Log to DB
```bash
python -m app.scripts.train --data-path data/churn_data.parquet
```
*This trains a RandomForest model, saves artifacts to `ml_models/`, and logs the version to the API.*

### Database Migrations

Migrations are automated in Docker. If you need to run them manually:
```bash
alembic upgrade head
```

---

## 📁 Project Structure

```text
.
├── app/
│   ├── main.py            # API Entry point
│   ├── models.py          # SQLAlchemy models
│   ├── schemas.py         # Pydantic models
│   ├── config.py          # Environment settings
│   ├── routers/           # API Endpoints
│   └── scripts/           # ML Data/Training scripts
├── alembic/               # Database migration scripts
├── tests/                 # Unit and Integration tests
├── Dockerfile             # Container definition
├── docker-compose.yml     # Service orchestration
├── pyproject.toml         # Dependencies
└── uv.lock                # Deterministic lockfile
```

## 🧪 Testing

Run the test suite using `pytest`:
```bash
# Unit tests
pytest tests/test_ml_service.py

# Integration tests (requires DB running on 5431)
pytest tests/test_utils_integration.py
```