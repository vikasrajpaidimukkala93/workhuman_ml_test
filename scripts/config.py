import os
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# Environment: DEV or PRD
ENV = os.getenv("ENV").upper()

# Local storage paths
LOCAL_DATA_DIR = "data"
LOCAL_FILE_PATH = os.path.join(LOCAL_DATA_DIR, "churn_data.parquet")

# AWS / PRD storage paths
S3_BUCKET = os.getenv("S3_BUCKET", "workhuman-churn-data-prd")
S3_KEY = "data/churn_data.parquet"

def get_storage_path():
    if ENV == "PRD":
        return f"s3://{S3_BUCKET}/{S3_KEY}"
    else:
        # Ensure local directory exists
        os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
        return LOCAL_FILE_PATH
