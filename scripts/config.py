import os
import logging
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Environment: DEV or PRD
ENV = os.getenv("ENV", "DEV").upper()

# Local storage paths
LOCAL_DATA_DIR = "data"
LOCAL_FILE_PATH = os.path.join(LOCAL_DATA_DIR, "churn_data.parquet")

# AWS / PRD storage paths
S3_BUCKET = os.getenv("S3_BUCKET", "workhuman-churn-data-prd")
S3_KEY = "data/churn_data.parquet"
CLOUDWATCH_LOG_GROUP = os.getenv("CLOUDWATCH_LOG_GROUP", "workhuman-ml-logs")

def get_storage_path():
    if ENV == "PRD":
        return f"s3://{S3_BUCKET}/{S3_KEY}"
    else:
        # Ensure local directory exists
        os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
        return LOCAL_FILE_PATH

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if ENV == "PRD":
        import boto3
        import watchtower
        
        # Try to use CloudWatch
        try:
            cw_handler = watchtower.CloudWatchLogHandler(
                log_group=CLOUDWATCH_LOG_GROUP,
                stream_name=f"{name}-{ENV}",
                boto3_client=boto3.client("logs", region_name=os.getenv("AWS_REGION", "us-east-1"))
            )
            cw_handler.setFormatter(formatter)
            logger.addHandler(cw_handler)
        except Exception as e:
            # no fall back as we will need to see logs in console for PRD
            logger.error(f"Failed to initialize CloudWatch logging, falling back to console: {e}")
    else:
        # DEV environment: Log to console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger
