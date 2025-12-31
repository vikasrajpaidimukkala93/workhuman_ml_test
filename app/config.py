import os
import logging
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env file        
load_dotenv()

class Settings(BaseSettings):
    app_name: str = "Workhuman Churn Prediction API"
    debug: bool = False
    
    # Environment: DEV or PRD
    ENV: str = os.getenv("ENV", "DEV").upper()
    
    # Database configuration
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres")
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "5432")
    DB_NAME: str = os.getenv("DB_NAME", "mydb")
    
    # Storage configuration
    LOCAL_DATA_DIR: str = "data"
    LOCAL_FILE_PATH: str = os.path.join("data", "churn_data.parquet")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "workhuman-churn-data-prd")
    S3_KEY: str = "data/churn_data.parquet"
    CLOUDWATCH_LOG_GROUP: str = os.getenv("CLOUDWATCH_LOG_GROUP", "workhuman-ml-logs")

    @property
    def database_url(self) -> str:
        if self.ENV == "PRD":
            return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        return "postgresql://postgres:postgres@localhost:5432/mydb"

    @property
    def storage_path(self) -> str:
        if self.ENV == "PRD":
            return f"s3://{self.S3_BUCKET}/{self.S3_KEY}"
        else:
            # Ensure local directory exists
            os.makedirs(self.LOCAL_DATA_DIR, exist_ok=True)
            return self.LOCAL_FILE_PATH

    def get_logger(self, name: str):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if self.ENV == "PRD":
            import boto3
            import watchtower
            
            # Try to use CloudWatch
            try:
                cw_handler = watchtower.CloudWatchLogHandler(
                    log_group=self.CLOUDWATCH_LOG_GROUP,
                    stream_name=f"{name}-{self.ENV}",
                    boto3_client=boto3.client("logs", region_name=os.getenv("AWS_REGION", "us-east-1"))
                )
                cw_handler.setFormatter(formatter)
                logger.addHandler(cw_handler)
            except Exception as e:
                # fall back to console for logs
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
                logger.error(f"Failed to initialize CloudWatch logging: {e}")
        else:
            # DEV environment: Log to console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger

settings = Settings()

# Alias for backward compatibility if needed, or just use settings.get_logger
def setup_logger(name):
    return settings.get_logger(name)
