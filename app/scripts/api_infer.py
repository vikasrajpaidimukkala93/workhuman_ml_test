from scipy.signal._signaltools import resample
from copyreg import pickle
from app.config import settings
from app.routers.utils import load_local_model, get_model_version
import pandas as pd
from requests import post
from app.config import setup_logger

logger = setup_logger(
    name="batch_inference",
)   

def infer_from_api(row):
    logger.info(f"Infering for customer {row.customer_id}")
    response = post(settings.CHURN_PRED_URL+ '/inferences/infer', json=row.to_dict()) 
    return response.json()
    

predict_churn_batch = pd.read_parquet(settings.LOCAL_FILE_PATH).head(10)

def main():
    for index, row in predict_churn_batch.iterrows():
        logger.info(f"Processing row {index}")
        logger.info(infer_from_api(row))

if __name__ == "__main__":
    main()



