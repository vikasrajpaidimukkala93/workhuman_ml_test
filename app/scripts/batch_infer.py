from copyreg import pickle
from app.config import settings
from app.routers.utils import load_local_model, get_model_version
import pandas as pd
import pickle
from app.config import setup_logger

logger = setup_logger(
    name="batch_inference",
)   

predict_churn_batch = pd.read_parquet(settings.PREDICT_CHURN_BATCH)
model_version = get_model_version()
model = load_local_model(model_version.version)


import joblib
import os
def load_model_artifacts(version: str):
    """
    Loads the model and its associated encoders for a specific version.

    Args:
        version (str): The version of the model to load.

    Returns:
        tuple: A tuple containing (model, encoders).
    """
    model = load_local_model(version)
    encoders_path = os.path.join(settings.MODEL_ARTIFACTS_DIR, version, "encoders.pkl")
    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)
    logger.info(f"Loaded model and encoders for version: {version}")
    return model, encoders


def preprocess_features(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    Preprocesses the features for inference.

    Args:
        df (pd.DataFrame): The input DataFrame containing features.
        encoders (dict): The encoders for feature preprocessing.

    Returns:
        pd.DataFrame: The preprocessed features.
    """
    for feature in df.columns:
        if feature in encoders:
            logger.info(f"Preprocessing feature: {feature}")
            df[feature] = encoders[feature].transform(df[feature])
    return df


# Prepare data and perform inference
X = predict_churn_batch.drop(columns=["customer_id", 'is_churned'], errors="ignore")

model, encoders = load_model_artifacts(model_version.version)
X = preprocess_features(X, encoders)
logger.info(f"Preprocessed features for {len(X)} samples")
logger.info(f"Performing inference for {len(X)} samples")
predictions = model['model'].predict_proba(X)
logger.info(f"Performed inference for {len(predictions)} samples")
# Save inference results
predict_churn_batch["churn_prediction"] = predictions[:, 1]
predict_churn_batch.to_parquet(settings.CHURN_BATCH_RESULTS)
logger.info(f"Saved inference results to {settings.CHURN_BATCH_RESULTS}")




