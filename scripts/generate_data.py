import pandas as pd
import numpy as np
import os
import argparse
import boto3
from botocore.exceptions import ClientError
from config import get_storage_path, ENV, S3_BUCKET, setup_logger

# Initialize logger
logger = setup_logger("generate_data")

def upload_to_s3(file_path, bucket, object_name=None):
    """
    Upload a file to an S3 bucket.
    """
    if object_name is None:
        object_name = os.path.basename(file_path)

    s3_client = boto3.client('s3')
    try:
        logger.info(f"Uploading {file_path} to s3://{bucket}/{object_name}...")
        s3_client.upload_file(file_path, bucket, object_name)
        logger.info(f"✓ Successfully uploaded to S3")
    except ClientError as e:
        logger.error(f"✗ Failed to upload to S3: {e}")
        return False
    return True

def generate_churn_data(n_samples=10000, random_seed=42):
    """
    Generate synthetic customer churn dataset.
    """
    np.random.seed(random_seed)
    
    logger.info(f"Generating {n_samples} customer records...")
    
    # Generate customer IDs
    customer_ids = [f"CUST_{i:06d}" for i in range(n_samples)]
    
    # Generate tenure (months with company)
    tenure_months = np.random.exponential(scale=24, size=n_samples).astype(int)
    tenure_months = np.clip(tenure_months, 1, 120)
    
    # Generate login activity (correlated with tenure)
    base_logins = np.random.poisson(lam=15, size=n_samples)
    tenure_factor = np.clip(tenure_months / 24, 0.5, 2.0)
    num_logins_last_30d = (base_logins * tenure_factor).astype(int)
    num_logins_last_30d = np.clip(num_logins_last_30d, 0, 100)
    
    # Generate support tickets
    num_tickets_last_90d = np.random.poisson(lam=2, size=n_samples)
    
    # Generate plan types
    plan_types = np.random.choice(
        ['basic', 'standard', 'premium'],
        size=n_samples,
        p=[0.5, 0.3, 0.2]
    )
    
    # Generate countries
    countries = np.random.choice(
        ['US', 'UK', 'CA', 'AU', 'DE', 'FR'],
        size=n_samples,
        p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
    )
    
    churn_prob = np.zeros(n_samples)
    churn_prob += (1 / (1 + tenure_months / 12)) * 0.3
    churn_prob += (1 / (1 + num_logins_last_30d / 10)) * 0.25
    churn_prob += np.clip(num_tickets_last_90d / 10, 0, 0.25)
    
    plan_churn_boost = {'basic': 0.15, 'standard': 0.05, 'premium': 0.0}
    churn_prob += np.array([plan_churn_boost[p] for p in plan_types])
    
    churn_prob += np.random.normal(0, 0.1, n_samples)
    churn_prob = np.clip(churn_prob, 0, 1)
    
    is_churned = (np.random.random(n_samples) < churn_prob).astype(int)
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'tenure_months': tenure_months,
        'num_logins_last_30d': num_logins_last_30d,
        'num_tickets_last_90d': num_tickets_last_90d,
        'plan_type': plan_types,
        'country': countries,
        'is_churned': is_churned
    })
    
    churn_rate = is_churned.mean()
    logger.info(f"✓ Generated {n_samples} records")
    logger.info(f"✓ Churn rate: {churn_rate:.2%} ({is_churned.sum()}/{len(is_churned)})")
    
    return df

def main():
    """Main function to generate and save data."""
    parser = argparse.ArgumentParser(description='Generate synthetic churn data')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Customer Churn Data Generator - ENV: {ENV}")
    logger.info("=" * 60)
    
    # Generate data
    df = generate_churn_data(n_samples=args.samples, random_seed=args.seed)
    
    # Determine storage path based on config
    storage_path = get_storage_path()
    
    if storage_path.startswith("s3://"):
        # PRD logic
        bucket = S3_BUCKET
        parquet_temp = 'data/temp_churn_data.parquet'
        os.makedirs('data', exist_ok=True)
        df.to_parquet(parquet_temp, index=False)
        upload_to_s3(parquet_temp, bucket, "data/churn_data.parquet")
        os.remove(parquet_temp)
    else:
        # DEV logic - save local
        os.makedirs('data', exist_ok=True)
        df.to_parquet(storage_path, index=False)
        logger.info(f"✓ Saved Parquet to {storage_path}")
        
        csv_path = storage_path.replace(".parquet", ".csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"✓ Saved CSV to {csv_path}")
    
    # Display sample data (logger for consistency, though head(10) is large)
    logger.info("\n" + "=" * 60)
    logger.info("Sample Data Preview:")
    logger.info("\n" + df.head(5).to_string(index=False))
    
    # Display summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("Summary Statistics:")
    logger.info("\n" + df.describe().to_string())
    
    # Display churn distribution by plan type
    logger.info("\n" + "=" * 60)
    logger.info("Churn Rate by Plan Type:")
    churn_by_plan = df.groupby('plan_type')['is_churned'].agg(['count', 'sum', 'mean'])
    churn_by_plan.columns = ['Total Customers', 'Churned', 'Churn Rate']
    churn_by_plan['Churn Rate'] = churn_by_plan['Churn Rate'].apply(lambda x: f"{x:.2%}")
    logger.info("\n" + churn_by_plan.to_string())
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Data generation complete!")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()