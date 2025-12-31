import pandas as pd
import numpy as np
import pickle
import argparse
import os
import sys
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)
import matplotlib
from app.scripts.plot_utils import plot_roc_curve, plot_pr_curve, plot_prediction_distribution
from app.routers.utils import log_model_to_db, get_model_version
from app.config import setup_logger, settings
matplotlib.use('Agg')  # Non-interactive backend

# Initialize logger
logger = setup_logger("train_model")


def load_data(data_path):
    """Load data from parquet or CSV file."""
    logger.info(f"Loading data from {data_path}...")
    
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Data file must be .parquet or .csv")
    
    logger.info(f"✓ Loaded {len(df)} records")
    return df


def preprocess_data(df):
    """Preprocess features and encode categorical variables."""
    logger.info("Preprocessing data...")
    
    # Separate features and target
    X = df.drop(['customer_id', 'is_churned'], axis=1)
    y = df['is_churned']
    
    # Encode categorical features
    encoders = {}
    categorical_cols = ['plan_type', 'country']
    
    for col in categorical_cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])
        encoders[col] = encoder
        logger.info(f"  ✓ Encoded {col}: {list(encoder.classes_)}")
    
    logger.info(f"✓ Features shape: {X.shape}")
    logger.info(f"✓ Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, encoders


def train_model(X_train, y_train, n_estimators=100, max_depth=10, random_state=42, tune_hyperparameters=False):
    """
    Train RandomForest classifier with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees (used if not tuning)
        max_depth: Max tree depth (used if not tuning)
        random_state: Random seed
        tune_hyperparameters: Whether to perform grid search
    
    Returns:
        Trained model and best parameters (if tuned)
    """
    if tune_hyperparameters:
        logger.info("Performing hyperparameter tuning with GridSearchCV...")
        logger.info("This may take a few minutes...")
        
        from sklearn.model_selection import GridSearchCV
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Base model
        base_model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info("✓ Hyperparameter tuning complete")
        logger.info(f"Best parameters:")
        for param, value in grid_search.best_params_.items():
            logger.info(f"  {param:20s}: {value}")
        logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    else:
        logger.info("Training RandomForest model...")
        logger.info(f"  - n_estimators: {n_estimators}")
        logger.info(f"  - max_depth: {max_depth}")
        logger.info(f"  - random_state: {random_state}")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        logger.info("✓ Model trained successfully")
        
        return model, model.get_params()


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and return predictions."""
    logger.info("Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba)
    }
    
    logger.info("Performance Metrics:")
    logger.info("=" * 40)
    for metric, value in metrics.items():
        logger.info(f"  {metric:12s}: {value:.4f}")
    
    logger.info("Classification Report:")
    logger.info("=" * 40)
    logger.info("\n" + classification_report(y_test, y_pred, target_names=['Active', 'Churned']))
    
    logger.info("Confusion Matrix:")
    logger.info("=" * 40)
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"  True Negatives:  {cm[0, 0]:5d}  |  False Positives: {cm[0, 1]:5d}")
    logger.info(f"  False Negatives: {cm[1, 0]:5d}  |  True Positives:  {cm[1, 1]:5d}")
    
    return metrics, y_pred, y_proba


def create_model_artifact(model, encoders, feature_names, metrics, y_test, y_proba, 
                          output_dir, best_params=None, model_version=None):
    """
    Create versioned model artifact with all metadata, metrics, and plots.
    
    Args:
        model: Trained model
        encoders: Feature encoders
        feature_names: List of feature names
        metrics: Dictionary of performance metrics
        y_test: True labels
        y_proba: Predicted probabilities
        output_dir: Base output directory
        best_params: Best hyperparameters (if tuned)
        model_version: Model version ID (auto-generated if None)
    
    Returns:
        Path to created artifact directory
    """
    # Generate version ID and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_version is None:
        model_version = f"v_{timestamp}"
    
    # Create artifact directory structure
    artifact_dir = os.path.join(output_dir, model_version)
    os.makedirs(artifact_dir, exist_ok=True)
    
    logger.info(f"Creating model artifact: {model_version}")
    logger.info(f"Artifact directory: {artifact_dir}")
    
    # 1. Save model pickle
    model_path = os.path.join(artifact_dir, 'model.pkl')
    model_package = {
        'model': model,
        'encoders': encoders, # Keeping for backward compatibility
        'feature_names': feature_names,
        'model_version': model_version,
        'trained_at': datetime.now().isoformat(),
        'model_type': 'RandomForestClassifier',
        'best_params': best_params
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    # 1b. Save encoders separately
    encoders_path = os.path.join(artifact_dir, 'encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    
    file_size = os.path.getsize(model_path) / 1024
    logger.info(f"  ✓ Model saved: model.pkl ({file_size:.1f} KB)")
    logger.info(f"  ✓ Encoders saved: encoders.pkl")
    
    # 2. Save metrics as JSON
    metrics_path = os.path.join(artifact_dir, 'metrics.json')
    metrics_data = {
        'model_version': model_version,
        'timestamp': datetime.now().isoformat(),
        'metrics': {k: float(v) for k, v in metrics.items()},
        'best_hyperparameters': best_params,
        'feature_names': feature_names
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    logger.info("  ✓ Metrics saved: metrics.json")
    
    # 3. Save metrics as human-readable text
    metrics_txt_path = os.path.join(artifact_dir, 'metrics.txt')
    with open(metrics_txt_path, 'w') as f:
        f.write(f"Model Version: {model_version}\n")
        f.write(f"Trained At: {datetime.now().isoformat()}\n")
        f.write(f"\n{'='*50}\n")
        f.write("PERFORMANCE METRICS\n")
        f.write(f"{'='*50}\n\n")
        for metric, value in metrics.items():
            f.write(f"{metric:20s}: {value:.6f}\n")
        
        if best_params:
            f.write(f"\n{'='*50}\n")
            f.write("BEST HYPERPARAMETERS\n")
            f.write(f"{'='*50}\n\n")
            for param, value in best_params.items():
                f.write(f"{param:20s}: {value}\n")
    logger.info("  ✓ Metrics saved: metrics.txt")
    
    # 4. Plot and save ROC curve
    roc_path = os.path.join(artifact_dir, 'roc_curve.png')
    plot_roc_curve(y_test, y_proba, roc_path)
    
    # 5. Plot and save PR curve
    pr_path = os.path.join(artifact_dir, 'pr_curve.png')
    plot_pr_curve(y_test, y_proba, pr_path)
    
    # 6. Plot and save prediction distribution (HTML)
    dist_path = os.path.join(artifact_dir, 'prediction_distribution.html')
    plot_prediction_distribution(y_test, y_proba, dist_path)
    
    # 7. Save feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_path = os.path.join(artifact_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        logger.info("  ✓ Feature importance saved: feature_importance.csv")
    
    # 8. Create README
    readme_path = os.path.join(artifact_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(f"# Model Artifact: {model_version}\n\n")
        f.write(f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Performance Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for metric, value in metrics.items():
            f.write(f"| {metric} | {value:.4f} |\n")
        f.write(f"\n## Files\n\n")
        f.write("- `model.pkl` - Trained model with encoders\n")
        f.write("- `metrics.json` - Metrics in JSON format\n")
        f.write("- `metrics.txt` - Metrics in text format\n")
        f.write("- `roc_curve.png` - ROC AUC curve plot\n")
        f.write("- `pr_curve.png` - Precision-Recall curve plot\n")
        f.write("- `prediction_distribution.html` - Interactive prediction probability distribution plot (open in browser)\n")
        f.write("- `feature_importance.csv` - Feature importance scores\n")
    logger.info("  ✓ README created: README.md")
    
    logger.info("✓ Model artifact created successfully!")
    logger.info(f"  Location: {artifact_dir}")
    
    return artifact_dir


def log_model_metadata(metrics):
    """Log model metadata to the database model_versions table."""
    model_version = get_model_version()
    logger.info(f"The current model version is: {model_version}")
    model_version += 1
    new_model_version = log_model_to_db(model_version, metrics)
    logger.info(f"The model version {new_model_version.version} has been logged to the database.")
    


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train churn prediction model')
    parser.add_argument('--data-path', type=str, default=settings.storage_path,
                       help='Path to training data')
    parser.add_argument('--output-path', type=str, default=os.path.join(settings.MODEL_ARTIFACTS_DIR, 'churn_model.pkl'),
                       help='Path to save trained model')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of trees (default: 100)')
    parser.add_argument('--max-depth', type=int, default=10,
                       help='Max tree depth (default: 10)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                       help='Perform hyperparameter tuning with GridSearchCV')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Churn Prediction Model Training")
    logger.info("=" * 60)
    
    # Load data
    df = load_data(args.data_path)
    
    # Preprocess
    X, y, encoders = preprocess_data(df)
    
    # Split data
    logger.info(f"Splitting data (test_size={args.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=args.test_size, 
        random_state=args.random_seed,
        stratify=y
    )
    logger.info(f"✓ Train set: {len(X_train)} samples")
    logger.info(f"✓ Test set:  {len(X_test)} samples")
    
    # Train model
    model, best_params = train_model(
        X_train, y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_seed,
        tune_hyperparameters=args.tune_hyperparameters
    )
    
    # Evaluate
    metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)
    
    # Feature importance
    logger.info("Top 10 Feature Importances:")
    logger.info("=" * 40)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']:25s}: {row['importance']:.4f}")
    
    # Create model artifact
    artifact_dir = create_model_artifact(
        model=model,
        encoders=encoders,
        feature_names=list(X.columns),
        metrics=metrics,
        y_test=y_test,
        y_proba=y_proba,
        output_dir=os.path.dirname(args.output_path) or 'ml_models',
        best_params=best_params
    )
    
    # Log to database
    logger.info("Logging model to database...")
    log_model_metadata(metrics)
    
    logger.info("=" * 60)
    logger.info("✓ Training complete!")
    logger.info("=" * 60)
    logger.info(f"Model artifact: {artifact_dir}")
    logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test F1 score: {metrics['f1_score']:.4f}")
    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"PR AUC: {metrics['pr_auc']:.4f}")


if __name__ == '__main__':
    main()