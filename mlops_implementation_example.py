#!/usr/bin/env python3
"""
MLOps Pipeline Implementation Example
Demonstrates integration of key MLOps components for the heart disease prediction service
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from mlflow.tracking import MlflowClient
from datetime import datetime
import joblib
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLOpsConfig:
    """Configuration for MLOps pipeline"""
    def __init__(self):
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.model_name = "heart_disease_classifier"
        self.data_path = "app/heart_cleveland_upload.csv"
        self.model_artifacts_path = "models/"
        self.baseline_metrics = {
            'accuracy': 0.75,
            'precision': 0.70,
            'recall': 0.70
        }

class DataValidator:
    """Data validation and quality checks"""
    
    def __init__(self):
        self.expected_columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
            'ca', 'thal', 'condition'
        ]
    
    def validate_schema(self, df):
        """Validate data schema"""
        missing_cols = set(self.expected_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Check data types and ranges
        validations = [
            (df['age'] >= 0) & (df['age'] <= 120),
            df['sex'].isin([0, 1]),
            df['cp'].isin([0, 1, 2, 3]),
            (df['trestbps'] >= 50) & (df['trestbps'] <= 250),
            (df['chol'] >= 100) & (df['chol'] <= 600)
        ]
        
        for i, validation in enumerate(validations):
            if not validation.all():
                logger.warning(f"Data validation failed for rule {i+1}")
        
        return df
    
    def check_data_drift(self, reference_df, current_df):
        """Simple data drift detection using statistical tests"""
        from scipy import stats
        
        drift_detected = {}
        numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        
        for col in numerical_columns:
            if col in reference_df.columns and col in current_df.columns:
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(
                    reference_df[col], current_df[col]
                )
                drift_detected[col] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'drift': p_value < 0.05  # Significant drift if p < 0.05
                }
        
        return drift_detected

class FeatureEngineer:
    """Feature engineering pipeline"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def create_features(self, df):
        """Create derived features"""
        df = df.copy()
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], 
                               bins=[0, 30, 50, 70, 100], 
                               labels=[0, 1, 2, 3])
        
        # Risk indicators
        df['cholesterol_risk'] = (df['chol'] > 240).astype(int)
        df['bp_risk'] = (df['trestbps'] > 140).astype(int)
        df['exercise_angina_risk'] = df['exang']
        
        # Interaction features
        df['age_chol_interaction'] = df['age'] * df['chol'] / 1000
        df['bp_age_interaction'] = df['trestbps'] * df['age'] / 1000
        
        return df
    
    def preprocess_data(self, df, fit=True):
        """Preprocess data for training/inference"""
        # Separate target if present
        if 'condition' in df.columns:
            X = df.drop('condition', axis=1)
            y = df['condition']
        else:
            X = df
            y = None
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled, y

class ModelTrainer:
    """Model training with MLflow integration"""
    
    def __init__(self, config):
        self.config = config
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        self.client = MlflowClient()
    
    def train_model(self, X_train, y_train, X_val, y_val, hyperparams=None):
        """Train model with experiment tracking"""
        
        if hyperparams is None:
            hyperparams = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42
            }
        
        with mlflow.start_run() as run:
            # Log hyperparameters
            mlflow.log_params(hyperparams)
            
            # Train model
            model = RandomForestClassifier(**hyperparams)
            model.fit(X_train, y_train)
            
            # Validate and log metrics
            y_pred_val = model.predict(X_val)
            
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred_val),
                'precision': precision_score(y_val, y_pred_val),
                'recall': recall_score(y_val, y_pred_val)
            }
            
            mlflow.log_metrics(metrics)
            
            # Log classification report
            report = classification_report(y_val, y_pred_val, output_dict=True)
            mlflow.log_dict(report, "classification_report.json")
            
            # Log model
            model_info = mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=self.config.model_name
            )
            
            logger.info(f"Model trained with metrics: {metrics}")
            return model, run.info.run_id, metrics

class ModelValidator:
    """Model validation and testing"""
    
    def __init__(self, config):
        self.config = config
    
    def validate_model(self, model, X_test, y_test):
        """Comprehensive model validation"""
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions)
        }
        
        # Check if model meets minimum requirements
        validation_passed = all(
            metrics[key] >= self.config.baseline_metrics[key] 
            for key in metrics
        )
        
        # Additional validation checks
        bias_check = self._check_fairness(model, X_test, y_test)
        
        validation_result = {
            'metrics': metrics,
            'validation_passed': validation_passed,
            'bias_check': bias_check,
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
        
        logger.info(f"Model validation {'PASSED' if validation_passed else 'FAILED'}")
        return validation_result
    
    def _check_fairness(self, model, X_test, y_test):
        """Basic fairness check across gender"""
        if 'sex' not in X_test.columns:
            return {"status": "skipped", "reason": "sex column not available"}
        
        fairness_metrics = {}
        for sex in [0, 1]:  # 0: female, 1: male
            mask = X_test['sex'] == sex
            if mask.sum() > 0:
                y_pred_sex = model.predict(X_test[mask])
                y_true_sex = y_test[mask]
                
                fairness_metrics[f'sex_{sex}'] = {
                    'accuracy': accuracy_score(y_true_sex, y_pred_sex),
                    'precision': precision_score(y_true_sex, y_pred_sex),
                    'recall': recall_score(y_true_sex, y_pred_sex)
                }
        
        return fairness_metrics

class ModelDeployer:
    """Model deployment and serving utilities"""
    
    def __init__(self, config):
        self.config = config
        self.client = MlflowClient()
    
    def promote_model(self, run_id, stage="Staging"):
        """Promote model to specified stage"""
        try:
            # Get model version
            model_version = self.client.search_model_versions(
                f"run_id='{run_id}'"
            )[0]
            
            # Transition to stage
            self.client.transition_model_version_stage(
                name=self.config.model_name,
                version=model_version.version,
                stage=stage
            )
            
            logger.info(f"Model version {model_version.version} promoted to {stage}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Model promotion failed: {e}")
            return None
    
    def get_latest_model(self, stage="Production"):
        """Get latest model from specified stage"""
        try:
            latest_version = self.client.get_latest_versions(
                self.config.model_name, 
                stages=[stage]
            )
            
            if latest_version:
                model_uri = f"models:/{self.config.model_name}/{stage}"
                model = mlflow.sklearn.load_model(model_uri)
                return model, latest_version[0].version
            else:
                logger.warning(f"No model found in {stage} stage")
                return None, None
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, None

class MonitoringSystem:
    """Model and system monitoring"""
    
    def __init__(self):
        self.prediction_log = []
        self.metrics_log = []
    
    def log_prediction(self, input_data, prediction, probability, model_version):
        """Log prediction for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input_data': input_data,
            'prediction': prediction,
            'probability': probability,
            'model_version': model_version
        }
        self.prediction_log.append(log_entry)
    
    def calculate_live_metrics(self, predictions, actuals):
        """Calculate live performance metrics"""
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")
        
        metrics = {
            'accuracy': accuracy_score(actuals, predictions),
            'precision': precision_score(actuals, predictions),
            'recall': recall_score(actuals, predictions),
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(predictions)
        }
        
        self.metrics_log.append(metrics)
        return metrics
    
    def check_alerts(self, current_metrics, baseline_metrics, threshold=0.05):
        """Check if alerts should be triggered"""
        alerts = []
        
        for metric_name, baseline_value in baseline_metrics.items():
            if metric_name in current_metrics:
                current_value = current_metrics[metric_name]
                if current_value < baseline_value - threshold:
                    alerts.append({
                        'metric': metric_name,
                        'current_value': current_value,
                        'baseline_value': baseline_value,
                        'threshold': threshold,
                        'severity': 'critical' if current_value < baseline_value - 2*threshold else 'warning'
                    })
        
        return alerts

def main():
    """Main MLOps pipeline execution"""
    logger.info("Starting MLOps Pipeline...")
    
    # Initialize configuration
    config = MLOpsConfig()
    
    # Initialize components
    data_validator = DataValidator()
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer(config)
    model_validator = ModelValidator(config)
    model_deployer = ModelDeployer(config)
    monitoring_system = MonitoringSystem()
    
    try:
        # 1. Data Loading and Validation
        logger.info("Loading and validating data...")
        df = pd.read_csv(config.data_path)
        df = data_validator.validate_schema(df)
        
        # 2. Feature Engineering
        logger.info("Engineering features...")
        df = feature_engineer.create_features(df)
        X, y = feature_engineer.preprocess_data(df, fit=True)
        
        # 3. Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Further split train into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # 4. Model Training
        logger.info("Training model...")
        model, run_id, training_metrics = model_trainer.train_model(
            X_train, y_train, X_val, y_val
        )
        
        # 5. Model Validation
        logger.info("Validating model...")
        validation_result = model_validator.validate_model(model, X_test, y_test)
        
        # 6. Model Deployment (if validation passed)
        if validation_result['validation_passed']:
            logger.info("Model validation passed. Promoting to staging...")
            version = model_deployer.promote_model(run_id, "Staging")
            
            # Save model artifacts locally
            os.makedirs(config.model_artifacts_path, exist_ok=True)
            joblib.dump(model, f"{config.model_artifacts_path}/model.pkl")
            joblib.dump(feature_engineer.scaler, f"{config.model_artifacts_path}/scaler.pkl")
            
            # Save validation results
            with open(f"{config.model_artifacts_path}/validation_results.json", 'w') as f:
                json.dump(validation_result, f, indent=2)
            
            logger.info("Model artifacts saved successfully")
            
        else:
            logger.error("Model validation failed. Deployment aborted.")
            return False
        
        # 7. Monitoring Setup (simulate)
        logger.info("Setting up monitoring...")
        
        # Simulate some predictions for monitoring
        sample_predictions = model.predict(X_test[:10])
        sample_actuals = y_test[:10].values
        
        live_metrics = monitoring_system.calculate_live_metrics(
            sample_predictions, sample_actuals
        )
        
        alerts = monitoring_system.check_alerts(
            live_metrics, config.baseline_metrics
        )
        
        if alerts:
            logger.warning(f"Alerts triggered: {alerts}")
        else:
            logger.info("No alerts triggered")
        
        # 8. Generate Pipeline Report
        pipeline_report = {
            'run_id': run_id,
            'model_version': version,
            'training_metrics': training_metrics,
            'validation_result': validation_result,
            'live_metrics': live_metrics,
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('mlops_pipeline_report.json', 'w') as f:
            json.dump(pipeline_report, f, indent=2)
        
        logger.info("MLOps Pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)