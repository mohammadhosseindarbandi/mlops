# MLOps Pipeline Implementation Guide

## Quick Start

This guide provides a practical implementation of an end-to-end MLOps pipeline for machine learning model deployment, based on the existing heart disease prediction service.

## 📁 File Structure

```
├── MLOps_Pipeline_Design.md          # Comprehensive design document
├── mlops_implementation_example.py   # Python implementation
├── kubernetes_manifests.yaml         # K8s deployment configs
├── app/                              # Existing application
│   ├── main.py                       # FastAPI app
│   ├── train_model.py               # Current training script
│   └── heart_cleveland_upload.csv   # Sample data
├── docker-compose.yaml              # Local development
├── Dockerfile                       # Container definition
└── requirements.txt                 # Dependencies
```

## 🚀 Implementation Steps

### Phase 1: Local Development Setup

#### 1. Enhanced Requirements
Update `requirements.txt` to include MLOps dependencies:

```bash
# Add to existing requirements.txt
mlflow>=2.8.0
evidently>=0.4.0
feast>=0.32.0
great-expectations>=0.17.0
pandera>=0.17.0
prometheus-client>=0.17.0
scipy>=1.11.0
hyperopt>=0.2.7
dvc>=3.0.0
```

#### 2. Local MLflow Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Run the MLOps pipeline
python mlops_implementation_example.py
```

#### 3. Test with Docker Compose
```bash
# Build and run locally
docker-compose up --build

# Test the API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'
```

### Phase 2: Kubernetes Deployment

#### 1. Prerequisites
- Kubernetes cluster (local or cloud)
- kubectl configured
- Helm (optional, for easier management)
- Prometheus Operator (for monitoring)

#### 2. Deploy to Kubernetes
```bash
# Apply all manifests
kubectl apply -f kubernetes_manifests.yaml

# Check deployment status
kubectl get pods -n mlops
kubectl get services -n mlops

# Check HPA status
kubectl get hpa -n mlops

# View logs
kubectl logs -f deployment/heart-disease-api -n mlops
```

#### 3. Access the Application
```bash
# Port forward for local access
kubectl port-forward service/heart-disease-service 8080:80 -n mlops

# Test the deployed API
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "sex": 0, "cp": 2, "trestbps": 130, "chol": 250, "fbs": 0, "restecg": 1, "thalach": 175, "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2}'
```

### Phase 3: CI/CD Pipeline Setup

#### 1. GitHub Actions Setup
The repository already includes `.github/workflows/mlops_pipeline.yaml`. Configure secrets:

```bash
# Set repository secrets in GitHub:
MLFLOW_TRACKING_URI
DOCKER_REGISTRY_URL
DOCKER_USERNAME
DOCKER_PASSWORD
KUBE_CONFIG_DATA
```

#### 2. GitLab CI Enhancement
Update `.gitlab-ci.yaml` to include MLOps stages:

```yaml
# Add to existing stages
stages:
  - lint
  - test
  - data_validation
  - model_training
  - model_validation
  - build
  - security_scan
  - deploy_staging
  - deploy_production

model_training:
  stage: model_training
  script:
    - python mlops_implementation_example.py
  artifacts:
    paths:
      - models/
      - mlops_pipeline_report.json
    expire_in: 1 week
```

### Phase 4: Monitoring and Observability

#### 1. Prometheus Setup
```bash
# Install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack

# Verify ServiceMonitor is created
kubectl get servicemonitor -n mlops
```

#### 2. Grafana Dashboard
Access Grafana and import the dashboard configuration from the design document.

#### 3. Set Up Alerting
```bash
# Check PrometheusRule
kubectl get prometheusrule -n mlops

# Test alerts
kubectl get alerts -n mlops
```

### Phase 5: Advanced Features

#### 1. Feature Store with Feast
```bash
# Initialize Feast project
feast init mlops_features
cd mlops_features

# Apply feature definitions
feast apply

# Materialize features
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

#### 2. Data Version Control with DVC
```bash
# Initialize DVC
dvc init

# Add data tracking
dvc add app/heart_cleveland_upload.csv

# Set up remote storage
dvc remote add -d storage s3://your-mlops-bucket/dvc-storage

# Push data
dvc push
```

#### 3. Automated Retraining
The CronJob in `kubernetes_manifests.yaml` handles weekly retraining. Monitor with:

```bash
# Check CronJob status
kubectl get cronjob -n mlops

# View job history
kubectl get jobs -n mlops

# Check logs of latest job
kubectl logs job/model-retrain-xxx -n mlops
```

## 🔧 Configuration Management

### Environment Variables
```bash
# Production
export MODEL_NAME="heart_disease_classifier"
export MODEL_STAGE="Production"
export MLFLOW_TRACKING_URI="http://mlflow:5000"
export LOG_LEVEL="INFO"

# Development
export MODEL_STAGE="Staging"
export LOG_LEVEL="DEBUG"
```

### Secrets Management
```bash
# Create secrets for sensitive data
kubectl create secret generic mlops-secrets \
  --from-literal=db-password="your-secure-password" \
  --from-literal=api-key="your-api-key" \
  -n mlops
```

## 📊 Monitoring Dashboard Metrics

### Key Performance Indicators (KPIs)
- **Model Performance**: Accuracy, Precision, Recall, F1-Score
- **System Performance**: Latency (p50, p95, p99), Throughput, Error Rate
- **Resource Usage**: CPU, Memory, Disk Usage
- **Business Metrics**: Prediction Volume, Model Confidence Distribution

### Alert Thresholds
- **Critical**: Accuracy < 70%, Error Rate > 5%, p95 Latency > 1s
- **Warning**: Accuracy < 75%, Error Rate > 2%, p95 Latency > 500ms
- **Info**: New model deployment, Weekly retraining completed

## 🔒 Security Best Practices

### Container Security
```dockerfile
# Use non-root user in Dockerfile
RUN adduser --disabled-password --gecos '' appuser
USER appuser
```

### Network Security
- Network policies restrict pod-to-pod communication
- TLS encryption for all external communications
- Regular security scanning with tools like Trivy

### Data Privacy
- PII anonymization before model training
- Encryption at rest for sensitive data
- GDPR compliance for data handling

## 🧪 Testing Strategy

### Unit Tests
```python
# tests/test_mlops.py
import pytest
from mlops_implementation_example import DataValidator, FeatureEngineer

def test_data_validation():
    validator = DataValidator()
    # Test schema validation
    
def test_feature_engineering():
    engineer = FeatureEngineer()
    # Test feature creation
```

### Integration Tests
```python
# tests/test_integration.py
def test_full_pipeline():
    # Test end-to-end pipeline
    
def test_api_endpoints():
    # Test FastAPI endpoints
```

### Load Testing
```bash
# Using the LoadTester class
python -c "
from mlops_implementation_example import LoadTester
tester = LoadTester('http://localhost:8080')
results = asyncio.run(tester.run_load_test(num_requests=1000))
print(results)
"
```

## 📈 Scaling Considerations

### Horizontal Scaling
- HPA configured for 3-20 replicas based on CPU/Memory
- Custom metrics scaling based on prediction volume
- Load balancer distributes traffic across pods

### Vertical Scaling
- Resource requests/limits defined in deployment
- VPA (Vertical Pod Autoscaler) for automatic right-sizing
- Node auto-scaling for cluster capacity

### Data Scaling
- Distributed training with Ray or Dask for large datasets
- Feature store for efficient feature serving
- Model sharding for very large models

## 🔄 Model Lifecycle Management

### Model Versioning
```python
# Automated versioning with MLflow
version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="heart_disease_classifier",
    tags={"environment": "production", "version": "v2.1.0"}
)
```

### A/B Testing
```python
# Traffic splitting for model comparison
if user_id % 100 < 10:  # 10% traffic
    model = load_model("staging")
else:
    model = load_model("production")
```

### Rollback Strategy
```bash
# Automated rollback on performance degradation
kubectl rollout undo deployment/heart-disease-api -n mlops

# Manual rollback to specific version
kubectl set image deployment/heart-disease-api \
  api=heart-disease-api:v1.2.3 -n mlops
```

## 🎯 Success Metrics

### Technical Metrics
- **Deployment Frequency**: Daily deployments
- **Lead Time**: < 2 hours from commit to production
- **Mean Time to Recovery**: < 15 minutes
- **Change Failure Rate**: < 5%

### Business Metrics
- **Model Accuracy**: Maintain > 75%
- **Prediction Latency**: p95 < 500ms
- **System Availability**: 99.9% uptime
- **Cost Efficiency**: Resource utilization > 70%

## 🚀 Next Steps

1. **Implement Advanced ML Techniques**
   - Online learning for continuous model updates
   - Ensemble methods for improved accuracy
   - Explainable AI for model interpretability

2. **Enhance Data Pipeline**
   - Real-time streaming data processing
   - Automated data quality monitoring
   - Advanced feature engineering with AutoML

3. **Improve Operations**
   - Chaos engineering for resilience testing
   - Multi-region deployment for disaster recovery
   - Advanced cost optimization strategies

4. **Compliance and Governance**
   - Model audit trails and lineage tracking
   - Regulatory compliance frameworks
   - Data governance policies

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Kubernetes MLOps Guide](https://kubernetes.io/docs/concepts/workloads/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [Feast Feature Store](https://docs.feast.dev/)
- [DVC Data Version Control](https://dvc.org/doc)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Run the full MLOps pipeline
5. Submit a pull request with detailed description

---

This MLOps pipeline provides a production-ready foundation for machine learning model deployment with comprehensive monitoring, automated deployment, and scalability features. The modular design allows for gradual implementation and customization based on specific requirements.