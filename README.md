# MLOps from Scratch 🧪

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**End-to-end MLOps pipeline** for Persian sentiment analysis. Demonstrates the complete ML lifecycle: data collection → automated training → A/B testing → drift monitoring.

## 🏗 Architecture

```
GitHub Actions → Data Collector → Training Pipeline → MLflow Registry
                                                         ↓
                                            FastAPI Service (A/B Test)
                                                         ↓
                                         Drift Monitor → GitHub Issue
```

## 🚀 Pipeline Stages

### 1. Data Collection
`src/data/collector.py` — Runs daily to collect Persian text samples.

### 2. Automated Training
`src/models/train.py` — TF-IDF + Logistic Regression with MLflow tracking.

### 3. Serving with A/B Testing
`src/api/main.py` — FastAPI service routing 90% traffic to production, 10% to canary.

### 4. Drift Monitoring
`src/monitoring/drift_detector.py` — Detects distribution shifts and alerts.

## 🧪 Quick Start

```bash
pip install -e .
python src/data/collector.py
python src/models/train.py
mlflow ui & uvicorn src.api.main:app --reload
```

## 💡 Design Decisions

- **Scikit-learn pipeline**: Simple, interpretable baseline before deep learning
- **MLflow tracking**: Industry-standard model registry
- **A/B testing**: Canary deployment pattern for safe rollouts
- **Statistical drift detection**: Distribution comparison without labels

## 🚀 Production Checklist

- [x] Automated data collection
- [x] MLflow experiment tracking
- [x] A/B testing service
- [x] Drift detection
- [ ] Kubernetes deployment
- [ ] Model versioning & rollback
- [ ] Alert system (Slack/Email)
- [ ] Load testing
