# Pipeline Simulation

This folder contains a complete anomaly detection pipeline for heating system data, featuring synthetic data generation, ensemble-based anomaly detection, and automated validation.

## Overview

The pipeline simulates a production environment for heating system anomaly detection using:
- **ARIMAX models** for synthetic data generation
- **Ensemble detection** combining multiple anomaly detection algorithms
- **Prefect workflows** for orchestration and monitoring
- **Automated validation** against ground truth labels

## Core Components

### 📊 Data Generation
- **[`arimax_generator.py`](arimax_generator.py)** - ARIMAX-based synthetic heating data generator
  - Generates realistic domestic hot water temperature patterns
  - Injects controlled anomalies (overheat, underheat, sensor errors)
  - Supports model persistence and retraining

### 🤖 Anomaly Detection
- **[`ensemble_detector.py`](ensemble_detector.py)** - Multi-algorithm ensemble detector
  - Isolation Forest for unsupervised detection
  - Statistical methods (Z-score) for threshold-based detection
  - Weighted ensemble combining multiple detectors

- **[`anomaly_labeling.py`](anomaly_labeling.py)** - Initial data labeling for training
  - Feature engineering for heating data
  - Unsupervised anomaly detection for ground truth creation
  - Domain-specific anomaly detection rules

### 🔄 Pipeline Orchestration
- **[`prefect_pipeline.py`](prefect_pipeline.py)** - Complete Prefect workflow
  - Automated data generation and detection
  - Validation against injected anomalies
  - HTML report generation with visualizations
  - Continuous monitoring with configurable intervals

### 📈 Analysis & Visualization
- **[`labeled_data_analysis.ipynb`](labeled_data_analysis.ipynb)** - Training data analysis
  - Anomaly statistics and distributions
  - Temporal patterns and correlations
  - Detector overlap analysis

## Quick Start

### 1. Initial Setup
```bash
# Generate labeled training data
python anomaly_labeling.py

# Train ARIMAX model (first time)
python arimax_generator.py
```

### 2. Run Pipeline

**Single Run:**
```bash
python prefect_pipeline.py once
```

**Continuous Monitoring:**
```bash
python prefect_pipeline.py continuous 1  # every minute
```

**Interactive Menu:**
```bash
python prefect_pipeline.py
```

### 3. View Results
- Reports are saved to `../reports/` as HTML files
- Models are persisted in `./models/`
- Synthetic data is saved to `../data/synthetic/`

## Pipeline Flow

```
📊 ARIMAX Generator
    ↓ (synthetic data with injected anomalies)
🧹 Data Cleaning
    ↓ (production-ready data)
🤖 Ensemble Detection
    ↓ (anomaly predictions)
📋 Ground Truth Extraction
    ↓ (known anomaly labels)
✅ Validation & Metrics
    ↓ (performance assessment)
📊 Report Generation
    ↓ (HTML visualization)
```

## Key Features

- **Data Simulation**: ARIMAX models trained on real heating system data
- **Batch Analysis**: Analysis of batch data
- **Controlled Testing**: Known anomaly injection for validation
- **Multiple Detectors**: Ensemble approach for robust detection
- **Automated Validation**: Precision, recall, F1-score tracking
- **Visual Reports**: Interactive Plotly visualizations
- **Continuous Monitoring**: Scheduled pipeline execution
- **Model Persistence**: Automatic model saving/loading

## Configuration

### ARIMAX Generator
- **Model Order**: `(1, 1, 1)` - Simple ARIMA configuration
- **Anomaly Rate**: `5%` - Configurable injection rate
- **Batch Size**: `1000` hours - Default generation size

### Ensemble Detector
- **Isolation Forest**: `contamination=0.1`
- **Statistical Detector**: `z_threshold=2.5`
- **Weights**: Equal weighting by default

### Validation Thresholds
- **Success Criteria**: Recall ≥ 0.7
- **Performance Tracking**: Precision, Recall, F1-Score
- **Report Generation**: Automatic HTML reports

## File Dependencies

```
📁 data/
  ├── labeled_training_data_cleaned.csv  # Training data
  └── synthetic/                         # Generated data

📁 models/
  ├── arimax_heating_model.pkl          # Trained ARIMAX
  ├── ensemble_*.pkl                     # Ensemble components
  └── ensemble_metadata.pkl             # Configuration

📁 reports/
  └── arimax_ensemble_validation_*.html # Validation reports
```

## Monitoring & Alerts

The pipeline provides:
- ✅ **Success indicators**: Metrics above threshold
- ❌ **Failure alerts**: Poor detection performance
- 📊 **Performance tracking**: Historical metrics
- 🔄 **Automatic retries**: Robust error handling

## Use Cases

1. **Model Validation**: Test detection algorithms on synthetic data
2. **Performance Monitoring**: Continuous validation of production systems
3. **Algorithm Development**: Benchmark new detection methods
4. **System Testing**: Validate complete anomaly detection pipelines

## Dependencies

- `prefect` - Workflow orchestration
- `pandas`, `numpy` - Data processing
- `scikit-learn` - Machine learning algorithms
- `statsmodels` - ARIMAX modeling
- `plotly` - Interactive visualizations
- `matplotlib`, `seaborn` - Static plots

## Notes

- Designed for **domestic hot water** systems initially
- Easily extensible to **forward flow temperature** data due to modular construction