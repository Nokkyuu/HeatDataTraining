# HeatDataTraining

This repository documents my approach to a heating system anomaly detection challenge. The project began with a **4 hour challenge** to analyze and detect anomalies in heating system data using Jupyter Notebooks and logging the thought process. After the initial challenge, I extended the work into a modular anomaly detection pipeline, including synthetic data generation, batch ETL, and ensemble ML-based detection.

---

## Workflow & Components

### 1. **Initial Challenge (4 hours)**
- **[`1_AnomalyDetectionEDA.ipynb`](1_AnomalyDetectionEDA.ipynb)**: Exploratory data analysis, failure mode identification and feature brainstorming/engineering .
- **[`2_AnomalyDetectionDetector.ipynb`](2_AnomalyDetectionDetector.ipynb)**: First anomaly detection logic, focusing on domestic hot water data as the low hanging fruit, using simple statistical methods.

### 2. **Pipeline Extension (Work in Progress)**
- **Synthetic Data Generation**:  
  - [`arimax_generator.py`](pipeline simulation/arimax_generator.py): Uses ARIMAX models to generate heating data, including injected anomalies.
- **Anomaly Labeling & Feature Engineering**:  
  - [`anomaly_labeling.py`](pipeline simulation/anomaly_labeling.py): Unsupervised labeling using Isolation Forest, statistical, and domain-specific rules to create a training dateset.
- **Ensemble Detection**:  
  - [`ensemble_detector.py`](pipeline simulation/ensemble_detector.py): Combines multiple ML models (Isolation Forest, SVM, statistical) for anomaly detection.
- **Batch ETL & Orchestration**:  
  - [`prefect_pipeline.py`](pipeline simulation/prefect_pipeline.py): Prefect-based workflow for batch data generation, anomaly detection, and reporting.
- **Analysis & Visualization**:  
  - [`labeled_data_analysis.ipynb`](pipeline simulation/labeled_data_analysis.ipynb): In-depth analysis of labeled training data and detection results.

---

## Status

- **Work in Progress**:  
  The pipeline currently supports batch data flow, synthetic data creation, and ensemble anomaly detection. Further improvements are planned for real-time streaming, advanced feature engineering, among other features.

---

## Notes

- The initial notebooks were completed under a strict 4-hour deadline as part of the challenge.
- All subsequent work aims to generalize, automate, and improve the anomaly detection process for heating as a simple exercise.