# Mining Analytics ML Pipeline

## 1. Project Overview

**Mining Analytics ML Pipeline** is a personal end-to-end analytics and Machine Learning project focused on predictive maintenance in mining operations. The project simulates operational data from mining equipment, processes the data with Pandas and PySpark, trains multiple Machine Learning models, evaluates them using maintenance-oriented metrics, exposes the selected model through a FastAPI service, and includes a basic monitoring module to detect possible data drift.

This project was designed to demonstrate practical skills for **Analytics, Data Science, and MLOps roles in the mining industry**, especially for positions related to operational analytics, maintenance analytics, reliability engineering, and predictive maintenance.

---

## 2. Business Problem

Mining operations depend on critical assets such as haul trucks, conveyor belts, pumps, crushers, and mills. When these assets fail unexpectedly, the business impact can be significant:

- Unplanned downtime
- Production losses
- Higher corrective maintenance costs
- Safety exposure for maintenance and operations teams
- Reduced reliability of the production process

Traditional maintenance strategies often rely on fixed schedules or reactive interventions. However, operational data such as temperature, vibration, load, current, pressure, dust level, and maintenance history can provide early warning signals before a failure occurs.

The business challenge is to use these operational signals to estimate whether an asset has a higher risk of failure in the next 7 days, allowing maintenance teams to prioritize inspections and preventive actions.

---

## 3. Project Objective

The objective of this project is to build a simple but complete Machine Learning pipeline that predicts the probability of failure of mining equipment in the next 7 days.

The project covers the full analytics lifecycle:

1. Synthetic data generation
2. Data cleaning and feature engineering
3. Batch processing with Pandas
4. Basic distributed processing with PySpark
5. Exploratory Data Analysis
6. Model training and evaluation
7. Experiment tracking with MLflow
8. Model serving with FastAPI
9. Basic monitoring for possible data drift
10. Containerization with Docker

---

## 4. Pipeline Architecture

```text
Synthetic Data Generation
        ↓
Raw Data Storage
        ↓
Data Processing with Pandas
        ↓
Feature Engineering
        ↓
Exploratory Data Analysis
        ↓
Model Training and Evaluation
        ↓
Experiment Tracking with MLflow
        ↓
Best Model Export with Joblib
        ↓
FastAPI Prediction Service
        ↓
Basic Model Monitoring
        ↓
Reports and Documentation
```

The pipeline follows a practical structure used in real analytics projects: raw data is stored separately from processed data, model artifacts are stored in a dedicated folder, reports are generated in Markdown, and the API is isolated inside the `src/api` module.

---

## 5. Technologies Used

| Area | Tools |
|---|---|
| Programming | Python |
| Data manipulation | Pandas, NumPy |
| Distributed processing | PySpark |
| Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Experiment tracking | MLflow |
| Model serialization | Joblib |
| API serving | FastAPI, Uvicorn |
| Containerization | Docker |
| Documentation | Markdown, Jupyter Notebook |

---

## 6. Folder Structure

```text
mining-analytics-ml-pipeline/
├── data/
│   ├── raw/
│   │   └── mining_equipment_data.csv
│   ├── processed/
│   │   ├── mining_equipment_processed.csv
│   │   └── pyspark_equipment_summary.csv
│   └── outputs/
│       └── data_generation_summary.json
├── models/
│   └── failure_prediction_model.pkl
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── reports/
│   ├── metrics_report.md
│   └── monitoring_report.md
├── src/
│   ├── api/
│   │   └── main.py
│   ├── data/
│   │   └── generate_mining_dataset.py
│   ├── models/
│   │   └── train_model.py
│   ├── monitoring/
│   │   └── model_monitoring.py
│   └── processing/
│       ├── pandas_processing.py
│       └── pyspark_processing.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 7. Synthetic Dataset

The dataset contains **10,000 synthetic records** representing operational conditions of mining equipment.

### Main Columns

| Column | Description |
|---|---|
| `equipment_id` | Unique equipment identifier |
| `equipment_type` | Type of mining asset: truck, conveyor belt, pump, crusher, or mill |
| `operation_hours` | Accumulated operating hours |
| `temperature_celsius` | Equipment operating temperature |
| `vibration_mm_s` | Vibration level in mm/s |
| `pressure_bar` | Operating pressure |
| `current_amp` | Electric current consumption |
| `load_percentage` | Equipment load percentage |
| `maintenance_days_since_last` | Days since the last maintenance activity |
| `failure_history_count` | Number of previous failures |
| `environment_dust_level` | Dust level: low, medium, or high |
| `shift` | Operating shift: day or night |
| `failure_next_7_days` | Target variable: 1 if failure is expected, 0 otherwise |

### Target Variable Logic

The target variable `failure_next_7_days` is generated using realistic risk logic. Failure risk increases when:

- Temperature is high
- Vibration is high
- Maintenance has not been performed recently
- The equipment has a higher failure history
- Dust level is high
- Operation hours are high
- Load percentage is high

This makes the dataset useful for demonstrating predictive maintenance concepts even without access to confidential industrial data.

---

## 8. Project Workflow

### 8.1 Data Generation

Script:

```bash
python src/data/generate_mining_dataset.py
```

Output:

```text
data/raw/mining_equipment_data.csv
```

This script creates a synthetic dataset with operational variables commonly used in industrial maintenance analytics.

---

### 8.2 Data Processing with Pandas

Script:

```bash
python src/processing/pandas_processing.py
```

Output:

```text
data/processed/mining_equipment_processed.csv
```

This step performs:

- Null validation
- Duplicate validation
- Categorical encoding
- Feature engineering
- Export of the final dataset for Machine Learning

Created features:

| Feature | Purpose |
|---|---|
| `high_temperature_flag` | Identifies equipment operating under high temperature |
| `high_vibration_flag` | Identifies equipment with high vibration |
| `maintenance_risk_flag` | Flags assets with possible maintenance risk |
| `operational_stress_score` | Combines operational stress signals into one score |

---

### 8.3 Data Processing with PySpark

Script:

```bash
python src/processing/pyspark_processing.py
```

Output:

```text
data/processed/pyspark_equipment_summary.csv
```

This step demonstrates basic distributed data processing skills using PySpark. It reads the raw CSV file, validates null values, creates new columns, and generates aggregated metrics by equipment type.

Aggregations include:

- Average temperature
- Average vibration
- Average operation hours
- Failure rate by equipment type

---

### 8.4 Exploratory Data Analysis

Notebook:

```text
notebooks/01_exploratory_analysis.ipynb
```

The notebook includes:

- Initial dataset inspection
- Descriptive statistics
- Failure distribution
- Average temperature by equipment type
- Average vibration by equipment type
- Relationship between maintenance days and failure
- Relationship between vibration and failure
- Business-oriented insights for predictive maintenance

The EDA helps translate technical patterns into operational decisions for maintenance teams.

---

### 8.5 Model Training

Script:

```bash
python src/models/train_model.py
```

Output:

```text
models/failure_prediction_model.pkl
reports/metrics_report.md
```

The script trains and compares three models:

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The best model is selected by prioritizing **recall** because, in predictive maintenance, missing a real failure is usually more critical than generating an additional preventive alert.

---

### 8.6 Model Evaluation

Current model comparison:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7325 | 0.3750 | 0.7354 | 0.4967 |
| Random Forest Classifier | 0.7895 | 0.4320 | 0.5487 | 0.4834 |
| Gradient Boosting Classifier | 0.8310 | 0.5755 | 0.2228 | 0.3213 |

Selected model:

```text
Logistic Regression
```

Reason:

The Logistic Regression model achieved the highest recall. This means it detected the largest proportion of real failure cases in the test dataset.

From a mining maintenance perspective, this is important because a false negative could represent a piece of equipment that is actually at risk but was not flagged by the model.

---

### 8.7 Experiment Tracking with MLflow

The training script also registers experiments in MLflow.

Experiment name:

```text
mining_failure_prediction_experiment
```

To open the MLflow UI locally:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Then open:

```text
http://localhost:5000
```

MLflow tracks:

- Model name
- Main model parameters
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix artifact
- Trained model artifact

---

### 8.8 API with FastAPI

Script:

```text
src/api/main.py
```

Run the API:

```bash
uvicorn src.api.main:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks if the API is running |
| POST | `/predict` | Predicts the failure risk of a mining asset |

---

### 8.9 Basic Model Monitoring

Script:

```bash
python src/monitoring/model_monitoring.py
```

Output:

```text
reports/monitoring_report.md
```

The monitoring module compares historical data against newly simulated data and checks possible changes in:

- `temperature_celsius`
- `vibration_mm_s`
- `operation_hours`
- `load_percentage`

This is a basic data drift monitoring approach. In production, monitoring is important because model performance can degrade when the operational behavior of the equipment changes over time.

---

## 9. Expected Results

This project is expected to produce:

- A synthetic mining equipment dataset with 10,000 records
- A cleaned and processed Machine Learning dataset
- Feature engineering variables related to maintenance risk
- EDA charts and business insights
- A trained classification model for failure prediction
- A metrics report comparing multiple models
- MLflow experiment tracking
- A FastAPI service for real-time predictions
- A basic monitoring report for possible data drift
- A Dockerfile to run the API in a container

---

## 10. How to Run the Project

### 10.1 Clone the Repository

```bash
git clone <your-repository-url>
cd mining-analytics-ml-pipeline
```

### 10.2 Create a Virtual Environment

Windows PowerShell:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 10.3 Install Dependencies

```bash
pip install -r requirements.txt
```

### 10.4 Generate Raw Data

```bash
python src/data/generate_mining_dataset.py
```

### 10.5 Process Data with Pandas

```bash
python src/processing/pandas_processing.py
```

### 10.6 Run PySpark Processing

```bash
python src/processing/pyspark_processing.py
```

### 10.7 Train the Model

```bash
python src/models/train_model.py
```

### 10.8 Launch MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open:

```text
http://localhost:5000
```

### 10.9 Run the API

```bash
uvicorn src.api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

### 10.10 Run Basic Monitoring

```bash
python src/monitoring/model_monitoring.py
```

---

## 11. Run with Docker

Build the Docker image:

```bash
docker build -t mining-analytics-api .
```

Run the container:

```bash
docker run -p 8000:8000 mining-analytics-api
```

Open:

```text
http://localhost:8000/docs
```

---

## 12. Prediction Example

Endpoint:

```text
POST /predict
```

Example request:

```json
{
  "equipment_type": "crusher",
  "operation_hours": 8500,
  "temperature_celsius": 96,
  "vibration_mm_s": 9.2,
  "pressure_bar": 8.0,
  "current_amp": 600,
  "load_percentage": 88,
  "maintenance_days_since_last": 120,
  "failure_history_count": 4,
  "environment_dust_level": "high",
  "shift": "night"
}
```

Example response:

```json
{
  "failure_probability": 0.99,
  "risk_level": "high",
  "recommendation": "Schedule a priority inspection. Review vibration, temperature, failure history, and maintenance condition before allowing extended operation."
}
```

Risk levels:

| Risk Level | Interpretation | Suggested Action |
|---|---|---|
| Low | Equipment condition appears stable | Continue normal monitoring |
| Medium | Some risk signals are present | Schedule preventive inspection |
| High | Strong failure risk signals are present | Schedule priority inspection |

---

## 13. Business Value for Mining Operations

This project demonstrates how analytics can support predictive maintenance decisions in mining by:

- Prioritizing assets with higher probability of failure
- Supporting preventive maintenance planning
- Reducing unplanned downtime risk
- Improving reliability of critical equipment
- Helping maintenance teams focus on high-risk assets
- Creating a foundation for future integration with sensor data, SCADA systems, historians, or maintenance management systems

Although the dataset is synthetic, the workflow reflects a realistic analytics use case for mining operations.

---

## 14. Key Learnings

This project helped develop practical skills in:

- Structuring an end-to-end Machine Learning project
- Creating synthetic industrial data with business logic
- Cleaning and transforming data with Pandas
- Performing basic distributed processing with PySpark
- Building visual analysis with Matplotlib
- Training and comparing classification models
- Selecting models using a business-oriented metric
- Tracking experiments with MLflow
- Serving a model using FastAPI
- Creating basic monitoring for data drift
- Preparing a project for GitHub and recruiter review
- Containerizing an API with Docker

---

## 15. Future Improvements

Planned improvements include:

### Azure

- Store data in Azure Data Lake Storage
- Deploy the API using Azure App Service or Azure Container Apps
- Use Azure Machine Learning for experiment tracking and model registry

### Databricks

- Move PySpark processing to Databricks notebooks
- Use Delta Lake for reliable data storage
- Build a more realistic batch pipeline for operational data

### CI/CD

- Add GitHub Actions for automated testing
- Validate code quality before merging changes
- Automate Docker image build and deployment

### Advanced Monitoring

- Add model performance monitoring
- Track prediction drift and feature drift over time
- Add alerts when drift exceeds thresholds
- Store monitoring history for trend analysis

### Real Data Integration

- Integrate sensor data from real mining equipment
- Connect with maintenance work orders
- Include failure labels from historical maintenance records
- Add time-series features for vibration, temperature, and load trends

---

## 16. Recruiter Summary

This project shows the ability to build a complete analytics solution for a mining maintenance use case. It combines data engineering, exploratory analysis, Machine Learning, MLOps fundamentals, API development, monitoring, and Docker-based deployment.

It is especially relevant for roles such as:

- Analytics Intern
- Data Science Intern
- Machine Learning Intern
- MLOps Intern
- Reliability Analytics Intern
- Predictive Maintenance Analytics Intern

---

## 17. Disclaimer

This project uses synthetic data created for educational and portfolio purposes. The logic is inspired by common predictive maintenance variables, but it does not represent confidential or real operational data from any mining company.
