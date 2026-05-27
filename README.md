# mining-analytics-ml-pipeline

End-to-end analytics and Machine Learning project for simulated mining equipment failure risk prediction.

## Business value for a mining company

Mining operations depend on the availability of critical equipment such as haul trucks, crushers, conveyors, mills, excavators, and drill rigs. Unexpected failures can affect production continuity, maintenance planning, safety exposure, and operating costs.

This project demonstrates how analytics can support predictive maintenance by transforming operational data into a failure risk prediction system. The final goal is to help maintenance and operations teams prioritize inspections, detect abnormal behavior earlier, and make data-driven decisions.

## Project objective

Build a complete analytics pipeline that:

1. Simulates operational data from mining equipment.
2. Processes the data using Python and PySpark.
3. Trains a Machine Learning model to predict `failure_next_7_days`.
4. Evaluates model performance and logs metrics with MLflow.
5. Saves the trained model with Joblib.
6. Exposes a prediction API using FastAPI.
7. Documents the project clearly for GitHub.

## Current project status

This first version includes:

- Project folder structure.
- `requirements.txt` with the main dependencies.
- Synthetic dataset generator with at least 10,000 records.
- Initial README documentation.

## Project structure

```text
mining-analytics-ml-pipeline/
├── data/
│   ├── raw/              # Original generated data
│   ├── processed/        # Cleaned and transformed data
│   └── outputs/          # Summaries, metrics, and generated outputs
├── notebooks/            # Exploratory analysis notebooks
├── src/
│   ├── data/             # Data generation and data loading scripts
│   ├── processing/       # Data cleaning and feature engineering scripts
│   ├── models/           # Model training and evaluation scripts
│   ├── api/              # FastAPI prediction service
│   └── monitoring/       # Monitoring and data drift scripts
├── models/               # Saved trained models
├── reports/              # Charts, reports, and final documentation
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Dataset

The synthetic dataset includes the following columns:

- `equipment_id`
- `equipment_type`
- `operation_hours`
- `temperature_celsius`
- `vibration_mm_s`
- `pressure_bar`
- `current_amp`
- `load_percentage`
- `maintenance_days_since_last`
- `failure_history_count`
- `environment_dust_level`
- `shift`
- `failure_next_7_days`

The target variable is `failure_next_7_days`, where:

- `0` means no expected failure in the next 7 days.
- `1` means higher risk of failure in the next 7 days.

## How to run the first step

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the synthetic dataset:

```bash
python src/data/generate_synthetic_data.py
```

Generate a larger dataset:

```bash
python src/data/generate_synthetic_data.py --rows 50000 --seed 123
```

Expected outputs:

```text
data/raw/mining_equipment_synthetic.csv
data/outputs/data_generation_summary.json
```

## Why this project is useful for a Practicante de Analytics role

This project shows practical skills in:

- Data simulation and understanding of industrial variables.
- Python data processing.
- Predictive maintenance analytics.
- Machine Learning for classification.
- Model evaluation and experiment tracking.
- API deployment basics.
- GitHub documentation and reproducible project structure.

## Next steps

1. Add exploratory data analysis in a notebook.
2. Build a PySpark processing pipeline.
3. Train a baseline classification model.
4. Evaluate the model using precision, recall, F1-score, ROC-AUC, and confusion matrix.
5. Track experiments with MLflow.
6. Save the best model with Joblib.
7. Create a FastAPI prediction endpoint.
8. Add Docker support.
