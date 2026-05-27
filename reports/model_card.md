# Model Card: Mining Equipment Failure Prediction Model

## 1. Model Name

**Mining Equipment Failure Prediction Model**

The selected model is a **Logistic Regression classifier** trained to estimate the probability that a mining asset may fail within the next 7 days.

---

## 2. Objective

The objective of this model is to support **predictive maintenance analytics** in a mining context by identifying equipment with a higher risk of failure in the short term.

The model is designed as part of an end-to-end analytics and MLOps demonstration pipeline that includes synthetic data generation, data processing, exploratory analysis, model training, evaluation, API deployment, and basic monitoring.

---

## 3. Problem Type

This is a **binary classification problem**.

The model predicts whether a mining equipment record belongs to one of two classes:

- `0`: No expected failure in the next 7 days.
- `1`: Possible failure in the next 7 days.

---

## 4. Target Variable

The target variable is:

```text
failure_next_7_days
```

This variable indicates whether the equipment is expected to fail within the next 7 days.

---

## 5. Input Variables

The model uses operational, maintenance, and environmental variables related to mining equipment behavior.

### Original Input Variables

```text
equipment_type
operation_hours
temperature_celsius
vibration_mm_s
pressure_bar
current_amp
load_percentage
maintenance_days_since_last
failure_history_count
environment_dust_level
shift
```

### Engineered Variables

```text
high_temperature_flag
high_vibration_flag
maintenance_risk_flag
operational_stress_score
```

### Encoded Categorical Variables

Categorical variables are transformed using one-hot encoding:

```text
equipment_type_conveyor_belt
equipment_type_crusher
equipment_type_mill
equipment_type_pump
equipment_type_truck
environment_dust_level_high
environment_dust_level_low
environment_dust_level_medium
shift_day
shift_night
```

---

## 6. Dataset Used

The model was trained using the processed dataset:

```text
data/processed/mining_equipment_processed.csv
```

The original dataset was generated synthetically from:

```text
data/raw/mining_equipment_data.csv
```

### Important Disclaimer

This dataset is **synthetic**. It was created for educational and portfolio purposes to simulate realistic mining equipment conditions such as temperature, vibration, operation hours, maintenance history, failure history, load percentage, current, pressure, environmental dust level, and work shift.

The model should **not** be used for real operational decisions without validation using real industrial data from sensors, maintenance systems, historians, or reliability databases.

---

## 7. Main Metrics

The model comparison prioritized **recall** because, in predictive maintenance, failing to detect a real potential failure can be more costly than generating an additional preventive alert.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7325 | 0.3750 | 0.7354 | 0.4967 |
| Random Forest Classifier | 0.7895 | 0.4320 | 0.5487 | 0.4834 |
| Gradient Boosting Classifier | 0.8310 | 0.5755 | 0.2228 | 0.3213 |

## Selected Model

**Logistic Regression** was selected because it achieved the highest recall among the evaluated models.

### Confusion Matrix for Selected Model

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 1201 | 440 |
| Actual 1 | 95 | 264 |

### Business Interpretation of Recall

The selected model detected approximately **73.54%** of the simulated failure cases in the test set. This is useful in a predictive maintenance scenario because the model is more sensitive to potential failures, reducing the chance of missing equipment that may require inspection.

---

## 8. Recommended Use

This model is recommended for **educational, demonstrative, and portfolio purposes**.

It can be used to demonstrate:

- Predictive maintenance analytics for mining equipment.
- Binary classification for failure risk prediction.
- Feature engineering for industrial data.
- Model evaluation using maintenance-oriented metrics.
- MLOps concepts such as experiment tracking, API deployment, and monitoring.
- How an analytics solution can support maintenance prioritization.

In a real mining operation, this type of model could support maintenance teams by ranking equipment according to failure probability and helping prioritize inspection routes.

---

## 9. Limitations

This model has important limitations:

- The dataset is synthetic and does not represent real equipment behavior with full accuracy.
- Sensor noise, calibration problems, missing values, shutdown periods, and abnormal operating modes are simplified.
- The model does not include time-series behavior or degradation trends over time.
- The model does not include maintenance work orders, spare parts availability, production constraints, or operator comments.
- The failure label was simulated using predefined risk logic, not real historical failures.
- The model should not be used for real safety-critical or production-critical decisions without industrial validation.

---

## 10. Risks

Potential risks include:

- **False negatives:** The model may classify a risky equipment condition as low risk, which could lead to missed inspections.
- **False positives:** The model may generate alerts for equipment that will not fail, increasing unnecessary maintenance workload.
- **Data drift:** Operational conditions may change over time, reducing model performance.
- **Overconfidence:** Users may trust the model output without engineering review.
- **Synthetic bias:** Since the dataset is simulated, the model may learn patterns that do not fully exist in real mining operations.

---

## 11. Ethical Considerations

This model should be used as a **decision-support tool**, not as a replacement for maintenance engineers, reliability specialists, planners, or field technicians.

Ethical considerations include:

- The model output should be reviewed by qualified personnel before operational action.
- Predictions should not be used to assign blame to operators, mechanics, or maintenance teams.
- The system should be transparent about its limitations and uncertainty.
- Real deployments should protect operational data and comply with company data governance policies.
- Any model used in production should be monitored, audited, and periodically validated.

---

## 12. Monitoring Considerations

The project includes a basic monitoring script that compares historical data with newly simulated data to detect possible data drift in:

```text
temperature_celsius
vibration_mm_s
operation_hours
load_percentage
```

Monitoring is important because mining equipment behavior can change due to aging, operating conditions, environmental factors, process changes, maintenance practices, or sensor issues.

If drift is detected, the model should be reviewed and may require retraining with more recent data.

---

## 13. Next Improvements

Recommended future improvements include:

- Train the model with real historical maintenance and sensor data.
- Add time-series features such as rolling averages, trends, and rate of change.
- Use more advanced models such as XGBoost, LightGBM, or survival analysis models.
- Add model explainability using SHAP or feature importance analysis.
- Improve threshold tuning based on maintenance cost and risk tolerance.
- Integrate the pipeline with Azure Machine Learning or Databricks.
- Add CI/CD for automated testing and deployment.
- Implement advanced monitoring for data drift, model drift, and prediction drift.
- Connect the API to real operational systems such as historians, CMMS, or dashboards.
- Create a Power BI dashboard to visualize equipment risk, alerts, and maintenance priorities.

---

## 14. Final Note

This model is part of a personal project called **Mining Analytics ML Pipeline**. Its purpose is to demonstrate applied analytics, machine learning, and MLOps skills for mining and predictive maintenance use cases.

The model is **not production-ready** and must be validated with real industrial data before being used in operational decision-making.
