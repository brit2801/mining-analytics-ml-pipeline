# Metrics Report - Failure Prediction Model

## Objective

Train and compare machine learning models to predict whether a mining asset may fail in the next 7 days.

## Business criterion

The best model was selected by prioritizing **recall**. In predictive maintenance, a false negative is risky because it means the system did not detect a possible failure. Missing a real failure can lead to unplanned downtime, production losses, safety exposure, and higher maintenance costs.

## Dataset split

- Training rows: 16000
- Testing rows: 4000
- Number of features used: 22

## Model comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7238 | 0.3689 | 0.7196 | 0.4877 |
| Random Forest Classifier | 0.7758 | 0.4168 | 0.5691 | 0.4812 |
| Gradient Boosting Classifier | 0.8330 | 0.6137 | 0.2326 | 0.3373 |

## Selected model

**Logistic Regression** was selected as the best model because it achieved the strongest recall-oriented performance among the evaluated models.

## Confusion matrices

### Logistic Regression

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 2369 | 900 |
| Actual 1 | 205 | 526 |


### Random Forest Classifier

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 2687 | 582 |
| Actual 1 | 315 | 416 |


### Gradient Boosting Classifier

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 3162 | 107 |
| Actual 1 | 561 | 170 |


## Features used by the model

The model was trained using the following numeric variables:

- operation_hours
- temperature_celsius
- vibration_mm_s
- pressure_bar
- current_amp
- load_percentage
- maintenance_days_since_last
- failure_history_count
- high_temperature_flag
- high_vibration_flag
- maintenance_risk_flag
- operational_stress_score
- equipment_type_conveyor_belt
- equipment_type_crusher
- equipment_type_mill
- equipment_type_pump
- equipment_type_truck
- environment_dust_level_high
- environment_dust_level_low
- environment_dust_level_medium
- shift_day
- shift_night

## Business interpretation

This model can support maintenance planning by identifying assets with higher probability of failure in the next 7 days. In a mining operation, this type of analytics can help prioritize inspections, schedule planned maintenance, reduce unplanned downtime, and focus attention on equipment showing abnormal operating conditions such as high vibration, high temperature, long operation hours, or extended time since last maintenance.
