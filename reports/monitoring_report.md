# Model Monitoring Report - Mining Failure Prediction

## Objective

Compare historical processed data against a new simulated sample to identify possible data drift in key operational variables used by the failure prediction model.

## Datasets compared

- Historical baseline rows: 20,000
- New simulated rows: 1,000
- Monitoring status: **Possible data drift detected**

## Variables monitored

The monitoring process focuses on variables that are operationally important for predictive maintenance in mining:

- `temperature_celsius`
- `vibration_mm_s`
- `operation_hours`
- `load_percentage`

## Drift comparison

| Variable | Baseline mean | New data mean | Absolute change | % change | Threshold | Possible drift |
|---|---:|---:|---:|---:|---:|:---:|
| temperature_celsius | 76.32 | 81.86 | 5.54 | 7.27% | 5.00% | Yes |
| vibration_mm_s | 6.19 | 7.30 | 1.12 | 18.07% | 10.00% | Yes |
| operation_hours | 4480.15 | 5185.71 | 705.55 | 15.75% | 8.00% | Yes |
| load_percentage | 74.40 | 81.97 | 7.56 | 10.16% | 5.00% | Yes |

## Business insights

- La temperatura promedio aumentó de forma relevante. Esto puede indicar mayor exigencia térmica, problemas de enfriamiento o condiciones de operación más severas.
- La vibración promedio cambió de forma importante. En mantenimiento predictivo, esto puede ser una señal temprana de desbalance, desalineación, desgaste mecánico o soltura.
- Las horas de operación promedio aumentaron. Esto puede significar que la flota o los activos están acumulando mayor envejecimiento operativo frente a la información usada para entrenar el modelo.
- La carga promedio aumentó de forma relevante. Una operación sostenida con mayor carga puede acelerar degradación, elevar temperatura y aumentar riesgo de falla.

## Why monitoring matters in production

A Machine Learning model is trained with historical data, but mining operations change over time. Equipment can age, operating loads can increase, environmental conditions can become harsher, and maintenance strategies can change. When the new production data becomes very different from the training data, model performance can degrade even if the original training metrics were good.

Monitoring helps the maintenance and analytics teams detect when the model may need review, recalibration or retraining. In predictive maintenance, this is important because an unreliable model could miss real failure risk, generate unnecessary alerts, or reduce trust from planners, reliability engineers and operations teams.

## Recommended actions

- Review assets with high temperature, vibration and load before long operating windows.
- Compare drift alerts with real maintenance work orders and inspection findings.
- Retrain the model if drift persists across multiple monitoring periods.
- Track recall over time because missing a real failure is critical in mining operations.
