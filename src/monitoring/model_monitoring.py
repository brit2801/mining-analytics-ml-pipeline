"""
Basic model monitoring script for the mining analytics ML pipeline.

This script compares the historical processed dataset against a new simulated
sample. The goal is to detect possible data drift in important operational
variables before the model is used in production.

Run from the project root:
    python src/monitoring/model_monitoring.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "mining_equipment_processed.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "monitoring_report.md"


# Variables that are monitored for possible data drift.
MONITORED_COLUMNS = [
    "temperature_celsius",
    "vibration_mm_s",
    "operation_hours",
    "load_percentage",
]


# Simple drift thresholds expressed as percentage change in the mean.
# These thresholds are intentionally simple so the project is easy to explain.
DRIFT_THRESHOLDS = {
    "temperature_celsius": 5.0,  # Temperature changes of 5% can be relevant.
    "vibration_mm_s": 10.0,     # Vibration is naturally more variable.
    "operation_hours": 8.0,     # Higher accumulated operation may indicate aging.
    "load_percentage": 5.0,     # Load changes affect equipment stress.
}


RANDOM_STATE = 42
NEW_SAMPLE_SIZE = 1_000


def load_base_dataset(file_path: Path) -> pd.DataFrame:
    """Load the historical processed dataset used as the monitoring baseline."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {file_path}\n"
            "First run: python src/processing/pandas_processing.py"
        )

    df = pd.read_csv(file_path)

    missing_columns = [column for column in MONITORED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing monitored columns: {missing_columns}")

    return df


def simulate_new_data(base_df: pd.DataFrame, sample_size: int = NEW_SAMPLE_SIZE) -> pd.DataFrame:
    """
    Create a new simulated sample with operational changes.

    In a real mining operation, this new data would come from sensors, dispatch
    systems, inspections, or maintenance records. Here we simulate a scenario
    where assets operate under more demanding conditions.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    # Sample historical rows to preserve the general structure of the data.
    sample_size = min(sample_size, len(base_df))
    new_df = base_df.sample(n=sample_size, random_state=RANDOM_STATE).copy()

    # Simulate harsher operating conditions.
    # These changes are intentional to demonstrate how monitoring detects drift.
    new_df["temperature_celsius"] = (
        new_df["temperature_celsius"] + rng.normal(loc=6.0, scale=2.0, size=sample_size)
    ).clip(lower=0)

    new_df["vibration_mm_s"] = (
        new_df["vibration_mm_s"] + rng.normal(loc=1.2, scale=0.4, size=sample_size)
    ).clip(lower=0)

    new_df["load_percentage"] = (
        new_df["load_percentage"] + rng.normal(loc=8.0, scale=3.0, size=sample_size)
    ).clip(lower=0, upper=100)

    new_df["operation_hours"] = (
        new_df["operation_hours"] + rng.normal(loc=700.0, scale=250.0, size=sample_size)
    ).clip(lower=0)

    # Keep risk flags consistent with the simulated values.
    if "high_temperature_flag" in new_df.columns:
        new_df["high_temperature_flag"] = (new_df["temperature_celsius"] >= 85).astype(int)

    if "high_vibration_flag" in new_df.columns:
        new_df["high_vibration_flag"] = (new_df["vibration_mm_s"] >= 7.5).astype(int)

    return new_df


def calculate_monitoring_summary(
    base_df: pd.DataFrame,
    new_df: pd.DataFrame,
    monitored_columns: List[str],
) -> pd.DataFrame:
    """Compare mean values between historical data and new simulated data."""
    rows = []

    for column in monitored_columns:
        baseline_mean = base_df[column].mean()
        new_mean = new_df[column].mean()
        absolute_change = new_mean - baseline_mean

        # Avoid division by zero in case a baseline mean is zero.
        if baseline_mean == 0:
            percentage_change = 0.0
        else:
            percentage_change = (absolute_change / baseline_mean) * 100

        threshold = DRIFT_THRESHOLDS[column]
        possible_drift = abs(percentage_change) >= threshold

        rows.append(
            {
                "variable": column,
                "baseline_mean": baseline_mean,
                "new_data_mean": new_mean,
                "absolute_change": absolute_change,
                "percentage_change": percentage_change,
                "threshold_percentage": threshold,
                "possible_drift": possible_drift,
            }
        )

    return pd.DataFrame(rows)


def build_business_interpretation(summary_df: pd.DataFrame) -> List[str]:
    """Create simple business insights based on detected drift."""
    insights = []

    drift_variables = summary_df.loc[summary_df["possible_drift"], "variable"].tolist()

    if not drift_variables:
        insights.append(
            "No se detectaron cambios relevantes en los promedios monitoreados. "
            "El comportamiento de los datos nuevos es similar a la línea base histórica."
        )
        return insights

    if "temperature_celsius" in drift_variables:
        insights.append(
            "La temperatura promedio aumentó de forma relevante. Esto puede indicar "
            "mayor exigencia térmica, problemas de enfriamiento o condiciones de operación más severas."
        )

    if "vibration_mm_s" in drift_variables:
        insights.append(
            "La vibración promedio cambió de forma importante. En mantenimiento predictivo, "
            "esto puede ser una señal temprana de desbalance, desalineación, desgaste mecánico o soltura."
        )

    if "operation_hours" in drift_variables:
        insights.append(
            "Las horas de operación promedio aumentaron. Esto puede significar que la flota o los activos "
            "están acumulando mayor envejecimiento operativo frente a la información usada para entrenar el modelo."
        )

    if "load_percentage" in drift_variables:
        insights.append(
            "La carga promedio aumentó de forma relevante. Una operación sostenida con mayor carga "
            "puede acelerar degradación, elevar temperatura y aumentar riesgo de falla."
        )

    return insights


def save_monitoring_report(
    report_path: Path,
    base_df: pd.DataFrame,
    new_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    insights: List[str],
) -> None:
    """Save the monitoring results as a Markdown report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    drift_count = int(summary_df["possible_drift"].sum())
    drift_status = "Possible data drift detected" if drift_count > 0 else "No relevant data drift detected"

    lines = [
        "# Model Monitoring Report - Mining Failure Prediction",
        "",
        "## Objective",
        "",
        "Compare historical processed data against a new simulated sample to identify "
        "possible data drift in key operational variables used by the failure prediction model.",
        "",
        "## Datasets compared",
        "",
        f"- Historical baseline rows: {len(base_df):,}",
        f"- New simulated rows: {len(new_df):,}",
        f"- Monitoring status: **{drift_status}**",
        "",
        "## Variables monitored",
        "",
        "The monitoring process focuses on variables that are operationally important "
        "for predictive maintenance in mining:",
        "",
        "- `temperature_celsius`",
        "- `vibration_mm_s`",
        "- `operation_hours`",
        "- `load_percentage`",
        "",
        "## Drift comparison",
        "",
        "| Variable | Baseline mean | New data mean | Absolute change | % change | Threshold | Possible drift |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]

    for _, row in summary_df.iterrows():
        drift_label = "Yes" if row["possible_drift"] else "No"
        lines.append(
            f"| {row['variable']} | "
            f"{row['baseline_mean']:.2f} | "
            f"{row['new_data_mean']:.2f} | "
            f"{row['absolute_change']:.2f} | "
            f"{row['percentage_change']:.2f}% | "
            f"{row['threshold_percentage']:.2f}% | "
            f"{drift_label} |"
        )

    lines.extend(
        [
            "",
            "## Business insights",
            "",
        ]
    )

    for insight in insights:
        lines.append(f"- {insight}")

    lines.extend(
        [
            "",
            "## Why monitoring matters in production",
            "",
            "A Machine Learning model is trained with historical data, but mining operations "
            "change over time. Equipment can age, operating loads can increase, environmental "
            "conditions can become harsher, and maintenance strategies can change. When the "
            "new production data becomes very different from the training data, model performance "
            "can degrade even if the original training metrics were good.",
            "",
            "Monitoring helps the maintenance and analytics teams detect when the model may need "
            "review, recalibration or retraining. In predictive maintenance, this is important "
            "because an unreliable model could miss real failure risk, generate unnecessary alerts, "
            "or reduce trust from planners, reliability engineers and operations teams.",
            "",
            "## Recommended actions",
            "",
            "- Review assets with high temperature, vibration and load before long operating windows.",
            "- Compare drift alerts with real maintenance work orders and inspection findings.",
            "- Retrain the model if drift persists across multiple monitoring periods.",
            "- Track recall over time because missing a real failure is critical in mining operations.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the basic model monitoring workflow."""
    print("Starting model monitoring...")

    base_df = load_base_dataset(BASE_DATA_PATH)
    new_df = simulate_new_data(base_df)
    summary_df = calculate_monitoring_summary(base_df, new_df, MONITORED_COLUMNS)
    insights = build_business_interpretation(summary_df)

    save_monitoring_report(REPORT_PATH, base_df, new_df, summary_df, insights)

    print("Monitoring summary:")
    print(summary_df.to_string(index=False))
    print(f"\nMonitoring report saved at: {REPORT_PATH}")


if __name__ == "__main__":
    main()