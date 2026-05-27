"""
Generate a synthetic dataset for mining equipment failure prediction.

Project: mining-analytics-ml-pipeline
Goal: Create operational data that can be used to train a Machine Learning
model to predict whether an equipment asset may fail in the next 7 days.

The dataset is synthetic, but the relationships are designed to look realistic:
- Higher temperature, vibration, current, load, dust, and overdue maintenance
  increase failure risk.
- Equipment with more historical failures has higher future risk.
- Some equipment types are naturally more critical than others.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "mining_equipment_synthetic.csv"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "outputs" / "data_generation_summary.json"


EQUIPMENT_TYPES = [
    "haul_truck",
    "excavator",
    "crusher",
    "conveyor_belt",
    "drill_rig",
    "ball_mill",
]

SHIFTS = ["day", "night"]
DUST_LEVELS = ["low", "medium", "high"]


# Baseline operating ranges by equipment type.
# These values are not intended to represent a specific mine; they are useful
# for a portfolio project because they create clear and explainable patterns.
EQUIPMENT_PROFILE = {
    "haul_truck": {
        "temperature_mean": 82,
        "vibration_mean": 4.8,
        "pressure_mean": 160,
        "current_mean": 420,
        "load_mean": 78,
        "risk_offset": 0.15,
    },
    "excavator": {
        "temperature_mean": 76,
        "vibration_mean": 5.2,
        "pressure_mean": 210,
        "current_mean": 360,
        "load_mean": 74,
        "risk_offset": 0.10,
    },
    "crusher": {
        "temperature_mean": 88,
        "vibration_mean": 7.4,
        "pressure_mean": 185,
        "current_mean": 510,
        "load_mean": 84,
        "risk_offset": 0.35,
    },
    "conveyor_belt": {
        "temperature_mean": 62,
        "vibration_mean": 3.2,
        "pressure_mean": 90,
        "current_mean": 180,
        "load_mean": 70,
        "risk_offset": -0.10,
    },
    "drill_rig": {
        "temperature_mean": 79,
        "vibration_mean": 6.1,
        "pressure_mean": 230,
        "current_mean": 390,
        "load_mean": 76,
        "risk_offset": 0.20,
    },
    "ball_mill": {
        "temperature_mean": 86,
        "vibration_mean": 6.8,
        "pressure_mean": 170,
        "current_mean": 560,
        "load_mean": 82,
        "risk_offset": 0.30,
    },
}


def sigmoid(value: np.ndarray) -> np.ndarray:
    """Convert a numeric score into a probability between 0 and 1."""
    return 1 / (1 + np.exp(-value))


def generate_synthetic_dataset(n_rows: int = 10_000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic mining equipment operational data.

    Args:
        n_rows: Number of records to generate.
        random_state: Seed used to make the dataset reproducible.

    Returns:
        A pandas DataFrame with operational variables and the target column.
    """
    rng = np.random.default_rng(random_state)

    equipment_type = rng.choice(
        EQUIPMENT_TYPES,
        size=n_rows,
        p=[0.22, 0.16, 0.17, 0.20, 0.13, 0.12],
    )

    equipment_id = [
        f"EQ-{eq_type.upper().replace('_', '-')}-{rng.integers(1, 151):03d}"
        for eq_type in equipment_type
    ]

    shift = rng.choice(SHIFTS, size=n_rows, p=[0.52, 0.48])
    environment_dust_level = rng.choice(DUST_LEVELS, size=n_rows, p=[0.25, 0.50, 0.25])

    operation_hours = rng.gamma(shape=5.0, scale=850.0, size=n_rows)
    operation_hours = np.clip(operation_hours, 50, 25_000).round(1)

    maintenance_days_since_last = rng.gamma(shape=2.2, scale=18.0, size=n_rows)
    maintenance_days_since_last = np.clip(maintenance_days_since_last, 0, 180).round(0).astype(int)

    failure_history_count = rng.poisson(lam=1.2, size=n_rows)
    failure_history_count = np.clip(failure_history_count, 0, 12)

    temperature_celsius = np.zeros(n_rows)
    vibration_mm_s = np.zeros(n_rows)
    pressure_bar = np.zeros(n_rows)
    current_amp = np.zeros(n_rows)
    load_percentage = np.zeros(n_rows)
    equipment_risk_offset = np.zeros(n_rows)

    for eq_type in EQUIPMENT_TYPES:
        mask = equipment_type == eq_type
        profile = EQUIPMENT_PROFILE[eq_type]
        count = int(mask.sum())

        dust_adjustment = np.select(
            [environment_dust_level[mask] == "medium", environment_dust_level[mask] == "high"],
            [2.0, 5.0],
            default=0.0,
        )
        night_adjustment = np.where(shift[mask] == "night", 1.5, 0.0)

        load_values = rng.normal(profile["load_mean"], 11, count)
        load_values = np.clip(load_values, 35, 110)

        temperature_celsius[mask] = rng.normal(profile["temperature_mean"], 7, count) + dust_adjustment + night_adjustment
        vibration_mm_s[mask] = rng.normal(profile["vibration_mean"], 1.4, count) + 0.018 * (load_values - 70)
        pressure_bar[mask] = rng.normal(profile["pressure_mean"], 22, count)
        current_amp[mask] = rng.normal(profile["current_mean"], 55, count) + 2.1 * (load_values - 70)
        load_percentage[mask] = load_values
        equipment_risk_offset[mask] = profile["risk_offset"]

    temperature_celsius = np.clip(temperature_celsius, 35, 125).round(2)
    vibration_mm_s = np.clip(vibration_mm_s, 0.2, 18).round(2)
    pressure_bar = np.clip(pressure_bar, 20, 320).round(2)
    current_amp = np.clip(current_amp, 40, 850).round(2)
    load_percentage = np.clip(load_percentage, 10, 120).round(2)

    dust_risk = np.select(
        [environment_dust_level == "medium", environment_dust_level == "high"],
        [0.20, 0.55],
        default=0.0,
    )
    shift_risk = np.where(shift == "night", 0.12, 0.0)

    # Risk score: intentionally simple and explainable for a beginner-friendly ML project.
    risk_score = (
        -4.20
        + 0.030 * (temperature_celsius - 70)
        + 0.260 * (vibration_mm_s - 4)
        + 0.006 * (current_amp - 300)
        + 0.020 * (load_percentage - 70)
        + 0.015 * (maintenance_days_since_last - 30)
        + 0.260 * failure_history_count
        + 0.00006 * (operation_hours - 4_000)
        + dust_risk
        + shift_risk
        + equipment_risk_offset
    )

    failure_probability = sigmoid(risk_score)
    failure_next_7_days = rng.binomial(n=1, p=failure_probability)

    dataframe = pd.DataFrame(
        {
            "equipment_id": equipment_id,
            "equipment_type": equipment_type,
            "operation_hours": operation_hours,
            "temperature_celsius": temperature_celsius,
            "vibration_mm_s": vibration_mm_s,
            "pressure_bar": pressure_bar,
            "current_amp": current_amp,
            "load_percentage": load_percentage,
            "maintenance_days_since_last": maintenance_days_since_last,
            "failure_history_count": failure_history_count,
            "environment_dust_level": environment_dust_level,
            "shift": shift,
            "failure_next_7_days": failure_next_7_days,
        }
    )

    return dataframe


def save_summary(dataframe: pd.DataFrame, summary_path: Path) -> None:
    """Save a small JSON summary to document the generated dataset."""
    summary = {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "target_column": "failure_next_7_days",
        "failure_rate": round(float(dataframe["failure_next_7_days"].mean()), 4),
        "equipment_type_distribution": dataframe["equipment_type"].value_counts(normalize=True).round(4).to_dict(),
        "shift_distribution": dataframe["shift"].value_counts(normalize=True).round(4).to_dict(),
        "dust_level_distribution": dataframe["environment_dust_level"].value_counts(normalize=True).round(4).to_dict(),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)


def parse_arguments() -> argparse.Namespace:
    """Read command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic mining equipment data for failure prediction."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10_000,
        help="Number of rows to generate. Default: 10000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV output path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"JSON summary output path. Default: {DEFAULT_SUMMARY_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the dataset and save it as a CSV file."""
    args = parse_arguments()

    if args.rows < 10_000:
        raise ValueError("The dataset must contain at least 10,000 rows for this project.")

    dataframe = generate_synthetic_dataset(n_rows=args.rows, random_state=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(args.output, index=False)
    save_summary(dataframe, args.summary_output)

    failure_rate = dataframe["failure_next_7_days"].mean()

    print("Synthetic dataset generated successfully.")
    print(f"Rows: {dataframe.shape[0]:,}")
    print(f"Columns: {dataframe.shape[1]}")
    print(f"Failure rate: {failure_rate:.2%}")
    print(f"CSV saved at: {args.output}")
    print(f"Summary saved at: {args.summary_output}")


if __name__ == "__main__":
    main()
