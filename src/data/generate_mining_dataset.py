"""
Generate a synthetic mining predictive maintenance dataset.

This script creates operational records for common mining equipment and saves
those records as a CSV file in data/raw/mining_equipment_data.csv.

The target variable, failure_next_7_days, is generated using a risk score.
The risk increases with realistic operational factors such as high temperature,
high vibration, long time since last maintenance, previous failures, high dust
level, and accumulated operation hours.

Run from the project root:
    python src/data/generate_mining_dataset.py

Optional:
    python src/data/generate_mining_dataset.py --rows 20000 --seed 123
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "mining_equipment_data.csv"


EQUIPMENT_TYPES = ["truck", "conveyor_belt", "pump", "crusher", "mill"]
DUST_LEVELS = ["low", "medium", "high"]
SHIFTS = ["day", "night"]


# Base operating profiles by equipment type.
# These values are not intended to represent a specific mine.
# They are reasonable synthetic ranges for a portfolio project.
EQUIPMENT_PROFILES = {
    "truck": {
        "temperature_mean": 78,
        "vibration_mean": 5.8,
        "pressure_mean": 6.0,
        "current_mean": 430,
        "load_mean": 74,
    },
    "conveyor_belt": {
        "temperature_mean": 62,
        "vibration_mean": 4.2,
        "pressure_mean": 4.0,
        "current_mean": 260,
        "load_mean": 68,
    },
    "pump": {
        "temperature_mean": 70,
        "vibration_mean": 4.8,
        "pressure_mean": 9.5,
        "current_mean": 310,
        "load_mean": 71,
    },
    "crusher": {
        "temperature_mean": 84,
        "vibration_mean": 7.2,
        "pressure_mean": 7.2,
        "current_mean": 520,
        "load_mean": 82,
    },
    "mill": {
        "temperature_mean": 80,
        "vibration_mean": 6.5,
        "pressure_mean": 6.8,
        "current_mean": 480,
        "load_mean": 79,
    },
}


def sigmoid(value: np.ndarray) -> np.ndarray:
    """Convert a risk score into a probability between 0 and 1."""
    return 1 / (1 + np.exp(-value))


def generate_dataset(rows: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset for mining predictive maintenance.

    Parameters
    ----------
    rows:
        Number of records to generate.
    seed:
        Random seed to make the result reproducible.

    Returns
    -------
    pd.DataFrame
        Synthetic mining equipment dataset.
    """
    rng = np.random.default_rng(seed)

    # Assign an equipment type to every record.
    equipment_type = rng.choice(
        EQUIPMENT_TYPES,
        size=rows,
        p=[0.25, 0.20, 0.20, 0.18, 0.17],
    )

    # Create IDs with a readable format, for example TRU-00001 or MIL-00008.
    type_prefix = {
        "truck": "TRU",
        "conveyor_belt": "CON",
        "pump": "PUM",
        "crusher": "CRU",
        "mill": "MIL",
    }
    equipment_id = [
        f"{type_prefix[equipment]}-{index + 1:05d}"
        for index, equipment in enumerate(equipment_type)
    ]

    # Operational variables.
    # Values are clipped to avoid unrealistic negative values or impossible ranges.
    operation_hours = rng.gamma(shape=5.0, scale=900.0, size=rows)
    operation_hours = np.clip(operation_hours, 100, 25_000).round(1)

    maintenance_days_since_last = rng.gamma(shape=3.0, scale=18.0, size=rows)
    maintenance_days_since_last = np.clip(maintenance_days_since_last, 1, 240).round(0).astype(int)

    failure_history_count = rng.poisson(lam=1.1, size=rows)
    failure_history_count = np.clip(failure_history_count, 0, 8)

    environment_dust_level = rng.choice(
        DUST_LEVELS,
        size=rows,
        p=[0.30, 0.45, 0.25],
    )

    shift = rng.choice(
        SHIFTS,
        size=rows,
        p=[0.55, 0.45],
    )

    # Generate numerical sensor variables based on the selected equipment profile.
    temperature_celsius = np.empty(rows)
    vibration_mm_s = np.empty(rows)
    pressure_bar = np.empty(rows)
    current_amp = np.empty(rows)
    load_percentage = np.empty(rows)

    for equipment in EQUIPMENT_TYPES:
        mask = equipment_type == equipment
        count = mask.sum()
        profile = EQUIPMENT_PROFILES[equipment]

        temperature_celsius[mask] = rng.normal(
            loc=profile["temperature_mean"],
            scale=8.5,
            size=count,
        )
        vibration_mm_s[mask] = rng.normal(
            loc=profile["vibration_mean"],
            scale=1.5,
            size=count,
        )
        pressure_bar[mask] = rng.normal(
            loc=profile["pressure_mean"],
            scale=1.2,
            size=count,
        )
        current_amp[mask] = rng.normal(
            loc=profile["current_mean"],
            scale=55,
            size=count,
        )
        load_percentage[mask] = rng.normal(
            loc=profile["load_mean"],
            scale=10,
            size=count,
        )

    # Add realistic stress effects:
    # equipment with higher load and longer operating hours tends to run hotter
    # and vibrate slightly more.
    temperature_celsius += (load_percentage - 70) * 0.12
    temperature_celsius += (operation_hours / 25_000) * 7

    vibration_mm_s += (operation_hours / 25_000) * 1.5
    vibration_mm_s += failure_history_count * 0.25

    current_amp += (load_percentage - 70) * 2.2

    # Clip variables to practical synthetic ranges.
    temperature_celsius = np.clip(temperature_celsius, 35, 125).round(2)
    vibration_mm_s = np.clip(vibration_mm_s, 0.5, 18).round(2)
    pressure_bar = np.clip(pressure_bar, 1, 18).round(2)
    current_amp = np.clip(current_amp, 80, 900).round(2)
    load_percentage = np.clip(load_percentage, 15, 100).round(2)

    # Convert categorical values into risk factors for the target variable.
    dust_risk = np.select(
        [
            environment_dust_level == "low",
            environment_dust_level == "medium",
            environment_dust_level == "high",
        ],
        [0.00, 0.35, 0.75],
        default=0.00,
    )

    equipment_risk = np.select(
        [
            equipment_type == "truck",
            equipment_type == "conveyor_belt",
            equipment_type == "pump",
            equipment_type == "crusher",
            equipment_type == "mill",
        ],
        [0.20, 0.10, 0.15, 0.45, 0.35],
        default=0.00,
    )

    night_shift_risk = np.where(shift == "night", 0.10, 0.00)

    # Risk score used to create the target variable.
    # Every term is normalized so the formula remains readable.
    # Higher values mean higher probability of failure in the next 7 days.
    risk_score = (
        -4.20
        + 0.035 * (temperature_celsius - 70)
        + 0.300 * (vibration_mm_s - 4)
        + 0.018 * (maintenance_days_since_last - 40)
        + 0.330 * failure_history_count
        + 0.00011 * (operation_hours - 4_000)
        + 0.018 * (load_percentage - 70)
        + dust_risk
        + equipment_risk
        + night_shift_risk
    )

    failure_probability = sigmoid(risk_score)

    # Sample the binary target using the calculated probability.
    failure_next_7_days = rng.binomial(n=1, p=failure_probability, size=rows)

    dataset = pd.DataFrame(
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

    return dataset


def save_dataset(dataset: pd.DataFrame, output_path: Path) -> None:
    """Save the generated dataset as a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic mining equipment data for predictive maintenance."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10_000,
        help="Number of rows to generate. Default: 10000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible results. Default: 42",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_arguments()

    dataset = generate_dataset(rows=args.rows, seed=args.seed)
    save_dataset(dataset, args.output)

    failure_rate = dataset["failure_next_7_days"].mean() * 100

    print("Synthetic mining dataset generated successfully.")
    print(f"Rows: {dataset.shape[0]:,}")
    print(f"Columns: {dataset.shape[1]}")
    print(f"Failure rate: {failure_rate:.2f}%")
    print(f"Output path: {args.output}")


if __name__ == "__main__":
    main()