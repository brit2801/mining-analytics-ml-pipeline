"""
Pandas processing script for the mining analytics ML pipeline.

This script reads the raw synthetic mining equipment dataset, performs basic
quality checks, creates simple risk features, encodes categorical variables,
and saves a processed dataset ready for Machine Learning experiments.

Run from the project root:
    python src/processing/pandas_processing.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "mining_equipment_data.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "mining_equipment_processed.csv"


# -----------------------------------------------------------------------------
# Expected columns in the raw dataset
# -----------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "equipment_id",
    "equipment_type",
    "operation_hours",
    "temperature_celsius",
    "vibration_mm_s",
    "pressure_bar",
    "current_amp",
    "load_percentage",
    "maintenance_days_since_last",
    "failure_history_count",
    "environment_dust_level",
    "shift",
    "failure_next_7_days",
]


NUMERIC_COLUMNS = [
    "operation_hours",
    "temperature_celsius",
    "vibration_mm_s",
    "pressure_bar",
    "current_amp",
    "load_percentage",
    "maintenance_days_since_last",
    "failure_history_count",
]


CATEGORICAL_COLUMNS = [
    "equipment_type",
    "environment_dust_level",
    "shift",
]


TARGET_COLUMN = "failure_next_7_days"


# -----------------------------------------------------------------------------
# Data loading and validation
# -----------------------------------------------------------------------------
def read_raw_data(file_path: Path) -> pd.DataFrame:
    """Read the raw mining equipment dataset from a CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {file_path}\n"
            "First run: python src/data/generate_mining_dataset.py"
        )

    return pd.read_csv(file_path)


def validate_required_columns(df: pd.DataFrame) -> None:
    """Validate that all expected columns are present in the dataset."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset using simple and understandable rules.

    Steps:
    - Remove duplicate rows.
    - Fill missing numeric values with the median.
    - Fill missing categorical values with the mode.
    - Validate that the target only contains 0 and 1.
    """
    df_clean = df.copy()

    # 1. Validate null values before cleaning.
    nulls_before = df_clean.isnull().sum().sum()
    print(f"Null values before cleaning: {nulls_before}")

    # 2. Remove duplicated rows if they exist.
    duplicated_before = df_clean.duplicated().sum()
    print(f"Duplicated rows before cleaning: {duplicated_before}")
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # 3. Fill missing values in numeric columns with the median.
    for column in NUMERIC_COLUMNS:
        if df_clean[column].isnull().any():
            df_clean[column] = df_clean[column].fillna(df_clean[column].median())

    # 4. Fill missing values in categorical columns with the most common value.
    for column in CATEGORICAL_COLUMNS:
        if df_clean[column].isnull().any():
            most_common_value = df_clean[column].mode()[0]
            df_clean[column] = df_clean[column].fillna(most_common_value)

    # 5. Validate the target column.
    valid_target_values = {0, 1}
    current_target_values = set(df_clean[TARGET_COLUMN].dropna().unique())

    if not current_target_values.issubset(valid_target_values):
        raise ValueError(
            f"Invalid target values found: {current_target_values}. "
            "The target must only contain 0 and 1."
        )

    nulls_after = df_clean.isnull().sum().sum()
    duplicated_after = df_clean.duplicated().sum()
    print(f"Null values after cleaning: {nulls_after}")
    print(f"Duplicated rows after cleaning: {duplicated_after}")

    return df_clean


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------
def min_max_scale(series: pd.Series) -> pd.Series:
    """
    Scale a numeric column between 0 and 1.

    This helper is used to combine different variables into one stress score.
    For example, temperature and current have different units, so scaling helps
    compare them on the same range.
    """
    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(np.zeros(len(series)), index=series.index)

    return (series - min_value) / (max_value - min_value)


def create_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple risk features for predictive maintenance."""
    df_features = df.copy()

    # Flag 1: high equipment temperature.
    # In this synthetic dataset, values above 85°C are considered risky.
    df_features["high_temperature_flag"] = (
        df_features["temperature_celsius"] >= 85
    ).astype(int)

    # Flag 2: high vibration level.
    # Values above 7.5 mm/s are considered a warning signal.
    df_features["high_vibration_flag"] = (
        df_features["vibration_mm_s"] >= 7.5
    ).astype(int)

    # Flag 3: maintenance risk.
    # Risk increases when the asset has gone many days without maintenance
    # or when it has a relevant previous failure history.
    df_features["maintenance_risk_flag"] = (
        (df_features["maintenance_days_since_last"] >= 90)
        | (df_features["failure_history_count"] >= 3)
    ).astype(int)

    # Feature 4: operational stress score.
    # This is a simple score from 0 to 1 that combines several operating factors.
    df_features["operational_stress_score"] = (
        0.25 * min_max_scale(df_features["operation_hours"])
        + 0.25 * min_max_scale(df_features["temperature_celsius"])
        + 0.20 * min_max_scale(df_features["vibration_mm_s"])
        + 0.15 * min_max_scale(df_features["load_percentage"])
        + 0.10 * min_max_scale(df_features["maintenance_days_since_last"])
        + 0.05 * min_max_scale(df_features["current_amp"])
    )

    return df_features


# -----------------------------------------------------------------------------
# Encoding categorical variables
# -----------------------------------------------------------------------------
def encode_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables using one-hot encoding.

    The equipment_id column is not encoded because it is an identifier, not a
    useful predictive feature. It is kept only for traceability.
    """
    df_encoded = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=False,
        dtype=int,
    )

    return df_encoded


# -----------------------------------------------------------------------------
# Save processed data
# -----------------------------------------------------------------------------
def save_processed_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save the processed dataset as a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
def main() -> None:
    """Execute the full Pandas processing pipeline."""
    print("Starting Pandas data processing...")

    raw_df = read_raw_data(RAW_DATA_PATH)
    validate_required_columns(raw_df)

    print(f"Raw dataset shape: {raw_df.shape}")

    clean_df = clean_data(raw_df)
    featured_df = create_risk_features(clean_df)
    processed_df = encode_categorical_variables(featured_df)

    save_processed_data(processed_df, PROCESSED_DATA_PATH)

    print(f"Processed dataset shape: {processed_df.shape}")
    print(f"Processed dataset saved at: {PROCESSED_DATA_PATH}")
    print("Pandas data processing completed successfully.")


if __name__ == "__main__":
    main()