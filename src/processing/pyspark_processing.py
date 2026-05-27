"""
PySpark processing script for the mining analytics ML pipeline.

Objective:
    Process the raw mining equipment dataset using PySpark to demonstrate
    basic distributed data processing skills.

Input:
    data/raw/mining_equipment_data.csv

Output:
    data/processed/pyspark_equipment_summary.csv

Run from the project root:
    python src/processing/pyspark_processing.py
"""

from pathlib import Path
import shutil

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "mining_equipment_data.csv"
OUTPUT_FILE_PATH = PROJECT_ROOT / "data" / "processed" / "pyspark_equipment_summary.csv"
TEMP_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "_pyspark_equipment_summary_temp"


# -----------------------------------------------------------------------------
# Expected columns
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


# -----------------------------------------------------------------------------
# Spark session
# -----------------------------------------------------------------------------
def create_spark_session() -> SparkSession:
    """
    Create a local Spark session.

    local[*] means Spark will use all available CPU cores in your machine.
    This is enough for learning and for a GitHub portfolio project.
    """
    spark = (
        SparkSession.builder
        .appName("MiningAnalyticsPySparkProcessing")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

    return spark


# -----------------------------------------------------------------------------
# Data loading and validation
# -----------------------------------------------------------------------------
def read_raw_data(spark: SparkSession, file_path: Path) -> DataFrame:
    """Read the raw CSV file using Spark."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {file_path}\n"
            "First run: python src/data/generate_mining_dataset.py"
        )

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(file_path))
    )

    return df


def validate_required_columns(df: DataFrame) -> None:
    """Validate that the dataset contains all required columns."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def show_null_counts(df: DataFrame) -> None:
    """
    Show the number of null or empty values per column.

    This is a basic data quality validation step. In real mining data,
    missing values are common because of sensor issues, communication failures,
    or manual entry problems.
    """
    null_count_expressions = [
        F.sum(
            F.when(
                F.col(column).isNull()
                | (F.trim(F.col(column).cast("string")) == ""),
                1,
            ).otherwise(0)
        ).alias(column)
        for column in df.columns
    ]

    print("\nNull or empty values by column:")
    df.select(null_count_expressions).show(truncate=False)


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------
def capped_ratio(column_name: str, max_expected_value: float):
    """
    Scale a numeric Spark column between 0 and 1 using an expected maximum value.

    Example:
        If temperature is 90 and max_expected_value is 120,
        the ratio is 90 / 120 = 0.75.

    The result is capped between 0 and 1 to avoid extreme values dominating
    the operational stress score.
    """
    return F.least(
        F.lit(1.0),
        F.greatest(F.lit(0.0), F.col(column_name).cast("double") / F.lit(max_expected_value)),
    )


def create_risk_features(df: DataFrame) -> DataFrame:
    """Create simple predictive maintenance risk features using PySpark."""
    df_features = (
        df
        # Flag 1: high equipment temperature.
        # In this synthetic dataset, values from 85 °C are considered risky.
        .withColumn(
            "high_temperature_flag",
            F.when(F.col("temperature_celsius") >= 85, 1).otherwise(0),
        )
        # Flag 2: high vibration level.
        # Values from 7.5 mm/s are considered a warning signal.
        .withColumn(
            "high_vibration_flag",
            F.when(F.col("vibration_mm_s") >= 7.5, 1).otherwise(0),
        )
        # Flag 3: maintenance risk.
        # Risk increases when many days have passed since the last maintenance
        # or when the equipment has previous failures.
        .withColumn(
            "maintenance_risk_flag",
            F.when(
                (F.col("maintenance_days_since_last") >= 90)
                | (F.col("failure_history_count") >= 3),
                1,
            ).otherwise(0),
        )
        # Feature 4: operational stress score.
        # This score combines several operating variables into one interpretable
        # value between 0 and 1. Higher values mean higher operational stress.
        .withColumn(
            "operational_stress_score",
            F.round(
                0.25 * capped_ratio("operation_hours", 30_000)
                + 0.25 * capped_ratio("temperature_celsius", 120)
                + 0.20 * capped_ratio("vibration_mm_s", 15)
                + 0.15 * capped_ratio("load_percentage", 100)
                + 0.10 * capped_ratio("maintenance_days_since_last", 180)
                + 0.05 * capped_ratio("current_amp", 600),
                4,
            ),
        )
    )

    return df_features


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------
def create_equipment_summary(df: DataFrame) -> DataFrame:
    """
    Group data by equipment type and calculate operational KPIs.

    The failure rate is calculated as the average of the binary target:
        0 = no failure
        1 = failure

    Example:
        failure_rate = 0.18 means 18% of records for that equipment type
        are labeled as failures in the next 7 days.
    """
    summary_df = (
        df.groupBy("equipment_type")
        .agg(
            F.count("*").alias("record_count"),
            F.round(F.avg("temperature_celsius"), 2).alias("avg_temperature_celsius"),
            F.round(F.avg("vibration_mm_s"), 2).alias("avg_vibration_mm_s"),
            F.round(F.avg("operation_hours"), 2).alias("avg_operation_hours"),
            F.round(F.avg("failure_next_7_days"), 4).alias("failure_rate"),
        )
        .orderBy("equipment_type")
    )

    return summary_df


# -----------------------------------------------------------------------------
# Output handling
# -----------------------------------------------------------------------------
def save_single_csv(df: DataFrame, output_file_path: Path, temp_output_dir: Path) -> None:
    """
    Save a Spark DataFrame as a single CSV file.

    By default, Spark writes CSV outputs as a folder with one or more part files.
    For a beginner-friendly GitHub project, this helper writes to a temporary
    folder and then renames the generated part file to the final CSV path.

    Note:
        For very large production datasets, it is better to keep Spark's
        partitioned output instead of forcing one single CSV file.
    """
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    if output_file_path.exists():
        output_file_path.unlink()

    if temp_output_dir.exists():
        shutil.rmtree(temp_output_dir)

    (
        df.coalesce(1)
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(str(temp_output_dir))
    )

    part_files = list(temp_output_dir.glob("part-*.csv"))

    if not part_files:
        raise FileNotFoundError("Spark did not generate a CSV part file.")

    shutil.move(str(part_files[0]), str(output_file_path))
    shutil.rmtree(temp_output_dir)


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
def main() -> None:
    """Execute the full local PySpark processing pipeline."""
    spark = create_spark_session()

    try:
        print("Starting PySpark data processing...")

        raw_df = read_raw_data(spark, RAW_DATA_PATH)
        validate_required_columns(raw_df)

        print("\nDataset schema:")
        raw_df.printSchema()

        record_count = raw_df.count()
        print(f"\nTotal records: {record_count}")

        show_null_counts(raw_df)

        featured_df = create_risk_features(raw_df)

        print("\nPreview with new risk features:")
        featured_df.select(
            "equipment_id",
            "equipment_type",
            "temperature_celsius",
            "vibration_mm_s",
            "maintenance_days_since_last",
            "failure_history_count",
            "high_temperature_flag",
            "high_vibration_flag",
            "maintenance_risk_flag",
            "operational_stress_score",
            "failure_next_7_days",
        ).show(10, truncate=False)

        summary_df = create_equipment_summary(featured_df)

        print("\nEquipment summary:")
        summary_df.show(truncate=False)

        save_single_csv(summary_df, OUTPUT_FILE_PATH, TEMP_OUTPUT_DIR)

        print(f"\nPySpark summary saved at: {OUTPUT_FILE_PATH}")
        print("PySpark data processing completed successfully.")

    finally:
        # Always stop Spark to release local resources.
        spark.stop()


if __name__ == "__main__":
    main()