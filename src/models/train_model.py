"""
Train machine learning models to predict equipment failure risk.

This script reads the processed mining equipment dataset, trains three
classification models, evaluates them, selects the best model prioritizing
recall, and saves the final model plus a metrics report.

Run from the project root:
    python src/models/train_model.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Reproducibility: using the same random seed helps obtain stable results.
RANDOM_STATE = 42
TARGET_COLUMN = "failure_next_7_days"
ID_COLUMN = "equipment_id"


def get_project_root() -> Path:
    """Return the project root path based on this file location."""
    return Path(__file__).resolve().parents[2]


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load the processed dataset and validate basic requirements."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {input_path}. "
            "Run src/processing/pandas_processing.py first."
        )

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("The processed dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in the dataset.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column '{TARGET_COLUMN}' contains null values.")

    return df


def split_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Separate predictors X and target y.

    equipment_id is removed because it is an identifier, not an operational
    signal. Keeping IDs as predictors can make the model learn noise instead
    of real maintenance patterns.
    """
    columns_to_drop = [TARGET_COLUMN]

    if ID_COLUMN in df.columns:
        columns_to_drop.append(ID_COLUMN)

    X = df.drop(columns=columns_to_drop)
    y = df[TARGET_COLUMN]

    # Keep only numeric columns to avoid training errors.
    X = X.select_dtypes(include=["number"])
    feature_columns = X.columns.tolist()

    if not feature_columns:
        raise ValueError("No numeric feature columns found for model training.")

    return X, y, feature_columns


def build_models() -> Dict[str, Pipeline]:
    """Create the models that will be compared.

    Logistic Regression uses scaling because it is sensitive to variable scale.
    Tree-based models do not require scaling, but they are still wrapped in a
    Pipeline to keep a consistent structure.
    """
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest Classifier": Pipeline(
            steps=[
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "Gradient Boosting Classifier": Pipeline(
            steps=[
                (
                    "model",
                    GradientBoostingClassifier(random_state=RANDOM_STATE),
                )
            ]
        ),
    }


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
    """Evaluate a trained model using classification metrics."""
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def select_best_model(results: Dict[str, Dict[str, object]]) -> str:
    """Select the best model prioritizing recall.

    In predictive maintenance, recall is critical because a false negative
    means the model failed to detect equipment that may fail soon.
    If two models have the same recall, the model with better F1-score is chosen.
    """
    return max(
        results,
        key=lambda model_name: (
            results[model_name]["recall"],
            results[model_name]["f1_score"],
            results[model_name]["precision"],
        ),
    )


def format_confusion_matrix(matrix) -> str:
    """Return a readable confusion matrix for the Markdown report."""
    tn, fp, fn, tp = matrix.ravel()
    return (
        "| Actual / Predicted | Predicted 0 | Predicted 1 |\n"
        "|---|---:|---:|\n"
        f"| Actual 0 | {tn} | {fp} |\n"
        f"| Actual 1 | {fn} | {tp} |\n"
    )


def save_metrics_report(
    output_path: Path,
    results: Dict[str, Dict[str, object]],
    best_model_name: str,
    feature_columns: List[str],
    train_rows: int,
    test_rows: int,
) -> None:
    """Save model evaluation results as a Markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Metrics Report - Failure Prediction Model",
        "",
        "## Objective",
        "",
        "Train and compare machine learning models to predict whether a mining "
        "asset may fail in the next 7 days.",
        "",
        "## Business criterion",
        "",
        "The best model was selected by prioritizing **recall**. In predictive "
        "maintenance, a false negative is risky because it means the system did "
        "not detect a possible failure. Missing a real failure can lead to "
        "unplanned downtime, production losses, safety exposure, and higher "
        "maintenance costs.",
        "",
        "## Dataset split",
        "",
        f"- Training rows: {train_rows}",
        f"- Testing rows: {test_rows}",
        f"- Number of features used: {len(feature_columns)}",
        "",
        "## Model comparison",
        "",
        "| Model | Accuracy | Precision | Recall | F1-score |",
        "|---|---:|---:|---:|---:|",
    ]

    for model_name, metrics in results.items():
        lines.append(
            f"| {model_name} | "
            f"{metrics['accuracy']:.4f} | "
            f"{metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | "
            f"{metrics['f1_score']:.4f} |"
        )

    lines.extend(
        [
            "",
            f"## Selected model",
            "",
            f"**{best_model_name}** was selected as the best model because it achieved "
            "the strongest recall-oriented performance among the evaluated models.",
            "",
            "## Confusion matrices",
            "",
        ]
    )

    for model_name, metrics in results.items():
        lines.extend(
            [
                f"### {model_name}",
                "",
                format_confusion_matrix(metrics["confusion_matrix"]),
                "",
            ]
        )

    lines.extend(
        [
            "## Features used by the model",
            "",
            "The model was trained using the following numeric variables:",
            "",
        ]
    )

    for feature in feature_columns:
        lines.append(f"- {feature}")

    lines.extend(
        [
            "",
            "## Business interpretation",
            "",
            "This model can support maintenance planning by identifying assets with "
            "higher probability of failure in the next 7 days. In a mining operation, "
            "this type of analytics can help prioritize inspections, schedule planned "
            "maintenance, reduce unplanned downtime, and focus attention on equipment "
            "showing abnormal operating conditions such as high vibration, high "
            "temperature, long operation hours, or extended time since last maintenance.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_best_model(
    output_path: Path,
    model: Pipeline,
    model_name: str,
    feature_columns: List[str],
) -> None:
    """Save the best trained model with useful metadata for future inference."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_package = {
        "model": model,
        "model_name": model_name,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
    }

    joblib.dump(model_package, output_path)


def main() -> None:
    """Run the full model training pipeline."""
    project_root = get_project_root()
    input_path = project_root / "data" / "processed" / "mining_equipment_processed.csv"
    model_output_path = project_root / "models" / "failure_prediction_model.pkl"
    report_output_path = project_root / "reports" / "metrics_report.md"

    print("Loading processed dataset...")
    df = load_dataset(input_path)

    print("Preparing features and target...")
    X, y, feature_columns = split_features_and_target(df)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_models()
    results: Dict[str, Dict[str, object]] = {}
    trained_models: Dict[str, Pipeline] = {}

    print("Training and evaluating models...")
    for model_name, model in models.items():
        print(f"- Training {model_name}...")
        model.fit(X_train, y_train)
        results[model_name] = evaluate_model(model, X_test, y_test)
        trained_models[model_name] = model

    best_model_name = select_best_model(results)
    best_model = trained_models[best_model_name]

    print(f"Best model selected: {best_model_name}")
    print(f"Best model recall: {results[best_model_name]['recall']:.4f}")

    save_best_model(
        output_path=model_output_path,
        model=best_model,
        model_name=best_model_name,
        feature_columns=feature_columns,
    )

    save_metrics_report(
        output_path=report_output_path,
        results=results,
        best_model_name=best_model_name,
        feature_columns=feature_columns,
        train_rows=len(X_train),
        test_rows=len(X_test),
    )

    print(f"Model saved to: {model_output_path}")
    print(f"Metrics report saved to: {report_output_path}")


if __name__ == "__main__":
    main()
