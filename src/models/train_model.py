"""
Train machine learning models to predict equipment failure risk with MLflow tracking.

This script reads the processed mining equipment dataset, trains three
classification models, evaluates them, registers metrics and artifacts in
MLflow, selects the best model prioritizing recall, and saves the final model
plus a Markdown metrics report.

Run from the project root:
    python src/models/train_model.py

Open MLflow UI from the project root:
    mlflow ui --backend-store-uri ./mlruns --port 5000
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
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
EXPERIMENT_NAME = "mining_failure_prediction_experiment"


def get_project_root() -> Path:
    """Return the project root path based on this file location."""
    return Path(__file__).resolve().parents[2]


def configure_mlflow(project_root: Path) -> None:
    """Configure MLflow to save runs locally inside the project.

    A local tracking folder keeps the project easy to run on a personal laptop
    and easy to demonstrate in GitHub or during an interview.
    """
    tracking_path = project_root / "mlruns"
    mlflow.set_tracking_uri(tracking_path.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


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


def get_model_params_for_mlflow(model: Pipeline) -> Dict[str, Any]:
    """Return the most important model parameters for MLflow.

    This avoids logging too many internal parameters and keeps the experiment
    easy to read for a beginner.
    """
    estimator = model.named_steps["model"]

    if isinstance(estimator, LogisticRegression):
        return {
            "model_type": "LogisticRegression",
            "max_iter": estimator.max_iter,
            "class_weight": estimator.class_weight,
            "random_state": estimator.random_state,
            "solver": estimator.solver,
            "C": estimator.C,
        }

    if isinstance(estimator, RandomForestClassifier):
        return {
            "model_type": "RandomForestClassifier",
            "n_estimators": estimator.n_estimators,
            "max_depth": estimator.max_depth,
            "class_weight": estimator.class_weight,
            "random_state": estimator.random_state,
            "n_jobs": estimator.n_jobs,
        }

    if isinstance(estimator, GradientBoostingClassifier):
        return {
            "model_type": "GradientBoostingClassifier",
            "n_estimators": estimator.n_estimators,
            "learning_rate": estimator.learning_rate,
            "max_depth": estimator.max_depth,
            "random_state": estimator.random_state,
        }

    return {"model_type": estimator.__class__.__name__}


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


def save_confusion_matrix_plot(
    matrix,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Save the confusion matrix as an image so MLflow can log it as an artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
    output_path = output_dir / file_name

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No failure", "Failure"],
    )
    display.plot(values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def log_run_to_mlflow(
    model_name: str,
    model: Pipeline,
    metrics: Dict[str, object],
    confusion_matrix_path: Path,
    train_rows: int,
    test_rows: int,
    feature_count: int,
) -> str:
    """Log one model training run to MLflow."""
    run_name = model_name.lower().replace(" ", "_")

    with mlflow.start_run(run_name=run_name) as run:
        # Tags are useful labels to understand the context of the run.
        mlflow.set_tag("project", "mining-analytics-ml-pipeline")
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("selection_metric", "recall")
        mlflow.set_tag("business_context", "predictive maintenance")

        # Parameters describe the configuration of the model and dataset split.
        mlflow.log_params(get_model_params_for_mlflow(model))
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("train_rows", train_rows)
        mlflow.log_param("test_rows", test_rows)
        mlflow.log_param("feature_count", feature_count)
        mlflow.log_param("random_state", RANDOM_STATE)

        # Metrics describe model performance.
        mlflow.log_metric("accuracy", float(metrics["accuracy"]))
        mlflow.log_metric("precision", float(metrics["precision"]))
        mlflow.log_metric("recall", float(metrics["recall"]))
        mlflow.log_metric("f1_score", float(metrics["f1_score"]))

        # The confusion matrix image helps explain false positives and false negatives.
        mlflow.log_artifact(str(confusion_matrix_path), artifact_path="confusion_matrices")

        # Log the trained scikit-learn pipeline as an MLflow model artifact.
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="trained_model",
            input_example=None,
        )

        return run.info.run_id


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
        "## MLflow experiment",
        "",
        f"- Experiment name: `{EXPERIMENT_NAME}`",
        "- Tracking folder: `mlruns/`",
        "- Artifacts logged: confusion matrix image and trained model for each run.",
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
            "## Selected model",
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
    confusion_matrix_dir = project_root / "reports" / "confusion_matrices"

    print("Configuring MLflow...")
    configure_mlflow(project_root)
    print(f"MLflow experiment: {EXPERIMENT_NAME}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

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
    mlflow_run_ids: Dict[str, str] = {}

    print("Training, evaluating, and logging models with MLflow...")
    for model_name, model in models.items():
        print(f"- Training {model_name}...")
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        results[model_name] = metrics
        trained_models[model_name] = model

        confusion_matrix_path = save_confusion_matrix_plot(
            matrix=metrics["confusion_matrix"],
            model_name=model_name,
            output_dir=confusion_matrix_dir,
        )

        run_id = log_run_to_mlflow(
            model_name=model_name,
            model=model,
            metrics=metrics,
            confusion_matrix_path=confusion_matrix_path,
            train_rows=len(X_train),
            test_rows=len(X_test),
            feature_count=len(feature_columns),
        )
        mlflow_run_ids[model_name] = run_id

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

    # Add extra information to the MLflow run of the selected model.
    with mlflow.start_run(run_id=mlflow_run_ids[best_model_name]):
        mlflow.set_tag("selected_as_best", "true")
        mlflow.log_artifact(str(model_output_path), artifact_path="best_model_joblib")
        mlflow.log_artifact(str(report_output_path), artifact_path="reports")

    print(f"Model saved to: {model_output_path}")
    print(f"Metrics report saved to: {report_output_path}")
    print("MLflow artifacts saved in the local 'mlruns' folder.")


if __name__ == "__main__":
    main()