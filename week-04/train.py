import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("Attribute DataSet.xlsx")
EXPERIMENT_NAME = "Dresses Attribute Sales - Experiments"


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train Logistic Regression experiments with MLflow."
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength.",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of training iterations.",
    )

    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "liblinear"],
        help="Optimization algorithm.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def load_data():
    """Load and clean the Dresses Attribute Sales dataset."""

    df = pd.read_excel(DATA_PATH)

    df_clean = df.copy()

    categorical_columns = [
        "Style",
        "Price",
        "Size",
        "Season",
        "NeckLine",
        "SleeveLength",
        "waiseline",
        "Material",
        "FabricType",
        "Decoration",
        "Pattern Type",
    ]

    for column in categorical_columns:
        df_clean[column] = (
            df_clean[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    df_clean["Size"] = df_clean["Size"].replace({
        "small": "s"
    })

    df_clean["Season"] = df_clean["Season"].replace({
        "automn": "autumn"
    })

    df_clean["SleeveLength"] = df_clean["SleeveLength"].replace({
        "sleeevless": "sleeveless",
        "sleveless": "sleeveless",
        "sleevless": "sleeveless",
        "threequarter": "threequarter",
        "thressqatar": "threequarter",
        "threequater": "threequarter",
        "cap-sleeves": "capsleeves",
    })

    for column in categorical_columns:
        df_clean[column] = df_clean[column].fillna("unknown")

    return df_clean


def build_pipeline(C, max_iter, solver, random_state):
    """Build the preprocessing and Logistic Regression pipeline."""

    X_columns = [
        "Style",
        "Price",
        "Size",
        "Season",
        "NeckLine",
        "SleeveLength",
        "waiseline",
        "Material",
        "FabricType",
        "Decoration",
        "Pattern Type",
    ]

    numeric_features = ["Rating"]

    categorical_features = X_columns

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features,
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_and_evaluate(df, args):
    """Split data, train the model and calculate evaluation metrics."""

    X = df.drop(
        columns=[
            "Recommendation",
            "Dress_ID",
        ]
    )

    y = df["Recommendation"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=args.random_state,
        stratify=y,
    )

    model_pipeline = build_pipeline(
        C=args.C,
        max_iter=args.max_iter,
        solver=args.solver,
        random_state=args.random_state,
    )

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
    }

    return (
        model_pipeline,
        y_test,
        y_pred,
        metrics,
    )


def save_artifacts(y_test, y_pred):
    """Create evaluation artifacts."""

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    report_path = artifacts_dir / "classification_report.txt"

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    matrix_path = artifacts_dir / "confusion_matrix.txt"

    matrix = confusion_matrix(
        y_test,
        y_pred,
    )

    matrix_path.write_text(
        str(matrix),
        encoding="utf-8",
    )

    return artifacts_dir


def run_experiment(args):
    """Execute one experiment and log it to MLflow."""

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    df = load_data()

    (
        model_pipeline,
        y_test,
        y_pred,
        metrics,
    ) = train_and_evaluate(
        df,
        args,
    )

    artifacts_dir = save_artifacts(
        y_test,
        y_pred,
    )

    run_name = (
        f"logreg-C-{args.C}"
        f"-iter-{args.max_iter}"
        f"-solver-{args.solver}"
    )

    with mlflow.start_run(
        run_name=run_name
    ):

        mlflow.log_param(
            "C",
            args.C,
        )

        mlflow.log_param(
            "max_iter",
            args.max_iter,
        )

        mlflow.log_param(
            "solver",
            args.solver,
        )

        mlflow.log_param(
            "random_state",
            args.random_state,
        )

        mlflow.log_param(
            "test_size",
            0.20,
        )

        mlflow.log_metric(
            "accuracy",
            metrics["accuracy"],
        )

        mlflow.log_metric(
            "precision",
            metrics["precision"],
        )

        mlflow.log_metric(
            "recall",
            metrics["recall"],
        )

        mlflow.log_metric(
            "f1_score",
            metrics["f1_score"],
        )

        mlflow.set_tag(
            "dataset",
            "Dresses Attribute Sales",
        )

        mlflow.set_tag(
            "model_type",
            "Logistic Regression",
        )

        mlflow.log_artifacts(
            artifacts_dir
        )

        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            name="model",
            skops_trusted_types=[
                "numpy.dtype"
            ],
        )

        run_id = (
            mlflow.active_run()
            .info.run_id
        )

        print(
            f"Run ID: {run_id}"
        )

        print(
            "Metrics:"
        )

        for name, value in metrics.items():
            print(
                f"  {name}: {value:.4f}"
            )


def main():
    args = parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()