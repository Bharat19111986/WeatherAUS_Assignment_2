from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from weather_ml import TARGET_COLUMN, build_features


ROOT = Path(__file__).resolve().parent
SOURCE_FILE = ROOT / "data" / "weatherAUS.csv"
MODEL_DIR = ROOT / "models"
TEST_FILE = ROOT / "test_data.csv"
METRICS_FILE = ROOT / "models" / "metrics.json"
METADATA_FILE = ROOT / "models" / "metadata.json"
RANDOM_STATE = 20261105
TRAINING_SAMPLE_SIZE = 20_000
TEST_SAMPLE_SIZE = 5_000
POSITIVE_LABEL = "Yes"


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    # Keep imputing and scaling inside the pipeline to prevent test-data leakage.
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                # Test data may contain categories not seen during training.
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        sparse_threshold=0,
    )


def make_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1_000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=11,
            weights="distance",
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        # Calibration gives the linear SVM probabilities for AUC evaluation.
        "Support Vector Machine": CalibratedClassifierCV(
            estimator=LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            cv=3,
            ensemble=False,
        ),
    }


def main() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {SOURCE_FILE}")

    source = pd.read_csv(SOURCE_FILE)
    if TARGET_COLUMN not in source.columns:
        raise ValueError(f"Dataset must include the {TARGET_COLUMN!r} target column.")

    source = source.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    source[TARGET_COLUMN] = source[TARGET_COLUMN].astype(str).str.strip()
    source = source[source[TARGET_COLUMN].isin(["Yes", "No"])].reset_index(drop=True)
    if source[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Both 'Yes' and 'No' target classes are required.")

    raw_features = source.drop(columns=[TARGET_COLUMN])
    features = build_features(source)
    labels = source[TARGET_COLUMN]
    all_indices = np.arange(len(source))
    # Split once so the training and test sets remain independent.
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # Stratified subsamples keep training practical and preserve class balance.
    training_indices, _ = train_test_split(
        train_indices,
        train_size=min(TRAINING_SAMPLE_SIZE, len(train_indices)),
        random_state=RANDOM_STATE,
        stratify=labels.iloc[train_indices],
    )
    _, selected_test_indices = train_test_split(
        test_indices,
        test_size=min(TEST_SAMPLE_SIZE, len(test_indices)),
        random_state=RANDOM_STATE,
        stratify=labels.iloc[test_indices],
    )

    x_train = features.iloc[training_indices]
    y_train = labels.iloc[training_indices]
    x_test = features.iloc[selected_test_indices]
    y_test = labels.iloc[selected_test_indices]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Save the original columns so the app can accept the source CSV format.
    test_data = raw_features.iloc[selected_test_indices].copy()
    test_data[TARGET_COLUMN] = y_test.to_numpy()
    test_data.to_csv(TEST_FILE, index=False)

    metrics: dict[str, dict[str, object]] = {}
    for model_name, classifier in make_models().items():
        pipeline = Pipeline(
            [
                ("preprocessor", make_preprocessor(x_train)),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(x_train, y_train)

        predictions = pipeline.predict(x_test)
        classes = pipeline.named_steps["classifier"].classes_
        # AUC uses scores for the positive "Yes" class.
        positive_index = list(classes).index(POSITIVE_LABEL)
        probabilities = pipeline.predict_proba(x_test)[:, positive_index]
        metrics[model_name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "auc": float(roc_auc_score(y_test, probabilities)),
            "precision": float(
                precision_score(y_test, predictions, pos_label=POSITIVE_LABEL)
            ),
            "recall": float(
                recall_score(y_test, predictions, pos_label=POSITIVE_LABEL)
            ),
            "f1": float(f1_score(y_test, predictions, pos_label=POSITIVE_LABEL)),
            "mcc": float(matthews_corrcoef(y_test, predictions)),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=["No", "Yes"]
            ).tolist(),
        }

        model_path = MODEL_DIR / f"{model_name.lower().replace(' ', '_')}.joblib"
        joblib.dump(pipeline, model_path, compress=3)
        print(
            f"{model_name}: accuracy={metrics[model_name]['accuracy']:.4f}, "
            f"AUC={metrics[model_name]['auc']:.4f}"
        )

    metrics["dataset"] = {
        "source_rows": int(len(source)),
        "training_rows": int(len(training_indices)),
        "test_rows": int(len(selected_test_indices)),
        "target": TARGET_COLUMN,
        "positive_label": POSITIVE_LABEL,
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    METADATA_FILE.write_text(
        json.dumps(
            {
                "target_column": TARGET_COLUMN,
                "positive_label": POSITIVE_LABEL,
                "raw_feature_columns": raw_features.columns.tolist(),
                "feature_columns": features.columns.tolist(),
                "class_labels": ["No", "Yes"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved holdout data to {TEST_FILE}")
    print(f"Saved model metrics to {METRICS_FILE}")


if __name__ == "__main__":
    main()
