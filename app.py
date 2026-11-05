from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from weather_ml import build_features


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
METADATA_PATH = MODEL_DIR / "metadata.json"
DEFAULT_TEST_PATH = ROOT / "test_data.csv"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


st.set_page_config(
    page_title="Rainfall Outlook | WeatherAUS",
    page_icon="🌦️",
    layout="wide",
)


@st.cache_resource
def load_model(model_name: str):
    return joblib.load(MODEL_DIR / MODEL_FILES[model_name])


@st.cache_data
def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def evaluate(model, data: pd.DataFrame, metadata: dict) -> dict:
    # Check the uploaded source columns before applying the saved preprocessing.
    target_column = metadata["target_column"]
    required_columns = metadata["raw_feature_columns"] + [target_column]
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    evaluation_data = data[required_columns].dropna(subset=[target_column]).copy()
    actual = evaluation_data[target_column].astype(str).str.strip()
    if not actual.isin(metadata["class_labels"]).all():
        raise ValueError("RainTomorrow must contain only 'Yes' and 'No' labels.")
    if actual.nunique() != 2:
        raise ValueError("The test data must contain both RainTomorrow classes.")

    features = build_features(evaluation_data)
    features = features[metadata["feature_columns"]]
    # Uploaded rows are evaluation data; the model is never retrained here.
    predicted = model.predict(features)
    classes = list(model.named_steps["classifier"].classes_)
    positive_index = classes.index(metadata["positive_label"])
    probabilities = model.predict_proba(features)[:, positive_index]

    return {
        "metrics": {
            "Accuracy": accuracy_score(actual, predicted),
            "AUC": roc_auc_score(actual, probabilities),
            "Precision": precision_score(
                actual, predicted, pos_label=metadata["positive_label"], zero_division=0
            ),
            "Recall": recall_score(
                actual, predicted, pos_label=metadata["positive_label"], zero_division=0
            ),
            "F1": f1_score(
                actual, predicted, pos_label=metadata["positive_label"], zero_division=0
            ),
            "MCC": matthews_corrcoef(actual, predicted),
        },
        "confusion": confusion_matrix(
            actual, predicted, labels=metadata["class_labels"]
        ),
        "report": classification_report(
            actual,
            predicted,
            labels=metadata["class_labels"],
            output_dict=True,
            zero_division=0,
        ),
        "rows": len(evaluation_data),
    }


st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .eyebrow {color: #168b83; font-size: .82rem; font-weight: 700;
              letter-spacing: .12em; text-transform: uppercase;}
    .hero {border-radius: 18px; padding: 1.5rem 1.8rem; margin-bottom: 1.4rem;
           background: linear-gradient(115deg, #102d3a 0%, #17615f 100%);
           color: #fff;}
    .hero p {color: #d6efeb; margin-bottom: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">WeatherAUS · binary classification</div>
      <h1>Rainfall Outlook</h1>
      <p>Explore whether tomorrow brings rain, using one of five trained classifiers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metadata = load_metadata()
with st.sidebar:
    st.markdown("### Evaluation setup")
    selected_model = st.selectbox("Choose a classifier", list(MODEL_FILES))
    uploaded_file = st.file_uploader(
        "Upload a labeled test CSV",
        type=["csv"],
        help="Include every WeatherAUS input column and the RainTomorrow target.",
    )
    st.caption(
        "No retraining happens here. The selected saved pipeline scores the uploaded "
        "rows directly."
    )

if uploaded_file is None:
    data = pd.read_csv(DEFAULT_TEST_PATH)
    data_source = "Included held-out sample"
else:
    try:
        data = pd.read_csv(uploaded_file)
        data_source = uploaded_file.name
    except (UnicodeDecodeError, pd.errors.ParserError, OSError) as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()

try:
    evaluations = {
        model_name: evaluate(load_model(model_name), data, metadata)
        for model_name in MODEL_FILES
    }
except (KeyError, ValueError, FileNotFoundError) as exc:
    st.error(f"Evaluation could not be completed: {exc}")
    st.stop()

result = evaluations[selected_model]
st.caption(f"Evaluating **{result['rows']:,}** labeled rows from **{data_source}**.")
metric_columns = st.columns(6)
for column, (name, value) in zip(metric_columns, result["metrics"].items()):
    column.metric(name, f"{value:.3f}")

comparison = pd.DataFrame.from_dict(
    {name: evaluation["metrics"] for name, evaluation in evaluations.items()},
    orient="index",
)
comparison.insert(
    0,
    "Selected",
    [name == selected_model for name in comparison.index],
)
comparison.index.name = "ML model"
st.subheader("All-model comparison")
st.dataframe(comparison.round(3), width="stretch")

left, right = st.columns([1, 1.25])
with left:
    st.subheader("Confusion matrix")
    st.caption("Rows are actual labels; columns are predicted labels.")
    st.dataframe(
        pd.DataFrame(
            result["confusion"],
            index=metadata["class_labels"],
            columns=metadata["class_labels"],
        ).rename_axis("Actual"),
        width="stretch",
    )

with right:
    st.subheader("Classification report")
    report = pd.DataFrame(result["report"]).transpose()
    st.dataframe(report.round(3), width="stretch")

st.markdown("---")
st.markdown(
    "The bundled sample is a held-out portion of WeatherAUS. To evaluate another "
    "labeled dataset, upload a CSV with the same input fields and `RainTomorrow`."
)
