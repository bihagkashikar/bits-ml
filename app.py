import hashlib
import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from models.evaluation import MODEL_NAMES, evaluate_models


st.set_page_config(page_title="Spambase model comparison", page_icon="📧", layout="wide")

st.title("Machine Learning - Classification Models")
st.subheader("Problem Statement")
st.markdown(
    "- Implement multiple classification models on the same dataset.\n"
    "- Build an interactive Streamlit web application to demonstrate the models.\n"
    "- Compare the models based on their performance metrics.\n"
)

repository_root = Path(__file__).resolve().parent
repository_csvs = []
for current_directory, directories, filenames in os.walk(repository_root):
    directories[:] = [
        directory
        for directory in directories
        if directory not in {".git", ".venv", "__pycache__"}
    ]
    for filename in filenames:
        if filename.lower().endswith(".csv"):
            path = Path(current_directory) / filename
            try:
                relative_path = path.resolve().relative_to(repository_root)
            except ValueError:
                continue
            repository_csvs.append(relative_path.as_posix())
repository_csvs.sort()

dataset_source = st.sidebar.radio(
    "Dataset source",
    options=["Repository CSV", "Upload CSV"],
)
if dataset_source == "Repository CSV":
    if not repository_csvs:
        st.error("No CSV files were found in the repository.")
        st.stop()
    default_dataset = "spambase.csv" if "spambase.csv" in repository_csvs else repository_csvs[0]
    selected_path = st.sidebar.selectbox(
        "CSV file in repository",
        options=repository_csvs,
        index=repository_csvs.index(default_dataset),
        help=(
            "The bundled `spambase.csv` is selected by default. "
            "[View its GitHub repository path]"
            "(https://github.com/bihagkashikar/bits-ml/blob/master/spambase.csv)."
        ),
    )
    if selected_path is None:
        st.error("Select a CSV file from the repository.")
        st.stop()
    dataset_path = (repository_root / selected_path).resolve()
    try:
        dataset_path.relative_to(repository_root)
    except ValueError:
        st.error("The selected dataset path must be inside the repository.")
        st.stop()
    dataset_bytes = dataset_path.read_bytes()
    st.sidebar.caption(f"Using `{selected_path}` from the repository.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload a labeled CSV dataset", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file or select one from the repository.")
        st.stop()
    dataset_bytes = uploaded_file.getvalue()

try:
    data = pd.read_csv(io.BytesIO(dataset_bytes))
except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as error:
    st.error(f"Could not read the CSV dataset: {error}")
    st.stop()

if data.empty:
    st.error("The dataset contains no data rows.")
    st.stop()
if len(data.columns) < 2:
    st.error("The dataset must contain at least one feature column and one target column.")
    st.stop()

target_options = data.columns.to_list()
preferred_target = "is_spam" if "is_spam" in target_options else target_options[-1]
target_column = st.sidebar.selectbox(
    "Target column",
    options=target_options,
    index=target_options.index(preferred_target),
)

st.subheader("Dataset Overview")
overview_columns = st.columns(4)
overview_columns[0].metric("Instances", f"{len(data):,}")
overview_columns[1].metric("Features", f"{len(data.columns) - 1:,}")
overview_columns[2].metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
overview_columns[3].metric(
    "Rows missing target", f"{int(data[target_column].isna().sum()):,}"
)
if target_column == "is_spam" and len(data.columns) == 58:
    st.caption(
        "UCI Spambase contains 4,601 email instances and 57 features. "
        "[View the UCI dataset](https://archive.ics.uci.edu/dataset/94)."
    )
st.caption(
    "Rows without a target are excluded. Missing or non-numeric feature values "
    "are imputed with medians learned from the training split."
)
st.dataframe(data.head(10), width="stretch")

st.subheader("Choose models and run evaluation")
run_mode = st.radio(
    "Evaluation mode",
    options=["Compare all five models", "Run one selected model"],
    horizontal=True,
)
selected_model = None
if run_mode == "Run one selected model":
    selected_model = st.selectbox("Model", MODEL_NAMES)
    models_to_run = [selected_model]
else:
    models_to_run = MODEL_NAMES

dataset_digest = hashlib.sha256(dataset_bytes).hexdigest()
run_signature = (dataset_digest, target_column, tuple(models_to_run))
run_clicked = st.button("Run evaluation", type="primary")
if run_clicked:
    with st.spinner("Training model(s) and evaluating the shared test split..."):
        try:
            results, details = evaluate_models(
                data,
                target_column,
                model_names=models_to_run,
            )
        except ValueError as error:
            st.error(str(error))
            st.stop()
    st.session_state["evaluation"] = {
        "signature": run_signature,
        "results": results,
        "details": details,
    }

evaluation = st.session_state.get("evaluation")
if not evaluation or evaluation["signature"] != run_signature:
    st.info("Choose an evaluation mode and click **Run evaluation** to train the model(s).")
    st.stop()

results = evaluation["results"]
details = evaluation["details"]
metric_columns = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC",
]

st.subheader("Model comparison" if len(results) > 1 else "Model evaluation")
st.caption(
    "All selected models use the same stratified 25% test split (random seed 42). "
    "Preprocessing is fitted on training data only."
)
comparison = pd.DataFrame.from_dict(results, orient="index")[metric_columns]
comparison.index.name = "ML Model Name"
st.dataframe(
    comparison.style.format("{:.3f}").highlight_max(
        subset=pd.Index(metric_columns), color="#d9f2d9"
    ),
    width="stretch",
)

st.subheader("Performance observations")
for name in results:
    with st.container(border=True):
        st.markdown(f"**{name}**")
        st.write(details[name]["observation"])

if len(results) > 1:
    winner = max(
        results,
        key=lambda name: (
            results[name]["F1"],
            results[name]["AUC"],
            results[name]["MCC"],
        ),
    )
    st.success(
        f"Overall winner on this holdout split: **{winner}** "
        "(ranked by F1, then AUC, then MCC)."
    )
    st.caption(
        "The winner is a split-specific result, not a guarantee of future performance. "
        "For deployment, validate with cross-validation and select a decision threshold "
        "based on the relative cost of missed spam and legitimate mail flagged as spam."
    )

st.subheader("Confusion matrix and classification report")
if len(results) > 1:
    inspected_model = st.selectbox("Inspect model", list(results))
else:
    inspected_model = next(iter(results))

selected_details = details[inspected_model]
selected_metrics = results[inspected_model]
detail_columns = st.columns(6)
for column, metric in zip(detail_columns, metric_columns):
    column.metric(metric, f"{selected_metrics[metric]:.3f}")

label_names = selected_details["class_names"]
matrix = confusion_matrix(
    selected_details["y_true"],
    selected_details["y_pred"],
    labels=[0, 1],
)
st.markdown("**Confusion matrix**")
st.dataframe(
    pd.DataFrame(
        matrix,
        index=pd.Index(label_names, name="Actual"),
        columns=pd.Index(label_names, name="Predicted"),
    ),
    width="stretch",
)
st.markdown("**Classification report**")
st.caption(
    "Class 1 is the positive class used for precision, recall, F1, and AUC. "
    f"Here it represents {label_names[1]}."
)
report = classification_report(
    selected_details["y_true"],
    selected_details["y_pred"],
    labels=[0, 1],
    target_names=label_names,
    output_dict=True,
    zero_division=0,
)
st.dataframe(pd.DataFrame(report).transpose().style.format("{:.3f}"))
