from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from models.evaluation import MODEL_NAMES, evaluate_models


st.set_page_config(page_title="Spambase model comparison", page_icon="📧", layout="wide")

st.title("Spambase: binary classification")
st.write(
    "Compare five classifiers on one stratified holdout split, inspect their "
    "evaluation metrics, and explore the errors made by each model."
)

uploaded_file = st.sidebar.file_uploader("Upload a labeled CSV dataset", type=["csv"])
if uploaded_file is None:
    default_path = Path(__file__).with_name("spambase.csv")
    if not default_path.exists():
        st.info("Upload a labeled CSV to begin. The bundled Spambase CSV was not found.")
        st.stop()
    data = pd.read_csv(default_path)
    st.sidebar.caption("Using the bundled spambase.csv dataset.")
else:
    try:
        data = pd.read_csv(uploaded_file)
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        st.stop()
    if data.empty:
        st.error("The uploaded CSV contains no data rows.")
        st.stop()

if len(data.columns) < 2:
    st.error("The dataset must contain at least one feature column and one target column.")
    st.stop()

preferred_target = "is_spam" if "is_spam" in data.columns else data.columns[-1]
target_column = st.sidebar.selectbox(
    "Target column",
    options=data.columns.tolist(),
    index=data.columns.get_loc(preferred_target),
)

st.subheader("Dataset overview")
overview_columns = st.columns(4)
overview_columns[0].metric("Rows", f"{len(data):,}")
overview_columns[1].metric("Columns", f"{len(data.columns):,}")
overview_columns[2].metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
overview_columns[3].metric(
    "Rows missing target", f"{int(data[target_column].isna().sum()):,}"
)
st.caption("Preview")
st.dataframe(data.head(10), use_container_width=True)
st.caption(
    "Rows without a target are excluded. Missing or non-numeric feature values "
    "are imputed with training-set medians."
)

with st.spinner("Training models and evaluating the shared test split..."):
    try:
        results, details = evaluate_models(data, target_column)
    except ValueError as error:
        st.error(str(error))
        st.stop()

st.subheader("Model comparison")
st.caption(
    "All models use the same stratified 25% test split (random seed 42). "
    "Preprocessing is fitted on training data only."
)
metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
comparison = pd.DataFrame.from_dict(results, orient="index")[metric_columns]
comparison.index.name = "ML Model Name"
st.dataframe(
    comparison.style.format("{:.3f}").highlight_max(
        subset=metric_columns, color="#d9f2d9"
    ),
    use_container_width=True,
)

st.subheader("Performance observations")
for name, metrics in results.items():
    with st.container(border=True):
        st.markdown(f"**{name}**")
        st.write(details[name]["observation"])

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

st.subheader("Inspect one model")
selected_model = st.selectbox("Model", MODEL_NAMES)
selected_details = details[selected_model]
selected_metrics = results[selected_model]
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
    use_container_width=True,
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
