from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from models.decision_tree import build_model as build_decision_tree
from models.knn import build_model as build_knn
from models.logistic_regression import build_model as build_logistic_regression
from models.naive_bayes import build_model as build_naive_bayes
from models.support_vector_machine import build_model as build_support_vector_machine

MODEL_BUILDERS: dict[str, Callable] = {
    "Logistic Regression": build_logistic_regression,
    "Decision Tree": build_decision_tree,
    "K-Nearest Neighbors": build_knn,
    "Naive Bayes (Gaussian)": build_naive_bayes,
    "Support Vector Machine": build_support_vector_machine,
}
MODEL_NAMES = list(MODEL_BUILDERS)
MINIMUM_ROWS = 8
TEST_SIZE = 0.25
RANDOM_STATE = 42


def _prepare_dataset(
    data: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series, list[object]]:
    if target_column not in data.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")

    frame = data.copy()
    frame = frame.dropna(subset=[target_column])
    if frame.empty:
        raise ValueError("No rows have a target value.")

    target_values = frame[target_column]
    classes = list(pd.unique(target_values))
    if len(classes) != 2:
        raise ValueError(
            f"The target must contain exactly two classes after removing missing "
            f"target values; found {len(classes)}."
        )
    classes.sort(key=lambda value: str(value))
    encoded_target = target_values.map({classes[0]: 0, classes[1]: 1}).astype(int)

    features = frame.drop(columns=[target_column])
    if features.shape[1] == 0:
        raise ValueError("At least one feature column is required.")
    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )

    unusable_columns = features.columns[features.isna().all()].tolist()
    if unusable_columns:
        raise ValueError(
            "These feature columns contain no usable numeric values: "
            + ", ".join(map(str, unusable_columns))
        )
    if len(frame) < MINIMUM_ROWS:
        raise ValueError(f"At least {MINIMUM_ROWS} labeled rows are required.")
    if encoded_target.value_counts().min() < 2:
        raise ValueError(
            "Each target class needs at least two labeled rows for a stratified split."
        )

    return features, encoded_target, classes


def _observation(metrics: dict[str, float]) -> str:
    if metrics["F1"] >= 0.9:
        summary = "strong balance between spam detection and avoiding false alarms"
    elif metrics["F1"] >= 0.75:
        summary = "useful but imperfect balance between spam detection and false alarms"
    else:
        summary = "weaker balance between spam detection and false alarms"

    return (
        f"On the holdout split, this model shows {summary} "
        f"(F1 {metrics['F1']:.3f}, recall {metrics['Recall']:.3f}, "
        f"precision {metrics['Precision']:.3f}, AUC {metrics['AUC']:.3f}). "
        f"It correctly classifies {metrics['Accuracy']:.1%} of records. "
        "Higher recall catches more positive cases; higher precision means fewer "
        "negative cases are incorrectly flagged as positive."
    )


def evaluate_models(
    data: pd.DataFrame,
    target_column: str,
    model_names: Sequence[str] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, object]]]:
    features, target, classes = _prepare_dataset(data, target_column)
    selected_models = MODEL_NAMES if model_names is None else list(model_names)
    unknown_models = [name for name in selected_models if name not in MODEL_BUILDERS]
    if unknown_models:
        raise ValueError(f"Unknown model name(s): {', '.join(unknown_models)}.")
    if not selected_models:
        raise ValueError("Select at least one model to evaluate.")

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    results: dict[str, dict[str, float]] = {}
    details: dict[str, dict[str, object]] = {}
    for name in selected_models:
        build_model = MODEL_BUILDERS[name]
        model = build_model()
        model.fit(train_features, train_target)
        predictions = model.predict(test_features)
        if hasattr(model, "predict_proba"):
            positive_scores = model.predict_proba(test_features)[:, 1]
        else:
            positive_scores = model.decision_function(test_features)
        metrics = {
            "Accuracy": float(accuracy_score(test_target, predictions)),
            "AUC": float(roc_auc_score(test_target, positive_scores)),
            "Precision": float(
                precision_score(test_target, predictions, zero_division=0)
            ),
            "Recall": float(recall_score(test_target, predictions, zero_division=0)),
            "F1": float(f1_score(test_target, predictions, zero_division=0)),
            "MCC": float(matthews_corrcoef(test_target, predictions)),
        }
        results[name] = metrics
        class_names = [f"Class 0 ({classes[0]})", f"Class 1 ({classes[1]})"]
        if target_column == "is_spam" and classes == [0, 1]:
            class_names = ["Not spam (0)", "Spam (1)"]
        details[name] = {
            "y_true": np.asarray(test_target, dtype=int),
            "y_pred": np.asarray(predictions, dtype=int),
            "class_names": class_names,
            "observation": _observation(metrics),
        }

    return results, details
