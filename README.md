# Spambase classifier comparison

This Streamlit app compares five binary classifiers on the UCI Spambase dataset:
Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes,
and Support Vector Machine. It reports accuracy, ROC AUC, precision, recall, F1,
and Matthews correlation coefficient (MCC), plus per-model observations,
confusion matrices, and classification reports.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

By default, the app loads the included `spambase.csv`. You can instead upload a
labeled CSV and select its target column in the sidebar. The target must have
exactly two non-missing classes, and the feature columns must contain numeric
values. Invalid numeric feature values and missing feature values are imputed
with training-set medians; rows without a target are excluded. At least eight
labeled rows and two examples of each class are required.

All models use the same stratified 75%/25% train/test split (random seed 42).
Scaling and imputation are fitted within each model's training pipeline to avoid
leaking test-set information. The positive class is the second target class
after a stable string-based sort; in the bundled dataset, this is `is_spam = 1`.
The comparison's overall winner is ranked by F1, with AUC and MCC as tie-breakers.
It is a holdout-set demonstration rather than a substitute for cross-validation
and deployment-specific threshold selection.

## Deploy on Streamlit Community Cloud

Push this repository, including `app.py`, `models/`, `requirements.txt`, and
`spambase.csv`, to GitHub. In Streamlit Community Cloud, create an app from the
repository, select `app.py` as the entry point, and deploy. No secrets or
external data services are required.