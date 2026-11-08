# Problem Statement

- Implement multiple classification models on the same dataset.
- Build an interactive Streamlit web application to demonstrate the models.
- Deploy the application on Streamlit Community Cloud (free tier).
- Share clickable links for evaluation.

**Live application:** [Spambase Classification Demo](https://bits-ml-2026aj05102.streamlit.app/)

## Dataset Description

This project uses the [UCI Spambase dataset](https://archive.ics.uci.edu/dataset/94), a collection of **4,601 email instances** described by **57 numeric features**. The features represent word frequencies, character frequencies, and capital-letter run-length statistics. The binary `is_spam` target identifies whether an email is spam.

The included [spambase.csv](./spambase.csv) is used by default in the app. The [get_data.py](./get_data.py) script retrieves dataset 94 from UCI via `ucimlrepo` and writes the features and target to `spambase.csv`.

## Machine Learning Classification Models and Evaluation Metrics

The five classifiers are implemented in the `models/` directory:

| Model | Python implementation |
|---|---|
| Logistic Regression | [models/logistic_regression.py](./models/logistic_regression.py) |
| Decision Tree Classifier | [models/decision_tree.py](./models/decision_tree.py) |
| K-Nearest Neighbor Classifier | [models/knn.py](./models/knn.py) |
| Gaussian Naive Bayes Classifier | [models/naive_bayes.py](./models/naive_bayes.py) |
| Support Vector Machine | [models/support_vector_machine.py](./models/support_vector_machine.py) |

Model training, shared train/test splitting, and metric calculation are handled by [models/evaluation.py](./models/evaluation.py). Each model is evaluated on the same stratified 25% holdout split (random seed 42). Preprocessing is fitted on training data only.

### Model Comparison

The following results are from the bundled Spambase dataset and the fixed holdout split. The app displays the same metrics for the selected dataset.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.928 | 0.973 | 0.930 | 0.883 | 0.906 | 0.848 |
| Decision Tree | 0.908 | 0.908 | 0.864 | 0.910 | 0.886 | 0.810 |
| K-Nearest Neighbor | 0.899 | 0.951 | 0.879 | 0.863 | 0.871 | 0.788 |
| Naive Bayes (Gaussian) | 0.831 | 0.947 | 0.713 | 0.954 | 0.816 | 0.689 |
| Support Vector Machine | 0.929 | 0.970 | 0.933 | 0.883 | **0.907** | **0.850** |

On this split, the Support Vector Machine has the highest F1 and MCC; Logistic Regression has the highest AUC. Gaussian Naive Bayes has the highest recall, which means it identifies the largest share of spam emails but also has lower precision. The app ranks its holdout winner by F1, using AUC and MCC as tie-breakers. Results may vary with a different dataset or split and should not be treated as a guarantee of future performance.

## Streamlit Application

The [live Streamlit app](https://bits-ml-2026aj05102.streamlit.app/) includes:

- A CSV upload option and target-column selector.
- An evaluation mode to run one selected classifier or compare all five.
- An explicit **Run evaluation** button so model training does not happen on every UI change.
- A comparison table for accuracy, AUC, precision, recall, F1, and MCC.
- Per-model performance observations.
- A confusion matrix and classification report for the selected model.

The app lists CSV files committed anywhere in the repository (excluding `.git` and local virtual-environment folders); `spambase.csv` is selected by default. You can also upload a CSV from your device. For either source, the dataset must have a binary target and numeric feature columns. Rows missing the target are excluded; missing or non-numeric feature values are imputed using medians learned from the training split. Select the evaluation mode and click **Run evaluation** to train the selected model or all five models.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Push this repository to GitHub with `app.py`, `models/`, `requirements.txt`, and `spambase.csv`. In [Streamlit Community Cloud](https://streamlit.io/cloud), create an app from the repository, choose `app.py` as the entry point, and deploy. The live deployment is available at [bits-ml-2026aj05102.streamlit.app](https://bits-ml-2026aj05102.streamlit.app/).
