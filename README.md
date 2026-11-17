# Telco Customer Churn Prediction (CM2604)

Name: Yuhan Thasmal | IIT ID: 20250383 | RGU ID: 2609847
GitHub URL: https://github.com/yunuxbro/telco-churn-cm2604

## 1. Overview
Binary classification of customer churn on the IBM/Kaggle *Telco Customer Churn* dataset, comparing
**K-Nearest Neighbours**, **Logistic Regression** and **Random Forest** under two experimental settings
(A: default hyperparameters, B: tuned with `GridSearchCV`, 5-fold stratified CV, F1 scoring).

## 2. Dataset
`data/Telco-Customer-Churn.csv` – 7,043 customers, 21 columns, 26.5% churn
(source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## 3. How to run
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook                  # run notebooks/01_EDA.ipynb, then notebooks/02_Models.ipynb
python tools/make_appendix.py     # regenerate the source-code appendix for the report
```
`02_Models.ipynb` takes about 15 minutes, mostly the Random Forest grid search.

| Path | Contents |
|---|---|
| `src/data_prep.py` | shared cleaning and feature engineering |
| `notebooks/01_EDA.ipynb` | Task 1 – exploratory data analysis |
| `notebooks/02_Models.ipynb` | Task 2 – preprocessing, models, tuning, evaluation, fairness check |
| `figures/`, `results/` | saved plots and result tables |
| `report/` | report draft and source-code appendix |

## 4. Methodology
Cleaning: `TotalCharges` to numeric (11 blanks belong to tenure-0 customers and are set to 0), `customerID` dropped,
"No internet/phone service" merged into "No", engineered `NumServices`. Stratified 80/20 split, then a
`Pipeline` with `StandardScaler` + `OneHotEncoder` (no leakage). Imbalance: `class_weight="balanced"` (LR, RF), SMOTE (KNN).

## 5. Results (test set, tuned models)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| KNN (K=41, Manhattan, SMOTE) | 0.727 | 0.492 | 0.832 | 0.618 | 0.828 |
| Logistic Regression (C=0.1, L2, balanced) | 0.740 | 0.507 | 0.786 | 0.616 | 0.841 |
| **Random Forest** (400 trees, depth 10, leaf 5, balanced) | 0.759 | 0.531 | 0.794 | **0.637** | **0.844** |

Main churn drivers: month-to-month contract, short tenure, fibre-optic internet, electronic-check payment.

## 6. AI ethics
Data minimisation, fairness audit across gender and SeniorCitizen, leakage-free evaluation, explainability
(coefficients, feature and permutation importance); post-deployment monitoring, drift detection, human-in-the-loop use. See report §5.

## 7. Limitations and future work
Single static dataset, missing behavioural drivers, default 0.5 threshold, no cost-sensitive evaluation.
Future work: gradient boosting, SMOTENC, cost-based threshold, SHAP, significance testing.

## 8. GenAI acknowledgement
Claude (Anthropic) was used to help structure the project, write and debug code, and draft the report (RGU GenAI category 5 – Full AI).
