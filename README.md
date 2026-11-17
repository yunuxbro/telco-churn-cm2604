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
"No internet/phone service" merged into "No", engineered `NumServices`. Stratified 70/30 split (`test_size=0.3, random_state=42`, as in the labs), then a
`Pipeline` with `StandardScaler` + `OneHotEncoder` (no leakage). Imbalance: `class_weight="balanced"` (LR, RF); SMOTE was searched for KNN but not selected.

## 5. Results (test set, tuned models)
| Model | Train acc. | Test acc. | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| KNN (K=41, Manhattan) | 0.809 | 0.787 | 0.603 | 0.579 | 0.862 | 0.591 | 0.834 |
| Logistic Regression (C=0.1, L2, balanced) | 0.749 | 0.747 | 0.515 | 0.797 | 0.729 | 0.626 | **0.844** |
| **Random Forest** (400 trees, depth 10, leaf 10, balanced) | 0.816 | 0.769 | 0.544 | 0.790 | 0.761 | **0.644** | 0.841 |

Test set = 2,113 customers (30%). Majority-class baseline accuracy = 0.735.

Main churn drivers: month-to-month contract, short tenure, fibre-optic internet, electronic-check payment.

## 6. AI ethics
Data minimisation, fairness audit across gender and SeniorCitizen, leakage-free evaluation, explainability
(coefficients, feature and permutation importance); post-deployment monitoring, drift detection, human-in-the-loop use. See report §5.

## 7. Limitations and future work
Single static dataset, missing behavioural drivers, default 0.5 threshold, no cost-sensitive evaluation.
Future work: gradient boosting, SMOTENC, cost-based threshold, SHAP, significance testing.

## 8. GenAI acknowledgement
Claude (Anthropic) was used to help structure the project, write and debug code, and draft the report (RGU GenAI category 5 – Full AI).
