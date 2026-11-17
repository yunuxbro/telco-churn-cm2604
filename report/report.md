# Predicting Telco Customer Churn with KNN, Logistic Regression and Random Forest

**Module:** CM2604 Machine Learning · **Student:** Yuhan Thasmal (IIT 20250383 / RGU 2609847)
**Git repository:** https://github.com/yunuxbro/telco-churn-cm2604

---

## 1. Introduction

Customer churn – a customer cancelling their service – is costly for telecom operators because winning a new customer usually costs more than keeping an existing one. This project treats churn as a **binary classification** problem: given a customer's account, service and billing attributes, predict whether they will churn (1) or stay (0). Three supervised algorithms required by the brief are implemented with scikit-learn and compared: K-Nearest Neighbours (KNN), Logistic Regression (LR) and Random Forest (RF).

The data is the public IBM "Telco Customer Churn" dataset (BlastChar, 2018): 7,043 customers and 21 columns covering demographics (gender, SeniorCitizen, Partner, Dependents), account details (tenure, Contract, PaymentMethod, PaperlessBilling), subscribed services (phone, internet and six add-ons) and charges (MonthlyCharges, TotalCharges). The target is `Churn` (Yes/No).

All work is version-controlled with Git. `notebooks/01_EDA.ipynb` covers Task 1, `notebooks/02_Models.ipynb` covers Task 2 and the fairness checks used in Task 3, and `src/data_prep.py` holds the shared cleaning rules.

## 2. Corpus Preparation

### 2.1 Data quality
`df.info()` reported no missing values, but `TotalCharges` was stored as text. Converting it with `pd.to_numeric(errors="coerce")` revealed **11 blank entries**. All 11 customers had a tenure of 0 months – they had not yet been billed – so their totals were set to **0** rather than imputed with the mean, which would have invented spending that never happened. There were no duplicate rows. 22 rows become identical once `customerID` is removed, but they are distinct customers on identical basic plans, so they were kept. The 1.5×IQR rule found no outliers in the numeric columns, and all values lay in valid ranges.

### 2.2 Exploratory data analysis – key insights
- **Class imbalance:** only **26.5%** of customers churn (Figure 1). A model that always predicts "stay" is 73.5% accurate, so accuracy alone is misleading.
- **Tenure** is the strongest numeric signal: churners have a median tenure of 10 months versus 38 for loyal customers. Churn is 47% in the first year and falls to 9.5% after four years (Figure 3).
- **Contract type** is the strongest categorical signal: month-to-month customers churn at **43%**, against 11% (one-year) and **3%** (two-year).
- **Fibre-optic** internet (42%), **electronic-check** payment (45%) and having **no OnlineSecurity/TechSupport** (≈42%) all roughly double the churn rate. Senior citizens churn at 42% versus 24%. Gender (27% vs 26%) and PhoneService make almost no difference (Figure 4).
- **Interactions:** month-to-month fibre customers churn at 55%, while two-year customers churn at ≤7% regardless of internet type (Figure 5). This suggests models that capture interactions may help.
- **Correlation:** `TotalCharges` correlates 0.83 with tenure (it is roughly tenure × monthly charge), a source of multicollinearity to remember when interpreting LR coefficients (Figure 7).
- **Unsupervised view:** PCA needs 15 of 24 encoded components to keep 90% of the variance, and churners overlap heavily with non-churners on the first two components. K-means (K = 3), without seeing labels, still separated segments with 8%, 15% and 45% churn (Figures 9–10).

### 2.3 Cleaning, feature engineering and encoding
`src/data_prep.py` applies fixed row-wise rules (none learned from data, so they cannot leak test information):
1. `TotalCharges` converted to numeric, with blanks set to 0.
2. `customerID` dropped: it has no predictive value, and dropping it follows data minimisation.
3. "No internet service" and "No phone service" mapped to "No", because `InternetService` and `PhoneService` already record those facts. This avoids duplicated columns.
4. `SeniorCitizen` mapped from 0/1 to No/Yes so every categorical column is treated alike.
5. **Engineered feature `NumServices`** (0–9 subscribed services). Its churn rate is clearly non-linear: 11% at one service, 45% at three and 5% at nine (Figure 6).

The data was split **70/30 with stratification** (`test_size=0.3, random_state=42`, as in the module labs): 4,930 training and 2,113 test customers, both 26.5% churn. Scaling and encoding were placed inside a scikit-learn `Pipeline` with a `ColumnTransformer`, so they are fitted only on training folds:
- **StandardScaler** on the four numeric columns. KNN distances and LR's regularised weights are otherwise dominated by large-valued columns such as TotalCharges.
- **One-hot encoding** of the 16 categorical columns (`drop="if_binary"`), giving 27 model inputs. Label encoding was rejected because nominal categories such as payment method have no order.

## 3. Methodology

### 3.1 Models
- **KNN** classifies a customer by majority vote of the K most similar training customers. It is a non-parametric "lazy learner" that can model local, non-linear boundaries but is sensitive to scaling and noise.
- **Logistic Regression** computes a weighted sum of features and passes it through the sigmoid function to output a churn probability. It is linear, fast and interpretable through coefficients and odds ratios.
- **Random Forest** is a bagging ensemble of decision trees, each trained on a bootstrap sample and considering a random subset of features at each split. Averaging many de-correlated trees reduces the variance of a single tree and captures interactions (Breiman, 2001).

### 3.2 Experimental settings
Two settings were run for every model:
- **Setting A – Baseline:** scikit-learn defaults (K = 5; C = 1; 100 fully grown trees), with no imbalance handling.
- **Setting B – Tuned:** `GridSearchCV` with stratified 5-fold cross-validation on the training set, optimising **F1** for the churn class.

| Model | Grid searched | Best parameters |
|---|---|---|
| KNN | K ∈ {5…51}, weights, p ∈ {1, 2}, SMOTE on/off | K = 41, Manhattan (p = 1), uniform, no SMOTE |
| LR | C ∈ {0.001…100}, L1/L2, class_weight | C = 0.1, L2, **balanced** |
| RF | trees {200, 400}, max_depth {6, 10, 15, None}, min_samples_leaf {1, 5, 10}, max_features {sqrt, 0.5}, class_weight | 400 trees, depth 10, leaf 10, max_features 0.5, **balanced** |

Each model received an imbalance strategy that suits it. LR and RF support `class_weight="balanced"`, which raises the cost of misclassifying a churner by about 2.8×. KNN has no class weights, so SMOTE (Chawla et al., 2002) was offered inside an imbalanced-learn pipeline (Lemaître, Nogueira and Aridas, 2017), which creates synthetic churners only inside training folds; the search ultimately preferred no resampling for KNN.

### 3.3 Evaluation criteria
The positive class is churn. **F1** (the harmonic mean of precision and recall) is the primary metric because it focuses on the minority class. **Recall** measures how many churners are caught, **precision** measures how many retention offers reach real churners, **specificity** measures how many loyal customers are correctly left alone, and **ROC-AUC** measures threshold-independent ranking quality. Accuracy is compared between training and test sets to detect overfitting, but not used for selection. Cross-validation results are given as mean ± standard deviation, and the test set was used once for final scores.

## 4. Results and Model Evaluation

**Table 1 – Cross-validation on the training set (F1 mean ± std over 5 folds)**

| Model | Setting | Train F1 | CV F1 | CV recall | CV precision | CV AUC |
|---|---|---|---|---|---|---|
| KNN | A | 0.678 | 0.547 ± 0.016 | 0.526 | 0.570 | 0.780 |
| LR | A | 0.600 | 0.595 ± 0.022 | 0.547 | 0.654 | 0.844 |
| RF | A | 0.997 | 0.562 ± 0.016 | 0.498 | 0.645 | 0.829 |
| KNN | B | 0.624 | 0.610 ± 0.026 | 0.589 | 0.633 | 0.837 |
| LR | B | 0.630 | 0.628 ± 0.019 | 0.804 | 0.515 | 0.843 |
| RF | B | 0.716 | **0.632 ± 0.021** | 0.775 | 0.534 | **0.846** |

**Table 2 – Final test-set results (2,113 unseen customers)**

| Model | Setting | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| KNN | A | 0.767 | 0.565 | 0.528 | 0.853 | 0.546 | 0.788 |
| LR | A | 0.809 | **0.668** | 0.560 | **0.899** | 0.609 | **0.845** |
| RF | A | 0.781 | 0.612 | 0.481 | 0.890 | 0.539 | 0.818 |
| KNN | B | 0.787 | 0.603 | 0.579 | 0.862 | 0.591 | 0.834 |
| LR | B | 0.747 | 0.515 | **0.797** | 0.729 | 0.626 | 0.844 |
| RF | B | 0.769 | 0.544 | 0.790 | 0.761 | **0.644** | 0.841 |

**Table 3 – Training vs test accuracy (overfitting check)**

| Model | Setting | Train accuracy | Test accuracy | Gap |
|---|---|---|---|---|
| Majority class ("stay") | – | 0.735 | 0.735 | 0.000 |
| KNN | A | 0.837 | 0.767 | 0.070 |
| LR | A | 0.804 | 0.809 | −0.005 |
| RF | A | 0.998 | 0.781 | **0.217** |
| KNN | B | 0.809 | 0.787 | 0.021 |
| LR | B | 0.749 | 0.747 | 0.002 |
| RF | B | 0.816 | 0.769 | 0.047 |

**Baseline behaviour.** Without imbalance handling, all three models missed about half of the churners (recall 0.48–0.56) despite 77–81% accuracy, barely above the 73.5% majority baseline. The baseline Random Forest shows classic **overfitting**: 99.8% training accuracy against 78.1% test accuracy (Table 3), because fully grown trees memorise the training data. KNN (K = 5) had a smaller gap; LR had none, as expected of a simple, low-variance model.

**Effect of tuning.** Tuning improved test F1 for every model: KNN +0.045, LR +0.017 and RF +0.105. For LR and RF, class weighting raised recall to 0.79–0.80, so four in five churners are now identified. The cost is precision of about 0.5 and lower specificity and accuracy – the expected trade-off when a model stops favouring the majority class. The tuning curves (Figure 11) show the bias–variance trade-off: larger K closed KNN's train/CV gap, and limiting tree depth and leaf size cut the forest's train/test accuracy gap from 22 to 5 points. LR was insensitive to C above 0.01.

**Model comparison.** The tuned **Random Forest** achieved the best CV F1 (0.632) and test F1 (0.644), with recall 0.790 and better precision than LR, so it is selected. The margin is small: LR is within one standard deviation on CV F1, has the highest recall (0.797) and a marginally higher AUC (0.844), showing that most of the signal is close to linear. The tuned KNN is the most accurate tuned model (0.787) but catches only 58% of churners and is slow at prediction time. LR is preferable if each decision must be explained. Because the models output probabilities, the decision threshold can also be moved along the precision–recall curve (Figure 14) to match the campaign budget.

**Explainability.** All three methods agree with the EDA (Figures 15–16). In LR, a month-to-month contract multiplies the churn odds by 1.84 and fibre optic by 1.44, while tenure (coefficient −1.00) and a two-year contract (−0.69) protect against churn. The forest ranks month-to-month contract, tenure and charges highest. Permutation importance confirms that Contract, tenure and InternetService dominate, while gender contributes nothing.

**Error analysis.** The 118 churners the forest missed differ sharply from the 443 it caught: median tenure 31.5 versus 6 months, 50% versus 99% on month-to-month contracts, and lower monthly charges. They look like loyal customers and probably leave for reasons not recorded in the data (relocation, competitor offers or service incidents).

## 5. AI Ethics (Task 3)

### 5.1 Strategies followed during development
- **Data minimisation and privacy (UK GDPR Art. 5):** `customerID` was removed, only attributes relevant to churn were used, and the public dataset contains no names or contact details.
- **Fairness auditing:** recall, precision and flag rates were compared across gender and SeniorCitizen (age is a protected characteristic under the Equality Act 2010). Gender showed no meaningful disparity: the forest flags 38–39% of both groups with the same false-alarm rate (24%). Seniors are flagged far more often (62% vs 34%) with higher recall (0.88 vs 0.76), but double the false-alarm rate (41% vs 21%), partly reflecting their genuinely higher churn rate. This is documented rather than hidden; senior status was kept because removing it would not remove proxies such as payment method.
- **Honest evaluation:** stratified splits, leakage-free pipelines, cross-validation with reported variance, a single use of the test set, and imbalance-aware metrics instead of an inflated accuracy figure.
- **Transparency and explainability:** coefficients, feature and permutation importance are reported, and limitations are stated openly (BCS Code of Conduct: competence and integrity).
- **Reproducibility and accountability:** fixed random seeds, pinned library versions, a public Git history and this report.

### 5.2 Post-deployment strategies
- **Human in the loop:** predictions should only prioritise *positive* retention outreach reviewed by staff. They should never trigger automatic price changes or service denial, and customers keep their right to contest automated decisions (UK GDPR Art. 22).
- **Monitoring:** track precision, recall and AUC on newly labelled data, and watch for **data drift** (changing input distributions such as new tariffs) and **concept drift** (a changing link between features and churn, for example after a competitor's launch). Retrain on a schedule or when alerts fire.
- **Fairness audits** after every retraining, using the same group metrics.
- **Feedback-loop control:** retention offers change who churns, so interventions must be logged and excluded from naïve retraining labels.
- **Governance and security:** a model card, versioned models and data, audit logs, access control, retention limits, and a pilot or A/B rollout with a rollback plan.

## 6. Limitations and Future Enhancements

**Limitations:** the data is a single, static snapshot from one (sample) company, so results may not transfer to other operators or time periods. Many churn drivers (service quality, competitor prices, complaints) are absent, which explains the missed long-tenure churners. SMOTE interpolates one-hot columns into fractional values. The decision threshold was left at 0.5, and no cost-sensitive (profit-based) evaluation was performed. Model differences are small relative to fold variance, and no formal significance test was applied.

**Future work:** gradient boosting (e.g. XGBoost) and stacking ensembles; SMOTENC for mixed data types; tuning the threshold to an explicit cost of a missed churner versus a wasted offer; probability calibration; SHAP explanations for individual customers; statistical comparison (McNemar's test or repeated CV); and survival analysis to predict *when* a customer will churn.

## 7. Conclusion

After careful cleaning, leakage-free preprocessing and imbalance-aware tuning, the tuned Random Forest and Logistic Regression identify around 80% of churners on unseen data, while the more accurate KNN catches fewer. The tuned Random Forest performed best (F1 0.644, ROC-AUC 0.841), with Logistic Regression a close and more interpretable alternative. Contract type, tenure and internet service are the dominant churn drivers, which points to concrete retention levers such as converting month-to-month fibre customers to longer contracts during their first year. Deployment should keep humans in the loop and include continuous monitoring and fairness audits.

---

## References
BCS (2022) *Code of Conduct for BCS Members*. Swindon: BCS, The Chartered Institute for IT.
BlastChar (2018) *Telco Customer Churn* [dataset]. Kaggle. Available at: https://www.kaggle.com/datasets/blastchar/telco-customer-churn.
Breiman, L. (2001) 'Random forests', *Machine Learning*, 45(1), pp. 5–32.
Chawla, N.V., Bowyer, K.W., Hall, L.O. and Kegelmeyer, W.P. (2002) 'SMOTE: synthetic minority over-sampling technique', *Journal of Artificial Intelligence Research*, 16, pp. 321–357.
Information Commissioner's Office (2023) *Guide to the UK General Data Protection Regulation (UK GDPR)*. Wilmslow: ICO.
Lemaître, G., Nogueira, F. and Aridas, C.K. (2017) 'Imbalanced-learn: a Python toolbox to tackle the curse of imbalanced datasets in machine learning', *Journal of Machine Learning Research*, 18(17), pp. 1–5.
Pedregosa, F. et al. (2011) 'Scikit-learn: machine learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825–2830.

## GenAI Acknowledgement
I acknowledge use of Claude (Anthropic) from https://claude.ai to help me complete my assessment in the following ways: to help me structure the project and notebooks, generate and debug Python code, interpret results, and draft and edit the wording of this report. *[Add the dates and the prompts you used, as required by the RGU template.]* All code was run, checked and understood by me, and I can explain every part of it.

## Figures
*(Insert these images from the `figures/` folder in the final Word/PDF version.)*
Figure 1 – `01_target_distribution.png` · Figure 2 – `02_numeric_distributions.png` · Figure 3 – `03_churn_by_tenure_group.png` · Figure 4 – `04_churn_by_category.png` · Figure 5 – `05_contract_internet_heatmap.png` · Figure 6 – `06_churn_by_num_services.png` · Figure 7 – `07_correlation_heatmap.png` · Figure 8 – `08_outlier_boxplots.png` · Figure 9 – `09_pca.png` · Figure 10 – `10_kmeans_segments.png` · Figure 11 – `11_tuning_curves.png` · Figure 12 – `12_baseline_vs_tuned.png` · Figure 13 – `13_confusion_matrices.png` · Figure 14 – `14_roc_pr_curves.png` · Figure 15 – `15_feature_importance.png` · Figure 16 – `16_permutation_importance.png`

## Appendix A – Source code
See `appendix_source_code.md` (all source code as text, generated by `tools/make_appendix.py`).
