# Appendix A – Source Code
Repository: https://github.com/yunuxbro/telco-churn-cm2604

## src/data_prep.py

```python
"""
Shared data loading and cleaning for the Telco Customer Churn project.

Both notebooks (01_EDA and 02_Models) call `load_clean_data()` so the data is
cleaned in exactly the same way everywhere.

Every step here is a fixed, row-by-row rule (it does not learn anything from the
data, such as a mean or a scale). That is why it is safe to run before the
train/test split: no information from the test rows can leak into training.
Anything that *learns* from the data (scaling, encoding, resampling) lives
inside the scikit-learn Pipeline in 02_Models, so it is fitted on training data only.
"""
import pandas as pd

# The six add-on services that only exist for customers who have internet
INTERNET_ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "NumServices"]
TARGET = "Churn"


def load_raw(path):
    """Read the CSV exactly as downloaded from Kaggle."""
    return pd.read_csv(path)


def clean(df):
    """Apply the cleaning rules decided during EDA. Returns a new DataFrame."""
    df = df.copy()

    # 1. TotalCharges is stored as text, and 11 rows contain a blank " ".
    #    All 11 have tenure = 0 (brand-new customers who have not been billed),
    #    so the correct total is 0, not an average.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(),
                                       errors="coerce").fillna(0.0)

    # 2. customerID is a unique label with no predictive meaning. Dropping it
    #    also follows the data-minimisation principle (UK GDPR).
    df = df.drop(columns=["customerID"])

    # 3. "No internet service" / "No phone service" mean the same as "No" for
    #    that add-on. InternetService / PhoneService already record whether the
    #    customer has the base service, so keeping a third category would only
    #    duplicate that information (extra, perfectly correlated columns).
    for col in INTERNET_ADDONS:
        df[col] = df[col].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    # 4. SeniorCitizen is 0/1 while every other yes/no column is text.
    #    Make it consistent so all categorical columns are encoded the same way.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # 5. Feature engineering: how many services the customer pays for.
    #    Idea from EDA: customers with more services are more "locked in".
    service_flags = [df["PhoneService"] == "Yes",
                     df["MultipleLines"] == "Yes",
                     df["InternetService"] != "No"]
    service_flags += [df[c] == "Yes" for c in INTERNET_ADDONS]
    df["NumServices"] = sum(flag.astype(int) for flag in service_flags)

    # 6. Target: "Yes"/"No" -> 1/0 (1 = the customer churned = positive class)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def load_clean_data(path):
    return clean(load_raw(path))


def categorical_columns(df):
    """All text columns except the target."""
    return [c for c in df.columns if c not in NUMERIC_COLS + [TARGET]]
```

## tools/make_appendix.py

```python
"""
Build report/appendix_source_code.md: every line of project source code as TEXT.

The brief says the appendix must contain all source code as text (screenshots are
rejected), so re-run this after any code change:   python tools/make_appendix.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "appendix_source_code.md"

parts = ["# Appendix A – Source Code\n",
         "Repository: https://github.com/yunuxbro/telco-churn-cm2604\n"]

for py in [ROOT / "src" / "data_prep.py", ROOT / "tools" / "make_appendix.py"]:
    parts.append(f"\n## {py.relative_to(ROOT).as_posix()}\n\n```python\n{py.read_text(encoding='utf-8').rstrip()}\n```\n")

for nb_path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    parts.append(f"\n## {nb_path.relative_to(ROOT).as_posix()} (code cells)\n")
    n = 0
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            n += 1
            src = "".join(cell["source"]).rstrip()
            parts.append(f"\n**Cell {n}**\n\n```python\n{src}\n```\n")

OUT.parent.mkdir(exist_ok=True)
OUT.write_text("".join(parts), encoding="utf-8")
print("Wrote", OUT)
```

## notebooks/01_EDA.ipynb (code cells)

**Cell 1**

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, "..")                 # so we can import our own src/ package
from src.data_prep import load_raw, clean, INTERNET_ADDONS

DATA_PATH = "../data/Telco-Customer-Churn.csv"
FIG_DIR = Path("../figures"); FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")
pd.set_option("display.max_columns", 30)

def save(name):
    """Save the current figure for the report, then show it."""
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()
```

**Cell 2**

```python
raw = load_raw(DATA_PATH)
print("Rows, columns:", raw.shape)
raw.head()
```

**Cell 3**

```python
raw.info()
```

**Cell 4**

```python
print("Missing values per column (as pandas sees them):", raw.isna().sum().sum())
print("Fully duplicated rows:", raw.duplicated().sum())
print("Duplicated rows if we ignore customerID:", raw.drop(columns="customerID").duplicated().sum())

# Find the hidden problem in TotalCharges
as_number = pd.to_numeric(raw["TotalCharges"].astype(str).str.strip(), errors="coerce")
blank = raw[as_number.isna()]
print("\nTotalCharges values that are not numbers:", len(blank))
blank[["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]]
```

**Cell 5**

```python
df = clean(raw)
print(df.shape)
print("Missing values after cleaning:", df.isna().sum().sum())
print("TotalCharges dtype:", df["TotalCharges"].dtype)
df.head()
```

**Cell 6**

```python
df.describe().T.round(2)
```

**Cell 7**

```python
counts = df["Churn"].value_counts().sort_index()
rate = df["Churn"].mean()
print(counts.rename({0: "Stayed (0)", 1: "Churned (1)"}))
print(f"Churn rate: {rate:.1%}")

fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x=df["Churn"].map({0: "Stayed", 1: "Churned"}), ax=ax)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
ax.set(title=f"Target distribution (churn rate {rate:.1%})", xlabel="", ylabel="Customers")
save("01_target_distribution")
```

**Cell 8**

```python
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
    sns.histplot(data=df, x=col, hue=df["Churn"].map({0: "Stayed", 1: "Churned"}),
                 bins=30, stat="density", common_norm=False, element="step", ax=ax)
    ax.set_title(f"{col} by churn")
save("02_numeric_distributions")

print("Median values by group:")
df.groupby(df["Churn"].map({0: "Stayed", 1: "Churned"}))[["tenure", "MonthlyCharges", "TotalCharges"]].median()
```

**Cell 9**

```python
bins = [-1, 12, 24, 48, 72]
labels = ["0-12", "13-24", "25-48", "49-72"]
tenure_group = pd.cut(df["tenure"], bins=bins, labels=labels)
by_group = df.groupby(tenure_group, observed=True)["Churn"].mean()
print(by_group.round(3))

ax = by_group.plot(kind="bar", color="#e07b5f", figsize=(5, 4), rot=0)
ax.set(title="Churn rate by tenure (months)", ylabel="Churn rate", xlabel="Tenure group")
ax.axhline(rate, ls="--", c="grey", label="overall rate"); ax.legend()
save("03_churn_by_tenure_group")
```

**Cell 10**

```python
cat_features = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
                "OnlineSecurity", "TechSupport", "SeniorCitizen", "Partner",
                "Dependents", "gender", "PhoneService", "StreamingTV"]

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
for ax, col in zip(axes.ravel(), cat_features):
    churn_rate = df.groupby(col)["Churn"].mean().sort_values(ascending=False)
    sns.barplot(x=churn_rate.index, y=churn_rate.values, ax=ax, color="#5f9ee0")
    ax.axhline(rate, ls="--", c="grey")
    ax.set(title=col, ylabel="Churn rate", xlabel="", ylim=(0, 0.6))
    ax.tick_params(axis="x", rotation=20)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.0%}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=9)
fig.suptitle("Churn rate per category (dashed line = overall 26.5%)", fontsize=14)
save("04_churn_by_category")
```

**Cell 11**

```python
pivot = pd.crosstab(df["Contract"], df["InternetService"], values=df["Churn"], aggfunc="mean")
plt.figure(figsize=(6, 3.8))
sns.heatmap(pivot, annot=True, fmt=".0%", cmap="Reds", cbar_kws={"label": "Churn rate"})
plt.title("Churn rate: Contract x InternetService")
save("05_contract_internet_heatmap")
```

**Cell 12**

```python
by_services = df.groupby("NumServices")["Churn"].agg(["mean", "count"])
print(by_services.round(3))
ax = by_services["mean"].plot(marker="o", figsize=(6, 4))
ax.set(title="Churn rate by number of services (engineered feature)", xlabel="NumServices", ylabel="Churn rate")
save("06_churn_by_num_services")
```

**Cell 13**

```python
num = df[["tenure", "MonthlyCharges", "TotalCharges", "NumServices", "Churn"]]
plt.figure(figsize=(6, 5))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Pearson correlation")
save("07_correlation_heatmap")
```

**Cell 14**

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
    sns.boxplot(x=df[col], ax=ax); ax.set_title(col)
    q1, q3 = df[col].quantile([0.25, 0.75]); iqr = q3 - q1
    n_out = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(f"{col}: {n_out} outliers by the 1.5*IQR rule")
save("08_outlier_boxplots")
```

**Cell 15**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

X_all = pd.get_dummies(df.drop(columns="Churn"), drop_first=True, dtype=float)
X_scaled = StandardScaler().fit_transform(X_all)

pca = PCA().fit(X_scaled)
cum = np.cumsum(pca.explained_variance_ratio_)
print("Encoded features:", X_all.shape[1])
print(f"Variance kept by PC1+PC2: {cum[1]:.1%}")
print("Components needed for 90% variance:", int(np.argmax(cum >= 0.90) + 1))

Z = pca.transform(X_scaled)[:, :2]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(1, len(cum) + 1), cum, marker="o")
axes[0].axhline(0.9, ls="--", c="grey"); axes[0].set(title="PCA scree plot (cumulative)", xlabel="Components", ylabel="Variance explained")
axes[1].scatter(Z[:, 0], Z[:, 1], c=df["Churn"], cmap="coolwarm", s=6, alpha=0.5)
axes[1].set(title="Customers on the first 2 principal components (red = churned)", xlabel="PC1", ylabel="PC2")
save("09_pca")
```

**Cell 16**

```python
inertia = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled).inertia_ for k in range(1, 9)]
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_scaled)
seg = df.assign(Segment=km.labels_)
print(seg.groupby("Segment").agg(customers=("Churn", "size"), churn_rate=("Churn", "mean"),
                                 tenure=("tenure", "median"), monthly=("MonthlyCharges", "median"),
                                 services=("NumServices", "median")).round(2))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(range(1, 9), inertia, marker="o"); axes[0].set(title="K-means elbow plot", xlabel="K", ylabel="SSE (inertia)")
axes[1].scatter(Z[:, 0], Z[:, 1], c=km.labels_, cmap="Set2", s=6)
axes[1].set(title="K-means segments (K=3) on PCA axes", xlabel="PC1", ylabel="PC2")
save("10_kmeans_segments")
```

## notebooks/02_Models.ipynb (code cells)

**Cell 1**

```python
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report,
                             RocCurveDisplay, PrecisionRecallDisplay)
from imblearn.pipeline import Pipeline as ImbPipeline   # a Pipeline that allows a resampling step
from imblearn.over_sampling import SMOTE

sys.path.insert(0, "..")
from src.data_prep import load_clean_data, NUMERIC_COLS, TARGET, categorical_columns

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="Set2")
FIG_DIR = Path("../figures"); FIG_DIR.mkdir(exist_ok=True)
RES_DIR = Path("../results"); RES_DIR.mkdir(exist_ok=True)
SEED = 42            # fixed random seed -> every run gives the same numbers (reproducibility)

def save(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()
```

**Cell 2**

```python
df = load_clean_data("../data/Telco-Customer-Churn.csv")   # same cleaning as the EDA notebook
X = df.drop(columns=TARGET)
y = df[TARGET]

num_cols = NUMERIC_COLS                  # tenure, MonthlyCharges, TotalCharges, NumServices
cat_cols = categorical_columns(df)       # the 16 text columns
print(f"{len(num_cols)} numeric:", num_cols)
print(f"{len(cat_cols)} categorical:", cat_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)

print("\nTrain:", X_train.shape, " Test:", X_test.shape)
print(f"Churn rate  train {y_train.mean():.3f} | test {y_test.mean():.3f} | full {y.mean():.3f}")
```

**Cell 3**

```python
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat_cols),
])

# Peek at what the models will actually receive
Xt = preprocessor.fit(X_train).transform(X_train)
feature_names = preprocessor.get_feature_names_out()
print("Shape after preprocessing:", Xt.shape)
print(list(feature_names))
```

**Cell 4**

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = {"f1": "f1", "recall": "recall", "precision": "precision", "roc_auc": "roc_auc", "accuracy": "accuracy"}

def test_metrics(model, name, setting):
    """Score a fitted model ONCE on the held-out test set."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {"model": name, "setting": setting,
            "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba)}

cv_rows, test_rows, fitted = [], [], {}
```

**Cell 5**

```python
baseline_models = {
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=SEED, n_jobs=-1),
}

for name, clf in baseline_models.items():
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
    cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, return_train_score=True)
    row = {"model": name, "setting": "A: baseline"}
    for m in SCORING:
        row[f"cv_{m}"] = cvres[f"test_{m}"].mean()
        row[f"cv_{m}_std"] = cvres[f"test_{m}"].std()
    row["train_f1"] = cvres["train_f1"].mean()
    cv_rows.append(row)

    pipe.fit(X_train, y_train)                       # refit on the whole training set
    fitted[(name, "A: baseline")] = pipe
    test_rows.append(test_metrics(pipe, name, "A: baseline"))

pd.DataFrame(cv_rows)[["model", "train_f1", "cv_f1", "cv_f1_std", "cv_recall", "cv_precision", "cv_roc_auc", "cv_accuracy"]].round(3)
```

**Cell 6**

```python
search_space = {
    "KNN": (
        ImbPipeline([("pre", preprocessor), ("sampler", "passthrough"), ("clf", KNeighborsClassifier())]),
        {"sampler": ["passthrough", SMOTE(random_state=SEED)],
         "clf__n_neighbors": [5, 11, 15, 21, 31, 41, 51],
         "clf__weights": ["uniform", "distance"],
         "clf__p": [1, 2]}),
    "Logistic Regression": (
        Pipeline([("pre", preprocessor), ("clf", LogisticRegression(solver="liblinear", max_iter=2000))]),
        {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
         "clf__penalty": ["l1", "l2"],
         "clf__class_weight": [None, "balanced"]}),
    "Random Forest": (
        Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(random_state=SEED, n_jobs=1))]),
        {"clf__n_estimators": [200, 400],
         "clf__max_depth": [6, 10, 15, None],
         "clf__min_samples_leaf": [1, 5, 10],
         "clf__max_features": ["sqrt", 0.5],
         "clf__class_weight": [None, "balanced"]}),
}

searches = {}
for name, (pipe, grid) in search_space.items():
    n_combos = int(np.prod([len(v) for v in grid.values()]))
    t0 = time.time()
    gs = GridSearchCV(pipe, grid, scoring=SCORING, refit="f1", cv=cv, n_jobs=-1, return_train_score=True)
    gs.fit(X_train, y_train)
    searches[name] = gs
    print(f"{name}: {n_combos} combinations x 5 folds = {n_combos * 5} fits in {time.time() - t0:.0f}s")
    print("   best params:", {k: (type(v).__name__ if hasattr(v, "fit_resample") else v) for k, v in gs.best_params_.items()})
    print(f"   best CV F1: {gs.best_score_:.3f}")

    i = gs.best_index_
    res = gs.cv_results_
    row = {"model": name, "setting": "B: tuned"}
    for m in SCORING:
        row[f"cv_{m}"] = res[f"mean_test_{m}"][i]
        row[f"cv_{m}_std"] = res[f"std_test_{m}"][i]
    row["train_f1"] = res["mean_train_f1"][i]
    cv_rows.append(row)
    fitted[(name, "B: tuned")] = gs.best_estimator_
    test_rows.append(test_metrics(gs.best_estimator_, name, "B: tuned"))
```

**Cell 7**

```python
def same(a, b):
    """Compare two hyperparameter values (handles 1 vs 1.0, None, and estimator objects)."""
    try:
        if a == b:
            return True
    except Exception:
        pass
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)

def curve(gs, param, values):
    """Train/CV F1 when `param` varies and all other params stay at their best values."""
    res = pd.DataFrame(gs.cv_results_)
    mask = np.ones(len(res), dtype=bool)
    for k, v in gs.best_params_.items():
        if k != param:
            mask &= res[f"param_{k}"].map(lambda a: same(a, v)).to_numpy(dtype=bool)
    sub = res[mask]
    rows = [sub[sub[f"param_{param}"].map(lambda a: same(a, v)).to_numpy(dtype=bool)].iloc[0] for v in values]
    out = pd.DataFrame(rows)[["mean_train_f1", "mean_test_f1", "std_test_f1"]]
    out.index = [str(v) for v in values]
    return out

panels = [("KNN", "clf__n_neighbors", "K (number of neighbours)"),
          ("Logistic Regression", "clf__C", "C (inverse regularisation, log scale)"),
          ("Random Forest", "clf__max_depth", "max_depth (None = unlimited)")]
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
for ax, (name, param, label) in zip(axes, panels):
    values = search_space[name][1][param]
    c = curve(searches[name], param, values)
    x = np.arange(len(c))
    ax.plot(x, c["mean_train_f1"], marker="o", label="training F1")
    ax.errorbar(x, c["mean_test_f1"], yerr=c["std_test_f1"], marker="o", capsize=3, label="cross-validation F1")
    ax.set_xticks(x, c.index); ax.set(title=name, xlabel=label, ylabel="F1"); ax.legend()
save("11_tuning_curves")
```

**Cell 8**

```python
cv_table = pd.DataFrame(cv_rows)
cv_table["cv_f1 (mean ± std)"] = cv_table.apply(lambda r: f"{r.cv_f1:.3f} ± {r.cv_f1_std:.3f}", axis=1)
cv_table["train-CV F1 gap"] = cv_table["train_f1"] - cv_table["cv_f1"]
cv_view = cv_table[["model", "setting", "train_f1", "cv_f1 (mean ± std)", "train-CV F1 gap", "cv_recall", "cv_precision", "cv_roc_auc", "cv_accuracy"]]
cv_view.to_csv(RES_DIR / "cv_results.csv", index=False)
print("Cross-validation on the TRAINING set (5 folds)")
cv_view.round(3)
```

**Cell 9**

```python
test_table = pd.DataFrame(test_rows).sort_values(["setting", "model"])
test_table.to_csv(RES_DIR / "test_results.csv", index=False)
print("Final scores on the untouched TEST set (1,409 customers)")
test_table.round(3)
```

**Cell 10**

```python
plot_df = test_table.melt(id_vars=["model", "setting"], value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
                          var_name="metric", value_name="score")
g = sns.catplot(data=plot_df, x="metric", y="score", hue="setting", col="model", kind="bar", height=4, aspect=1.1)
g.set(ylim=(0, 1))
for ax in g.axes.flat:
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", fontsize=8)
g.fig.suptitle("Test-set metrics: baseline (A) vs tuned (B)", y=1.03)
plt.savefig(FIG_DIR / "12_baseline_vs_tuned.png", dpi=150, bbox_inches="tight"); plt.show()
```

**Cell 11**

```python
names = list(search_space)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for r, setting in enumerate(["A: baseline", "B: tuned"]):
    for c, name in enumerate(names):
        model = fitted[(name, setting)]
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["Stay", "Churn"],
                                              cmap="Blues", colorbar=False, ax=axes[r, c])
        axes[r, c].set_title(f"{name} – {setting}")
save("13_confusion_matrices")

for name in names:
    print(f"===== {name} (tuned) =====")
    print(classification_report(y_test, fitted[(name, "B: tuned")].predict(X_test), target_names=["Stay", "Churn"], digits=3))
```

**Cell 12**

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for name in names:
    m = fitted[(name, "B: tuned")]
    RocCurveDisplay.from_estimator(m, X_test, y_test, name=name, ax=axes[0])
    PrecisionRecallDisplay.from_estimator(m, X_test, y_test, name=name, ax=axes[1])
axes[0].plot([0, 1], [0, 1], "k--", label="random (AUC = 0.5)"); axes[0].legend(); axes[0].set_title("ROC curve")
axes[1].axhline(y_test.mean(), ls="--", c="k", label=f"random (= churn rate {y_test.mean():.2f})"); axes[1].legend(); axes[1].set_title("Precision-Recall curve")
save("14_roc_pr_curves")
```

**Cell 13**

```python
lr = fitted[("Logistic Regression", "B: tuned")]
coefs = pd.Series(lr.named_steps["clf"].coef_[0], index=lr.named_steps["pre"].get_feature_names_out())
coefs.index = coefs.index.str.replace(r"^(num|cat)__", "", regex=True)
top_coefs = pd.concat([coefs.nlargest(8), coefs.nsmallest(8)]).sort_values()

rf = fitted[("Random Forest", "B: tuned")]
rf_imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=rf.named_steps["pre"].get_feature_names_out())
rf_imp.index = rf_imp.index.str.replace(r"^(num|cat)__", "", regex=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
top_coefs.plot(kind="barh", ax=axes[0], color=np.where(top_coefs > 0, "#e07b5f", "#5f9ee0"))
axes[0].set(title="Logistic Regression coefficients (red = raises churn odds)", xlabel="coefficient (log-odds)")
rf_imp.nlargest(15).sort_values().plot(kind="barh", ax=axes[1], color="#6bbf8a")
axes[1].set(title="Random Forest feature importance (top 15)", xlabel="mean decrease in impurity")
save("15_feature_importance")
print("Odds ratios (e^coef) for the 5 strongest positive coefficients:")
print(np.exp(coefs.nlargest(5)).round(2))
```

**Cell 14**

```python
perm_rows = {}
for name in names:
    pi = permutation_importance(fitted[(name, "B: tuned")], X_test, y_test, scoring="f1",
                                n_repeats=10, random_state=SEED, n_jobs=-1)
    perm_rows[name] = pd.Series(pi.importances_mean, index=X_test.columns)
perm = pd.DataFrame(perm_rows).sort_values("Random Forest", ascending=False)
perm.to_csv(RES_DIR / "permutation_importance.csv")

perm.head(10).iloc[::-1].plot(kind="barh", figsize=(9, 6))
plt.title("Permutation importance on the test set (drop in F1 when a column is shuffled)")
plt.xlabel("mean F1 decrease")
save("16_permutation_importance")
perm.head(10).round(3)
```

**Cell 15**

```python
best_name = test_table[test_table.setting == "B: tuned"].sort_values("f1", ascending=False).iloc[0]["model"]
best = fitted[(best_name, "B: tuned")]
print("Best tuned model by test F1:", best_name)

res = X_test.copy()
res["actual"] = y_test.values
res["pred"] = best.predict(X_test)
churners = res[res.actual == 1].copy()
churners["outcome"] = np.where(churners.pred == 1, "caught (TP)", "missed (FN)")
summary = churners.groupby("outcome").agg(
    customers=("tenure", "size"), median_tenure=("tenure", "median"), median_monthly=("MonthlyCharges", "median"),
    month_to_month=("Contract", lambda s: (s == "Month-to-month").mean()),
    fibre=("InternetService", lambda s: (s == "Fiber optic").mean()),
    electronic_check=("PaymentMethod", lambda s: (s == "Electronic check").mean()))
summary.round(2)
```

**Cell 16**

```python
fair_rows = []
for name in names:
    pred = fitted[(name, "B: tuned")].predict(X_test)
    for attr in ["gender", "SeniorCitizen"]:
        for group, idx in X_test.groupby(attr).groups.items():
            yt, yp = y_test.loc[idx], pd.Series(pred, index=X_test.index).loc[idx]
            fair_rows.append({"model": name, "attribute": attr, "group": group, "customers": len(idx),
                              "actual churn rate": yt.mean(), "flag rate": yp.mean(),
                              "recall": recall_score(yt, yp), "precision": precision_score(yt, yp)})
fair = pd.DataFrame(fair_rows)
fair.to_csv(RES_DIR / "fairness.csv", index=False)
fair.round(3)
```
