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
