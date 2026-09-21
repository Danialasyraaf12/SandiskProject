"""
Sprint 1 Data Preprocessing Pipeline
SanDisk KPI Intelligence Engine (Danial - AI Engine)

Implements the protocol from Section 6.4.3 of the proposal:
  1. Missing value imputation (median for <10% missing; MICE for 10-50% missing)
  2. Outlier detection & treatment (IQR + Isolation Forest)
  3. Feature normalization (StandardScaler)
  4. Class imbalance handling (SMOTE)

Currently configured against the SECOM fallback dataset (data/secom.csv) as a
stand-in for real SanDisk manufacturing data. Swap DATA_PATH + column names
once the real dataset arrives - the pipeline logic stays the same.
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (required to unlock IterativeImputer)
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from preprocessing.quality import compute_quality_score


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = "data/secom.csv"
LABEL_COL = "Pass/Fail"          # will become e.g. "fg_blocked" for real data
DROP_COLS = ["Time"]              # non-feature columns to set aside
HIGH_MISSING_THRESHOLD = 0.50     # drop columns missing more than this
MICE_THRESHOLD = 0.10             # use MICE instead of median above this
CONTAMINATION = 0.05              # expected outlier fraction for Isolation Forest


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load_data] Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_columns(df: pd.DataFrame, label_col: str, drop_cols: list):
    """Separate features from label and any non-feature columns (e.g. timestamps)."""
    y = df[label_col].copy()
    X = df.drop(columns=[label_col] + drop_cols)
    print(f"[split_columns] Features: {X.shape[1]} | Label: '{label_col}'")
    return X, y


def drop_uninformative_columns(X: pd.DataFrame, high_missing_threshold: float) -> pd.DataFrame:
    """Drop columns that are mostly missing or have zero variance (constant)."""
    missing_pct = X.isnull().mean()
    high_missing_cols = missing_pct[missing_pct > high_missing_threshold].index.tolist()

    constant_cols = X.columns[X.nunique(dropna=True) <= 1].tolist()

    cols_to_drop = list(set(high_missing_cols + constant_cols))
    X_clean = X.drop(columns=cols_to_drop)

    print(f"[drop_uninformative_columns] Dropped {len(high_missing_cols)} columns "
          f">{high_missing_threshold*100:.0f}% missing")
    print(f"[drop_uninformative_columns] Dropped {len(constant_cols)} constant columns")
    print(f"[drop_uninformative_columns] Remaining: {X_clean.shape[1]} columns")
    return X_clean


def impute_missing_values(X: pd.DataFrame, mice_threshold: float) -> pd.DataFrame:
    """
    Median imputation for columns with light missingness.
    MICE (IterativeImputer) for columns with moderate missingness (10-50%),
    per the proposal's preprocessing protocol.
    """
    missing_pct = X.isnull().mean()
    mice_cols = missing_pct[missing_pct > mice_threshold].index.tolist()
    median_cols = missing_pct[(missing_pct > 0) & (missing_pct <= mice_threshold)].index.tolist()

    X_imputed = X.copy()

    if median_cols:
        median_imputer = SimpleImputer(strategy="median")
        X_imputed[median_cols] = median_imputer.fit_transform(X_imputed[median_cols])
        print(f"[impute_missing_values] Median-imputed {len(median_cols)} columns (<= {mice_threshold*100:.0f}% missing)")

    if mice_cols:
        mice_imputer = IterativeImputer(max_iter=10, random_state=42)
        X_imputed[mice_cols] = mice_imputer.fit_transform(X_imputed[mice_cols])
        print(f"[impute_missing_values] MICE-imputed {len(mice_cols)} columns (> {mice_threshold*100:.0f}% missing)")

    remaining_na = X_imputed.isnull().sum().sum()
    print(f"[impute_missing_values] Remaining missing values: {remaining_na}")
    return X_imputed


def treat_outliers_iqr(X: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    Univariate outlier treatment via IQR capping (winsorization).
    Values beyond [Q1 - factor*IQR, Q3 + factor*IQR] are clipped to the bound
    rather than dropped, to preserve row count for downstream modelling.
    """
    X_capped = X.copy()
    capped_counts = {}

    for col in X_capped.columns:
        q1 = X_capped[col].quantile(0.25)
        q3 = X_capped[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        n_capped = ((X_capped[col] < lower) | (X_capped[col] > upper)).sum()
        if n_capped > 0:
            capped_counts[col] = n_capped
            X_capped[col] = X_capped[col].clip(lower, upper)

    total_capped = sum(capped_counts.values())
    print(f"[treat_outliers_iqr] Capped {total_capped} outlier values across {len(capped_counts)} columns")
    return X_capped


def flag_multivariate_outliers(X: pd.DataFrame, contamination: float) -> pd.Series:
    """
    Multivariate anomaly detection via Isolation Forest.
    Returns a boolean Series (True = flagged as anomalous row).
    Rows are flagged, not dropped - final decision (drop vs keep vs review)
    is a modelling choice made downstream.
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(X)  # -1 = anomaly, 1 = normal
    is_outlier = pd.Series(predictions == -1, index=X.index)

    print(f"[flag_multivariate_outliers] Flagged {is_outlier.sum()} rows "
          f"({is_outlier.mean()*100:.1f}%) as multivariate outliers")
    return is_outlier


def normalize_features(X: pd.DataFrame) -> pd.DataFrame:
    """StandardScaler normalization (zero mean, unit variance) for classification inputs."""
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    print(f"[normalize_features] Scaled {X_scaled.shape[1]} columns (mean=0, std=1)")
    return X_scaled


def balance_classes(X: pd.DataFrame, y: pd.Series):
    """SMOTE oversampling of the minority class to address class imbalance."""
    print(f"[balance_classes] Before SMOTE: {dict(y.value_counts())}")
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)
    print(f"[balance_classes] After SMOTE:  {dict(pd.Series(y_bal).value_counts())}")
    return X_bal, y_bal


def run_pipeline():
    print("=" * 70)
    print("SPRINT 1 PREPROCESSING PIPELINE")
    print("=" * 70)

    df = load_data(DATA_PATH)
    X, y = split_columns(df, LABEL_COL, DROP_COLS)

    quality_score = compute_quality_score(X)  # scored on raw input
    print(f"[compute_quality_score] {quality_score}")

    X = drop_uninformative_columns(X, HIGH_MISSING_THRESHOLD)
    X = impute_missing_values(X, MICE_THRESHOLD)
    X = treat_outliers_iqr(X)

    outlier_flags = flag_multivariate_outliers(X, CONTAMINATION)
    # Rows are flagged only, not dropped, at this stage - kept for engineer/mentor review

    X_scaled = normalize_features(X)
    X_balanced, y_balanced = balance_classes(X_scaled, y)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Final dataset shape: {X_balanced.shape}")
    print(f"Quality score: {quality_score['composite_score']}  "
          f"(Sprint 1 gate requires >= 0.80)")
    print(f"Issues detected: {len(quality_score['issues'])}")

    return X_balanced, y_balanced, outlier_flags, quality_score


if __name__ == "__main__":
    X_final, y_final, outliers, score = run_pipeline()

    # Save outputs for the next stage (model training)
    X_final.to_csv("data/processed_features.csv", index=False)
    pd.Series(y_final).to_csv("data/processed_labels.csv", index=False)
    print("\nSaved: data/processed_features.csv, data/processed_labels.csv")