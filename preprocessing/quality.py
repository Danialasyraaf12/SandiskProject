"""
Data Quality Scoring Module
SanDisk KPI Intelligence Engine (Danial - AI Engine)

Standalone, reusable quality scoring logic - imported by both:
  - preprocessing/pipeline.py   (Sprint 1 batch processing)
  - api/main.py                  (live /api/data/quality-score endpoint)

Mirrors the Data Quality Scoring Model described in the proposal (Section 6.5.2):
  completeness, consistency, accuracy -> composite score + issue list.

This is a heuristic version for Sprint 1/2. Replace `accuracy` placeholder
with real ground-truth validation rules once the SanDisk mentor defines them.
"""

import pandas as pd
from datetime import datetime, timezone


HIGH_MISSING_THRESHOLD = 0.50   # flag as "high" severity issue above this
MEDIUM_MISSING_THRESHOLD = 0.10  # flag as "medium" severity issue above this


def compute_quality_score(X: pd.DataFrame) -> dict:
    """
    Compute completeness, consistency, accuracy sub-scores and a composite
    quality score for a feature dataframe (label column should be excluded
    before calling this).

    Returns a dict matching the api-contract.md response shape for
    GET /api/data/quality-score.
    """
    completeness = 1 - X.isnull().mean().mean()
    consistency = 1 - (X.nunique(dropna=True) <= 1).mean()  # penalise constant cols
    accuracy = 1.0  # placeholder until ground-truth validation rules exist (Sprint 2)

    composite = round((completeness + consistency + accuracy) / 3, 3)

    issues = _detect_issues(X)

    result = {
        "composite_score": composite,
        "dimensions": {
            "completeness": round(completeness, 3),
            "consistency": round(consistency, 3),
            "accuracy": round(accuracy, 3),
        },
        "issues": issues,
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    return result


def _detect_issues(X: pd.DataFrame) -> list:
    """
    Build a per-column issue list (missing values, constant columns),
    matching the "issues" array shape in api-contract.md.
    Capped at 20 entries so the API response doesn't balloon on wide datasets.
    """
    issues = []
    missing_pct = X.isnull().mean()

    for col, pct in missing_pct.items():
        if pct > HIGH_MISSING_THRESHOLD:
            issues.append({
                "field": col,
                "issue": f"{pct*100:.1f}% missing values",
                "severity": "high"
            })
        elif pct > MEDIUM_MISSING_THRESHOLD:
            issues.append({
                "field": col,
                "issue": f"{pct*100:.1f}% missing values",
                "severity": "medium"
            })

    constant_cols = X.columns[X.nunique(dropna=True) <= 1]
    for col in constant_cols:
        issues.append({
            "field": col,
            "issue": "constant value (zero variance)",
            "severity": "low"
        })

    # Sort worst first, cap the list length for API response size
    severity_order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: severity_order[i["severity"]])
    return issues[:20]


if __name__ == "__main__":
    # quick manual check
    df = pd.read_csv("data/secom.csv")
    X = df.drop(columns=["Time", "Pass/Fail"])
    score = compute_quality_score(X)
    import json
    print(json.dumps(score, indent=2, default=str))
