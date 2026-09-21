"""
SanDisk KPI Intelligence Engine - API
Danial (AI Engine)

FastAPI backend exposing AI engine capabilities to the Web KPI Dashboard
(Zulfaqa) and Mobile KPI Monitoring app (Samini), per api-contract.md.

Run:
    uvicorn api.main:app --reload

Docs (auto-generated):
    http://localhost:8000/docs
"""

import os
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import pandas as pd

from preprocessing.quality import compute_quality_score

app = FastAPI(title="SanDisk KPI Intelligence Engine")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
UPLOAD_DIR = "data/uploads"
FALLBACK_DATA_PATH = "data/secom.csv"   # used when no upload_id is given
FALLBACK_LABEL_COL = "Pass/Fail"
FALLBACK_NON_FEATURE_COLS = ["Time"]

os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory registry mapping upload_id -> file metadata.
# Fine for Sprint 1/local dev. Replace with the shared DB (per api-contract.md
# "Shared DB schema ownership" open question) once that's agreed with the team -
# this in-memory store resets every time the server restarts.
UPLOAD_REGISTRY: dict[str, dict] = {}


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 1a. Upload
# ---------------------------------------------------------------------------
@app.post("/api/data/upload")
async def upload_data(file: UploadFile = File(...), source: str = Form("manual_upload")):
    """
    Accepts a CSV or XLSX file, saves it, and registers it under a new
    upload_id. Matches api-contract.md section 1.1.
    """
    filename = file.filename or ""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in ("csv", "xlsx", "xls"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a .csv or .xlsx file."
        )

    upload_id = f"UPL-{datetime.now(timezone.utc):%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"
    saved_path = os.path.join(UPLOAD_DIR, f"{upload_id}.{ext}")

    contents = await file.read()
    with open(saved_path, "wb") as f:
        f.write(contents)

    # Peek at the file to report row/column counts back to the caller
    try:
        if ext == "csv":
            df_preview = pd.read_csv(saved_path)
        else:
            df_preview = pd.read_excel(saved_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse uploaded file: {e}")

    UPLOAD_REGISTRY[upload_id] = {
        "upload_id": upload_id,
        "path": saved_path,
        "source": source,
        "original_filename": filename,
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "rows": df_preview.shape[0],
        "columns": list(df_preview.columns),
    }

    return {
        "upload_id": upload_id,
        "status": "received",
        "rows_detected": df_preview.shape[0],
        "columns_detected": list(df_preview.columns)[:10],  # cap for response size
        "next_step": "preprocessing_queued"
    }


# ---------------------------------------------------------------------------
# 1b. Quality score
# ---------------------------------------------------------------------------
@app.get("/api/data/quality-score")
def get_quality_score(upload_id: str | None = None):
    """
    Returns the composite data quality score (completeness, consistency,
    accuracy) plus a ranked issue list, per api-contract.md section 1.2.

    - If upload_id is provided, scores that specific uploaded dataset.
    - If omitted, falls back to the SECOM dev dataset (useful for quick
      testing without needing to upload a file first).
    """
    if upload_id:
        record = UPLOAD_REGISTRY.get(upload_id)
        if not record:
            raise HTTPException(
                status_code=404,
                detail=f"No upload found with upload_id '{upload_id}'. "
                       f"Did the server restart since you uploaded it?"
            )
        path = record["path"]
        df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
        # Best-effort: drop an obvious label/id-like column if present, else score all columns
        drop_cols = [c for c in df.columns if c.lower() in
                     ("pass/fail", "fg_blocked", "label", "time", "timestamp", "fg_id")]
        X = df.drop(columns=drop_cols) if drop_cols else df
        freshness_hours = round(
            (datetime.now(timezone.utc) -
             datetime.fromisoformat(record["uploaded_at"].replace("Z", "+00:00"))
             ).total_seconds() / 3600, 1
        )
    else:
        try:
            df = pd.read_csv(FALLBACK_DATA_PATH)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"No upload_id given and fallback dataset not found at {FALLBACK_DATA_PATH}."
            )
        drop_cols = [c for c in FALLBACK_NON_FEATURE_COLS + [FALLBACK_LABEL_COL] if c in df.columns]
        X = df.drop(columns=drop_cols)
        upload_id = "DEV-SECOM-FALLBACK"
        freshness_hours = None

    score = compute_quality_score(X)

    return {
        "upload_id": upload_id,
        "composite_score": score["composite_score"],
        "dimensions": score["dimensions"],
        "issues": score["issues"],
        "data_freshness_hours": freshness_hours,
        "scored_at": score["scored_at"],
    }