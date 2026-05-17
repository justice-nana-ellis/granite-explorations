"""TimesFM endpoints — Google Research time-series foundation model.

These endpoints are independent of all existing RAG / chat / upload routes.

Endpoints
---------
GET  /timesfm/status
    Model load status and model ID.

POST /timesfm/forecast
    Upload a CSV or Excel file.  Specify which column is the date and which
    columns to forecast.  Returns point + quantile forecasts with chart-ready data.

POST /timesfm/rag-forecast
    Use data already stored in a RAG (by rag_id) to run a TimesFM forecast
    without re-uploading anything.
"""
from __future__ import annotations

import asyncio
import io
import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.services import timesfm_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/timesfm", tags=["timesfm"])


# ── Status ─────────────────────────────────────────────────────────────────────

@router.get("/status")
async def timesfm_status():
    """Check whether the TimesFM model is loaded and ready."""
    return timesfm_service.get_status()


# ── File upload forecast ───────────────────────────────────────────────────────

@router.post("/forecast")
async def timesfm_forecast(
    file: UploadFile = File(...),
    date_col: str = Form(..., description="Name of the date/time column"),
    value_cols: str = Form(
        ...,
        description="Comma-separated list of numeric columns to forecast",
    ),
    horizon: int = Form(12, description="Number of future periods to forecast (max 256)"),
):
    """
    Upload a CSV or Excel file and get TimesFM point + quantile forecasts.

    **Parameters (multipart form)**
    - `file`       — CSV or Excel file
    - `date_col`   — name of the date column (e.g. `"Period"`, `"Date"`)
    - `value_cols` — comma-separated column names to forecast (e.g. `"AUM,NetFlow"`)
    - `horizon`    — how many future periods to project (default 12, max 256)

    **Response**
    ```json
    {
      "model": "google/timesfm-2.5-200m-pytorch",
      "date_col": "Period",
      "value_cols": ["AUM"],
      "horizon": 12,
      "forecasts": {
        "AUM": {
          "series": "AUM",
          "historical_rows": 40,
          "forecast_horizon": 12,
          "point_forecast": [...],
          "quantile_10": [...],
          "quantile_50": [...],
          "quantile_90": [...],
          "chart_data": { "type": "line", "labels": [...], "series": [...] }
        }
      }
    }
    ```
    """
    horizon = min(max(horizon, 1), 256)

    fname = (file.filename or "").lower()
    if not any(fname.endswith(ext) for ext in (".csv", ".xlsx", ".xls")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Upload a .csv, .xlsx, or .xls file.",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        if fname.endswith(".csv"):
            df = await asyncio.to_thread(
                pd.read_csv, io.StringIO(raw.decode("utf-8")), low_memory=False
            )
        else:
            df = await asyncio.to_thread(
                pd.read_excel, io.BytesIO(raw), engine="openpyxl"
            )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {exc}")

    if date_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Date column '{date_col}' not found. Available columns: {list(df.columns)}",
        )

    cols = [c.strip() for c in value_cols.split(",") if c.strip()]
    if not cols:
        raise HTTPException(status_code=400, detail="value_cols must not be empty.")

    try:
        forecasts = await timesfm_service.forecast_dataframe(
            df=df, date_col=date_col, value_cols=cols, horizon=horizon
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"TimesFM is not installed: {exc}. "
                "Run: pip install timesfm[torch]"
            ),
        )
    except Exception as exc:
        logger.exception("TimesFM forecast failed for file %s", file.filename)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "model":      timesfm_service.MODEL_ID,
        "date_col":   date_col,
        "value_cols": list(forecasts.keys()),
        "horizon":    horizon,
        "forecasts":  forecasts,
    }


# ── RAG-based forecast ─────────────────────────────────────────────────────────

class RagForecastRequest(BaseModel):
    rag_id: str
    date_col: str
    value_cols: list[str]
    horizon: int = 12
    max_rows: int = 16000


@router.post("/rag-forecast")
async def timesfm_rag_forecast(body: RagForecastRequest):
    """
    Run a TimesFM forecast on data already ingested into a RAG.

    No file upload needed — the data is retrieved directly from the vector
    store by `rag_id`, reconstructed from the stored metadata, and fed
    straight into TimesFM.

    **Body**
    ```json
    {
      "rag_id":    "2f4217f8-e831-4aea-a307-5885db2c55e4",
      "date_col":  "Period",
      "value_cols": ["AUM", "NetFlow"],
      "horizon":   12,
      "max_rows":  16000
    }
    ```
    """
    body.horizon  = min(max(body.horizon, 1), 256)
    body.max_rows = min(max(body.max_rows, 1), 16000)

    from app.services import rag_service

    try:
        rows = await rag_service.fetch_all_metadata(body.rag_id, limit=body.max_rows)
    except Exception as exc:
        logger.exception("TimesFM: failed to fetch metadata for rag_id=%s", body.rag_id)
        raise HTTPException(status_code=500, detail=str(exc))

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for rag_id '{body.rag_id}'. Ingest files first.",
        )

    df = pd.DataFrame(rows)

    if body.date_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Date column '{body.date_col}' not found. "
                f"Available columns: {list(df.columns)}"
            ),
        )

    try:
        forecasts = await timesfm_service.forecast_dataframe(
            df=df,
            date_col=body.date_col,
            value_cols=body.value_cols,
            horizon=body.horizon,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"TimesFM is not installed: {exc}. Run: pip install timesfm[torch]",
        )
    except Exception as exc:
        logger.exception("TimesFM rag-forecast failed for rag_id=%s", body.rag_id)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "model":      timesfm_service.MODEL_ID,
        "rag_id":     body.rag_id,
        "source_rows": len(rows),
        "date_col":   body.date_col,
        "value_cols": list(forecasts.keys()),
        "horizon":    body.horizon,
        "forecasts":  forecasts,
    }
