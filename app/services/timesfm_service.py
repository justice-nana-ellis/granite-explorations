"""TimesFM (Google Research) time-series forecasting service.

Install from GitHub source (required for v2.5 model):
    pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm.git"

The model (~200 MB) is downloaded from HuggingFace on the first request.
All subsequent calls reuse the in-process singleton — no reload needed.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_model: Any | None = None
_model_error: str | None = None
_load_lock = threading.Lock()

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ── Model lifecycle ────────────────────────────────────────────────────────────

def _do_load() -> None:
    """Blocking load — runs inside asyncio.to_thread."""
    global _model, _model_error
    with _load_lock:
        if _model is not None:
            return
        try:
            import timesfm

            logger.info("Loading TimesFM model %s …", MODEL_ID)
            m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
            # max_context=512 (16 patches × 32), max_horizon=256 (2 output patches × 128)
            m.compile(timesfm.ForecastConfig(max_context=512, max_horizon=256))
            _model = m
            logger.info("TimesFM model ready.")
        except Exception as exc:
            _model_error = str(exc)
            logger.error("TimesFM model load failed: %s", exc)
            raise


async def ensure_loaded() -> None:
    """Ensure the model is loaded, downloading if necessary."""
    if _model is None:
        await asyncio.to_thread(_do_load)


def get_status() -> dict:
    return {
        "loaded": _model is not None,
        "model_id": MODEL_ID,
        "error": _model_error,
    }


# ── Core forecast ──────────────────────────────────────────────────────────────

def _run_forecast(inputs: list[np.ndarray], horizon: int) -> tuple:
    """Blocking forecast call — runs inside asyncio.to_thread."""
    point_fc, quantile_fc = _model.forecast(horizon=horizon, inputs=inputs)
    return point_fc, quantile_fc


async def forecast_dataframe(
    df: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
    horizon: int = 12,
) -> dict:
    """Forecast each value column using TimesFM.

    Returns a dict keyed by column name, each containing chart-ready data
    with historical series, point forecast, and 10/50/90 quantile bands.
    """
    await ensure_loaded()

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid rows after parsing date column '{date_col}'.")

    date_labels = df[date_col].dt.strftime("%Y-%m-%d").tolist()

    inputs: list[np.ndarray] = []
    valid_cols: list[str] = []
    for col in value_cols:
        if col not in df.columns:
            logger.warning("Column '%s' not found in data — skipping.", col)
            continue
        series = pd.to_numeric(df[col], errors="coerce").ffill().dropna()
        if len(series) < 4:
            logger.warning("Column '%s' has < 4 valid rows — skipping.", col)
            continue
        inputs.append(series.values.astype(np.float32))
        valid_cols.append(col)

    if not inputs:
        raise ValueError(
            f"No valid numeric series found in columns: {value_cols}. "
            "Check that the columns exist and contain numeric data."
        )

    point_fc, quantile_fc = await asyncio.to_thread(_run_forecast, inputs, horizon)
    # point_fc:    (n, horizon)
    # quantile_fc: (n, horizon, len(QUANTILE_LEVELS))

    # Future date labels
    last_date = df[date_col].iloc[-1]
    try:
        freq = pd.infer_freq(df[date_col]) or "D"
    except Exception:
        freq = "D"
    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
    future_labels = future_dates.strftime("%Y-%m-%d").tolist()
    all_labels = date_labels + future_labels

    n_hist = len(date_labels)
    null_hist = [None] * n_hist
    null_fut  = [None] * horizon

    results: dict[str, dict] = {}
    for i, col in enumerate(valid_cols):
        historical   = df[col].tolist()
        point        = [round(float(v), 4) for v in point_fc[i]]
        q10          = [round(float(v), 4) for v in quantile_fc[i, :, 0]]
        q50          = [round(float(v), 4) for v in quantile_fc[i, :, 4]]
        q90          = [round(float(v), 4) for v in quantile_fc[i, :, -1]]

        results[col] = {
            "series":              col,
            "historical_rows":     n_hist,
            "forecast_horizon":    horizon,
            "point_forecast":      point,
            "quantile_10":         q10,
            "quantile_50":         q50,
            "quantile_90":         q90,
            "chart_data": {
                "type":   "line",
                "labels": all_labels,
                "series": [
                    {
                        "id":          "historical",
                        "label":       f"{col} — Historical",
                        "data":        historical + null_fut,
                        "borderColor": "#2563eb",
                        "dashed":      False,
                    },
                    {
                        "id":          "point",
                        "label":       f"{col} — Point Forecast",
                        "data":        null_hist + point,
                        "borderColor": "#16a34a",
                        "dashed":      True,
                    },
                    {
                        "id":          "q90",
                        "label":       "90th Percentile",
                        "data":        null_hist + q90,
                        "borderColor": "#84cc16",
                        "dashed":      True,
                    },
                    {
                        "id":          "q50",
                        "label":       "Median (50th)",
                        "data":        null_hist + q50,
                        "borderColor": "#f59e0b",
                        "dashed":      True,
                    },
                    {
                        "id":          "q10",
                        "label":       "10th Percentile",
                        "data":        null_hist + q10,
                        "borderColor": "#dc2626",
                        "dashed":      True,
                    },
                ],
            },
        }

    return results
