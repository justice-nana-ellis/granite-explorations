"""Full fund management forecasting endpoint.

POST /forecast/full
  Fetches all data stored under one or more rag_ids, computes analytics,
  optionally runs TimesFM, then calls Claude for a complete structured report.
  Returns a single JSON response with everything — all 6 use cases + chart data.
"""
from __future__ import annotations

import asyncio
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.fund_forecast_service import (
    ColumnMap,
    build_analytics,
    build_chart_data,
    build_claude_context,
    cache_analytics,
    get_cached_analytics,
    inject_charts_into_forecast,
    run_timesfm_forecasts,
)
from app.utils.prompt_library import RAG_FORECASTING_SYSTEM_PROMPT

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/forecast", tags=["forecast"])


# ── Request model ──────────────────────────────────────────────────────────────

class FullForecastRequest(BaseModel):
    rag_ids: list[str]

    forecast_horizon: str = "6m"        # "3m", "6m", "1y", "2y"
    msg: str = ""                        # additional user focus / question

    # Column overrides — defaults match GOLD_MANDATES.xlsx
    date_col:           str = "nav_date"
    aum_col:            str = "market_value_eur"
    subscription_col:   str = "subscription_gross"
    redemption_col:     str = "redemption_gross"
    net_flow_col:       str = "net_sales_gross"
    revenue_col:        str = "revenues"
    portfolio_col:      str = "portfolio_ik"
    portfolio_name_col: str = "portfolio_name"
    client_col:         str = "crm_account_id"
    asset_class_col:    str = "asset_class"
    region_col:         str = "client_region"
    mgmt_category_col:  str = "management_category"
    product_family_col: str = "product_family"
    currency_col:       str = "currency"
    alert_col:          str = "alert"
    alert_desc_col:     str = "alert_description"

    max_rows:   int = 16000
    max_tokens: int = 16384


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_horizon(h: str) -> int:
    h = h.lower().strip()
    if h.endswith("y"):
        return int(h[:-1]) * 12
    if h.endswith("m"):
        return int(h[:-1])
    return int(h)


# ── Forecast tool schema ───────────────────────────────────────────────────────

FUND_FORECAST_TOOL: dict = {
    "name": "submit_fund_forecast",
    "description": (
        "Submit the complete quantitative fund management forecasting report. "
        "Every field must be populated. Cover all 6 use cases without exception."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "executive_summary": {"type": "string"},
            "data_period": {
                "type": "object",
                "properties": {
                    "earliest":          {"type": "string"},
                    "latest":            {"type": "string"},
                    "months_of_history": {"type": "integer"},
                },
                "required": ["earliest", "latest", "months_of_history"],
            },
            "current_snapshot": {"type": "object", "additionalProperties": True},
            "forecasts": {
                "type": "object",
                "properties": {
                    "aum_trend":           {"type": "object", "additionalProperties": True},
                    "net_flows":           {"type": "object", "additionalProperties": True},
                    "revenue":             {"type": "object", "additionalProperties": True},
                    "churn_risk":          {"type": "object", "additionalProperties": True},
                    "portfolio_mix_drift": {"type": "object", "additionalProperties": True},
                    "data_quality":        {"type": "object", "additionalProperties": True},
                },
                "required": [
                    "aum_trend", "net_flows", "revenue",
                    "churn_risk", "portfolio_mix_drift", "data_quality",
                ],
            },
            "top_10_performers": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
            },
            "top_10_at_risk": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
            },
            "strategic_recommendations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Minimum 6 specific, actionable recommendations referencing exact figures.",
            },
            "methodology":   {"type": "string"},
            "data_warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "executive_summary", "data_period", "current_snapshot", "forecasts",
            "top_10_performers", "top_10_at_risk",
            "strategic_recommendations", "methodology", "data_warnings",
        ],
    },
}


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post("/full")
async def full_forecast(body: FullForecastRequest):
    """
    Combined Claude + TimesFM full fund management forecast.

    Returns a single complete JSON object covering all 6 use cases:
      1. AUM Forecasting
      2. Net Flow Forecasting (Subscriptions / Redemptions)
      3. Revenue Forecasting
      4. Client Churn Risk
      5. Portfolio Mix Drift
      6. Data Quality / Operational Forecast

    Every section includes chart-ready data.
    """
    from app.services import rag_service
    from app.services.claude_service import claude_service
    from app.config import settings

    if not body.rag_ids:
        raise HTTPException(status_code=400, detail="At least one rag_id is required.")

    horizon = _parse_horizon(body.forecast_horizon)
    col = ColumnMap(
        date=body.date_col,
        aum=body.aum_col,
        subscription=body.subscription_col,
        redemption=body.redemption_col,
        net_flow=body.net_flow_col,
        revenue=body.revenue_col,
        portfolio=body.portfolio_col,
        portfolio_name=body.portfolio_name_col,
        client=body.client_col,
        asset_class=body.asset_class_col,
        region=body.region_col,
        mgmt_category=body.mgmt_category_col,
        product_family=body.product_family_col,
        currency=body.currency_col,
        alert=body.alert_col,
        alert_desc=body.alert_desc_col,
    )

    # ── Stages 1-5: fetch, compute analytics, build context + charts ─────────
    # Check the analytics cache first — pre-warmed after ingest, so most
    # requests skip straight to the Claude call.
    cached = get_cached_analytics(body.rag_ids, horizon, col)

    if cached:
        source_rows     = cached["source_rows"]
        analytics       = cached["analytics"]
        context         = cached["context"]
        charts          = cached["charts"]
        timesfm_results = cached["timesfm_results"]

        # Cache was pre-warmed without TimesFM but USE_TIMESFM is now on — run it
        if timesfm_results is None and settings.use_timesfm:
            timesfm_results = await run_timesfm_forecasts(analytics, horizon)
            if timesfm_results:
                charts = await asyncio.to_thread(build_chart_data, analytics, timesfm_results, horizon)
                cache_analytics(body.rag_ids, horizon, col, {
                    **cached,
                    "charts":          charts,
                    "timesfm_results": timesfm_results,
                })

        if body.msg.strip():
            context += f"\n\nADDITIONAL USER FOCUS: {body.msg.strip()}"
    else:
        # ── Stage 1: fetch all rag_ids in parallel ────────────────────────
        per_limit = max(1, body.max_rows // len(body.rag_ids))

        async def _fetch(rid: str) -> list[dict]:
            try:
                rows = await rag_service.fetch_all_metadata(rid, limit=per_limit)
                for r in rows:
                    r["_rag_id"] = rid
                logger.info("Fetched %d rows for rag_id=%s", len(rows), rid)
                return rows
            except Exception as exc:
                logger.warning("Full forecast: failed to fetch rag_id=%s — %s", rid, exc)
                return []

        fetched  = await asyncio.gather(*[_fetch(rid) for rid in body.rag_ids])
        all_rows = [r for batch in fetched for r in batch]

        if not all_rows:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for rag_ids {body.rag_ids}. Ingest your files first via POST /rag/ingest.",
            )

        source_rows = len(all_rows)
        df = pd.DataFrame(all_rows)
        logger.info("Full forecast: %d total rows, %d columns", source_rows, len(df.columns))

        # ── Stage 2: compute analytics ────────────────────────────────────
        try:
            analytics = await asyncio.to_thread(build_analytics, df, col)
        except Exception as exc:
            logger.exception("Full forecast: analytics failed")
            raise HTTPException(status_code=500, detail=f"Analytics computation failed: {exc}")

        # ── Stages 3+4 in parallel: TimesFM and Claude context ───────────
        async def _maybe_timesfm() -> dict | None:
            if not settings.use_timesfm:
                return None
            result = await run_timesfm_forecasts(analytics, horizon)
            if result:
                logger.info("TimesFM: forecasted %d series", len(result))
            return result

        async def _build_context() -> str:
            ctx = await asyncio.to_thread(build_claude_context, analytics, col, horizon)
            if body.msg.strip():
                ctx += f"\n\nADDITIONAL USER FOCUS: {body.msg.strip()}"
            return ctx

        timesfm_results, context = await asyncio.gather(_maybe_timesfm(), _build_context())

        # ── Stage 5: build chart data ─────────────────────────────────────
        charts = await asyncio.to_thread(build_chart_data, analytics, timesfm_results, horizon)

        # Store for next request (without the user-specific msg appended)
        base_context = await asyncio.to_thread(build_claude_context, analytics, col, horizon)
        cache_analytics(body.rag_ids, horizon, col, {
            "source_rows":      source_rows,
            "analytics":        analytics,
            "context":          base_context,
            "charts":           charts,
            "timesfm_results":  timesfm_results,
        })

    # ── Stage 6: Claude forecast ──────────────────────────────────────────────
    messages = [{
        "role": "user",
        "content": (
            f"[FUND MANAGEMENT DATA — FULL HISTORY]\n{context}\n\n"
            f"[DIRECTIVE] Produce a complete, expert-grade fund management forecasting "
            f"report covering ALL 6 sections without exception: AUM trend, net flows, "
            f"revenue, churn risk, portfolio mix drift, and data quality. "
            f"Use every data point provided. Leave no section empty. "
            f"The forecast horizon is {body.forecast_horizon} from {analytics['date_last']}."
        ),
    }]

    try:
        claude_result = await claude_service.complete_with_tool(
            messages=messages,
            system=RAG_FORECASTING_SYSTEM_PROMPT,
            tool=FUND_FORECAST_TOOL,
            model=settings.claude_forecast_model,
            max_tokens=body.max_tokens or settings.claude_forecast_max_tokens,
        )
    except Exception as exc:
        logger.exception("Full forecast: Claude call failed rag_ids=%s", body.rag_ids)
        raise HTTPException(status_code=500, detail=str(exc))

    # ── Stage 7: inject charts and return ─────────────────────────────────────
    full_report = inject_charts_into_forecast(claude_result, charts)

    return {
        "rag_ids":          body.rag_ids,
        "forecast_horizon": body.forecast_horizon,
        "source_rows":      source_rows,
        "timesfm_used":     timesfm_results is not None,
        "analytics": {
            "date_first":   analytics["date_first"],
            "date_last":    analytics["date_last"],
            "n_periods":    analytics["n_periods"],
            "n_portfolios": analytics["n_portfolios"],
            "n_clients":    analytics["n_clients"],
            "n_regions":    analytics["n_regions"],
            "aum_latest":   analytics["aum_latest"],
            "aum_earliest": analytics["aum_earliest"],
            "aum_growth_pct": analytics["aum_growth_pct"],
        },
        "forecast": full_report,
    }
