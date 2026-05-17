"""Fund management analytics service.

Aggregates raw row-level fund data into time-series analytics that feed both
Claude (for written interpretation) and TimesFM (for statistical forecasts).
Covers all 6 use cases:
  1. AUM Forecasting
  2. Net Flow Forecasting
  3. Revenue Forecasting
  4. Client Churn Risk
  5. Portfolio Mix Drift
  6. Data Quality / Operational Forecast
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Analytics cache ────────────────────────────────────────────────────────────
# Stores (analytics, context, charts, source_rows) keyed by
# (sorted rag_ids, horizon, col_hash).  Pre-warmed after ingest so the
# endpoint only has to do the Claude call.

_analytics_cache: dict[tuple, tuple[dict, float]] = {}
_ANALYTICS_CACHE_TTL = 300  # seconds — 5 minutes


def _col_hash(col) -> str:
    return hashlib.md5(
        json.dumps(dataclasses.asdict(col), sort_keys=True).encode()
    ).hexdigest()[:8]


def _analytics_key(rag_ids: list[str], horizon: int, col) -> tuple:
    return (tuple(sorted(rag_ids)), horizon, _col_hash(col))


def get_cached_analytics(rag_ids: list[str], horizon: int, col) -> dict | None:
    entry = _analytics_cache.get(_analytics_key(rag_ids, horizon, col))
    if entry:
        data, ts = entry
        if time.time() - ts < _ANALYTICS_CACHE_TTL:
            logger.info("Analytics cache hit rag_ids=%s horizon=%d", rag_ids, horizon)
            return data
    return None


def cache_analytics(rag_ids: list[str], horizon: int, col, data: dict) -> None:
    _analytics_cache[_analytics_key(rag_ids, horizon, col)] = (data, time.time())
    logger.info("Analytics cached rag_ids=%s horizon=%d", rag_ids, horizon)


def invalidate_analytics_cache(rag_id: str) -> None:
    keys = [k for k in list(_analytics_cache) if rag_id in k[0]]
    for k in keys:
        del _analytics_cache[k]
    if keys:
        logger.info("Analytics cache invalidated rag_id=%s (%d entries)", rag_id, len(keys))


# ── Column map ─────────────────────────────────────────────────────────────────

@dataclass
class ColumnMap:
    date:           str = "nav_date"
    aum:            str = "market_value_eur"
    subscription:   str = "subscription_gross"
    redemption:     str = "redemption_gross"
    net_flow:       str = "net_sales_gross"
    revenue:        str = "revenues"
    portfolio:      str = "portfolio_ik"
    portfolio_name: str = "portfolio_name"
    client:         str = "crm_account_id"
    asset_class:    str = "asset_class"
    region:         str = "client_region"
    mgmt_category:  str = "management_category"
    product_family: str = "product_family"
    currency:       str = "currency"
    alert:          str = "alert"
    alert_desc:     str = "alert_description"


def _col(df: pd.DataFrame, name: str) -> str | None:
    return name if name in df.columns else None


def _fmt(v: float, scale: str = "B") -> str:
    if pd.isna(v):
        return "N/A"
    if scale == "B":
        return f"€{v / 1e9:.1f}B"
    if scale == "M":
        return f"€{v / 1e6:.1f}M"
    return f"{v:,.0f}"


def _pct(v: float) -> str:
    return f"{v:+.1f}%" if not pd.isna(v) else "N/A"


# ── Core analytics ─────────────────────────────────────────────────────────────

def build_analytics(df: pd.DataFrame, col: ColumnMap) -> dict:
    """Compute every metric needed for context and charts from row-level data."""
    df = df.copy()

    # ── Date handling
    df[col.date] = pd.to_datetime(df[col.date], errors="coerce")
    df = df.dropna(subset=[col.date]).sort_values(col.date)
    df["_period"] = df[col.date].dt.to_period("M")
    periods = sorted(df["_period"].unique())
    labels  = [str(p) for p in periods]

    def agg_series(group_col: str | None, value_col: str | None) -> pd.Series | None:
        if not value_col or value_col not in df.columns:
            return None
        if group_col:
            return df.groupby("_period")[value_col].sum().reindex(periods)
        return df.groupby("_period")[value_col].sum().reindex(periods)

    # ── AUM time series
    aum_ts = agg_series(None, _col(df, col.aum))

    # ── Flow time series
    subs_ts = agg_series(None, _col(df, col.subscription))
    reds_ts  = agg_series(None, _col(df, col.redemption))
    nf_ts    = agg_series(None, _col(df, col.net_flow))

    # ── Revenue time series
    rev_ts = agg_series(None, _col(df, col.revenue))

    # ── AUM by asset class
    ac_pivot: dict[str, list] = {}
    if _col(df, col.asset_class) and _col(df, col.aum):
        pivot = (
            df.groupby(["_period", col.asset_class])[col.aum]
            .sum()
            .unstack(col.asset_class)
            .reindex(periods)
        )
        for ac in pivot.columns:
            ac_pivot[str(ac)] = [
                round(float(v) / 1e9, 3) if not pd.isna(v) else None
                for v in pivot[ac]
            ]

    # ── AUM by region
    region_pivot: dict[str, list] = {}
    if _col(df, col.region) and _col(df, col.aum):
        pivot = (
            df.groupby(["_period", col.region])[col.aum]
            .sum()
            .unstack(col.region)
            .reindex(periods)
        )
        for r in pivot.columns:
            region_pivot[str(r)] = [
                round(float(v) / 1e9, 3) if not pd.isna(v) else None
                for v in pivot[r]
            ]

    # ── Portfolio-level stats (latest period)
    portfolio_stats: list[dict] = []
    if _col(df, col.portfolio) and _col(df, col.aum):
        latest_p = periods[-1]
        prev_p   = periods[-2] if len(periods) >= 2 else None
        latest_df = df[df["_period"] == latest_p].groupby(col.portfolio)[col.aum].sum()
        prev_df   = (
            df[df["_period"] == prev_p].groupby(col.portfolio)[col.aum].sum()
            if prev_p else pd.Series(dtype=float)
        )

        # Include all 16 months for growth calcs
        first_p   = periods[0]
        first_df  = df[df["_period"] == first_p].groupby(col.portfolio)[col.aum].sum()

        for pid, cur_aum in latest_df.items():
            prev_aum  = prev_df.get(pid, np.nan)
            first_aum = first_df.get(pid, np.nan)
            mom_pct   = (cur_aum - prev_aum) / prev_aum * 100 if not pd.isna(prev_aum) and prev_aum else np.nan
            total_pct = (cur_aum - first_aum) / first_aum * 100 if not pd.isna(first_aum) and first_aum else np.nan
            portfolio_stats.append({
                "id":        str(pid),
                "aum":       float(cur_aum),
                "mom_pct":   float(mom_pct) if not np.isnan(mom_pct) else None,
                "total_pct": float(total_pct) if not np.isnan(total_pct) else None,
            })
        portfolio_stats.sort(key=lambda x: x["aum"], reverse=True)

    top_10    = portfolio_stats[:10]
    bottom_10 = sorted(
        [p for p in portfolio_stats if p.get("total_pct") is not None],
        key=lambda x: x["total_pct"]
    )[:10]

    # ── Client churn risk
    churn_signals: list[dict] = []
    if _col(df, col.client) and _col(df, col.aum):
        client_aum = (
            df.groupby(["_period", col.client])[col.aum]
            .sum()
            .unstack(col.client)
            .reindex(periods)
        )
        for cid in client_aum.columns:
            series = client_aum[cid].dropna()
            if len(series) < 3:
                continue
            recent = series.tail(3).values
            diffs  = np.diff(recent)
            consecutive_decline = int((diffs < 0).sum())
            latest_aum = float(series.iloc[-1])
            first_aum  = float(series.iloc[0])
            total_change_pct = (latest_aum - first_aum) / first_aum * 100 if first_aum else 0
            flow_vol = float(series.pct_change().std() * 100) if len(series) > 2 else 0

            if consecutive_decline >= 2 or total_change_pct < -5:
                churn_signals.append({
                    "client_id":    str(cid),
                    "latest_aum":   latest_aum,
                    "total_change_pct": round(total_change_pct, 2),
                    "consecutive_decline_months": consecutive_decline,
                    "flow_volatility_pct": round(flow_vol, 2),
                    "risk_score": round(
                        min(1.0, (abs(total_change_pct) / 50) + (consecutive_decline / 3) + (flow_vol / 100)),
                        3,
                    ),
                })
        churn_signals.sort(key=lambda x: x["risk_score"], reverse=True)

    # ── Portfolio mix current + latest allocation
    mix_current: list[dict] = []
    mix_first: list[dict] = []
    if ac_pivot and aum_ts is not None:
        latest_total = float(aum_ts.iloc[-1]) if not pd.isna(aum_ts.iloc[-1]) else 1
        first_total  = float(aum_ts.iloc[0])  if not pd.isna(aum_ts.iloc[0])  else 1
        for ac, vals in ac_pivot.items():
            cur_v   = vals[-1]
            first_v = vals[0]
            mix_current.append({
                "segment": ac,
                "aum_eur_bn": cur_v,
                "pct": round((cur_v or 0) * 1e9 / latest_total * 100, 2),
            })
            mix_first.append({
                "segment": ac,
                "aum_eur_bn": first_v,
                "pct": round((first_v or 0) * 1e9 / first_total * 100, 2),
            })

    # ── Alert / data quality
    alert_ts: list[int] = []
    alert_by_type: dict[str, int] = {}
    if _col(df, col.alert):
        alert_counts = (
            df[df[col.alert].notna()].groupby("_period").size().reindex(periods, fill_value=0)
        )
        alert_ts = [int(v) for v in alert_counts]
        by_type = df[df[col.alert].notna()][col.alert].value_counts()
        alert_by_type = {str(k): int(v) for k, v in by_type.items()}

    # ── Key summary stats
    latest_aum   = float(aum_ts.iloc[-1])   if aum_ts is not None and not pd.isna(aum_ts.iloc[-1])  else 0
    earliest_aum = float(aum_ts.iloc[0])    if aum_ts is not None and not pd.isna(aum_ts.iloc[0])   else 0
    aum_growth   = (latest_aum - earliest_aum) / earliest_aum * 100 if earliest_aum else 0

    return {
        # metadata
        "labels":         labels,
        "n_periods":      len(periods),
        "n_portfolios":   int(df[col.portfolio].nunique()) if _col(df, col.portfolio) else 0,
        "n_clients":      int(df[col.client].nunique())    if _col(df, col.client)    else 0,
        "n_regions":      int(df[col.region].nunique())    if _col(df, col.region)    else 0,
        "date_first":     labels[0]  if labels else "",
        "date_last":      labels[-1] if labels else "",
        # AUM
        "aum_ts":         [round(v / 1e9, 3) if not pd.isna(v) else None for v in aum_ts] if aum_ts is not None else [],
        "aum_latest":     latest_aum,
        "aum_earliest":   earliest_aum,
        "aum_growth_pct": round(aum_growth, 2),
        # Flows
        "subs_ts":  [round(v / 1e9, 3) if not pd.isna(v) else None for v in subs_ts]  if subs_ts is not None else [],
        "reds_ts":  [round(v / 1e9, 3) if not pd.isna(v) else None for v in reds_ts]  if reds_ts is not None else [],
        "nf_ts":    [round(v / 1e9, 3) if not pd.isna(v) else None for v in nf_ts]    if nf_ts  is not None else [],
        # Revenue
        "rev_ts":   [round(v / 1e9, 3) if not pd.isna(v) else None for v in rev_ts]   if rev_ts is not None else [],
        # Breakdowns
        "ac_pivot":      ac_pivot,
        "region_pivot":  region_pivot,
        "mix_current":   mix_current,
        "mix_first":     mix_first,
        # Portfolios
        "top_10":        top_10,
        "bottom_10":     bottom_10,
        # Churn
        "churn_signals": churn_signals[:10],
        # Data quality
        "alert_ts":      alert_ts,
        "alert_by_type": alert_by_type,
    }


# ── Claude context builder ─────────────────────────────────────────────────────

def build_claude_context(analytics: dict, col: ColumnMap, horizon: int) -> str:
    a   = analytics
    lbl = a["labels"]
    sep = "━" * 50

    def ts_block(name: str, vals: list, unit: str = "€Bn") -> str:
        if not vals:
            return f"  {name}: data not available\n"
        lines = [f"  {name} ({unit}):"]
        for lbl_i, v in zip(lbl, vals):
            lines.append(f"    {lbl_i}: {v if v is not None else 'N/A'}")
        return "\n".join(lines) + "\n"

    parts = [
        f"{sep}",
        f"FUND DATA ANALYTICS — COMPLETE HISTORY",
        f"{sep}",
        f"Period:       {a['date_first']} → {a['date_last']}  ({a['n_periods']} months)",
        f"Portfolios:   {a['n_portfolios']}",
        f"Clients:      {a['n_clients']}",
        f"Countries:    {a['n_regions']}",
        f"Total AUM now: {_fmt(a['aum_latest'])}  (was {_fmt(a['aum_earliest'])}, "
        f"growth {_pct(a['aum_growth_pct'])})",
        f"Forecast:     next {horizon} months",
        "",
        f"{sep}",
        "1. AUM TIME SERIES (€Bn, monthly total)",
        f"{sep}",
        ts_block("Total AUM", a["aum_ts"]),
    ]

    if a["ac_pivot"]:
        parts.append("  AUM by Asset Class (€Bn):")
        for ac, vals in a["ac_pivot"].items():
            row = "  ".join(
                f"{lbl[i]}:{v}" for i, v in enumerate(vals) if v is not None
            )
            parts.append(f"    {ac}: {row}")
        parts.append("")

    if a["region_pivot"]:
        parts.append("  AUM by Country/Region (€Bn):")
        for r, vals in a["region_pivot"].items():
            latest = next((v for v in reversed(vals) if v is not None), None)
            parts.append(f"    {r}: latest {latest}€Bn")
        parts.append("")

    parts += [
        f"{sep}",
        "2. NET FLOWS (€Bn, monthly totals)",
        f"{sep}",
        ts_block("Subscriptions", a["subs_ts"]),
        ts_block("Redemptions",   a["reds_ts"]),
        ts_block("Net Flow",      a["nf_ts"]),
    ]

    parts += [
        f"{sep}",
        "3. REVENUE (€Bn, monthly total fee income)",
        f"{sep}",
        ts_block("Revenue", a["rev_ts"]),
    ]

    parts += [
        f"{sep}",
        "4. TOP 10 PORTFOLIOS (current AUM, full period growth)",
        f"{sep}",
    ]
    for i, p in enumerate(a["top_10"], 1):
        parts.append(
            f"  {i:2d}. {p['id']:40s}  AUM {_fmt(p['aum'])}  "
            f"MoM {_pct(p['mom_pct'])}  Period {_pct(p['total_pct'])}"
        )

    parts += [
        "",
        f"{sep}",
        "5. BOTTOM 10 PORTFOLIOS (largest AUM decline over full period)",
        f"{sep}",
    ]
    for i, p in enumerate(a["bottom_10"], 1):
        parts.append(
            f"  {i:2d}. {p['id']:40s}  AUM {_fmt(p['aum'])}  "
            f"Period {_pct(p['total_pct'])}  Consec declines: {p.get('consecutive_decline', 'N/A')}"
        )

    parts += [
        "",
        f"{sep}",
        "6. CLIENT CHURN RISK SIGNALS (top 10 highest-risk clients)",
        f"{sep}",
    ]
    if a["churn_signals"]:
        for i, c in enumerate(a["churn_signals"], 1):
            parts.append(
                f"  {i:2d}. Client {c['client_id']}  "
                f"AUM {_fmt(c['latest_aum'])}  "
                f"Change {_pct(c['total_change_pct'])}  "
                f"Consecutive declines: {c['consecutive_decline_months']}  "
                f"Flow vol: {c['flow_volatility_pct']:.1f}%  "
                f"Risk score: {c['risk_score']:.3f}"
            )
    else:
        parts.append("  No significant churn signals detected.")

    parts += [
        "",
        f"{sep}",
        "7. PORTFOLIO MIX (asset class allocation)",
        f"{sep}",
        f"  Start ({a['date_first']}):",
    ]
    for m in a["mix_first"]:
        parts.append(f"    {m['segment']}: €{m['aum_eur_bn']}Bn  ({m['pct']}%)")
    parts.append(f"  Latest ({a['date_last']}):")
    for m in a["mix_current"]:
        parts.append(f"    {m['segment']}: €{m['aum_eur_bn']}Bn  ({m['pct']}%)")

    parts += [
        "",
        f"{sep}",
        "8. DATA QUALITY ALERTS (monthly counts)",
        f"{sep}",
    ]
    if a["alert_ts"]:
        for lbl_i, cnt in zip(lbl, a["alert_ts"]):
            if cnt:
                parts.append(f"  {lbl_i}: {cnt} alerts")
        if a["alert_by_type"]:
            parts.append("  By type:")
            for t, cnt in a["alert_by_type"].items():
                parts.append(f"    {t}: {cnt}")
    else:
        parts.append("  No alert data available.")

    return "\n".join(parts)


# ── TimesFM integration ────────────────────────────────────────────────────────

async def run_timesfm_forecasts(analytics: dict, horizon: int) -> dict | None:
    """Run TimesFM on all key aggregated time series.

    Returns None (silently) if TimesFM is not installed or fails.
    """
    try:
        from app.services.timesfm_service import ensure_loaded, _run_forecast
        await ensure_loaded()
    except Exception as exc:
        logger.info("TimesFM not available — skipping statistical forecast: %s", exc)
        return None

    import numpy as np

    def _to_array(vals: list) -> np.ndarray | None:
        arr = [v for v in vals if v is not None]
        if len(arr) < 4:
            return None
        return np.array(arr, dtype=np.float32)

    series_map: dict[str, np.ndarray] = {}
    for key in ("aum_ts", "subs_ts", "reds_ts", "nf_ts", "rev_ts"):
        arr = _to_array(analytics.get(key, []))
        if arr is not None:
            series_map[key] = arr

    for ac, vals in analytics.get("ac_pivot", {}).items():
        arr = _to_array(vals)
        if arr is not None:
            series_map[f"ac_{ac}"] = arr

    if not series_map:
        return None

    keys    = list(series_map.keys())
    inputs  = [series_map[k] for k in keys]
    try:
        point_fc, quantile_fc = await asyncio.to_thread(_run_forecast, inputs, horizon)
    except Exception as exc:
        logger.warning("TimesFM forecast failed: %s", exc)
        return None

    results: dict[str, dict] = {}
    for i, key in enumerate(keys):
        results[key] = {
            "point":  [round(float(v), 4) for v in point_fc[i]],
            "q10":    [round(float(v), 4) for v in quantile_fc[i, :, 0]],
            "q50":    [round(float(v), 4) for v in quantile_fc[i, :, 4]],
            "q90":    [round(float(v), 4) for v in quantile_fc[i, :, -1]],
        }
    return results


# ── Chart builders ────────────────────────────────────────────────────────────

def _make_future_labels(labels: list[str], horizon: int) -> list[str]:
    try:
        last = pd.Period(labels[-1], freq="M")
        return [str(last + i) for i in range(1, horizon + 1)]
    except Exception:
        return [f"+{i}m" for i in range(1, horizon + 1)]


def build_chart_data(analytics: dict, timesfm: dict | None, horizon: int) -> dict:
    """Build all chart_data objects — merging TimesFM outputs where available."""
    lbl      = analytics["labels"]
    fut_lbl  = _make_future_labels(lbl, horizon)
    all_lbl  = lbl + fut_lbl
    n_hist   = len(lbl)
    nullh    = [None] * n_hist
    nullf    = [None] * horizon

    def _chart(hist_vals, point, q10, q90, series_label: str, unit: str = "€Bn") -> dict:
        return {
            "type":   "line",
            "labels": all_lbl,
            "series": [
                {
                    "id":          "historical",
                    "label":       f"{series_label} — Historical",
                    "data":        hist_vals + nullf,
                    "borderColor": "#2563eb",
                    "dashed":      False,
                },
                {
                    "id":          "point",
                    "label":       f"{series_label} — Forecast",
                    "data":        nullh + point,
                    "borderColor": "#16a34a",
                    "dashed":      True,
                },
                {
                    "id":          "q90",
                    "label":       "90th Percentile",
                    "data":        nullh + q90,
                    "borderColor": "#84cc16",
                    "dashed":      True,
                },
                {
                    "id":          "q10",
                    "label":       "10th Percentile",
                    "data":        nullh + q10,
                    "borderColor": "#dc2626",
                    "dashed":      True,
                },
            ],
        }

    def _bar_chart(hist_vals, series_label: str) -> dict:
        return {
            "type":   "bar",
            "labels": lbl,
            "series": [{"id": "hist", "label": series_label, "data": hist_vals, "backgroundColor": "#2563eb"}],
        }

    def _doughnut(segments: list[dict]) -> dict:
        return {
            "type":   "doughnut",
            "labels": [s["segment"] for s in segments],
            "series": [{"id": "current", "label": "Current Allocation", "data": [s["pct"] for s in segments]}],
        }

    tfm = timesfm or {}

    def _fc(key: str, hist: list):
        d = tfm.get(key, {})
        n = horizon
        if d:
            return d["point"], d["q10"], d["q90"]
        # Claude-fallback: simple linear extrapolation
        clean = [v for v in hist if v is not None]
        if len(clean) >= 2:
            slope = (clean[-1] - clean[-2])
            pt = [round(clean[-1] + slope * (i + 1), 3) for i in range(n)]
        else:
            last = clean[-1] if clean else 0
            pt = [round(last, 3)] * n
        q10 = [round(v * 0.97, 3) for v in pt]
        q90 = [round(v * 1.03, 3) for v in pt]
        return pt, q10, q90

    charts: dict[str, dict] = {}

    # AUM trend
    if analytics["aum_ts"]:
        pt, q10, q90 = _fc("aum_ts", analytics["aum_ts"])
        charts["aum_trend"] = _chart(analytics["aum_ts"], pt, q10, q90, "Total AUM")

    # Net flows
    if analytics["nf_ts"]:
        charts["net_flows"] = _bar_chart(analytics["nf_ts"], "Net Flow (€Bn)")
    if analytics["subs_ts"]:
        charts["subscriptions"] = _bar_chart(analytics["subs_ts"], "Subscriptions (€Bn)")
    if analytics["reds_ts"]:
        charts["redemptions"] = _bar_chart(analytics["reds_ts"], "Redemptions (€Bn)")

    # Revenue
    if analytics["rev_ts"]:
        pt, q10, q90 = _fc("rev_ts", analytics["rev_ts"])
        charts["revenue"] = _chart(analytics["rev_ts"], pt, q10, q90, "Revenue")

    # Asset class breakdown over time (multi-line)
    if analytics["ac_pivot"]:
        ac_series = []
        colors = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed"]
        for i, (ac, vals) in enumerate(analytics["ac_pivot"].items()):
            pt, _, _ = _fc(f"ac_{ac}", vals)
            ac_series.append({
                "id":          f"ac_{ac}",
                "label":       ac,
                "data":        vals + nullf,
                "borderColor": colors[i % len(colors)],
                "dashed":      False,
            })
            ac_series.append({
                "id":          f"ac_{ac}_fc",
                "label":       f"{ac} (Forecast)",
                "data":        nullh + pt,
                "borderColor": colors[i % len(colors)],
                "dashed":      True,
            })
        charts["asset_class_trend"] = {"type": "line", "labels": all_lbl, "series": ac_series}

    # Portfolio mix doughnut
    if analytics["mix_current"]:
        charts["portfolio_mix_current"] = _doughnut(analytics["mix_current"])
    if analytics["mix_first"]:
        charts["portfolio_mix_start"] = _doughnut(analytics["mix_first"])

    # Data quality
    if analytics["alert_ts"]:
        charts["data_quality_alerts"] = {
            "type":   "bar",
            "labels": lbl,
            "series": [{"id": "alerts", "label": "Alert Count", "data": analytics["alert_ts"], "backgroundColor": "#f59e0b"}],
        }

    return charts


def _ensure_dict(value):
    """Parse a value that Claude may have returned as a JSON string instead of a dict."""
    if isinstance(value, str):
        import json as _json
        try:
            return _json.loads(value)
        except Exception:
            return {}
    return value if isinstance(value, dict) else {}


def inject_charts_into_forecast(forecast: dict, charts: dict) -> dict:
    """Merge pre-computed charts into the Claude forecast structure."""
    forecasts = _ensure_dict(forecast.get("forecasts", {}))

    # Also ensure every sub-section is a dict, not a string
    for key in ("aum_trend", "net_flows", "revenue", "churn_risk", "portfolio_mix_drift", "data_quality"):
        if key in forecasts:
            forecasts[key] = _ensure_dict(forecasts[key])

    if "aum_trend" in charts and "aum_trend" in forecasts:
        forecasts["aum_trend"]["chart_data"] = charts["aum_trend"]
    if "net_flows" in charts and "net_flows" in forecasts:
        forecasts["net_flows"]["chart_data"] = charts["net_flows"]
    if "revenue" in charts and "revenue" in forecasts:
        forecasts["revenue"]["chart_data"] = charts["revenue"]
    if "portfolio_mix_drift" in forecasts and "portfolio_mix_current" in charts:
        forecasts["portfolio_mix_drift"]["chart_data"] = {
            "current": charts.get("portfolio_mix_current"),
            "start":   charts.get("portfolio_mix_start"),
            "trend":   charts.get("asset_class_trend"),
        }

    # Add supplemental charts at top level
    forecast["supplemental_charts"] = {
        k: v for k, v in charts.items()
        if k not in ("aum_trend", "net_flows", "revenue",
                     "portfolio_mix_current", "portfolio_mix_start")
    }
    forecast["forecasts"] = forecasts
    return forecast
