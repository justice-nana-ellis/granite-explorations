"""Chart generation (Matplotlib PNG + Chart.js HTML) and forecasting."""
import asyncio
import html
import io
import json
import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from fastapi import HTTPException

from app.models.session import SessionData

logger = logging.getLogger(__name__)

PALETTE = ["#2563eb", "#16a34a", "#d97706", "#7c3aed", "#dc2626", "#0891b2"]

# In-memory chart store: session_id/filename → PNG bytes, served directly from memory.
chart_store: dict[str, bytes] = {}


# ── Matplotlib chart helpers ──────────────────────────────────────────────────

def make_chart_bytes(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str) -> bytes:
    fig, ax = plt.subplots(figsize=(11, 6))
    working = df[[x_col, y_col]].dropna().copy()

    if chart_type in ("auto", "bar"):
        is_date_like = any(k in x_col.lower() for k in ("date", "month", "time", "year"))
        chosen = "line" if is_date_like else "bar"
    else:
        chosen = chart_type

    if chosen == "line":
        working.plot(x=x_col, y=y_col, kind="line", marker="o", ax=ax, legend=False, color="#2563eb")
        ax.fill_between(range(len(working)), working[y_col], alpha=0.08, color="#2563eb")
    elif chosen == "scatter":
        working.plot(x=x_col, y=y_col, kind="scatter", ax=ax, color="#7c3aed")
    else:
        working.head(50).plot(
            x=x_col, y=y_col, kind="bar", ax=ax, legend=False, color="#2563eb", edgecolor="#1d4ed8"
        )

    ax.set_title(f"{y_col} by {x_col}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(y_col, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def make_forecast_chart_bytes(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    date_col: str,
    value_col: str,
) -> bytes:
    fig, ax = plt.subplots(figsize=(11, 6))
    actual_df.plot(x=date_col, y=value_col, kind="line", marker="o", ax=ax, label="Actual")
    forecast_df.plot(
        x=date_col, y=value_col, kind="line", marker="o",
        linestyle="--", ax=ax, label="Forecast", color="#d97706",
    )
    ax.set_title(f"Forecast for {value_col}")
    ax.set_xlabel(date_col)
    ax.set_ylabel(value_col)
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ── Column resolution ─────────────────────────────────────────────────────────

def resolve_chart_specs(df: pd.DataFrame, chart_type: str, max_charts: int) -> list[tuple[str, str, str]]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()
    if not numeric_cols or len(all_cols) < 2:
        raise HTTPException(status_code=400, detail="Not enough usable columns to create charts.")

    x_candidates = [c for c in all_cols if c not in numeric_cols]
    x_col = x_candidates[0] if x_candidates else all_cols[0]
    chosen_chart = "bar" if chart_type == "auto" else chart_type

    specs: list[tuple[str, str, str]] = []
    for y_col in numeric_cols:
        if y_col == x_col:
            continue
        specs.append((chosen_chart, x_col, y_col))
        if len(specs) >= max_charts:
            break

    if not specs:
        raise HTTPException(status_code=400, detail="Could not determine chart columns from uploaded data.")
    return specs


def find_forecast_columns(df: pd.DataFrame) -> tuple[str, str, pd.DataFrame]:
    working = df.copy()
    date_candidates = [c for c in working.columns if any(k in c.lower() for k in ("date", "month", "time"))]
    if not date_candidates:
        raise HTTPException(status_code=400, detail="Could not find a date-like column for forecasting.")

    date_col = date_candidates[0]
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working = working.dropna(subset=[date_col])

    numeric_cols = [c for c in working.select_dtypes(include="number").columns.tolist() if c != date_col]
    if not numeric_cols:
        raise HTTPException(status_code=400, detail="Could not find a numeric column for forecasting.")

    value_col = numeric_cols[0]
    grouped = working.groupby(date_col, as_index=False)[value_col].sum().sort_values(date_col)
    if len(grouped) < 2:
        raise HTTPException(status_code=400, detail="Need at least two dated data points to build a forecast.")
    return date_col, value_col, grouped


# ── VisualizationService ──────────────────────────────────────────────────────

class VisualizationService:
    async def ensure_df(self, session: SessionData) -> pd.DataFrame:
        if session.get("df") is None and session.get("cloudinary_url"):
            from app.services.session_service import session_service
            await session_service.reload_df(session)
        df = session.get("df")
        if df is None:
            raise HTTPException(
                status_code=400,
                detail="No tabular data found for this session. Upload CSV/Excel first.",
            )
        return df

    async def build_artifacts(
        self,
        session_id: str,
        session: SessionData,
        chart_type: str = "auto",
        max_charts: int = 3,
    ) -> list[dict]:
        df = await self.ensure_df(session)
        specs = resolve_chart_specs(df, chart_type=chart_type, max_charts=max_charts)

        async def _one(chosen_chart: str, x_col: str, y_col: str) -> dict:
            key = f"{session_id}/{chosen_chart}_{y_col}"
            chart_bytes = await asyncio.to_thread(make_chart_bytes, df, chosen_chart, x_col, y_col)
            chart_store[key] = chart_bytes
            return {
                "title": f"{chosen_chart.title()} - {y_col} by {x_col}",
                "chart_type": chosen_chart,
                "x": x_col,
                "y": y_col,
                "public_id": key,
                "url": f"/charts/{key}",
            }

        return list(await asyncio.gather(*[_one(c, x, y) for c, x, y in specs]))

    async def build_forecast(self, session_id: str, session: SessionData, periods: int = 3) -> dict:
        df = await self.ensure_df(session)
        date_col, value_col, grouped = find_forecast_columns(df)

        recent = grouped.tail(min(4, len(grouped))).reset_index(drop=True)
        slope = (recent[value_col].iloc[-1] - recent[value_col].iloc[0]) / max(len(recent) - 1, 1)
        last_date = recent[date_col].iloc[-1]
        step = recent[date_col].diff().dropna().median()
        if pd.isna(step) or step <= pd.Timedelta(0):
            step = pd.Timedelta(days=30)

        forecast_rows = [
            {date_col: last_date + (step * i), value_col: float(recent[value_col].iloc[-1] + slope * i)}
            for i in range(1, periods + 1)
        ]
        forecast_df = pd.DataFrame(forecast_rows)

        chart_bytes = await asyncio.to_thread(
            make_forecast_chart_bytes, recent, forecast_df, date_col, value_col
        )
        key = f"{session_id}/forecast_{value_col}"
        chart_store[key] = chart_bytes

        serialized_rows = [
            {date_col: pd.Timestamp(row[date_col]).strftime("%Y-%m-%d"), value_col: row[value_col]}
            for row in forecast_rows
        ]
        return {
            "title": f"Forecast - {value_col}",
            "chart_type": "forecast",
            "x": date_col,
            "y": value_col,
            "public_id": key,
            "url": f"/charts/{key}",
            "forecast_rows": serialized_rows,
            "summary": f"Forecasted {value_col} for the next {periods} period(s) based on the recent trend.",
        }

    async def build_visual_bundle(self, session_id: str, session: SessionData, base_url: str) -> dict:
        artifacts = await self.build_artifacts(session_id, session, chart_type="auto", max_charts=3)
        forecast = await self.build_forecast(session_id, session)
        return {
            "session_id": session_id,
            "file": session.get("file"),
            "source_file_url": session.get("cloudinary_url"),
            "dashboard_url": f"{base_url}/dashboard/{session_id}/render",
            "artifacts_url": f"{base_url}/artifacts/{session_id}",
            "forecast_url": f"{base_url}/forecast/{session_id}",
            "artifacts": artifacts,
            "forecast": forecast,
        }

    # ── Chart.js / HTML helpers ───────────────────────────────────────────────

    def build_kpi_cards_html(self, session: SessionData) -> str:
        df = session.get("df")
        if df is None:
            return ""
        numeric_cols = df.select_dtypes(include="number").columns.tolist()[:6]
        if not numeric_cols:
            return ""

        def _fmt(v: float) -> str:
            if abs(v) >= 1_000_000:
                return f"{v/1_000_000:,.2f}M"
            if abs(v) >= 1_000:
                return f"{v/1_000:,.1f}K"
            return f"{v:,.2f}"

        cards = ""
        for i, col in enumerate(numeric_cols):
            data = df[col].dropna()
            color = PALETTE[i % len(PALETTE)]
            cards += (
                f"<div style='background:#fff;border-top:4px solid {color};border-radius:14px;padding:16px 18px;"
                f"box-shadow:0 1px 6px rgba(0,0,0,.06);min-width:140px'>"
                f"<div style='font:600 11px Arial,sans-serif;color:#6b7280;text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px'>{html.escape(col)}</div>"
                f"<div style='font:700 24px Arial,sans-serif;color:#111827;line-height:1.1'>{_fmt(data.sum())}</div>"
                f"<div style='font:500 12px Arial,sans-serif;color:#6b7280;margin-top:4px'>avg {_fmt(data.mean())}</div>"
                f"</div>"
            )
        return f"<div style='display:flex;flex-wrap:wrap;gap:14px'>{cards}</div>"

    def build_chartjs_html(self, session: SessionData) -> tuple[str, str]:
        df = session.get("df")
        if df is None:
            return ("", "")
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        x_candidates = [c for c in all_cols if c not in numeric_cols]
        if not numeric_cols or not x_candidates:
            return ("", "")

        x_col = x_candidates[0]
        is_ts = any(k in x_col.lower() for k in ("date", "month", "time", "year", "period"))
        chart_cols = numeric_cols[:6]
        labels_json = json.dumps([str(v) for v in df[x_col].fillna("").tolist()[:100]])
        chart_type_js = "line" if is_ts else "bar"
        inits: list[str] = []

        # Overview multi-series chart
        overview_datasets = []
        for i, col in enumerate(chart_cols):
            vals_json = json.dumps([round(float(v), 4) if pd.notna(v) else None for v in df[col].tolist()[:100]])
            color = PALETTE[i % len(PALETTE)]
            fill_val = "true" if (is_ts and i == 0) else "false"
            pt_r = 3 if len(df) <= 40 else 1
            overview_datasets.append(
                "{label:" + json.dumps(col) + ",data:" + vals_json + ","
                "borderColor:'" + color + "',backgroundColor:'" + color + "33',"
                "fill:" + fill_val + ",tension:0.35,pointRadius:" + str(pt_r) + ",borderWidth:2.5}"
            )
        inits.append(
            "(function(){var el=document.getElementById('chartOverview');"
            "if(!el)return;if(el._ci){el._ci.resize();return;}"
            "el._ci=new Chart(el.getContext('2d'),{type:'" + chart_type_js + "',"
            "data:{labels:" + labels_json + ",datasets:[" + ",".join(overview_datasets) + "]},"
            "options:{responsive:true,maintainAspectRatio:false,"
            "interaction:{mode:'index',intersect:false},"
            "plugins:{legend:{position:'top',labels:{font:{size:12},padding:14}},"
            "tooltip:{callbacks:{label:function(c){return ' '+c.dataset.label+': '+Number(c.raw).toLocaleString();}}}},"
            "scales:{x:{ticks:{maxRotation:45,font:{size:11},maxTicksLimit:24}},"
            "y:{ticks:{callback:function(v){return v>=1e6?(v/1e6).toFixed(1)+'M':v>=1e3?(v/1e3).toFixed(1)+'K':v;},font:{size:11}}}}}});"
            "if(!window._chartInstances)window._chartInstances=[];"
            "window._chartInstances.push(el._ci);})();"
        )

        # Doughnut totals
        totals = [round(float(df[c].sum()), 2) for c in chart_cols]
        totals_json = json.dumps(totals)
        donut_labels_json = json.dumps(chart_cols)
        donut_colors_json = json.dumps(PALETTE[:len(chart_cols)])
        inits.append(
            "(function(){var el=document.getElementById('chartDonut');"
            "if(!el)return;if(el._ci){el._ci.resize();return;}"
            "el._ci=new Chart(el.getContext('2d'),{type:'doughnut',"
            "data:{labels:" + donut_labels_json + ",datasets:[{data:" + totals_json + ","
            "backgroundColor:" + donut_colors_json + ",borderWidth:3,borderColor:'#fff',hoverOffset:10}]},"
            "options:{responsive:true,maintainAspectRatio:false,cutout:'62%',"
            "plugins:{legend:{position:'right',labels:{font:{size:12},padding:14}},"
            "tooltip:{callbacks:{label:function(c){"
            "var t=c.chart.data.datasets[0].data.reduce(function(a,b){return a+b;},0);"
            "return ' '+c.label+': '+Number(c.raw).toLocaleString()+' ('+(c.raw/t*100).toFixed(1)+'%)';}}}}}});"
            "if(!window._chartInstances)window._chartInstances=[];"
            "window._chartInstances.push(el._ci);})();"
        )

        # Horizontal bar snapshot
        last_row = df[chart_cols].dropna(how="all").tail(1)
        if len(last_row) > 0:
            snap_vals = [round(float(last_row[c].iloc[0]), 2) if pd.notna(last_row[c].iloc[0]) else 0 for c in chart_cols]
        else:
            snap_vals = [round(float(df[c].mean()), 2) for c in chart_cols]
        snap_vals_json = json.dumps(snap_vals)
        hbar_colors_json = json.dumps([p + "cc" for p in PALETTE[:len(chart_cols)]])
        inits.append(
            "(function(){var el=document.getElementById('chartHBar');"
            "if(!el)return;if(el._ci){el._ci.resize();return;}"
            "el._ci=new Chart(el.getContext('2d'),{type:'bar',"
            "data:{labels:" + donut_labels_json + ",datasets:[{label:'Latest snapshot',"
            "data:" + snap_vals_json + ",backgroundColor:" + hbar_colors_json + ",borderRadius:6,borderWidth:0}]},"
            "options:{responsive:true,maintainAspectRatio:false,indexAxis:'y',"
            "plugins:{legend:{display:false},"
            "tooltip:{callbacks:{label:function(c){return ' '+Number(c.raw).toLocaleString();}}}},"
            "scales:{x:{ticks:{callback:function(v){return v>=1e6?(v/1e6).toFixed(1)+'M':v>=1e3?(v/1e3).toFixed(1)+'K':v;},font:{size:11}}},"
            "y:{ticks:{font:{size:12}}}}}});"
            "if(!window._chartInstances)window._chartInstances=[];"
            "window._chartInstances.push(el._ci);})();"
        )

        # Stacked chart (time-series only)
        stacked_html = ""
        if is_ts and len(chart_cols) >= 2:
            stacked_datasets = []
            for i, col in enumerate(chart_cols[:4]):
                vals_json = json.dumps([round(float(v), 4) if pd.notna(v) else 0 for v in df[col].tolist()[:100]])
                color = PALETTE[i % len(PALETTE)]
                stacked_datasets.append(
                    "{label:" + json.dumps(col) + ",data:" + vals_json + ","
                    "backgroundColor:'" + color + "99',borderColor:'" + color + "',"
                    "borderWidth:1,borderRadius:2,stack:'s1'}"
                )
            inits.append(
                "(function(){var el=document.getElementById('chartStacked');"
                "if(!el)return;if(el._ci){el._ci.resize();return;}"
                "el._ci=new Chart(el.getContext('2d'),{type:'bar',"
                "data:{labels:" + labels_json + ",datasets:[" + ",".join(stacked_datasets) + "]},"
                "options:{responsive:true,maintainAspectRatio:false,"
                "interaction:{mode:'index',intersect:false},"
                "plugins:{legend:{position:'top',labels:{font:{size:12}}},"
                "tooltip:{callbacks:{label:function(c){return ' '+c.dataset.label+': '+Number(c.raw).toLocaleString();}}}},"
                "scales:{x:{stacked:true,ticks:{maxRotation:45,font:{size:11},maxTicksLimit:24}},"
                "y:{stacked:true,ticks:{callback:function(v){return v>=1e6?(v/1e6).toFixed(1)+'M':v>=1e3?(v/1e3).toFixed(1)+'K':v;},font:{size:11}}}}}});"
                "if(!window._chartInstances)window._chartInstances=[];"
                "window._chartInstances.push(el._ci);})();"
            )
            stacked_html = (
                "<div style='background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:16px;margin-bottom:18px'>"
                "<div style='font:700 13px Arial,sans-serif;color:#1e3a8a;margin-bottom:10px'>&#9641; Stacked Period Comparison</div>"
                "<div style='position:relative;height:300px'><canvas id='chartStacked'></canvas></div>"
                "</div>"
            )

        # Individual sparklines
        spark_canvases = []
        for i, col in enumerate(chart_cols):
            cid = f"chartC{i}"
            vals_json = json.dumps([round(float(v), 4) if pd.notna(v) else None for v in df[col].tolist()[:100]])
            color = PALETTE[i % len(PALETTE)]
            fill_val = "true" if (is_ts and i == 0) else "false"
            pt_r = 3 if len(df) <= 40 else 1
            spark_canvases.append(
                f"<div style='background:#fafafa;border:1px solid #e5e7eb;border-radius:14px;padding:14px'>"
                f"<div style='font:700 12px Arial,sans-serif;color:#374151;margin-bottom:8px'><strong>{html.escape(col)}</strong></div>"
                f"<div style='position:relative;height:190px'><canvas id='{cid}'></canvas></div>"
                f"</div>"
            )
            inits.append(
                "(function(){var el=document.getElementById('" + cid + "');"
                "if(!el)return;if(el._ci){el._ci.resize();return;}"
                "el._ci=new Chart(el.getContext('2d'),{type:'" + chart_type_js + "',"
                "data:{labels:" + labels_json + ",datasets:[{label:" + json.dumps(col) + ",data:" + vals_json + ","
                "borderColor:'" + color + "',backgroundColor:'" + color + "22',"
                "fill:" + fill_val + ",tension:0.35,pointRadius:" + str(pt_r) + ",borderWidth:2}]},"
                "options:{responsive:true,maintainAspectRatio:false,"
                "plugins:{legend:{display:false},"
                "tooltip:{callbacks:{label:function(c){return c.dataset.label+': '+Number(c.raw).toLocaleString();}}}},"
                "scales:{x:{ticks:{maxRotation:45,font:{size:10},maxTicksLimit:16}},"
                "y:{ticks:{callback:function(v){return v>=1e6?(v/1e6).toFixed(1)+'M':v>=1e3?(v/1e3).toFixed(1)+'K':v;},font:{size:10}}}}}});"
                "if(!window._chartInstances)window._chartInstances=[];"
                "window._chartInstances.push(el._ci);})();"
            )

        overview_block = (
            "<div style='background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:16px;margin-bottom:18px'>"
            "<div style='font:700 14px Arial,sans-serif;color:#1e3a8a;margin-bottom:12px'>&#128200; All Metrics Overview</div>"
            "<div style='position:relative;height:340px'><canvas id='chartOverview'></canvas></div>"
            "</div>"
        )
        two_col_block = (
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:18px'>"
            "<div style='background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:16px'>"
            "<div style='font:700 13px Arial,sans-serif;color:#1e3a8a;margin-bottom:10px'>&#11835; Total Distribution</div>"
            "<div style='position:relative;height:260px'><canvas id='chartDonut'></canvas></div>"
            "</div>"
            "<div style='background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:16px'>"
            "<div style='font:700 13px Arial,sans-serif;color:#1e3a8a;margin-bottom:10px'>&#128293; Metric Snapshot</div>"
            "<div style='position:relative;height:260px'><canvas id='chartHBar'></canvas></div>"
            "</div>"
            "</div>"
        )
        spark_grid = (
            "<div style='font:700 14px Arial,sans-serif;color:#1e3a8a;margin:4px 0 12px'>&#9889; Individual Trends</div>"
            "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-bottom:4px'>"
            + "".join(spark_canvases)
            + "</div>"
        )

        init_js = "window._initIC=function(){"
        init_js += "if(typeof Chart==='undefined'){setTimeout(window._initIC,200);return;}"
        init_js += "".join(inits)
        init_js += "};"
        init_js += (
            "if(document.readyState==='loading'){"
            "document.addEventListener('DOMContentLoaded',function(){setTimeout(window._initIC,50)});"
            "}else{setTimeout(window._initIC,50);}"
            "window.addEventListener('load',function(){setTimeout(window._initIC,50);});"
        )

        return (overview_block + two_col_block + stacked_html + spark_grid, init_js)

    def build_data_table_html(self, session: SessionData, max_rows: int = 10) -> str:
        df = session.get("df")
        if df is None:
            return ""
        preview = df.head(max_rows)
        th_cells = "".join(f"<th>{html.escape(str(c))}</th>" for c in preview.columns)
        rows_html = ""
        for _, row in preview.iterrows():
            cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
            rows_html += f"<tr>{cells}</tr>"
        return (
            f"<div style='background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:20px;"
            f"box-shadow:0 1px 6px rgba(0,0,0,.05);overflow-x:auto'>"
            f"<div style='font:700 15px Arial,sans-serif;color:#1e3a8a;margin-bottom:12px'>&#128196; Data Preview (first {len(preview)} rows)</div>"
            f"<table style='width:100%;border-collapse:collapse;font-size:13px;white-space:nowrap'>"
            f"<thead><tr style='background:#1e3a8a;color:#fff'>{th_cells}</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            f"</table>"
            f"<div style='font:500 12px Arial,sans-serif;color:#9ca3af;margin-top:8px'>{len(df):,} total rows × {len(df.columns)} columns</div>"
            f"</div>"
        )

    def build_jsx_snippet(self, session: SessionData, analysis: str) -> str:
        df = session.get("df")
        cols = list(df.columns) if df is not None else []
        numeric_cols = [c for c in cols if df is not None and pd.api.types.is_numeric_dtype(df[c])]
        cols_json = json.dumps(cols)
        numeric_json = json.dumps(numeric_cols[:4])
        jsx = (
            f"import {{ BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer }} from 'recharts';\n\n"
            f"export default function Dashboard({{ data }}) {{\n"
            f"  // data: array of row objects with keys: {cols_json}\n"
            f"  // numeric columns: {numeric_json}\n"
            f"  return (\n"
            f"    <div className=\"p-6 bg-gray-50 min-h-screen\">\n"
            f"      <h1 className=\"text-2xl font-bold text-blue-900 mb-6\">Analytics Dashboard</h1>\n"
            f"      <div className=\"grid grid-cols-1 md:grid-cols-2 gap-6\">\n"
            f"        <div className=\"bg-white rounded-2xl p-4 shadow\">\n"
            f"          <h2 className=\"font-semibold text-gray-700 mb-3\">{(numeric_cols[:1] or ['value'])[0]} Over Time</h2>\n"
            f"          <ResponsiveContainer width=\"100%\" height={{300}}>\n"
            f"            <LineChart data={{data}}>\n"
            f"              <CartesianGrid strokeDasharray=\"3 3\" />\n"
            f"              <XAxis dataKey=\"{(cols[0] if cols else 'x')}\" />\n"
            f"              <YAxis />\n"
            f"              <Tooltip />\n"
            f"              <Legend />\n"
            + "".join(f'              <Line type=\"monotone\" dataKey=\"{c}\" stroke=\"#2563eb\" dot={{false}} />\n' for c in numeric_cols[:2])
            + f"            </LineChart>\n"
            f"          </ResponsiveContainer>\n"
            f"        </div>\n"
            f"        <div className=\"bg-white rounded-2xl p-4 shadow\">\n"
            f"          <h2 className=\"font-semibold text-gray-700 mb-3\">Breakdown</h2>\n"
            f"          <ResponsiveContainer width=\"100%\" height={{300}}>\n"
            f"            <BarChart data={{data}}>\n"
            f"              <CartesianGrid strokeDasharray=\"3 3\" />\n"
            f"              <XAxis dataKey=\"{(cols[0] if cols else 'x')}\" />\n"
            f"              <YAxis />\n"
            f"              <Tooltip />\n"
            f"              <Legend />\n"
            + "".join(f'              <Bar dataKey=\"{c}\" fill=\"#7c3aed\" />\n' for c in numeric_cols[:2])
            + f"            </BarChart>\n"
            f"          </ResponsiveContainer>\n"
            f"        </div>\n"
            f"      </div>\n"
            f"    </div>\n"
            f"  );\n"
            f"}}"
        )
        safe_jsx = html.escape(jsx)
        return (
            "<div style='background:#1e293b;border-radius:16px;padding:20px;box-shadow:0 1px 6px rgba(0,0,0,.15)'>"
            "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px'>"
            "<div style='font:700 14px Arial,sans-serif;color:#94a3b8'>&#128196; React JSX Component Scaffold</div>"
            "<button onclick=\"navigator.clipboard.writeText(document.getElementById('jsxCode').innerText)\" "
            "style='background:#2563eb;color:#fff;border:none;border-radius:8px;padding:6px 14px;font:600 12px Arial,sans-serif;cursor:pointer'>Copy</button>"
            "</div>"
            f"<pre id='jsxCode' style='font:13px/1.6 monospace;color:#e2e8f0;overflow-x:auto;white-space:pre-wrap'>{safe_jsx}</pre>"
            "</div>"
        )


visualization_service = VisualizationService()
