import argparse
import os
import sys
import webbrowser
from datetime import datetime

# Force UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf

# ----------------------------------------
# 1. Load portfolio from text file
# ----------------------------------------
def load_portfolio(path: str) -> dict:
    weights = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Markdown table format: | TICKER | WEIGHT |
            if line.startswith("|"):
                parts = [p.strip() for p in line.strip("|").split("|")]
                parts = [p for p in parts if p]
                if len(parts) < 2:
                    continue
                ticker, weight_str = parts[0], parts[1]
                # Skip header and separator rows (e.g. "Asset", "-----")
                if not ticker or set(ticker).issubset(set("- ")) or not weight_str[0].isdigit():
                    continue
            # Plain format: TICKER WEIGHT
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                ticker, weight_str = parts[0], parts[1]

            try:
                weights[ticker.upper()] = float(weight_str.rstrip("%")) / 100.0
            except ValueError:
                continue  # skip any unparseable rows

    if not weights:
        raise ValueError(f"No valid ticker/weight rows found in '{path}'")

    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}

parser = argparse.ArgumentParser(description="Portfolio stress tester")
parser.add_argument(
    "portfolio_file",
    nargs="?",
    default="portfolio.txt",
    help="Path to portfolio text file (default: portfolio.txt)",
)
args = parser.parse_args()

print(f"Loading portfolio: {args.portfolio_file}")
weights = load_portfolio(args.portfolio_file)
tickers = list(weights.keys())

# derive a clean name for the report title / output filename
portfolio_name = os.path.splitext(os.path.basename(args.portfolio_file))[0].replace("_", " ").title()
report_stem    = os.path.splitext(args.portfolio_file)[0]
report_path    = report_stem + "_report.html"

# ----------------------------------------
# 2. Download historical data
# ----------------------------------------
start_date = "2005-01-01"
end_date   = "2024-12-31"

raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

if isinstance(raw.columns, pd.MultiIndex):
    data = raw["Close"]
else:
    data = raw[["Close"]]

data = data.ffill().dropna(how="all")

# ----------------------------------------
# 2b. Download benchmark data (SPY and 60/40)
# ----------------------------------------
_bench_raw = yf.download(["SPY", "VTI", "BND"], start=start_date, end=end_date,
                         auto_adjust=True, progress=False)
if isinstance(_bench_raw.columns, pd.MultiIndex):
    bench_data = _bench_raw["Close"]
else:
    bench_data = _bench_raw[["Close"]]
bench_data = bench_data.ffill()

warnings = []
for t in tickers:
    if t not in data.columns:
        msg = f"No data for {t} — excluded from portfolio."
        print(f"WARNING: {msg}")
        warnings.append(msg)
    else:
        first = data[t].first_valid_index()
        if first is not None and first > pd.Timestamp("2007-01-01"):
            msg = f"{t} history starts {first.date()} — early stress windows may be affected."
            print(f"NOTE: {msg}")
            warnings.append(msg)

active      = [t for t in tickers if t in data.columns]
data        = data[active]
weights_vec = np.array([weights[t] for t in active])
weights_vec = weights_vec / weights_vec.sum()

# ----------------------------------------
# 3. Compute returns
# ----------------------------------------
returns          = data.pct_change().dropna()
portfolio_returns = (returns * weights_vec).sum(axis=1)
portfolio_cum     = (1 + portfolio_returns).cumprod()
drawdown_series   = portfolio_cum / portfolio_cum.cummax() - 1

# Benchmark returns aligned to portfolio dates
_bench_ret = bench_data.pct_change().reindex(portfolio_returns.index).ffill()

# SPY
spy_ret = _bench_ret["SPY"].dropna() if "SPY" in _bench_ret.columns else pd.Series(dtype=float)
spy_cum = (1 + spy_ret).cumprod()

# 60/40 blend (VTI 60% + BND 40%)
b6040_ret = pd.Series(dtype=float)
if "VTI" in _bench_ret.columns and "BND" in _bench_ret.columns:
    _b = _bench_ret[["VTI", "BND"]].dropna()
    b6040_ret = _b["VTI"] * 0.6 + _b["BND"] * 0.4
b6040_cum = (1 + b6040_ret).cumprod()

# ---- VaR / CVaR (historical, 95% confidence, 1-day) ----
var_95  = float(portfolio_returns.quantile(0.05))
cvar_95 = float(portfolio_returns[portfolio_returns <= var_95].mean())

# ---- Best / worst 10 single trading days ----
best10  = portfolio_returns.nlargest(10).reset_index()
worst10 = portfolio_returns.nsmallest(10).reset_index()
best10.columns  = ["Date", "Return"]
worst10.columns = ["Date", "Return"]

# ---- Rolling 12-month (monthly) returns heatmap data ----
try:
    monthly_ret = portfolio_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
except ValueError:
    monthly_ret = portfolio_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)

month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
heat_years  = sorted(monthly_ret.index.year.unique())
heat_z, heat_text = [], []
for yr in heat_years:
    row_z, row_t = [], []
    for mo in range(1, 13):
        mask = (monthly_ret.index.year == yr) & (monthly_ret.index.month == mo)
        if mask.any():
            v = float(monthly_ret[mask].iloc[0]) * 100
            row_z.append(v)
            row_t.append(f"{v:+.1f}%")
        else:
            row_z.append(None)
            row_t.append("")
    heat_z.append(row_z)
    heat_text.append(row_t)

# ---- Correlation matrix ----
corr_matrix = returns.corr()
corr_tickers = corr_matrix.columns.tolist()
corr_z    = corr_matrix.values.tolist()
corr_text = [
    [f"{corr_matrix.iloc[r, c]:.2f}" for c in range(len(corr_tickers))]
    for r in range(len(corr_tickers))
]

# ---- Rolling 252-day Sharpe ----
roll_window = 252
roll_sharpe = (
    portfolio_returns.rolling(roll_window).mean() /
    portfolio_returns.rolling(roll_window).std()
) * np.sqrt(roll_window)
roll_sharpe = roll_sharpe.dropna()

# ----------------------------------------
# 4. Stat helpers
# ----------------------------------------
def max_drawdown(series):
    return (series / series.cummax() - 1).min()

def cagr(series):
    years = (series.index[-1] - series.index[0]).days / 365.25
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1

def annual_vol(ret, periods=252):
    return ret.std() * np.sqrt(periods)

def sharpe(ret, risk_free=0.0, periods=252):
    excess = ret - risk_free / periods
    return float("nan") if excess.std() == 0 else (excess.mean() / excess.std()) * np.sqrt(periods)

def get_stress_stats(start, end):
    sub     = portfolio_cum.loc[start:end]
    sub_ret = portfolio_returns.loc[start:end]
    if len(sub) == 0:
        return None
    return {
        "actual_start": sub.index[0].date(),
        "actual_end":   sub.index[-1].date(),
        "total_return":  sub.iloc[-1] / sub.iloc[0] - 1,
        "max_drawdown":  max_drawdown(sub),
        "annual_vol":    annual_vol(sub_ret),
        "sharpe":        sharpe(sub_ret),
    }

# ----------------------------------------
# 5. Compute all stats
# ----------------------------------------
full_stats = {
    "cagr":       cagr(portfolio_cum),
    "max_dd":     max_drawdown(portfolio_cum),
    "vol":        annual_vol(portfolio_returns),
    "sharpe":     sharpe(portfolio_returns),
    "var_95":     var_95,
    "cvar_95":    cvar_95,
    "start_date": data.index[0].date(),
    "end_date":   data.index[-1].date(),
}

def _safe_cagr(cum): return cagr(cum) if len(cum) > 1 else float("nan")
def _safe_dd(cum):   return max_drawdown(cum) if len(cum) > 1 else float("nan")

spy_stats = {
    "cagr":   _safe_cagr(spy_cum),
    "max_dd": _safe_dd(spy_cum),
    "sharpe": sharpe(spy_ret),
}
b6040_stats = {
    "cagr":   _safe_cagr(b6040_cum),
    "max_dd": _safe_dd(b6040_cum),
    "sharpe": sharpe(b6040_ret),
}

# ----------------------------------------
# 5b. Portfolio grade / rating
# ----------------------------------------
def _grade_sharpe(s):
    if s >= 1.5: return 30, "Excellent Sharpe ratio"
    if s >= 1.0: return 24, "Good risk-adjusted return"
    if s >= 0.7: return 18, "Acceptable risk-adjusted return"
    if s >= 0.4: return 12, "Below-average risk-adjusted return"
    if s >= 0.0: return  6, "Poor risk-adjusted return"
    return 0, "Negative risk-adjusted return"

def _grade_alpha(a):
    if a >= 0.03: return 20, f"Strong outperformance vs SPY (+{a:.1%})"
    if a >= 0.01: return 16, f"Moderate outperformance vs SPY (+{a:.1%})"
    if a >= 0.00: return 12, f"Roughly in-line with SPY ({a:+.1%})"
    if a >= -0.02: return 6, f"Slight underperformance vs SPY ({a:+.1%})"
    return 0, f"Meaningful underperformance vs SPY ({a:+.1%})"

def _grade_drawdown(dd):
    if dd >= -0.10: return 20, "Very shallow max drawdown"
    if dd >= -0.20: return 16, "Controlled max drawdown"
    if dd >= -0.30: return 12, "Moderate max drawdown"
    if dd >= -0.40: return  6, "Deep max drawdown"
    return 2, "Severe max drawdown"

def _grade_vol(v):
    if v <= 0.10: return 15, "Very low volatility"
    if v <= 0.15: return 12, "Low-to-moderate volatility"
    if v <= 0.20: return  9, "Moderate volatility"
    if v <= 0.25: return  5, "Above-average volatility"
    return 2, "High volatility"

def _grade_var(var):
    if var >= -0.010: return 15, "Very low daily tail risk (VaR)"
    if var >= -0.015: return 12, "Low daily tail risk (VaR)"
    if var >= -0.020: return  9, "Moderate daily tail risk (VaR)"
    if var >= -0.025: return  5, "Elevated daily tail risk (VaR)"
    return 2, "High daily tail risk (VaR)"

_spy_alpha         = full_stats['cagr'] - spy_stats['cagr']
_pts_sharpe, _txt_sharpe = _grade_sharpe(full_stats['sharpe'])
_pts_alpha,  _txt_alpha  = _grade_alpha(_spy_alpha)
_pts_dd,     _txt_dd     = _grade_drawdown(full_stats['max_dd'])
_pts_vol,    _txt_vol    = _grade_vol(full_stats['vol'])
_pts_var,    _txt_var    = _grade_var(full_stats['var_95'])

grade_score = _pts_sharpe + _pts_alpha + _pts_dd + _pts_vol + _pts_var  # 0-100

if   grade_score >= 90: grade_letter, grade_color = "A+", "#10b981"
elif grade_score >= 80: grade_letter, grade_color = "A",  "#10b981"
elif grade_score >= 70: grade_letter, grade_color = "B+", "#6366f1"
elif grade_score >= 60: grade_letter, grade_color = "B",  "#6366f1"
elif grade_score >= 50: grade_letter, grade_color = "C+", "#f59e0b"
elif grade_score >= 40: grade_letter, grade_color = "C",  "#f59e0b"
elif grade_score >= 30: grade_letter, grade_color = "D",  "#ef4444"
else:                   grade_letter, grade_color = "F",  "#ef4444"

grade_breakdown = [
    ("Sharpe",     _pts_sharpe, 30, _txt_sharpe),
    ("Alpha",      _pts_alpha,  20, _txt_alpha),
    ("Drawdown",   _pts_dd,     20, _txt_dd),
    ("Volatility", _pts_vol,    15, _txt_vol),
    ("Tail Risk",  _pts_var,    15, _txt_var),
]
_grade_bullets_html = "".join(f"<li>{t}</li>" for _, _, _, t in grade_breakdown)
_grade_bars_html = "".join(
    f'<div class="gr-row"><span class="gr-label">{name}</span>'
    f'<div class="gr-bar"><div class="gr-bar-fill" style="width:{pts/mx*100:.0f}%"></div></div>'
    f'<span class="gr-pts">{pts}/{mx}</span></div>'
    for name, pts, mx, _ in grade_breakdown
)

stress_windows = [
    ("2008 GFC",         "2007-10-01", "2009-03-31"),
    ("COVID Crash 2020", "2020-02-01", "2020-04-30"),
    ("Rate Shock 2022",  "2022-01-01", "2022-10-31"),
]
stress_results = [(lbl, s, e, get_stress_stats(s, e)) for lbl, s, e in stress_windows]

# ----------------------------------------
# 6. Terminal output
# ----------------------------------------
LINE = "=" * 50
print(f"\n{LINE}")
print(f"  {portfolio_name}")
print(f"  FULL PERIOD  ({full_stats['start_date']} \u2192 {full_stats['end_date']})")
print(LINE)
print(f"  CAGR           : {full_stats['cagr']:>8.2%}")
print(f"  Max Drawdown   : {full_stats['max_dd']:>8.2%}")
print(f"  Annualized Vol : {full_stats['vol']:>8.2%}")
print(f"  Sharpe Ratio   : {full_stats['sharpe']:>8.2f}")
print(f"  VaR  (95% 1d)  : {full_stats['var_95']:>8.2%}")
print(f"  CVaR (95% 1d)  : {full_stats['cvar_95']:>8.2%}")
print(f"\n  Benchmarks (same period):")
print(f"    SPY   CAGR {spy_stats['cagr']:.2%}  MaxDD {spy_stats['max_dd']:.2%}  Sharpe {spy_stats['sharpe']:.2f}")
print(f"    60/40 CAGR {b6040_stats['cagr']:.2%}  MaxDD {b6040_stats['max_dd']:.2%}  Sharpe {b6040_stats['sharpe']:.2f}")
print(f"\n  Weights (normalized):")
for t, w in zip(active, weights_vec):
    print(f"    {t:<8}: {w:.2%}")

for label, _s, _e, stats in stress_results:
    if stats is None:
        print(f"\n{label}: No data in this window.")
        continue
    print(f"\n{LINE}")
    print(f"  {label}  ({stats['actual_start']} \u2192 {stats['actual_end']})")
    print(LINE)
    print(f"  Total Return   : {stats['total_return']:>8.2%}")
    print(f"  Max Drawdown   : {stats['max_drawdown']:>8.2%}")
    print(f"  Annualized Vol : {stats['annual_vol']:>8.2%}")
    print(f"  Sharpe Ratio   : {stats['sharpe']:>8.2f}")

# ----------------------------------------
# 7. Build Plotly figures
# ----------------------------------------
ACCENT      = "#6366f1"
DANGER      = "#ef4444"
STRESS_CLRS = ["#f59e0b", "#10b981", "#3b82f6"]
GRID_CLR    = "rgba(255,255,255,0.06)"
BG          = "#0f172a"
CARD_BG     = "#1e293b"
TEXT        = "#e2e8f0"

LAYOUT_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=CARD_BG,
    font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=13),
    margin=dict(l=60, r=30, t=50, b=50),
    xaxis=dict(gridcolor=GRID_CLR, zeroline=False, showspikes=True,
               spikecolor=ACCENT, spikethickness=1),
    yaxis=dict(gridcolor=GRID_CLR, zeroline=False),
    hovermode="x unified",
)

# 7a. Cumulative growth with stress window shading
fig_growth = go.Figure()
shade_meta = [
    ("2007-10-01", "2009-03-31", "rgba(239,68,68,0.12)",   "2008 GFC"),
    ("2020-02-01", "2020-04-30", "rgba(245,158,11,0.12)",  "COVID 2020"),
    ("2022-01-01", "2022-10-31", "rgba(59,130,246,0.12)",  "Rate Shock 2022"),
]
for sd, ed, sc, sl in shade_meta:
    fig_growth.add_vrect(x0=sd, x1=ed, fillcolor=sc, line_width=0,
                         annotation_text=sl, annotation_position="top left",
                         annotation_font=dict(size=11, color=TEXT))

# SPY benchmark trace
if len(spy_cum) > 0:
    _spy_norm = spy_cum / spy_cum.iloc[0]
    fig_growth.add_trace(go.Scatter(
        x=_spy_norm.index, y=_spy_norm.values,
        mode="lines", name="SPY",
        line=dict(color="#f59e0b", width=1.8, dash="dot"),
        hovertemplate="%{x|%b %d %Y}<br>SPY: %{y:.3f}<extra></extra>",
    ))

# 60/40 benchmark trace
if len(b6040_cum) > 0:
    _b6040_norm = b6040_cum / b6040_cum.iloc[0]
    fig_growth.add_trace(go.Scatter(
        x=_b6040_norm.index, y=_b6040_norm.values,
        mode="lines", name="60/40 (VTI+BND)",
        line=dict(color="#10b981", width=1.8, dash="dash"),
        hovertemplate="%{x|%b %d %Y}<br>60/40: %{y:.3f}<extra></extra>",
    ))

fig_growth.add_trace(go.Scatter(
    x=portfolio_cum.index, y=portfolio_cum.values,
    mode="lines", name="Portfolio",
    line=dict(color=ACCENT, width=2.5),
    fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
    hovertemplate="%{x|%b %d %Y}<br>Value: %{y:.3f}<extra></extra>",
))
fig_growth.update_layout(
    **LAYOUT_BASE,
    title=dict(text=f"Cumulative Growth — {portfolio_name}", font=dict(size=18, color=TEXT)),
    yaxis_title="Cumulative Return (base = 1.0)",
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)

# 7b. Drawdown chart
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=drawdown_series.index, y=drawdown_series.values * 100,
    mode="lines", name="Drawdown",
    line=dict(color=DANGER, width=2),
    fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
    hovertemplate="%{x|%b %d %Y}<br>Drawdown: %{y:.2f}%<extra></extra>",
))
fig_dd.update_layout(
    **LAYOUT_BASE,
    title=dict(text="Portfolio Drawdown (%)", font=dict(size=18, color=TEXT)),
    yaxis_title="Drawdown (%)", yaxis_ticksuffix="%",
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)

# 7c. Stress windows grouped bar chart
stress_labels_plot, total_rets, max_dds, vols = [], [], [], []
for label, _s, _e, stats in stress_results:
    if stats is None:
        continue
    stress_labels_plot.append(label)
    total_rets.append(stats["total_return"] * 100)
    max_dds.append(stats["max_drawdown"] * 100)
    vols.append(stats["annual_vol"] * 100)

fig_stress = go.Figure()
for metric_name, vals, color in [
    ("Total Return (%)",    total_rets, STRESS_CLRS[1]),
    ("Max Drawdown (%)",    max_dds,   DANGER),
    ("Ann. Volatility (%)", vols,      STRESS_CLRS[2]),
]:
    fig_stress.add_trace(go.Bar(
        name=metric_name, x=stress_labels_plot, y=vals,
        marker_color=color,
        hovertemplate=f"<b>%{{x}}</b><br>{metric_name}: %{{y:.2f}}%<extra></extra>",
    ))
fig_stress.update_layout(
    **LAYOUT_BASE,
    title=dict(text="Stress Window Comparison", font=dict(size=18, color=TEXT)),
    barmode="group", yaxis_ticksuffix="%", yaxis_title="Value (%)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# 7d. Allocation donut
fig_alloc = go.Figure(go.Pie(
    labels=active,
    values=(weights_vec * 100).tolist(),
    hole=0.55,
    textinfo="label+percent",
    textfont=dict(size=13, color=TEXT),
    marker=dict(
        colors=["#6366f1","#8b5cf6","#a855f7","#ec4899","#3b82f6","#10b981","#f59e0b"],
        line=dict(color=BG, width=2),
    ),
    hovertemplate="<b>%{label}</b><br>Weight: %{value:.2f}%<extra></extra>",
))
fig_alloc.update_layout(
    paper_bgcolor=BG,
    font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=13),
    title=dict(text="Portfolio Allocation", font=dict(size=18, color=TEXT)),
    margin=dict(l=20, r=20, t=60, b=20),
    showlegend=True,
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

# 7e. Rolling monthly returns heatmap
fig_heatmap = go.Figure(go.Heatmap(
    z=heat_z,
    x=month_names,
    y=[str(y) for y in heat_years],
    text=heat_text,
    texttemplate="%{text}",
    textfont=dict(size=11),
    colorscale=[
        [0.0,  "#7f1d1d"],
        [0.35, "#ef4444"],
        [0.48, "#fca5a5"],
        [0.50, "#1e293b"],
        [0.52, "#86efac"],
        [0.65, "#10b981"],
        [1.0,  "#064e3b"],
    ],
    zmid=0,
    colorbar=dict(
        title="Return", ticksuffix="%",
        thickness=14, len=0.8,
        tickfont=dict(size=11, color=TEXT),
    ),
    hovertemplate="<b>%{y} %{x}</b><br>Return: %{text}<extra></extra>",
))
fig_heatmap.update_layout(
    paper_bgcolor=BG, plot_bgcolor=CARD_BG,
    font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=12),
    title=dict(text="Monthly Returns Heatmap", font=dict(size=18, color=TEXT)),
    margin=dict(l=60, r=80, t=50, b=40),
    xaxis=dict(side="top", gridcolor=GRID_CLR, tickfont=dict(size=12)),
    yaxis=dict(gridcolor=GRID_CLR, autorange="reversed"),
)

# 7f. Daily returns distribution with VaR / CVaR lines
fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(
    x=portfolio_returns.values * 100,
    nbinsx=80,
    name="Daily Returns",
    marker_color=ACCENT,
    opacity=0.8,
    hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
))
fig_dist.add_vline(
    x=var_95 * 100, line_color=DANGER, line_dash="dash", line_width=2,
    annotation_text=f"VaR 95%: {var_95*100:.2f}%",
    annotation_position="top left",
    annotation_font=dict(color=DANGER, size=12),
)
fig_dist.add_vline(
    x=cvar_95 * 100, line_color="#f87171", line_dash="dot", line_width=2,
    annotation_text=f"CVaR 95%: {cvar_95*100:.2f}%",
    annotation_position="bottom left",
    annotation_font=dict(color="#f87171", size=12),
)
fig_dist.update_layout(
    **LAYOUT_BASE,
    title=dict(text="Daily Returns Distribution & Risk Thresholds", font=dict(size=18, color=TEXT)),
    xaxis_title="Daily Return (%)", xaxis_ticksuffix="%",
    yaxis_title="Frequency",
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    bargap=0.05,
)

# 7g. Correlation heatmap
_corr_palette = [
    [0.0,  "#1e3a5f"],
    [0.25, "#2563eb"],
    [0.45, "#93c5fd"],
    [0.50, "#1e293b"],
    [0.55, "#fca5a5"],
    [0.75, "#ef4444"],
    [1.0,  "#7f1d1d"],
]
fig_corr = go.Figure(go.Heatmap(
    z=corr_z,
    x=corr_tickers,
    y=corr_tickers,
    text=corr_text,
    texttemplate="%{text}",
    textfont=dict(size=12),
    colorscale=_corr_palette,
    zmin=-1, zmax=1, zmid=0,
    colorbar=dict(
        title="\u03c1",
        thickness=14, len=0.85,
        tickvals=[-1, -0.5, 0, 0.5, 1],
        tickfont=dict(size=11, color=TEXT),
    ),
    hovertemplate="<b>%{y} \u2194 %{x}</b><br>Correlation: %{text}<extra></extra>",
))
fig_corr.update_layout(
    paper_bgcolor=BG, plot_bgcolor=CARD_BG,
    font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=13),
    title=dict(text="Ticker Correlation Matrix (Full Period)", font=dict(size=18, color=TEXT)),
    margin=dict(l=80, r=80, t=60, b=60),
    xaxis=dict(gridcolor=GRID_CLR, tickangle=-35),
    yaxis=dict(gridcolor=GRID_CLR, autorange="reversed"),
)

# 7h. Rolling 252-day Sharpe ratio
fig_roll_sharpe = go.Figure()
fig_roll_sharpe.add_hline(
    y=1.0, line_color="#10b981", line_dash="dot", line_width=1.5,
    annotation_text="Sharpe = 1.0",
    annotation_font=dict(color="#10b981", size=11),
)
fig_roll_sharpe.add_hline(
    y=0.0, line_color=DANGER, line_dash="dot", line_width=1.5,
    annotation_text="Sharpe = 0",
    annotation_font=dict(color=DANGER, size=11),
)
fig_roll_sharpe.add_trace(go.Scatter(
    x=roll_sharpe.index, y=roll_sharpe.values,
    mode="lines", name="Rolling Sharpe (252d)",
    line=dict(color=ACCENT, width=2),
    fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
    hovertemplate="%{x|%b %d %Y}<br>Sharpe: %{y:.2f}<extra></extra>",
))
fig_roll_sharpe.update_layout(
    **LAYOUT_BASE,
    title=dict(text="Rolling 252-Day Sharpe Ratio", font=dict(size=18, color=TEXT)),
    yaxis_title="Sharpe Ratio",
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)
json_growth      = pio.to_json(fig_growth,      validate=False)
json_dd          = pio.to_json(fig_dd,          validate=False)
json_stress      = pio.to_json(fig_stress,      validate=False)
json_alloc       = pio.to_json(fig_alloc,       validate=False)
json_heatmap     = pio.to_json(fig_heatmap,     validate=False)
json_dist        = pio.to_json(fig_dist,        validate=False)
json_corr        = pio.to_json(fig_corr,        validate=False)
json_roll_sharpe = pio.to_json(fig_roll_sharpe, validate=False)

# ----------------------------------------
# 9. HTML helpers
# ----------------------------------------
def pct(v):       return f"{v:+.2f}%" if v == v else "N/A"
def pct_plain(v): return f"{v:.2f}%"  if v == v else "N/A"
def num(v, d=2):  return f"{v:.{d}f}" if v == v else "N/A"
def color_class(v, positive_good=True):
    if v != v: return ""
    return "positive" if (v > 0) == positive_good else "negative"

stress_rows_html = ""
for label, _s, _e, stats in stress_results:
    if stats is None:
        stress_rows_html += (
            f'<tr><td><span class="badge badge-warn">{label}</span></td>'
            f'<td colspan="4" class="muted">No data available for this window</td></tr>'
        )
        continue
    stress_rows_html += f"""
        <tr>
          <td><strong>{label}</strong><br>
              <small class="muted">{stats['actual_start']} \u2192 {stats['actual_end']}</small></td>
          <td class="{color_class(stats['total_return'], True)}">{pct(stats['total_return']*100)}</td>
          <td class="{color_class(stats['max_drawdown'], False)}">{pct(stats['max_drawdown']*100)}</td>
          <td>{pct_plain(stats['annual_vol']*100)}</td>
          <td class="{color_class(stats['sharpe'], True)}">{num(stats['sharpe'])}</td>
        </tr>"""

weights_rows_html = ""
for t, w in zip(active, weights_vec):
    bar_w = int(w * 100 * 2)
    weights_rows_html += f"""
        <tr>
          <td><strong>{t}</strong></td>
          <td>{pct_plain(w*100)}</td>
          <td><div class="bar-track"><div class="bar-fill" style="width:{bar_w}%"></div></div></td>
        </tr>"""

warn_html = ""
if warnings:
    items = "".join(f'<div class="warn-item">\u26a0 {w}</div>' for w in warnings)
    warn_html = f'<div class="warn-box">{items}</div>'

def _days_rows(df, positive_good):
    rows = ""
    for _, row in df.iterrows():
        d = row["Date"].strftime("%b %d, %Y") if hasattr(row["Date"], "strftime") else str(row["Date"])[:10]
        v = row["Return"] * 100
        cls = color_class(v, positive_good)
        rows += f'<tr><td>{d}</td><td class="{cls}"><strong>{pct(v)}</strong></td></tr>'
    return rows

best10_rows  = _days_rows(best10,  positive_good=True)
worst10_rows = _days_rows(worst10, positive_good=False)

generated_at = datetime.now().strftime("%B %d, %Y at %I:%M %p")

# ----------------------------------------
# 10. Generate HTML report
# ----------------------------------------
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{portfolio_name} \u2014 Stress Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0f172a;--surface:#1e293b;--surface2:#273348;--border:rgba(255,255,255,0.07);
  --accent:#6366f1;--accent2:#818cf8;--green:#10b981;--red:#ef4444;--amber:#f59e0b;
  --text:#e2e8f0;--muted:#94a3b8;--radius:14px;--shadow:0 4px 24px rgba(0,0,0,0.4);
  --font:"Inter",system-ui,-apple-system,sans-serif;
}}
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
body{{background:var(--bg);color:var(--text);font-family:var(--font);min-height:100vh;padding:0 0 60px}}
.header{{background:linear-gradient(135deg,#1e1b4b 0%,#0f172a 60%);border-bottom:1px solid var(--border);padding:40px 48px 32px}}
.header-inner{{max-width:1280px;margin:0 auto;display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:16px}}
.header h1{{font-size:clamp(22px,3vw,34px);font-weight:700;background:linear-gradient(90deg,#a5b4fc,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.header .subtitle{{color:var(--muted);font-size:14px;margin-top:6px}}
.badge{{display:inline-block;padding:4px 12px;border-radius:999px;font-size:12px;font-weight:600;letter-spacing:.04em}}
.badge-accent{{background:rgba(99,102,241,.2);color:var(--accent2);border:1px solid rgba(99,102,241,.3)}}
.badge-warn{{background:rgba(245,158,11,.15);color:var(--amber);border:1px solid rgba(245,158,11,.25)}}
.main{{max-width:1280px;margin:0 auto;padding:40px 48px 0}}
.section-title{{font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--accent2);margin-bottom:16px}}
.stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px;margin-bottom:40px}}
.stat-card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:24px 24px 20px;box-shadow:var(--shadow);position:relative;overflow:hidden;transition:transform .2s,box-shadow .2s}}
.stat-card:hover{{transform:translateY(-2px);box-shadow:0 8px 32px rgba(0,0,0,.5)}}
.stat-card::before{{content:"";position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--accent),var(--accent2));border-radius:var(--radius) var(--radius) 0 0}}
.stat-card.danger::before{{background:linear-gradient(90deg,var(--red),#f87171)}}
.stat-card.success::before{{background:linear-gradient(90deg,var(--green),#34d399)}}
.stat-card .label{{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}}
.stat-card .value{{font-size:30px;font-weight:700;line-height:1}}
.stat-card .sub{{font-size:12px;color:var(--muted);margin-top:6px}}
.positive{{color:var(--green)}}.negative{{color:var(--red)}}.muted{{color:var(--muted)}}
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:40px}}
.chart-full{{grid-column:1 / -1}}
.chart-card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:8px;box-shadow:var(--shadow)}}
.table-card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden;margin-bottom:40px}}
.table-card table{{width:100%;border-collapse:collapse;font-size:14px}}
.table-card thead th{{background:var(--surface2);padding:14px 20px;text-align:left;font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--border)}}
.table-card tbody tr{{border-bottom:1px solid var(--border);transition:background .15s}}
.table-card tbody tr:last-child{{border-bottom:none}}
.table-card tbody tr:hover{{background:rgba(255,255,255,.03)}}
.table-card tbody td{{padding:16px 20px;vertical-align:middle}}
.bar-track{{background:rgba(255,255,255,.06);border-radius:999px;height:8px;width:160px}}
.bar-fill{{background:linear-gradient(90deg,var(--accent),var(--accent2));height:100%;border-radius:999px}}
.warn-box{{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.2);border-radius:var(--radius);padding:16px 20px;margin-bottom:40px;display:flex;flex-direction:column;gap:8px}}
.warn-item{{font-size:13px;color:var(--amber)}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.two-col-header{{padding:14px 20px;border-bottom:1px solid var(--border);font-size:13px;font-weight:600}}
.grade-card{{grid-column:1 / -1;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:28px 32px;box-shadow:var(--shadow);position:relative;overflow:hidden;display:flex;align-items:flex-start;gap:40px;flex-wrap:wrap;transition:transform .2s,box-shadow .2s}}
.grade-card:hover{{transform:translateY(-2px);box-shadow:0 8px 32px rgba(0,0,0,.5)}}
.grade-card::before{{content:"";position:absolute;top:0;left:0;right:0;height:3px;background:var(--gc,var(--accent));border-radius:var(--radius) var(--radius) 0 0}}
.grade-letter{{font-size:88px;font-weight:800;line-height:1;color:var(--gc,var(--accent));min-width:96px;text-align:center}}
.grade-title-label{{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:8px;text-align:center}}
.grade-meta{{flex:1;min-width:200px}}
.grade-meta .label{{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}}
.grade-score-line{{font-size:15px;font-weight:700;color:var(--gc,var(--accent));margin-bottom:14px}}
.grade-bullets{{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:6px}}
.grade-bullets li{{font-size:13px;color:var(--muted)}}
.grade-bullets li::before{{content:"\203a  ";color:var(--gc,var(--accent))}}
.grade-breakdown{{display:flex;flex-direction:column;gap:10px;min-width:240px}}
.gr-section-label{{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:2px}}
.gr-row{{display:grid;grid-template-columns:90px 1fr 44px;align-items:center;gap:10px}}
.gr-label{{font-size:12px;color:var(--muted)}}
.gr-bar{{height:6px;border-radius:999px;background:rgba(255,255,255,.07);overflow:hidden}}
.gr-bar-fill{{height:100%;border-radius:999px;background:var(--gc,var(--accent))}}
.gr-pts{{font-size:12px;color:var(--text);text-align:right;font-weight:600}}
.footer{{text-align:center;color:var(--muted);font-size:12px;margin-top:60px;padding-top:24px;border-top:1px solid var(--border)}}
@media(max-width:768px){{.header{{padding:28px 20px}}.main{{padding:24px 20px 0}}.chart-grid{{grid-template-columns:1fr}}.two-col{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="header">
  <div class="header-inner">
    <div>
      <h1>{portfolio_name} \u2014 Stress Report</h1>
      <p class="subtitle">Full period: {full_stats['start_date']} \u2192 {full_stats['end_date']} &nbsp;\u00b7&nbsp; Generated {generated_at}</p>
    </div>
    <span class="badge badge-accent">Portfolio Stress Engine</span>
  </div>
</div>
<div class="main">
  {warn_html}
  <p class="section-title">Full-Period Metrics</p>
  <div class="stat-grid">
    <div class="stat-card success">
      <div class="label">CAGR</div>
      <div class="value positive">{pct_plain(full_stats['cagr']*100)}</div>
      <div class="sub">Compound Annual Growth Rate</div>
    </div>
    <div class="stat-card danger">
      <div class="label">Max Drawdown</div>
      <div class="value negative">{pct(full_stats['max_dd']*100)}</div>
      <div class="sub">Peak-to-trough decline</div>
    </div>
    <div class="stat-card">
      <div class="label">Annualized Volatility</div>
      <div class="value">{pct_plain(full_stats['vol']*100)}</div>
      <div class="sub">Daily std \u00d7 \u221a252</div>
    </div>
    <div class="stat-card">
      <div class="label">Sharpe Ratio</div>
      <div class="value {color_class(full_stats['sharpe'], True)}">{num(full_stats['sharpe'])}</div>
      <div class="sub">Risk-adjusted return (rf\u00a0=\u00a00)</div>
    </div>
    <div class="stat-card danger">
      <div class="label">VaR (95%, 1-Day)</div>
      <div class="value negative">{pct(full_stats['var_95']*100)}</div>
      <div class="sub">Historical worst day at 95% confidence</div>
    </div>
    <div class="stat-card danger">
      <div class="label">CVaR (95%, 1-Day)</div>
      <div class="value negative">{pct(full_stats['cvar_95']*100)}</div>
      <div class="sub">Avg loss beyond VaR (tail risk)</div>
    </div>
    <div class="stat-card">
      <div class="label">Alpha vs SPY</div>
      <div class="value {color_class((full_stats['cagr'] - spy_stats['cagr']), True)}">{pct((full_stats['cagr'] - spy_stats['cagr'])*100)}</div>
      <div class="sub">CAGR outperformance vs S&amp;P 500</div>
    </div>
    <div class="stat-card">
      <div class="label">Alpha vs 60/40</div>
      <div class="value {color_class((full_stats['cagr'] - b6040_stats['cagr']), True)}">{pct((full_stats['cagr'] - b6040_stats['cagr'])*100)}</div>
      <div class="sub">CAGR outperformance vs 60/40 blend</div>
    </div>
    <div class="grade-card" style="--gc:{grade_color}">
      <div>
        <div class="grade-title-label">Portfolio Grade</div>
        <div class="grade-letter">{grade_letter}</div>
      </div>
      <div class="grade-meta">
        <div class="label">What drove this grade</div>
        <div class="grade-score-line">Score: {grade_score} / 100</div>
        <ul class="grade-bullets">{_grade_bullets_html}</ul>
      </div>
      <div class="grade-breakdown">
        <div class="gr-section-label">Component Scores</div>
        {_grade_bars_html}
      </div>
    </div>
  </div>
  <p class="section-title">Performance Charts</p>
  <div class="chart-grid">
    <div class="chart-card chart-full"><div id="chart-growth" style="height:420px"></div></div>
    <div class="chart-card"><div id="chart-dd" style="height:340px"></div></div>
    <div class="chart-card"><div id="chart-alloc" style="height:340px"></div></div>
  </div>
  <p class="section-title">Stress Window Analysis</p>
  <div class="chart-grid" style="margin-bottom:20px">
    <div class="chart-card chart-full"><div id="chart-stress" style="height:360px"></div></div>
  </div>
  <div class="table-card">
    <table>
      <thead>
        <tr>
          <th>Period</th><th>Total Return</th><th>Max Drawdown</th>
          <th>Ann. Volatility</th><th>Sharpe</th>
        </tr>
      </thead>
      <tbody>{stress_rows_html}</tbody>
    </table>
  </div>
  <p class="section-title">Risk Analysis</p>
  <div class="chart-grid" style="margin-bottom:40px">
    <div class="chart-card chart-full"><div id="chart-heatmap" style="height:min(520px, calc(28px * {len(heat_years)} + 100px))"></div></div>
    <div class="chart-card chart-full"><div id="chart-dist" style="height:360px"></div></div>
  </div>
  <p class="section-title">Diversification &amp; Risk-Adjusted Quality</p>
  <div class="chart-grid" style="margin-bottom:40px">
    <div class="chart-card"><div id="chart-corr" style="height:420px"></div></div>
    <div class="chart-card"><div id="chart-roll-sharpe" style="height:420px"></div></div>
  </div>
  <p class="section-title">Best &amp; Worst Single Trading Days</p>
  <div class="two-col" style="margin-bottom:40px">
    <div class="table-card" style="margin-bottom:0">
      <div class="two-col-header positive">&#9650; Top 10 Best Days</div>
      <table>
        <thead><tr><th>Date</th><th>Return</th></tr></thead>
        <tbody>{best10_rows}</tbody>
      </table>
    </div>
    <div class="table-card" style="margin-bottom:0">
      <div class="two-col-header negative">&#9660; Top 10 Worst Days</div>
      <table>
        <thead><tr><th>Date</th><th>Return</th></tr></thead>
        <tbody>{worst10_rows}</tbody>
      </table>
    </div>
  </div>
  <p class="section-title">Holdings &amp; Weights</p>
  <div class="table-card" style="margin-bottom:0">
    <table>
      <thead><tr><th>Ticker</th><th>Weight</th><th>Allocation</th></tr></thead>
      <tbody>{weights_rows_html}</tbody>
    </table>
  </div>
  <div class="footer">
    Portfolio Stress Engine &nbsp;\u00b7&nbsp; Data via Yahoo Finance &nbsp;\u00b7&nbsp;
    For informational purposes only \u2014 not financial advice.
  </div>
</div>
<script>
(function(){{
  var cfg={{responsive:true,displayModeBar:'hover',displaylogo:false}};
  Plotly.newPlot('chart-growth',      {json_growth},      cfg);
  Plotly.newPlot('chart-dd',          {json_dd},          cfg);
  Plotly.newPlot('chart-stress',      {json_stress},      cfg);
  Plotly.newPlot('chart-alloc',       {json_alloc},       cfg);
  Plotly.newPlot('chart-heatmap',     {json_heatmap},     cfg);
  Plotly.newPlot('chart-dist',        {json_dist},        cfg);
  Plotly.newPlot('chart-corr',        {json_corr},        cfg);
  Plotly.newPlot('chart-roll-sharpe', {json_roll_sharpe}, cfg);
}})();
</script>
</body>
</html>
"""

# ----------------------------------------
# 11. Write report & open in browser
# ----------------------------------------
with open(report_path, "w", encoding="utf-8") as fh:
    fh.write(HTML)

abs_path = os.path.abspath(report_path)
print(f"\nReport saved \u2192 {abs_path}")
webbrowser.open(f"file:///{abs_path}")
