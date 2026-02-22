# Portfolio Stress Engine

A DIY investor's Python tool that pulls historical price data via **yfinance**, stress-tests any portfolio through major market crises, and generates a **self-contained, modern HTML report** with interactive charts.

## Features

| Category | What's included |
|---|---|
| **Performance** | CAGR, Max Drawdown, Annualized Volatility, Sharpe Ratio |
| **Risk metrics** | VaR (95%, 1-day), CVaR (tail risk) |
| **Benchmarks** | SPY and 60/40 (VTI+BND) overlay on growth chart + alpha cards |
| **Stress windows** | 2008 GFC · COVID Crash 2020 · Rate Shock 2022 |
| **Charts** | Cumulative growth · Drawdown · Monthly heatmap · Returns distribution · Correlation matrix · Rolling Sharpe |
| **Tables** | Best/worst 10 days · Stress window stats · Holdings & weights |

## Requirements

```
Python 3.10+
yfinance
pandas
numpy
plotly
```

Install all dependencies:

```bash
pip install yfinance pandas numpy plotly
```

## Usage

### 1. Define your portfolio

Create a plain text file with one holding per line. Two formats are supported:

**Plain format** (`portfolio.txt`):
```
# Ticker  Weight(%)
FZROX    41
XLV      13
SMH      12
SCHD     12
VXUS     10
VRT       4
VGIT      7
```

**Markdown table format** (paste directly from a doc):
```
| Asset | Weight |
| ----- | ------ |
| VTI   | 28%    |
| SCHD  | 12%    |
```

Weights are automatically normalized if they don't sum to 100.

### 2. Run the stress test

```bash
# Default — uses portfolio.txt
python portfolio_stress.py

# Named portfolio — each gets its own report file
python portfolio_stress.py Roth_IRA.txt
python portfolio_stress.py Rollover_IRA.txt
```

### 3. View the report

The report is saved as `<portfolio_name>_report.html` next to your `.txt` file and opens automatically in your default browser.

## Report Sections

1. **KPI cards** — CAGR, Max Drawdown, Vol, Sharpe, VaR, CVaR, Alpha vs SPY, Alpha vs 60/40
2. **Cumulative Growth** — portfolio vs SPY vs 60/40 with stress window shading
3. **Drawdown chart** — full peak-to-trough history
4. **Monthly Returns Heatmap** — year × month calendar grid
5. **Daily Returns Distribution** — histogram with VaR/CVaR threshold lines
6. **Correlation Matrix** — reveals hidden concentration between holdings
7. **Rolling 252-Day Sharpe** — tracks risk-adjusted return quality over time
8. **Stress Window Comparison** — grouped bar chart + detailed table
9. **Best & Worst 10 Days** — side-by-side table
10. **Holdings & Weights** — normalized allocation with inline bar

## Example output

```
FULL PERIOD  (2018-08-03 → 2024-12-30)
  CAGR           :   14.54%
  Max Drawdown   :  -31.17%
  Annualized Vol :   18.78%
  Sharpe Ratio   :     0.82
  VaR  (95% 1d)  :   -1.72%
  CVaR (95% 1d)  :   -2.81%

  Benchmarks:
    SPY   CAGR 13.86%  MaxDD -33.72%  Sharpe 0.76
    60/40 CAGR  8.77%  MaxDD -22.70%  Sharpe 0.72
```

## Notes

- Data is sourced from Yahoo Finance via `yfinance`. Some tickers (e.g. FZROX) have limited history — the script warns you and adjusts automatically.
- The 2008 GFC stress window requires tickers with history back to 2007. Use proxies (e.g. `VTI` instead of `FZROX`) to enable it.
- For informational purposes only — not financial advice.
