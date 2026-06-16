# 🔍 Google Trends Alpha — NSE Large Caps

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![pytrends](https://img.shields.io/badge/data-pytrends-orange)](https://github.com/GeneralMills/pytrends)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An alternative data strategy using Google Search volume as a retail investor attention signal on NSE large caps — with Granger causality testing and a market-neutral long-short backtest.

---

## Hypothesis

Retail investor **search interest** on Google is a proxy for attention and sentiment. Stocks experiencing abnormally high search volume may attract capital inflows in subsequent weeks — a measurable behavioural effect.

We test this formally with **Granger causality** and build a **long-short alpha** based on cross-sectional ranking of the signal.

---

## Universe

| Stock | Ticker |
|---|---|
| Reliance Industries | RELIANCE.NS |
| Tata Consultancy Services | TCS.NS |
| HDFC Bank | HDFCBANK.NS |
| Infosys | INFY.NS |
| ICICI Bank | ICICIBANK.NS |
| Wipro | WIPRO.NS |
| State Bank of India | SBIN.NS |
| Bajaj Finance | BAJFINANCE.NS |

---

## Methodology

### 1. Data
- **Google Trends:** Weekly search volume index (0–100) via `pytrends`, geo=IN
- **Prices:** Weekly Friday-close prices via `yfinance`
- **Period:** 2018 – 2024

### 2. Signal Processing
```python
# Rolling z-score normalisation (52-week window)
trends_z = (trends - trends.rolling(52).mean()) / trends.rolling(52).std()

# Signal: last week's z-score predicts this week's return
signal = trends_z.shift(1)
```

### 3. Granger Causality Test
```python
# H0: search volume does NOT Granger-cause returns
# Reject H0 (p < 0.05) → search volume has predictive content
grangercausalitytests(df[['returns', 'trends']], maxlag=4)
```

### 4. Long-Short Portfolio
- **Long:** Top 3 stocks by signal rank
- **Short:** Bottom 3 stocks by signal rank
- **Weighting:** Equal weight within each leg
- **Rebalance:** Weekly

---

## Results

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|---|---|---|---|---|
| Long-Short Trends Alpha | — | — | — | — |
| Equal Weight Buy & Hold | — | — | — | — |

*Run the notebook to populate exact figures.*

**Signal Quality:**
- Mean IC (Spearman): —
- IC IR: —
- % weeks IC > 0: —

---

## Key Concepts

**Granger Causality** tests whether past search volume improves the forecast of returns beyond lagged returns alone. It's a predictive test, not true causality.

**Information Coefficient (IC)** is the Spearman rank correlation between signal and next-period returns — a model-free measure of signal quality independent of position sizing.

**Long-short structure** removes market beta — returns come purely from the cross-sectional ranking, not from market direction.

**Rolling z-score** normalises raw trends (which are relative, 0–100) across stocks so they can be compared cross-sectionally.

---

## Project Structure

```
google-trends-alpha-nse/
├── notebooks/
│   └── google_trends_alpha.ipynb   # Full walkthrough
├── src/
│   └── google_trends_alpha.py      # Reusable utilities
├── data/                           # Cached trends + prices (gitignored)
├── figures/                        # Auto-generated plots
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/sabarishvar/google-trends-alpha-nse.git
cd google-trends-alpha-nse
pip install -r requirements.txt
jupyter notebook notebooks/google_trends_alpha.ipynb
```

> **Note:** pytrends may occasionally hit Google's rate limits. The notebook caches results to `data/` after the first run — subsequent runs load from cache instantly.

---

## Interview Questions

**Q: What does Granger causality actually test?**  
Whether past values of X improve the forecast of Y beyond what past Y alone explains. It's predictive, not causal.

**Q: Why z-score normalise the trends signal?**  
Raw trends are relative (0–100 index within each keyword). Z-scoring puts all stocks on the same scale for cross-sectional comparison.

**Q: What's IC and why does it matter?**  
IC is the Spearman rank correlation between signal and next-period returns. It measures signal quality independently of position sizing. IC > 0.05 is meaningful for weekly signals.

**Q: Why long-short and not just long?**  
Long-short removes market beta — alpha comes purely from cross-sectional signal, not market direction. More portable and easier to evaluate in isolation.

---

## References

- Da, Engelberg & Gao (2011) — *In Search of Attention*, Journal of Finance
- Preis, Moat & Stanley (2013) — *Quantifying Trading Behavior in Financial Markets Using Google Trends*
