"""
google_trends_alpha.py
======================
Reusable utilities for the Google Trends Alpha strategy.
"""

import time
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_trends(keywords, start, end, geo='IN', sleep=2):
    """
    Fetch weekly Google Trends for a list of keywords via pytrends.
    Batches of 5 (API limit), sleeps between calls to avoid rate limiting.
    """
    from pytrends.request import TrendReq
    pytrends = TrendReq(hl='en-US', tz=330)
    all_dfs  = []
    batches  = [keywords[i:i+5] for i in range(0, len(keywords), 5)]

    for batch in batches:
        pytrends.build_payload(batch, timeframe=f'{start} {end}', geo=geo)
        df = pytrends.interest_over_time()
        if 'isPartial' in df.columns:
            df = df.drop(columns=['isPartial'])
        all_dfs.append(df)
        time.sleep(sleep)

    return pd.concat(all_dfs, axis=1)


def fetch_prices_weekly(tickers, ticker_to_name, start, end):
    """Download weekly Friday-close prices from yfinance."""
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]
    close.columns = [ticker_to_name.get(c, c) for c in close.columns]
    return close.resample('W-FRI').last()


# ---------------------------------------------------------------------------
# Signal Processing
# ---------------------------------------------------------------------------

def rolling_zscore(df, window=52):
    """Normalise each column to rolling z-score."""
    return (df - df.rolling(window).mean()) / df.rolling(window).std()


def rank_signal(row, long_n=3, short_n=3):
    """Long top N, short bottom N, zero otherwise (equal weight)."""
    ranked   = row.rank(ascending=False)
    n        = len(row.dropna())
    weights  = pd.Series(0.0, index=row.index)
    weights[ranked <= long_n]        =  1.0 / long_n
    weights[ranked > (n - short_n)] = -1.0 / short_n
    return weights


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def granger_pvalue(trends_series, returns_series, max_lag=4):
    """
    Min p-value across lags for H0: trends do NOT Granger-cause returns.
    Reject H0 (p < 0.05) → trends have predictive content.
    """
    df = pd.DataFrame({'returns': returns_series,
                       'trends':  trends_series}).dropna()
    if len(df) < max_lag * 3:
        return np.nan
    try:
        result = grangercausalitytests(df[['returns', 'trends']],
                                       maxlag=max_lag, verbose=False)
        pvals = [result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]
        return min(pvals)
    except Exception:
        return np.nan


def compute_ic(signal_df, returns_df):
    """
    Compute weekly Spearman IC between signal and next-period returns.
    Returns a pd.Series indexed by date.
    """
    ic_records = []
    for date in signal_df.index:
        if date not in returns_df.index:
            continue
        s = signal_df.loc[date].dropna()
        r = returns_df.loc[date].dropna()
        common = s.index.intersection(r.index)
        if len(common) < 4:
            continue
        ic, _ = stats.spearmanr(s[common], r[common])
        ic_records.append({'date': date, 'ic': ic})
    df = pd.DataFrame(ic_records).set_index('date')
    return df['ic']


def performance_metrics(returns, label='Strategy', freq=52):
    """Annualised return, vol, Sharpe, max drawdown, Calmar."""
    ann_ret = returns.mean() * freq
    ann_vol = returns.std()  * np.sqrt(freq)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum     = (1 + returns).cumprod()
    max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar  = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    return dict(label=label, ann_ret=ann_ret, ann_vol=ann_vol,
                sharpe=sharpe, max_dd=max_dd, calmar=calmar)
