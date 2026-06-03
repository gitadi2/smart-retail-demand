"""Forecasting metrics. WMAPE is the headline number used in retail demand."""
from __future__ import annotations

import numpy as np


def _arr(x):
    return np.asarray(x, dtype=float)


def mae(y, yhat):
    return float(np.mean(np.abs(_arr(y) - _arr(yhat))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((_arr(y) - _arr(yhat)) ** 2)))


def r2(y, yhat):
    y, yhat = _arr(y), _arr(yhat)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def mape(y, yhat, eps=1.0):
    """MAPE with a floor on the denominator (zero-demand days are common)."""
    y, yhat = _arr(y), _arr(yhat)
    return float(np.mean(np.abs(y - yhat) / np.maximum(np.abs(y), eps)) * 100)


def wmape(y, yhat):
    """Weighted MAPE = sum|err| / sum|actual|. Robust to zero-demand days."""
    y, yhat = _arr(y), _arr(yhat)
    denom = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - yhat)) / denom * 100) if denom > 0 else float("nan")


def bias(y, yhat):
    """Mean forecast error. Positive => over-forecasting."""
    return float(np.mean(_arr(yhat) - _arr(y)))


def all_metrics(y, yhat) -> dict:
    return {
        "wmape": round(wmape(y, yhat), 3),
        "mae": round(mae(y, yhat), 4),
        "rmse": round(rmse(y, yhat), 4),
        "r2": round(r2(y, yhat), 4),
        "mape": round(mape(y, yhat), 3),
        "bias": round(bias(y, yhat), 4),
    }
