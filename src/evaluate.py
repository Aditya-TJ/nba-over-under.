"""Regression metrics and betting evaluation against the sportsbook line."""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import BREAK_EVEN, SEED

WIN_PAYOUT = 100 / 110          # profit on a winning 1-unit bet at -110
THRESHOLDS = np.round(np.arange(0.0, 0.101, 0.01), 3)


def regression_metrics(total, pred_total):
    return {
        "MSE": mean_squared_error(total, pred_total),
        "RMSE": np.sqrt(mean_squared_error(total, pred_total)),
        "MAE": mean_absolute_error(total, pred_total),
    }


def bet_profit(pick_over, total, line):
    """Profit (in units) of a 1-unit bet on each game; pushes are refunded."""
    pick_over = np.asarray(pick_over, bool)
    total, line = np.asarray(total), np.asarray(line)
    won = np.where(pick_over, total > line, total < line)
    push = total == line
    return np.where(push, 0.0, np.where(won, WIN_PAYOUT, -1.0))


def betting_summary(pick_over, total, line):
    profit = bet_profit(pick_over, total, line)
    wins, losses = int((profit > 0).sum()), int((profit < 0).sum())
    decided = wins + losses
    out = {"bets": len(profit), "wins": wins, "losses": losses,
           "hit_rate": wins / decided if decided else np.nan,
           "roi": profit.mean() if len(profit) else np.nan,
           "units": profit.sum()}
    if decided:
        out["p_vs_coinflip"] = binomtest(wins, decided, 0.5, alternative="greater").pvalue
        out["p_vs_breakeven"] = binomtest(wins, decided, BREAK_EVEN, alternative="greater").pvalue
        lo, hi = bootstrap_ci(profit)
        out["roi_ci_low"], out["roi_ci_high"] = lo, hi
    return out


def bootstrap_ci(profit, n_boot=2000, alpha=0.05):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(profit), size=(n_boot, len(profit)))
    means = profit[idx].mean(axis=1)
    return np.quantile(means, alpha / 2), np.quantile(means, 1 - alpha / 2)


def selective_curve(p_over, total, line, thresholds=THRESHOLDS):
    """Bet only when the model is at least `t` away from a coin flip: |P(over) - 0.5| >= t."""
    rows = []
    for t in thresholds:
        mask = np.abs(p_over - 0.5) >= t
        if mask.sum() == 0:
            continue
        s = betting_summary(p_over[mask] > 0.5, total[mask], line[mask])
        rows.append({"threshold": t, **s})
    return pd.DataFrame(rows)


def choose_threshold(p_over_val, total_val, line_val, min_bets=100):
    """Pick the confidence threshold with the best ROI on the validation season."""
    curve = selective_curve(p_over_val, total_val, line_val)
    curve = curve[curve.bets >= min_bets]
    return float(curve.loc[curve.roi.idxmax(), "threshold"])
