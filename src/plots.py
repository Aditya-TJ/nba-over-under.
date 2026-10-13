"""Figures for the write-up and slides (static PNGs, light theme).

Colors follow a fixed, colorblind-validated categorical order; text stays in neutral ink.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import BREAK_EVEN, FIG_DIR  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "lines.linewidth": 2, "legend.frameon": False,
    "legend.fontsize": 9, "font.size": 10, "text.color": INK,
})


def _save(fig, name, rect=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(FIG_DIR / name, dpi=160)
    plt.close(fig)


def scoring_trend(df):
    by = df.groupby("season")[["total", "line"]].mean()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(by))
    ax.plot(x, by.total, color=SERIES[0], marker="o", markersize=6, label="Actual total")
    ax.plot(x, by.line, color=SERIES[1], marker="o", markersize=6, label="Sportsbook line")
    ax.set_xticks(x, by.index)
    ax.set_ylabel("Points per game (both teams)")
    ax.set_title("Scoring keeps rising, and the line tracks it")
    ax.legend(loc="upper left")
    _save(fig, "scoring_trend.png")


def model_rmse(table, book_rmse, name, title):
    t = table.sort_values("RMSE")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(t) + 1.2))
    y = np.arange(len(t))
    ax.barh(y, t.RMSE, color=SERIES[0], height=0.6)
    ax.axvline(book_rmse, color=SERIES[1], linewidth=2, linestyle="--")
    ax.text(book_rmse, 1.0, f" Sportsbook line {book_rmse:.2f}", color=INK_2, fontsize=9, va="bottom",
            transform=ax.get_xaxis_transform())
    for yi, v in zip(y, t.RMSE):
        ax.text(v + 0.05, yi, f"{v:.2f}", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(y, t.index)
    ax.invert_yaxis()
    lo = min(t.RMSE.min(), book_rmse)
    ax.set_xlim(lo - 1, t.RMSE.max() + 1)
    ax.set_xlabel("Test RMSE (points), lower is better")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, pad=18)
    _save(fig, name)


def tree_depth_curve(depths, train_mse, val_mse):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(depths, np.sqrt(train_mse), color=SERIES[0], marker="o", markersize=6, label="Training")
    ax.plot(depths, np.sqrt(val_mse), color=SERIES[1], marker="o", markersize=6, label="Validation")
    best = depths[int(np.argmin(val_mse))]
    ax.axvline(best, color=REF, linestyle=":", linewidth=1.5)
    ax.text(best, ax.get_ylim()[1], f" best depth = {best}", color=INK_2, fontsize=9, va="top")
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("RMSE (points)")
    ax.set_title("Decision tree: deeper trees overfit")
    ax.legend(loc="lower left")
    _save(fig, "tree_depth_overfitting.png")


def optimizer_curves(histories):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, (name, h) in enumerate(histories.items()):
        ax.plot(np.arange(1, len(h["val"]) + 1), np.sqrt(np.array(h["val"])), color=SERIES[i], label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE (standardised)")
    ax.set_ylim(top=1.2)
    ax.set_title("ANN optimizers: validation loss per epoch")
    ax.legend(loc="upper right", ncol=5)
    _save(fig, "ann_optimizers.png")


def roc_curves(curves):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=REF, linestyle=":", linewidth=1.5, label="Coin flip (AUC 0.50)")
    for i, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        ax.plot(fpr, tpr, color=SERIES[i], label=f"{name} (AUC {auc:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC: over/under classifiers (test)")
    ax.legend(loc="lower right")
    _save(fig, "roc_curves.png")


def selective_betting(curves):
    """Hit rate and number of bets vs confidence threshold: two panels, one axis each."""
    fig, (top, bot) = plt.subplots(2, 1, figsize=(7, 5.4), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    for i, (name, c) in enumerate(curves.items()):
        top.plot(c.threshold, 100 * c.hit_rate, color=SERIES[i], marker="o", markersize=5, label=name)
        bot.plot(c.threshold, c.bets, color=SERIES[i], marker="o", markersize=5)
    top.axhline(100 * BREAK_EVEN, color=INK_2, linestyle="--", linewidth=1.5)
    top.text(1.0, 100 * BREAK_EVEN, " break-even\n 52.4%", color=INK_2, fontsize=9, va="center",
             transform=top.get_yaxis_transform())
    top.axhline(50, color=REF, linestyle=":", linewidth=1.2)
    top.set_ylabel("Hit rate (%)")
    top.set_title("Selective betting on the test seasons")
    fig.legend(*top.get_legend_handles_labels(), loc="lower center", ncol=4)
    bot.set_ylabel("Bets placed")
    bot.set_xlabel("Minimum confidence |P(over) - 0.5|")
    _save(fig, "selective_betting.png", rect=(0, 0.05, 1, 1))


def walk_forward(table):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(table))
    for i, col in enumerate([c for c in table.columns if c.endswith("hit_rate")]):
        ax.plot(x, 100 * table[col], color=SERIES[i], marker="o", markersize=6,
                label=col.replace("_hit_rate", ""))
    ax.axhline(100 * BREAK_EVEN, color=INK_2, linestyle="--", linewidth=1.5)
    ax.text(1.0, 100 * BREAK_EVEN, " break-even\n 52.4%", color=INK_2, fontsize=9, va="center",
            transform=ax.get_yaxis_transform())
    ax.axhline(50, color=REF, linestyle=":", linewidth=1.2)
    ax.set_xticks(x, table.index)
    ax.set_ylabel("Hit rate (%)")
    ax.set_title("Walk-forward: train on all past seasons, test on the next")
    ax.legend(loc="lower left")
    _save(fig, "walk_forward.png")


def feature_importance(series, name, title):
    s = series.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.arange(len(s)), s.values, color=SERIES[0], height=0.6)
    ax.set_yticks(np.arange(len(s)), s.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Increase in test MSE when the feature is shuffled")
    ax.set_title(title)
    _save(fig, name)
