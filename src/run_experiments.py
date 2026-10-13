"""Run every experiment and write tables to results/ and figures to results/figures/.

  Exp 1  Predict the total from team stats only (the paper's setup), compare to the line.
  Exp 2  Model against the line: predict (total - line) using stats + line features.
  Exp 3  Classify over vs under directly.
  Betting  Flat and selective betting at -110, threshold chosen on the validation season.
  Extras   Decision-tree depth curve, ANN optimizer comparison, walk-forward validation,
           permutation feature importance.

Usage:  uv run python -m src.run_experiments [--quick]
"""
import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src import plots
from src.ann import OPTIMIZERS, MLPRegressor
from src.config import RESULTS_DIR, SEASONS, SEED
from src.evaluate import (betting_summary, choose_threshold, regression_metrics,
                          selective_curve)
from src.features import line_features, load_dataset, split, stat_features
from src.models import build, classifiers, regressors, scores, tune


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def shrink(grid, quick):
    return {k: v[:1] for k, v in grid.items()} if quick else grid


def pretty(name):
    name = name.replace("h_", "Home ").replace("a_", "Away ").replace("exp_", "Expected ")
    for w in (5, 10, 20):
        name = name.replace(f"_r{w}", f" (last {w})")
    return name.replace("_", " ")


def fit_models(zoo, train, val, trval, X_cols, target, task, quick):
    """Tune each model on the validation season, then refit on train+val.
    Returns {name: (params, val score, model fitted on train only, model fitted on train+val)}."""
    out = {}
    for name, (est, grid) in zoo.items():
        t0 = time.time()
        params, val_score = tune(est, shrink(grid, quick), train[X_cols], train[target],
                                 val[X_cols], val[target], task)
        val_model = build(est, params).fit(train[X_cols], train[target])
        final = build(est, params).fit(trval[X_cols], trval[target])
        out[name] = (params, val_score, val_model, final)
        log(f"  {name:22s} {params}  ({time.time() - t0:.0f}s)")
    return out


def main(quick=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    train, val, test = split(df)
    trval = pd.concat([train, val])
    S, L = stat_features(df), line_features(df)
    SL = S + L
    log(f"train {len(train)}  val {len(val)}  test {len(test)} games; "
        f"{len(S)} stat features, {len(L)} line features")
    plots.scoring_trend(df)

    book = regression_metrics(test.total, test.line)
    tables = {}

    # ---------------- Exp 1: total points from team stats only ----------------
    log("Exp 1: predict total points from team stats (no line)")
    fitted1 = fit_models(regressors(), train, val, trval, S, "total", "regression", quick)
    rows = {"Sportsbook line": {**book, "hit_rate": np.nan, "params": ""}}
    offset = (trval.total - trval.exp_total_r10).mean()
    naive = {
        "League average (last 21 days)": test.league_total_21d,
        "Pace x efficiency formula": test.exp_total_r10 + offset,
    }
    for name, pred in naive.items():
        rows[name] = {**regression_metrics(test.total, pred),
                      "hit_rate": betting_summary(pred > test.line, test.total, test.line)["hit_rate"],
                      "params": ""}
    for name, (params, _, _, model) in fitted1.items():
        pred = model.predict(test[S])
        rows[name] = {**regression_metrics(test.total, pred),
                      "hit_rate": betting_summary(pred > test.line, test.total, test.line)["hit_rate"],
                      "params": str(params)}
    tables["exp1_regression"] = pd.DataFrame(rows).T
    plots.model_rmse(tables["exp1_regression"].drop("Sportsbook line").astype({"RMSE": float}),
                     book["RMSE"], "exp1_rmse.png", "Exp 1: predicting total points without the line")

    # ---------------- Exp 2: model against the line (residual) ----------------
    log("Exp 2: predict (total - line) using stats + line features")
    fitted2 = fit_models(regressors(), train, val, trval, SL, "resid", "regression", quick)
    rows = {"Sportsbook line": {**book, "hit_rate": np.nan, "params": ""}}
    prob_models = {}   # name -> (P(over) on val from train-only model, P(over) on test)
    for name, (params, _, val_model, model) in fitted2.items():
        pred = model.predict(test[SL])
        rows[name] = {**regression_metrics(test.total, test.line + pred),
                      "hit_rate": betting_summary(pred > 0, test.total, test.line)["hit_rate"],
                      "params": str(params)}
        # Turn the residual prediction into P(over) with a Gaussian error model whose
        # spread is estimated on the validation season.
        val_pred = val_model.predict(val[SL])
        sigma = np.std(val.resid - val_pred)
        prob_models[f"{name} (residual)"] = (norm.cdf(val_pred / sigma), norm.cdf(pred / sigma))
    tables["exp2_vs_line"] = pd.DataFrame(rows).T
    plots.model_rmse(tables["exp2_vs_line"].drop("Sportsbook line").astype({"RMSE": float}),
                     book["RMSE"], "exp2_rmse.png", "Exp 2: line + predicted correction")

    # ---------------- Exp 3: over / under classification ----------------
    log("Exp 3: classify over vs under")
    tr_np, val_np = train[train.push == 0], val[val.push == 0]
    trval_np, test_np = trval[trval.push == 0], test[test.push == 0]
    fitted3 = fit_models(classifiers(), tr_np, val_np, trval_np, SL, "over", "classification", quick)
    rows, rocs = {}, {}
    for name, (params, _, val_model, model) in fitted3.items():
        y_hat = model.predict(test_np[SL])
        s = scores(model, test_np[SL])
        auc = roc_auc_score(test_np.over, s)
        rows[name] = {"accuracy": accuracy_score(test_np.over, y_hat),
                      "precision": precision_score(test_np.over, y_hat, zero_division=0),
                      "recall": recall_score(test_np.over, y_hat, zero_division=0),
                      "F1": f1_score(test_np.over, y_hat, zero_division=0),
                      "AUC": auc, "params": str(params)}
        if hasattr(model, "predict_proba"):
            prob_models[name] = (val_model.predict_proba(val[SL])[:, 1], model.predict_proba(test[SL])[:, 1])
        if name in ("Logistic Regression", "ANN (MLP)", "Random Forest", "Gradient Boosting*"):
            fpr, tpr, _ = roc_curve(test_np.over, s)
            rocs[name] = (fpr, tpr, auc)
    tables["exp3_classification"] = pd.DataFrame(rows).T
    plots.roc_curves(rocs)

    # ---------------- Betting ----------------
    log("Betting evaluation")
    rows, curves = {}, {}
    for name, (p_val, p_test) in prob_models.items():
        flat = betting_summary(p_test > 0.5, test.total.values, test.line.values)
        t = choose_threshold(p_val, val.total.values, val.line.values)
        mask = np.abs(p_test - 0.5) >= t
        sel = betting_summary(p_test[mask] > 0.5, test.total.values[mask], test.line.values[mask])
        rows[name] = {"all_bets": flat["bets"], "all_hit_rate": flat["hit_rate"], "all_roi": flat["roi"],
                      "all_p_vs_breakeven": flat["p_vs_breakeven"],
                      "threshold": t, "sel_bets": sel["bets"], "sel_hit_rate": sel["hit_rate"],
                      "sel_roi": sel["roi"], "sel_roi_ci": f"[{sel['roi_ci_low']:.3f}, {sel['roi_ci_high']:.3f}]",
                      "sel_p_vs_breakeven": sel["p_vs_breakeven"]}
        if name in ("Logistic Regression", "ANN (MLP)", "Ridge Regression (residual)", "Gradient Boosting*"):
            curves[name] = selective_curve(p_test, test.total.values, test.line.values)
    tables["betting"] = pd.DataFrame(rows).T
    plots.selective_betting(curves)

    # ---------------- Extras ----------------
    log("Extra: decision tree depth vs over-fitting")
    depths = list(range(1, 16))
    tr_mse, va_mse = [], []
    for d in depths:
        m = DecisionTreeRegressor(max_depth=d, random_state=SEED).fit(train[S], train.total)
        tr_mse.append(np.mean((m.predict(train[S]) - train.total) ** 2))
        va_mse.append(np.mean((m.predict(val[S]) - val.total) ** 2))
    plots.tree_depth_curve(depths, tr_mse, va_mse)

    log("Extra: ANN optimizer comparison")
    scaler = StandardScaler().fit(train[S])
    Xtr, Xva = scaler.transform(train[S]), scaler.transform(val[S])
    histories, rows = {}, {}
    for opt in OPTIMIZERS:
        t0 = time.time()
        m = MLPRegressor(hidden=(64, 32), dropout=0.2, weight_decay=1e-4, optimizer=opt,
                         epochs=60 if quick else 200, patience=0)
        m.fit(Xtr, train.total, eval_set=(Xva, val.total))
        histories[opt] = m.history_
        best = min(m.history_["val"])
        rows[opt] = {"best_val_RMSE_points": np.sqrt(best) * m.y_std_, "best_epoch": m.best_epoch_,
                     "seconds": round(time.time() - t0, 1)}
    tables["ann_optimizers"] = pd.DataFrame(rows).T
    plots.optimizer_curves(histories)

    log("Extra: walk-forward validation")
    C = fitted3["Logistic Regression"][0]["C"]
    alpha = fitted2["Ridge Regression"][0]["alpha"]
    rows = {}
    for season in SEASONS[2:]:
        past = df[df.season < season]
        nxt = df[df.season == season]
        past_np = past[past.push == 0]
        sc = StandardScaler().fit(past[SL])
        lr = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(past_np[SL]), past_np.over)
        rg = Ridge(alpha=alpha).fit(sc.transform(past[SL]), past.resid)
        p = lr.predict_proba(sc.transform(nxt[SL]))[:, 1]
        r = rg.predict(sc.transform(nxt[SL]))
        rows[season] = {
            "games": len(nxt),
            "Logistic Regression_hit_rate": betting_summary(p > 0.5, nxt.total, nxt.line)["hit_rate"],
            "Ridge (residual)_hit_rate": betting_summary(r > 0, nxt.total, nxt.line)["hit_rate"],
            "line_RMSE": np.sqrt(np.mean((nxt.total - nxt.line) ** 2)),
            "ridge_RMSE": np.sqrt(np.mean((nxt.total - nxt.line - r) ** 2)),
        }
    tables["walk_forward"] = pd.DataFrame(rows).T
    plots.walk_forward(tables["walk_forward"])

    log("Extra: permutation feature importance (Exp 1, best model on validation)")
    best_name = min(fitted1, key=lambda n: fitted1[n][1])
    model = fitted1[best_name][3]
    pi = permutation_importance(model, test[S], test.total, scoring="neg_mean_squared_error",
                                n_repeats=3 if quick else 10, random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=[pretty(c) for c in S])
    tables["feature_importance"] = imp.sort_values(ascending=False).to_frame("mse_increase")
    plots.feature_importance(imp, "feature_importance.png", f"What drives the total? ({best_name}, Exp 1)")

    for name, t in tables.items():
        t.to_csv(RESULTS_DIR / f"{name}.csv", float_format="%.4f")
    write_summary(tables, book, len(test))
    log("done")


def write_summary(tables, book, n_test):
    def md(t, digits=3):
        return t.map(lambda v: f"{v:.{digits}f}" if isinstance(v, float) else v).to_markdown()

    out = [
        "# Results summary", "",
        f"Test set: {n_test} games (2024-25 and 2025-26 up to Jan 2026). "
        f"Sportsbook line test MSE {book['MSE']:.1f} (RMSE {book['RMSE']:.2f}).",
        "`*` = beyond the course syllabus, included for comparison only.", "",
        "## Exp 1: predict total points from team stats only", md(tables["exp1_regression"]), "",
        "## Exp 2: model against the line (predict total - line)", md(tables["exp2_vs_line"]), "",
        "## Exp 3: over / under classification (pushes excluded)", md(tables["exp3_classification"]), "",
        "## Betting at -110 (break-even 52.38%)", md(tables["betting"]), "",
        "## ANN optimizer comparison (Exp 1 setup)", md(tables["ann_optimizers"]), "",
        "## Walk-forward validation", md(tables["walk_forward"]), "",
        "## Top 15 features (permutation importance)", md(tables["feature_importance"].head(15), 2), "",
    ]
    (RESULTS_DIR / "summary.md").write_text("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny grids, for a smoke test")
    main(ap.parse_args().quick)
