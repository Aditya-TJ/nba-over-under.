"""Model zoo and hyperparameter tuning on the validation season.

Each entry is (estimator, parameter grid). Every model sits in a pipeline behind a
StandardScaler: needed for KNN, SVM, logistic regression and the ANN, harmless for trees.
"""
import itertools

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.ann import MLPClassifier, MLPRegressor
from src.config import SEED

ANN_GRID = {"hidden": [(64, 32), (128, 64)], "dropout": [0.2, 0.5], "weight_decay": [1e-4, 1e-2]}


def regressors():
    return {
        "Ridge Regression": (Ridge(), {"alpha": [1, 10, 100, 1000]}),
        "KNN": (KNeighborsRegressor(), {"n_neighbors": [25, 50, 100, 200]}),
        "Decision Tree": (DecisionTreeRegressor(random_state=SEED),
                          {"max_depth": [3, 4, 6, 8], "min_samples_leaf": [20, 50, 100]}),
        "Random Forest": (RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED),
                          {"max_depth": [6, 10, None], "min_samples_leaf": [5, 20, 50],
                           "max_features": [0.33, "sqrt"]}),
        "ANN (MLP)": (MLPRegressor(), ANN_GRID),
        "Gradient Boosting*": (HistGradientBoostingRegressor(max_iter=300, random_state=SEED),
                               {"learning_rate": [0.03, 0.1], "max_depth": [3, 6],
                                "l2_regularization": [0.0, 1.0]}),
    }


def classifiers():
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000), {"C": [0.001, 0.01, 0.1, 1]}),
        "SVM (linear)": (LinearSVC(), {"C": [0.0001, 0.001, 0.01, 0.1]}),
        "SVM (polynomial)": (SVC(kernel="poly", random_state=SEED),
                             {"degree": [2, 3], "C": [0.01, 0.1, 1]}),
        "KNN": (KNeighborsClassifier(), {"n_neighbors": [25, 50, 100, 200]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
                          {"max_depth": [2, 3, 4, 6], "min_samples_leaf": [20, 50, 100]}),
        "Random Forest": (RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
                          {"max_depth": [4, 8, None], "min_samples_leaf": [20, 50],
                           "max_features": [0.33, "sqrt"]}),
        "ANN (MLP)": (MLPClassifier(), ANN_GRID),
        "Gradient Boosting*": (HistGradientBoostingClassifier(max_iter=300, random_state=SEED),
                               {"learning_rate": [0.03, 0.1], "max_depth": [3, 6],
                                "l2_regularization": [0.0, 1.0]}),
    }


def build(estimator, params):
    est = clone(estimator).set_params(**params)
    return make_pipeline(StandardScaler(), est)


def score_for_tuning(model, X, y, task):
    """Lower is better. Regression: MSE. Classification: 1 - AUC (works for SVMs too)."""
    if task == "regression":
        return mean_squared_error(y, model.predict(X))
    return 1 - roc_auc_score(y, scores(model, X))


def scores(model, X):
    """Continuous score for the positive class: probability if available, else margin."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def tune(estimator, grid, X_tr, y_tr, X_val, y_val, task):
    """Grid search: fit on training seasons, score on the validation season."""
    keys = list(grid)
    best_params, best_score = None, np.inf
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        model = build(estimator, params).fit(X_tr, y_tr)
        s = score_for_tuning(model, X_val, y_val, task)
        if s < best_score:
            best_params, best_score = params, s
    return best_params, best_score
