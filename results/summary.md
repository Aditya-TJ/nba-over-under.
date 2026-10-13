# Results summary

Test set: 1779 games (2024-25 and 2025-26 up to Jan 2026). Sportsbook line test MSE 320.4 (RMSE 17.90).
`*` = beyond the course syllabus, included for comparison only.

## Exp 1: predict total points from team stats only
|                               |     MSE |   RMSE |    MAE |   hit_rate | params                                                            |
|:------------------------------|--------:|-------:|-------:|-----------:|:------------------------------------------------------------------|
| Sportsbook line               | 320.413 | 17.9   | 14.352 |    nan     |                                                                   |
| League average (last 21 days) | 390.898 | 19.771 | 15.844 |      0.504 |                                                                   |
| Pace x efficiency formula     | 352.253 | 18.768 | 15.037 |      0.51  |                                                                   |
| Ridge Regression              | 347.74  | 18.648 | 14.922 |      0.499 | {'alpha': 1000}                                                   |
| KNN                           | 359.343 | 18.956 | 15.175 |      0.495 | {'n_neighbors': 100}                                              |
| Decision Tree                 | 356.775 | 18.888 | 15.18  |      0.495 | {'max_depth': 3, 'min_samples_leaf': 20}                          |
| Random Forest                 | 348.788 | 18.676 | 14.971 |      0.498 | {'max_depth': None, 'min_samples_leaf': 20, 'max_features': 0.33} |
| ANN (MLP)                     | 348.237 | 18.661 | 14.913 |      0.51  | {'hidden': (64, 32), 'dropout': 0.2, 'weight_decay': 0.01}        |
| Gradient Boosting*            | 351.145 | 18.739 | 15.038 |      0.501 | {'learning_rate': 0.03, 'max_depth': 3, 'l2_regularization': 1.0} |

## Exp 2: model against the line (predict total - line)
|                    |     MSE |   RMSE |    MAE |   hit_rate | params                                                            |
|:-------------------|--------:|-------:|-------:|-----------:|:------------------------------------------------------------------|
| Sportsbook line    | 320.413 | 17.9   | 14.352 |    nan     |                                                                   |
| Ridge Regression   | 322.421 | 17.956 | 14.348 |      0.523 | {'alpha': 1000}                                                   |
| KNN                | 320.814 | 17.911 | 14.349 |      0.51  | {'n_neighbors': 200}                                              |
| Decision Tree      | 326.276 | 18.063 | 14.496 |      0.502 | {'max_depth': 4, 'min_samples_leaf': 20}                          |
| Random Forest      | 320.307 | 17.897 | 14.339 |      0.511 | {'max_depth': 6, 'min_samples_leaf': 5, 'max_features': 'sqrt'}   |
| ANN (MLP)          | 321.864 | 17.941 | 14.331 |      0.523 | {'hidden': (128, 64), 'dropout': 0.5, 'weight_decay': 0.0001}     |
| Gradient Boosting* | 323.42  | 17.984 | 14.389 |      0.525 | {'learning_rate': 0.03, 'max_depth': 3, 'l2_regularization': 0.0} |

## Exp 3: over / under classification (pushes excluded)
|                     |   accuracy |   precision |   recall |    F1 |   AUC | params                                                            |
|:--------------------|-----------:|------------:|---------:|------:|------:|:------------------------------------------------------------------|
| Logistic Regression |      0.518 |       0.524 |    0.664 | 0.586 | 0.514 | {'C': 0.001}                                                      |
| SVM (linear)        |      0.518 |       0.525 |    0.66  | 0.585 | 0.513 | {'C': 0.0001}                                                     |
| SVM (polynomial)    |      0.516 |       0.528 |    0.54  | 0.534 | 0.512 | {'degree': 2, 'C': 1}                                             |
| KNN                 |      0.493 |       0.507 |    0.504 | 0.505 | 0.486 | {'n_neighbors': 25}                                               |
| Decision Tree       |      0.506 |       0.519 |    0.517 | 0.518 | 0.5   | {'max_depth': 4, 'min_samples_leaf': 100}                         |
| Random Forest       |      0.515 |       0.524 |    0.592 | 0.556 | 0.51  | {'max_depth': None, 'min_samples_leaf': 50, 'max_features': 0.33} |
| ANN (MLP)           |      0.519 |       0.525 |    0.66  | 0.585 | 0.521 | {'hidden': (64, 32), 'dropout': 0.5, 'weight_decay': 0.0001}      |
| Gradient Boosting*  |      0.523 |       0.534 |    0.548 | 0.541 | 0.518 | {'learning_rate': 0.1, 'max_depth': 6, 'l2_regularization': 1.0}  |

## Betting at -110 (break-even 52.38%)
|                               |   all_bets |   all_hit_rate |   all_roi |   all_p_vs_breakeven |   threshold |   sel_bets |   sel_hit_rate |   sel_roi | sel_roi_ci      |   sel_p_vs_breakeven |
|:------------------------------|-----------:|---------------:|----------:|---------------------:|------------:|-----------:|---------------:|----------:|:----------------|---------------------:|
| Ridge Regression (residual)   |       1779 |          0.523 |    -0.001 |                0.53  |        0.08 |        175 |          0.537 |     0.025 | [-0.116, 0.167] |                0.391 |
| KNN (residual)                |       1779 |          0.51  |    -0.026 |                0.879 |        0.05 |        181 |          0.464 |    -0.114 | [-0.251, 0.023] |                0.954 |
| Decision Tree (residual)      |       1779 |          0.502 |    -0.042 |                0.97  |        0    |       1779 |          0.502 |    -0.042 | [-0.086, 0.003] |                0.97  |
| Random Forest (residual)      |       1779 |          0.511 |    -0.024 |                0.859 |        0.03 |        179 |          0.536 |     0.024 | [-0.115, 0.173] |                0.398 |
| ANN (MLP) (residual)          |       1779 |          0.523 |    -0.001 |                0.53  |        0    |       1779 |          0.523 |    -0.001 | [-0.045, 0.044] |                0.53  |
| Gradient Boosting* (residual) |       1779 |          0.525 |     0.002 |                0.473 |        0.07 |        270 |          0.534 |     0.019 | [-0.098, 0.128] |                0.398 |
| Logistic Regression           |       1779 |          0.518 |    -0.012 |                0.709 |        0.06 |        180 |          0.564 |     0.077 | [-0.066, 0.220] |                0.157 |
| KNN                           |       1779 |          0.493 |    -0.058 |                0.995 |        0.1  |        422 |          0.478 |    -0.086 | [-0.174, 0.008] |                0.972 |
| Decision Tree                 |       1779 |          0.506 |    -0.034 |                0.939 |        0.1  |         86 |          0.488 |    -0.068 | [-0.290, 0.132] |                0.778 |
| Random Forest                 |       1779 |          0.515 |    -0.017 |                0.785 |        0.01 |       1323 |          0.517 |    -0.013 | [-0.064, 0.039] |                0.696 |
| ANN (MLP)                     |       1779 |          0.519 |    -0.009 |                0.658 |        0.02 |        934 |          0.527 |     0.007 | [-0.053, 0.073] |                0.425 |
| Gradient Boosting*            |       1779 |          0.523 |    -0.002 |                0.548 |        0.1  |       1003 |          0.512 |    -0.023 | [-0.082, 0.034] |                0.79  |

## ANN optimizer comparison (Exp 1 setup)
|          |   best_val_RMSE_points |   best_epoch |   seconds |
|:---------|-----------------------:|-------------:|----------:|
| GD       |                 18.36  |          200 |       3.5 |
| SGD      |                 18.332 |           26 |      20.7 |
| Momentum |                 18.319 |            4 |      22.7 |
| Adagrad  |                 18.303 |            1 |      25.6 |
| Adam     |                 18.311 |            4 |      29.6 |

## Walk-forward validation
|         |   games |   Logistic Regression_hit_rate |   Ridge (residual)_hit_rate |   line_RMSE |   ridge_RMSE |
|:--------|--------:|-------------------------------:|----------------------------:|------------:|-------------:|
| 2019-20 |    1053 |                          0.497 |                       0.501 |      19.051 |       19.126 |
| 2020-21 |    1073 |                          0.499 |                       0.513 |      18.625 |       18.684 |
| 2021-22 |    1224 |                          0.488 |                       0.494 |      18.376 |       18.474 |
| 2022-23 |    1228 |                          0.5   |                       0.526 |      18.102 |       18.249 |
| 2023-24 |    1214 |                          0.505 |                       0.509 |      17.472 |       17.528 |
| 2024-25 |    1228 |                          0.528 |                       0.531 |      17.568 |       17.633 |
| 2025-26 |     551 |                          0.486 |                       0.519 |      18.62  |       18.635 |

## Top 15 features (permutation importance)
|                            |   mse_increase |
|:---------------------------|---------------:|
| Away pace (last 5)         |           1.8  |
| Expected pace (last 20)    |           1.63 |
| Away pace (last 20)        |           1.51 |
| Expected pace (last 5)     |           1.43 |
| Home game total (last 20)  |           1.29 |
| Home pace (last 10)        |           1.27 |
| Expected total (last 20)   |           1.27 |
| Expected pace (last 10)    |           0.93 |
| Away efg (last 20)         |           0.9  |
| Away days rest             |           0.81 |
| Away game total (last 20)  |           0.81 |
| Expected total (last 10)   |           0.78 |
| Home days rest             |           0.77 |
| Home game total (last 10)  |           0.74 |
| Home opp orb pct (last 20) |           0.74 |
