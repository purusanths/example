# Assumes these already exist in your session:
#   X_train, y_train, X_val, y_val   (pandas DataFrames / Series, binary target)
# pip install optuna xgboost scikit-learn

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

spw_max = (y_train == 0).sum() / (y_train == 1).sum()   # fully balanced weight

def objective(trial):
    params = dict(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_estimators=3000,
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=-1,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        min_child_weight=trial.suggest_float("min_child_weight", 1, 200, log=True),
        gamma=trial.suggest_float("gamma", 1e-3, 10, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 50, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 100, log=True),
        scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, max(spw_max, 1.5), log=True),
    )
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    p = model.predict_proba(X_val)[:, 1]
    trial.set_user_attr("best_n_estimators", model.best_iteration + 1)
    return average_precision_score(y_val, p)          # swap for roc_auc_score if preferred

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100, show_progress_bar=True)

best = study.best_trial
print("Best PR-AUC:", round(best.value, 4))
print(best.params)

# Refit with best params on train (early stopping against val)
final = xgb.XGBClassifier(
    objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
    n_estimators=3000, early_stopping_rounds=100, random_state=42, n_jobs=-1,
    **best.params,
)
final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

p_val = final.predict_proba(X_val)[:, 1]
print("Val AUC:", round(roc_auc_score(y_val, p_val), 4),
      "| Val PR-AUC:", round(average_precision_score(y_val, p_val), 4))
