import sys
sys.path.insert(0, "src")

import pandas as pd
import numpy as np
import mlflow
from evaluate import evaluate_public

mlflow.set_tracking_uri("http://127.0.0.1:5000")

train = pd.read_csv("data/train.csv")
X_train = train.drop("target", axis=1)
y_train = train["target"]
feature_names = sorted(X_train.columns.tolist())
X_train = X_train[feature_names]
val_X = pd.read_csv("data/val_public_X.csv")[feature_names]

DIRECTION_RATIONALE = (
    "GBDT models are the gold standard for tabular binary classification. "
    "EDA shows nonlinear interactions (f02, f03 high RF importance, low linear correlation). "
    "This direction tunes LightGBM, XGBoost, CatBoost and ensembles them."
)
BRANCH = "exp/gbdt-ensemble-tuning"

# --- CatBoost Tuned: deeper, more iterations, lower lr ---
print("=== CatBoost Tuned v1: deeper + more iters ===")
from catboost import CatBoostClassifier

cb_params_v1 = {
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 5.0,
    "subsample": 0.8,
    "random_seed": 42,
    "verbose": 0,
    "eval_metric": "AUC",
    "grow_policy": "SymmetricTree",
    "bootstrap_type": "Bernoulli",
}

cb_model_v1 = CatBoostClassifier(**cb_params_v1)
cb_model_v1.fit(X_train, y_train)
cb_preds_v1 = cb_model_v1.predict_proba(val_X)[:, 1]
cb_score_v1 = evaluate_public(cb_preds_v1)
print(f"CatBoost v1: {cb_score_v1:.6f}")

with mlflow.start_run(run_name="catboost_tuned_v1") as run:
    mlflow.set_tags({
        "git.branch": BRANCH,
        "direction_rationale": DIRECTION_RATIONALE,
        "run_rationale": "CatBoost tuned: deeper trees (depth=8), 3000 iters, lr=0.03, stronger regularization (l2=5). Aiming to capture more complex interactions.",
        "run_analysis": f"CatBoost v1: {cb_score_v1:.6f} AUC. Compared to baseline 0.7800.",
    })
    mlflow.log_params(cb_params_v1)
    mlflow.log_metric("public_val_score", cb_score_v1)
    mlflow.catboost.log_model(cb_model_v1, "model")
    print(f"  run ID: {run.info.run_id}")

# --- CatBoost Tuned v2: even deeper with more leaves ---
print("\n=== CatBoost Tuned v2: depth=10, more regularization ===")
cb_params_v2 = {
    "iterations": 3000,
    "learning_rate": 0.02,
    "depth": 10,
    "l2_leaf_reg": 10.0,
    "subsample": 0.75,
    "random_seed": 42,
    "verbose": 0,
    "eval_metric": "AUC",
    "grow_policy": "SymmetricTree",
    "bootstrap_type": "Bernoulli",
    "min_data_in_leaf": 20,
}

cb_model_v2 = CatBoostClassifier(**cb_params_v2)
cb_model_v2.fit(X_train, y_train)
cb_preds_v2 = cb_model_v2.predict_proba(val_X)[:, 1]
cb_score_v2 = evaluate_public(cb_preds_v2)
print(f"CatBoost v2: {cb_score_v2:.6f}")

with mlflow.start_run(run_name="catboost_tuned_v2") as run:
    mlflow.set_tags({
        "git.branch": BRANCH,
        "direction_rationale": DIRECTION_RATIONALE,
        "run_rationale": "CatBoost deeper: depth=10, lr=0.02, 3000 iters, heavy reg (l2=10, min_data_in_leaf=20). Testing if deeper trees help.",
        "run_analysis": f"CatBoost v2: {cb_score_v2:.6f} AUC. Compared to v1 and baseline 0.7800.",
    })
    mlflow.log_params(cb_params_v2)
    mlflow.log_metric("public_val_score", cb_score_v2)
    mlflow.catboost.log_model(cb_model_v2, "model")
    print(f"  run ID: {run.info.run_id}")

# --- XGBoost Tuned ---
print("\n=== XGBoost Tuned: deeper + more trees ===")
import xgboost as xgb

xgb_params_v1 = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "min_child_weight": 10,
    "gamma": 0.1,
    "random_state": 42,
    "verbosity": 0,
}

xgb_model_v1 = xgb.XGBClassifier(**xgb_params_v1)
xgb_model_v1.fit(X_train, y_train)
xgb_preds_v1 = xgb_model_v1.predict_proba(val_X)[:, 1]
xgb_score_v1 = evaluate_public(xgb_preds_v1)
print(f"XGBoost v1: {xgb_score_v1:.6f}")

with mlflow.start_run(run_name="xgb_tuned_v1") as run:
    mlflow.set_tags({
        "git.branch": BRANCH,
        "direction_rationale": DIRECTION_RATIONALE,
        "run_rationale": "XGBoost tuned: depth=8, lr=0.03, 2000 trees, stronger reg (alpha=0.5, lambda=5), gamma=0.1",
        "run_analysis": f"XGBoost v1: {xgb_score_v1:.6f} AUC. Compared to baseline 0.7680.",
    })
    mlflow.log_params({k: v for k, v in xgb_params_v1.items()})
    mlflow.log_metric("public_val_score", xgb_score_v1)
    mlflow.xgboost.log_model(xgb_model_v1, "model")
    print(f"  run ID: {run.info.run_id}")

# --- LightGBM Tuned ---
print("\n=== LightGBM Tuned: more leaves + deeper ===")
import lightgbm as lgb

lgb_params_v1 = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "num_leaves": 127,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "min_child_samples": 30,
    "random_state": 42,
    "subsample_freq": 1,
}

lgb_model_v1 = lgb.LGBMClassifier(**lgb_params_v1)
lgb_model_v1.fit(X_train, y_train)
lgb_preds_v1 = lgb_model_v1.predict_proba(val_X)[:, 1]
lgb_score_v1 = evaluate_public(lgb_preds_v1)
print(f"LightGBM v1: {lgb_score_v1:.6f}")

with mlflow.start_run(run_name="lgbm_tuned_v1") as run:
    mlflow.set_tags({
        "git.branch": BRANCH,
        "direction_rationale": DIRECTION_RATIONALE,
        "run_rationale": "LightGBM tuned: 127 leaves, depth=8, lr=0.03, 2000 trees, stronger reg. Testing if more capacity helps.",
        "run_analysis": f"LightGBM v1: {lgb_score_v1:.6f} AUC. Compared to baseline 0.7626.",
    })
    mlflow.log_params(lgb_params_v1)
    mlflow.log_metric("public_val_score", lgb_score_v1)
    mlflow.lightgbm.log_model(lgb_model_v1, "model")
    print(f"  run ID: {run.info.run_id}")

print(f"\n=== Tuning Summary ===")
print(f"CatBoost v1 (d8, 3k):   {cb_score_v1:.6f}")
print(f"CatBoost v2 (d10, 3k):  {cb_score_v2:.6f}")
print(f"XGBoost v1 (d8, 2k):    {xgb_score_v1:.6f}")
print(f"LightGBM v1 (l127, 2k): {lgb_score_v1:.6f}")
