import sys
sys.path.insert(0, "src")

import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import roc_auc_score
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

# --- LightGBM Baseline ---
print("=== LightGBM Baseline ===")
import lightgbm as lgb

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_samples": 20,
    "random_state": 42,
}

lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict_proba(val_X)[:, 1]
lgb_score = evaluate_public(lgb_preds)
print(f"LightGBM public_val_score: {lgb_score:.6f}")

with mlflow.start_run(run_name="lgbm_baseline") as run:
    mlflow.set_tags({
        "git.branch": BRANCH,
        "direction_rationale": DIRECTION_RATIONALE,
        "run_rationale": "LightGBM baseline with reasonable defaults: lr=0.05, 1000 trees, depth=6, num_leaves=63, subsample=0.8",
        "run_analysis": f"LightGBM baseline achieves {lgb_score:.6f} ROC AUC. This is our first GBDT baseline to compare against.",
    })
    mlflow.log_params({k: v for k, v in lgb_params.items()})
    mlflow.log_metric("public_val_score", lgb_score)
    mlflow.lightgbm.log_model(lgb_model, "model")
    lgb_run_id = run.info.run_id
    print(f"LightGBM run ID: {lgb_run_id}")

# --- XGBoost Baseline ---
print("\n=== XGBoost Baseline ===")
import xgboost as xgb

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "random_state": 42,
    "verbosity": 0,
}

xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict_proba(val_X)[:, 1]
xgb_score = evaluate_public(xgb_preds)
print(f"XGBoost public_val_score: {xgb_score:.6f}")

with mlflow.start_run(run_name="xgb_baseline") as run:
    mlflow.set_tags({
        "git.branch": BRANCH,
        "direction_rationale": DIRECTION_RATIONALE,
        "run_rationale": "XGBoost baseline with reasonable defaults: lr=0.05, 1000 trees, depth=6, subsample=0.8",
        "run_analysis": f"XGBoost baseline achieves {xgb_score:.6f} ROC AUC.",
    })
    mlflow.log_params({k: v for k, v in xgb_params.items()})
    mlflow.log_metric("public_val_score", xgb_score)
    mlflow.xgboost.log_model(xgb_model, "model")
    xgb_run_id = run.info.run_id
    print(f"XGBoost run ID: {xgb_run_id}")

# --- CatBoost Baseline ---
print("\n=== CatBoost Baseline ===")
from catboost import CatBoostClassifier

cb_params = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "subsample": 0.8,
    "random_seed": 42,
    "verbose": 0,
    "eval_metric": "AUC",
}

cb_model = CatBoostClassifier(**cb_params)
cb_model.fit(X_train, y_train)
cb_preds = cb_model.predict_proba(val_X)[:, 1]
cb_score = evaluate_public(cb_preds)
print(f"CatBoost public_val_score: {cb_score:.6f}")

with mlflow.start_run(run_name="catboost_baseline") as run:
    mlflow.set_tags({
        "git.branch": BRANCH,
        "direction_rationale": DIRECTION_RATIONALE,
        "run_rationale": "CatBoost baseline with reasonable defaults: lr=0.05, 1000 iters, depth=6, l2_leaf_reg=3",
        "run_analysis": f"CatBoost baseline achieves {cb_score:.6f} ROC AUC.",
    })
    mlflow.log_params({k: v for k, v in cb_params.items()})
    mlflow.log_metric("public_val_score", cb_score)
    mlflow.catboost.log_model(cb_model, "model")
    cb_run_id = run.info.run_id
    print(f"CatBoost run ID: {cb_run_id}")

print(f"\n=== Summary ===")
print(f"LightGBM:  {lgb_score:.6f}  (run: {lgb_run_id})")
print(f"XGBoost:   {xgb_score:.6f}  (run: {xgb_run_id})")
print(f"CatBoost:  {cb_score:.6f}  (run: {cb_run_id})")
