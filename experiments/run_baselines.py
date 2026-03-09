import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from evaluate import evaluate_public

mlflow.set_tracking_uri("http://127.0.0.1:5000")

train = pd.read_csv("data/train.csv")
val_X = pd.read_csv("data/val_public_X.csv")

feature_cols = [c for c in train.columns if c != "target"]
X_train = train[feature_cols].values
y_train = train["target"].values
X_val = val_X[feature_cols].values

BRANCH = "exp/stacked-generalization"
DIRECTION_RATIONALE = (
    "Stacked generalization with diverse base learners: train LightGBM, XGBoost, "
    "CatBoost, Extra Trees, and Logistic Regression using K-fold CV, then combine "
    "via a meta-learner to capture complementary inductive biases."
)

# --- Run 1: LightGBM baseline ---
import lightgbm as lgb

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 1000,
    "verbose": -1,
    "random_state": 42,
}

print("=== LightGBM Baseline ===")
with mlflow.start_run(run_name="lgb_baseline") as run:
    mlflow.set_tag("git.branch", BRANCH)
    mlflow.set_tag("direction_rationale", DIRECTION_RATIONALE)
    mlflow.set_tag("run_rationale", "LightGBM baseline with solid defaults to establish performance floor for stacking")
    mlflow.log_params(lgb_params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    val_preds = np.zeros(len(X_val))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        dtrain = lgb.Dataset(X_train[tr_idx], y_train[tr_idx])
        dval = lgb.Dataset(X_train[va_idx], y_train[va_idx], reference=dtrain)
        model = lgb.train(
            {k: v for k, v in lgb_params.items() if k not in ["n_estimators", "random_state"]},
            dtrain,
            num_boost_round=lgb_params["n_estimators"],
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof_preds[va_idx] = model.predict(X_train[va_idx])
        val_preds += model.predict(X_val) / 5
        fold_auc = roc_auc_score(y_train[va_idx], oof_preds[va_idx])
        mlflow.log_metric(f"fold_{fold}_auc", fold_auc, step=fold)
        print(f"  Fold {fold}: AUC={fold_auc:.6f}")

    oof_auc = roc_auc_score(y_train, oof_preds)
    mlflow.log_metric("oof_auc", oof_auc)
    print(f"  OOF AUC: {oof_auc:.6f}")

    score = evaluate_public(val_preds)
    mlflow.log_metric("public_val_score", score)
    print(f"  Public val AUC: {score:.6f}")

    mlflow.lightgbm.log_model(model, "model")
    mlflow.set_tag("run_analysis", f"LightGBM baseline OOF={oof_auc:.6f}, public_val={score:.6f}. This establishes the performance floor for the stacking direction.")
    lgb_run_id = run.info.run_id
    print(f"  Run ID: {lgb_run_id}")


# --- Run 2: XGBoost baseline ---
import xgboost as xgb

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 1000,
    "tree_method": "hist",
    "random_state": 42,
}

print("\n=== XGBoost Baseline ===")
with mlflow.start_run(run_name="xgb_baseline") as run:
    mlflow.set_tag("git.branch", BRANCH)
    mlflow.set_tag("direction_rationale", DIRECTION_RATIONALE)
    mlflow.set_tag("run_rationale", "XGBoost baseline with solid defaults to establish performance floor for stacking")
    mlflow.log_params(xgb_params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    val_preds = np.zeros(len(X_val))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(
            X_train[tr_idx], y_train[tr_idx],
            eval_set=[(X_train[va_idx], y_train[va_idx])],
            verbose=False,
        )
        oof_preds[va_idx] = clf.predict_proba(X_train[va_idx])[:, 1]
        val_preds += clf.predict_proba(X_val)[:, 1] / 5
        fold_auc = roc_auc_score(y_train[va_idx], oof_preds[va_idx])
        mlflow.log_metric(f"fold_{fold}_auc", fold_auc, step=fold)
        print(f"  Fold {fold}: AUC={fold_auc:.6f}")

    oof_auc = roc_auc_score(y_train, oof_preds)
    mlflow.log_metric("oof_auc", oof_auc)
    print(f"  OOF AUC: {oof_auc:.6f}")

    score = evaluate_public(val_preds)
    mlflow.log_metric("public_val_score", score)
    print(f"  Public val AUC: {score:.6f}")

    mlflow.xgboost.log_model(clf, "model")
    mlflow.set_tag("run_analysis", f"XGBoost baseline OOF={oof_auc:.6f}, public_val={score:.6f}.")
    xgb_run_id = run.info.run_id
    print(f"  Run ID: {xgb_run_id}")


# --- Run 3: CatBoost baseline ---
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
    "early_stopping_rounds": 50,
}

print("\n=== CatBoost Baseline ===")
with mlflow.start_run(run_name="cb_baseline") as run:
    mlflow.set_tag("git.branch", BRANCH)
    mlflow.set_tag("direction_rationale", DIRECTION_RATIONALE)
    mlflow.set_tag("run_rationale", "CatBoost baseline with ordered boosting defaults for stacking")
    mlflow.log_params(cb_params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    val_preds = np.zeros(len(X_val))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        clf = CatBoostClassifier(**cb_params)
        clf.fit(X_train[tr_idx], y_train[tr_idx], eval_set=(X_train[va_idx], y_train[va_idx]))
        oof_preds[va_idx] = clf.predict_proba(X_train[va_idx])[:, 1]
        val_preds += clf.predict_proba(X_val)[:, 1] / 5
        fold_auc = roc_auc_score(y_train[va_idx], oof_preds[va_idx])
        mlflow.log_metric(f"fold_{fold}_auc", fold_auc, step=fold)
        print(f"  Fold {fold}: AUC={fold_auc:.6f}")

    oof_auc = roc_auc_score(y_train, oof_preds)
    mlflow.log_metric("oof_auc", oof_auc)
    print(f"  OOF AUC: {oof_auc:.6f}")

    score = evaluate_public(val_preds)
    mlflow.log_metric("public_val_score", score)
    print(f"  Public val AUC: {score:.6f}")

    mlflow.catboost.log_model(clf, "model")
    mlflow.set_tag("run_analysis", f"CatBoost baseline OOF={oof_auc:.6f}, public_val={score:.6f}.")
    cb_run_id = run.info.run_id
    print(f"  Run ID: {cb_run_id}")

print("\n=== Baseline Summary ===")
print(f"LightGBM: {lgb_run_id}")
print(f"XGBoost:  {xgb_run_id}")
print(f"CatBoost: {cb_run_id}")
