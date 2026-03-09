"""XGBoost with internal CV for optimal rounds, then retrain on full data."""
import sys
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost

sys.path.insert(0, "src")
from evaluate import evaluate_public

mlflow.set_tracking_uri("http://127.0.0.1:5000")

DIRECTION_RATIONALE = (
    "EDA shows only f00/f01 have notable marginal correlation with target (~0.16); "
    "most features near zero. This suggests interaction-driven classification. "
    "XGBoost can capture interactions via tree splits, augmented with engineered "
    "interaction features and systematic hyperparameter tuning."
)


def load_data():
    train = pd.read_csv("data/train.csv")
    val_X = pd.read_csv("data/val_public_X.csv")
    feature_cols = [c for c in train.columns if c != "target"]
    X_train = train[feature_cols]
    y_train = train["target"]
    return X_train, y_train, val_X[feature_cols], feature_cols


def run_cv_experiment(params, run_rationale):
    X_train, y_train, X_val, feature_cols = load_data()

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "verbosity": 0,
        **params,
    }

    num_rounds = xgb_params.pop("num_boost_round", 5000)
    nfold = xgb_params.pop("nfold", 5)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val)

    cv_results = xgb.cv(
        xgb_params,
        dtrain,
        num_boost_round=num_rounds,
        nfold=nfold,
        early_stopping_rounds=100,
        verbose_eval=False,
        seed=42,
    )

    best_rounds = len(cv_results)
    cv_auc = cv_results["test-auc-mean"].iloc[-1]
    cv_std = cv_results["test-auc-std"].iloc[-1]

    print(f"CV best rounds: {best_rounds}, CV AUC: {cv_auc:.6f} +/- {cv_std:.6f}")

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=best_rounds,
        verbose_eval=False,
    )

    preds = model.predict(dval)
    score = evaluate_public(preds)

    with mlflow.start_run() as run:
        mlflow.set_tag("git.branch", "exp/xgboost-interactions")
        mlflow.set_tag("direction_rationale", DIRECTION_RATIONALE)
        mlflow.set_tag("run_rationale", run_rationale)

        mlflow.log_params({k: v for k, v in xgb_params.items()})
        mlflow.log_param("num_boost_round_max", num_rounds)
        mlflow.log_param("best_rounds_cv", best_rounds)
        mlflow.log_param("nfold", nfold)

        mlflow.log_metric("cv_auc_mean", cv_auc)
        mlflow.log_metric("cv_auc_std", cv_std)
        mlflow.log_metric("public_val_score", score)
        mlflow.log_metric("num_trees", model.num_boosted_rounds())

        mlflow.xgboost.log_model(model, "model")

        run_id = run.info.run_id

    print(f"Run {run_id}: public_val_score={score:.6f}, trees={best_rounds}")
    return run_id, score, best_rounds, cv_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--rationale", type=str, required=True)
    args = parser.parse_args()

    params = json.loads(args.config)
    run_id, score, best_rounds, cv_auc = run_cv_experiment(params, args.rationale)
