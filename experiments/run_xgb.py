"""XGBoost experiments for the xgboost-interactions direction."""
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


def add_interaction_features(X_train, X_val, feature_cols, interaction_pairs):
    """Add pairwise interaction (product) features."""
    X_train = X_train.copy()
    X_val = X_val.copy()
    new_cols = []
    for f1, f2 in interaction_pairs:
        col_name = f"{f1}_x_{f2}"
        X_train[col_name] = X_train[f1] * X_train[f2]
        X_val[col_name] = X_val[f1] * X_val[f2]
        new_cols.append(col_name)
    return X_train, X_val, new_cols


def run_experiment(params, run_rationale, interaction_pairs=None, extra_features_fn=None):
    X_train, y_train, X_val, feature_cols = load_data()

    new_cols = []
    if interaction_pairs:
        X_train, X_val, new_cols = add_interaction_features(
            X_train, X_val, feature_cols, interaction_pairs
        )

    if extra_features_fn:
        X_train, X_val, extra_cols = extra_features_fn(X_train, X_val, feature_cols)
        new_cols.extend(extra_cols)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val)

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "verbosity": 0,
        **params,
    }

    num_rounds = xgb_params.pop("num_boost_round", 1000)
    early_stop = xgb_params.pop("early_stopping_rounds", 50)

    evals_result = {}
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dtrain, "train")],
        evals_result=evals_result,
        verbose_eval=False,
    )

    preds = model.predict(dval)
    score = evaluate_public(preds)

    with mlflow.start_run() as run:
        mlflow.set_tag("git.branch", "exp/xgboost-interactions")
        mlflow.set_tag("direction_rationale", DIRECTION_RATIONALE)
        mlflow.set_tag("run_rationale", run_rationale)

        mlflow.log_params({k: v for k, v in xgb_params.items()})
        mlflow.log_param("num_boost_round", num_rounds)
        mlflow.log_param("early_stopping_rounds", early_stop)
        mlflow.log_param("n_interaction_features", len(new_cols))
        if interaction_pairs:
            mlflow.log_param("interaction_pairs", str(interaction_pairs))

        for i, auc_val in enumerate(evals_result["train"]["auc"]):
            mlflow.log_metric("train_auc", auc_val, step=i)

        mlflow.log_metric("public_val_score", score)
        mlflow.log_metric("num_trees", model.num_boosted_rounds())

        mlflow.xgboost.log_model(model, "model")

        run_id = run.info.run_id

    print(f"Run {run_id}: public_val_score={score:.6f}, trees={model.num_boosted_rounds()}")
    return run_id, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="JSON config string or path")
    parser.add_argument("--rationale", type=str, required=True)
    parser.add_argument("--interactions", type=str, default=None, help="JSON list of pairs")
    parser.add_argument("--analysis", type=str, default=None, help="Post-run analysis")
    args = parser.parse_args()

    if args.config.endswith(".json"):
        with open(args.config) as f:
            params = json.load(f)
    else:
        params = json.loads(args.config)

    interactions = json.loads(args.interactions) if args.interactions else None

    run_id, score = run_experiment(params, args.rationale, interaction_pairs=interactions)

    if args.analysis:
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tag("run_analysis", args.analysis)
