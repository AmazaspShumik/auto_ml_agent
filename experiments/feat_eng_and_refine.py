import sys
sys.path.insert(0, "src")

import pandas as pd
import numpy as np
import mlflow
from evaluate import evaluate_public
from catboost import CatBoostClassifier

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

def add_interaction_features(df):
    df = df.copy()
    top_feats = ["f00", "f01", "f02", "f03"]
    for i, f1 in enumerate(top_feats):
        for f2 in top_feats[i+1:]:
            df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
            df[f"{f1}_p_{f2}"] = df[f1] + df[f2]
            df[f"{f1}_d_{f2}"] = df[f1] - df[f2]
    df["f00_sq"] = df["f00"] ** 2
    df["f01_sq"] = df["f01"] ** 2
    df["f02_sq"] = df["f02"] ** 2
    df["f03_sq"] = df["f03"] ** 2
    df["f00_f01_f02"] = df["f00"] * df["f01"] * df["f02"]
    df["top4_sum"] = df["f00"] + df["f01"] + df["f02"] + df["f03"]
    df["top4_std"] = df[top_feats].std(axis=1)
    return df

X_train_fe = add_interaction_features(X_train)
val_X_fe = add_interaction_features(val_X)
print(f"Features after engineering: {X_train_fe.shape[1]}")

# --- Run 1: CatBoost with feature engineering, baseline-like params ---
print("\n=== CatBoost + Feature Engineering ===")
cb_params_fe = {
    "iterations": 1500,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "subsample": 0.8,
    "random_seed": 42,
    "verbose": 0,
    "eval_metric": "AUC",
}

cb_model_fe = CatBoostClassifier(**cb_params_fe)
cb_model_fe.fit(X_train_fe, y_train)
cb_preds_fe = cb_model_fe.predict_proba(val_X_fe)[:, 1]
cb_score_fe = evaluate_public(cb_preds_fe)
print(f"CatBoost + FE: {cb_score_fe:.6f}")

with mlflow.start_run(run_name="catboost_feat_eng") as run:
    mlflow.set_tags({
        "git.branch": BRANCH,
        "direction_rationale": DIRECTION_RATIONALE,
        "run_rationale": "CatBoost with engineered features: pairwise interactions (multiply, add, subtract) for top-4 features, squares, triple interaction, sum/std aggregates. Depth=6 kept from best baseline.",
        "run_analysis": f"CatBoost+FE: {cb_score_fe:.6f} AUC.",
    })
    mlflow.log_params({**cb_params_fe, "feature_engineering": "top4_interactions"})
    mlflow.log_metric("public_val_score", cb_score_fe)
    mlflow.catboost.log_model(cb_model_fe, "model")
    run_id_fe = run.info.run_id
    print(f"  run ID: {run_id_fe}")

# --- Run 2: CatBoost refined at depth=6, lower lr, more iters ---
print("\n=== CatBoost Refined (depth=6, lr=0.03, 2000 iters) ===")
cb_params_ref = {
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "subsample": 0.85,
    "random_seed": 42,
    "verbose": 0,
    "eval_metric": "AUC",
    "bootstrap_type": "Bernoulli",
}

cb_model_ref = CatBoostClassifier(**cb_params_ref)
cb_model_ref.fit(X_train, y_train)
cb_preds_ref = cb_model_ref.predict_proba(val_X)[:, 1]
cb_score_ref = evaluate_public(cb_preds_ref)
print(f"CatBoost refined: {cb_score_ref:.6f}")

with mlflow.start_run(run_name="catboost_refined_d6") as run:
    mlflow.set_tags({
        "git.branch": BRANCH,
        "direction_rationale": DIRECTION_RATIONALE,
        "run_rationale": "CatBoost keeping depth=6 (sweet spot), but lr=0.03 with 2000 iters to fit slower. Subsample=0.85.",
        "run_analysis": f"CatBoost refined: {cb_score_ref:.6f} AUC.",
    })
    mlflow.log_params(cb_params_ref)
    mlflow.log_metric("public_val_score", cb_score_ref)
    mlflow.catboost.log_model(cb_model_ref, "model")
    run_id_ref = run.info.run_id
    print(f"  run ID: {run_id_ref}")

# --- Run 3: CatBoost with Lossguide (leaf-wise) growth ---
print("\n=== CatBoost Lossguide (leaf-wise) ===")
cb_params_lg = {
    "iterations": 1500,
    "learning_rate": 0.05,
    "max_leaves": 31,
    "l2_leaf_reg": 3.0,
    "subsample": 0.8,
    "random_seed": 42,
    "verbose": 0,
    "eval_metric": "AUC",
    "grow_policy": "Lossguide",
    "bootstrap_type": "Bernoulli",
    "min_data_in_leaf": 10,
}

cb_model_lg = CatBoostClassifier(**cb_params_lg)
cb_model_lg.fit(X_train, y_train)
cb_preds_lg = cb_model_lg.predict_proba(val_X)[:, 1]
cb_score_lg = evaluate_public(cb_preds_lg)
print(f"CatBoost Lossguide: {cb_score_lg:.6f}")

with mlflow.start_run(run_name="catboost_lossguide") as run:
    mlflow.set_tags({
        "git.branch": BRANCH,
        "direction_rationale": DIRECTION_RATIONALE,
        "run_rationale": "CatBoost with Lossguide (leaf-wise) growth instead of SymmetricTree. Max 31 leaves. Testing if LightGBM-style growth helps in CatBoost.",
        "run_analysis": f"CatBoost Lossguide: {cb_score_lg:.6f} AUC.",
    })
    mlflow.log_params(cb_params_lg)
    mlflow.log_metric("public_val_score", cb_score_lg)
    mlflow.catboost.log_model(cb_model_lg, "model")
    run_id_lg = run.info.run_id
    print(f"  run ID: {run_id_lg}")

print(f"\n=== Summary ===")
print(f"CatBoost + FE:           {cb_score_fe:.6f}")
print(f"CatBoost refined d6:     {cb_score_ref:.6f}")
print(f"CatBoost Lossguide:      {cb_score_lg:.6f}")
print(f"(Best so far: 0.780019)")
