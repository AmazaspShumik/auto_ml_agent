"""
Deep learning for tabular binary classification.
MLP with batch normalization, dropout, and weight decay.
"""
import argparse
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.pytorch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from evaluate import evaluate_public

mlflow.set_tracking_uri("http://127.0.0.1:5000")


class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.3, use_batchnorm=True, activation="relu"):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def add_interaction_features(X, feature_cols, top_k=5):
    """Add pairwise product interactions for the top_k features by index."""
    top_indices = list(range(top_k))
    new_feats = []
    new_names = []
    for i in range(len(top_indices)):
        for j in range(i + 1, len(top_indices)):
            fi, fj = top_indices[i], top_indices[j]
            new_feats.append(X[:, fi] * X[:, fj])
            new_names.append(f"{feature_cols[fi]}x{feature_cols[fj]}")
    if new_feats:
        X = np.column_stack([X] + [f.reshape(-1, 1) for f in new_feats])
    return X, feature_cols + new_names


def load_data(add_interactions=False, interaction_top_k=5):
    train_df = pd.read_csv(PROJECT_ROOT / "data" / "train.csv")
    val_X = pd.read_csv(PROJECT_ROOT / "data" / "val_public_X.csv")

    feature_cols = [c for c in train_df.columns if c != "target"]
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["target"].values.astype(np.float32)
    X_val = val_X[feature_cols].values.astype(np.float32)

    if add_interactions:
        X_train, feature_cols = add_interaction_features(X_train, feature_cols, interaction_top_k)
        X_val, _ = add_interaction_features(X_val, feature_cols[:30], interaction_top_k)

    return X_train, y_train, X_val, feature_cols


def train_model(config):
    X_train, y_train, X_val, feature_cols = load_data(
        add_interactions=config.get("add_interactions", False),
        interaction_top_k=config.get("interaction_top_k", 5),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    X_t = torch.tensor(X_train_scaled, device=device)
    y_t = torch.tensor(y_train, device=device)
    X_v = torch.tensor(X_val_scaled, device=device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)

    model = TabularMLP(
        input_dim=X_train_scaled.shape[1],
        hidden_dims=config["hidden_dims"],
        dropout_rate=config["dropout_rate"],
        use_batchnorm=config["use_batchnorm"],
        activation=config.get("activation", "relu"),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    scheduler = None
    if config.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=config["lr"] * 0.01
        )
    elif config.get("scheduler") == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config["lr"], epochs=config["epochs"],
            steps_per_epoch=len(loader)
        )

    label_smoothing = config.get("label_smoothing", 0.0)
    if label_smoothing > 0:
        criterion = nn.BCEWithLogitsLoss()
        def smooth_criterion(logits, targets):
            targets_smooth = targets * (1 - label_smoothing) + 0.5 * label_smoothing
            return nn.functional.binary_cross_entropy_with_logits(logits, targets_smooth)
        criterion = smooth_criterion
    else:
        criterion = nn.BCEWithLogitsLoss()

    use_swa = config.get("use_swa", False)
    swa_model = None
    swa_scheduler = None
    swa_start_epoch = config.get("swa_start_epoch", config["epochs"] // 2)
    if use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=config.get("swa_lr", config["lr"] * 0.5))

    best_val_score = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(config["epochs"]):
        model.train()
        train_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if config.get("grad_clip", 0) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            if config.get("scheduler") == "onecycle" and scheduler:
                scheduler.step()
            train_losses.append(loss.item())

        if use_swa and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        elif config.get("scheduler") == "cosine" and scheduler:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            train_logits = model(X_t)
            train_probs = torch.sigmoid(train_logits).cpu().numpy()

        val_score = evaluate_public(val_probs)
        from sklearn.metrics import roc_auc_score
        train_score = roc_auc_score(y_train, train_probs)

        mlflow.log_metrics({
            "train_loss": np.mean(train_losses),
            "train_auc": train_score,
            "val_auc": val_score,
        }, step=epoch)

        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == config["epochs"] - 1:
            print(f"Epoch {epoch:3d} | train_loss={np.mean(train_losses):.4f} | "
                  f"train_auc={train_score:.4f} | val_auc={val_score:.4f} | best={best_val_score:.4f}")

        if patience_counter >= config.get("patience", 50):
            print(f"Early stopping at epoch {epoch}")
            break

    if use_swa and swa_model is not None:
        torch.optim.swa_utils.update_bn(loader, swa_model, device=device)
        swa_model.eval()
        with torch.no_grad():
            swa_val_probs = torch.sigmoid(swa_model(X_v)).cpu().numpy()
        swa_score = evaluate_public(swa_val_probs)
        mlflow.log_metric("swa_val_auc", swa_score)
        print(f"SWA val AUC: {swa_score:.4f} (best checkpoint: {best_val_score:.4f})")
        if swa_score > best_val_score:
            best_val_score = swa_score
            val_probs = swa_val_probs
            return swa_model.module, best_val_score, val_probs, scaler

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_v)).cpu().numpy()

    return model, best_val_score, val_probs, scaler


def run_experiment(config, run_rationale, direction_rationale):
    with mlflow.start_run(run_name=config.get("run_name", "nn_run")):
        mlflow.set_tag("git.branch", "exp/deep-learning-tabular")
        mlflow.set_tag("direction_rationale", direction_rationale)
        mlflow.set_tag("run_rationale", run_rationale)

        for k, v in config.items():
            if k == "hidden_dims":
                mlflow.log_param(k, json.dumps(v))
            elif k not in ("run_name",):
                mlflow.log_param(k, v)

        model, best_val_score, val_probs, scaler = train_model(config)

        mlflow.log_metric("public_val_score", best_val_score)

        mlflow.pytorch.log_model(model, "model")

        np.save(PROJECT_ROOT / "experiments" / "scaler_mean.npy", scaler.mean_)
        np.save(PROJECT_ROOT / "experiments" / "scaler_scale.npy", scaler.scale_)
        mlflow.log_artifact(str(PROJECT_ROOT / "experiments" / "scaler_mean.npy"), "preprocessing")
        mlflow.log_artifact(str(PROJECT_ROOT / "experiments" / "scaler_scale.npy"), "preprocessing")

        run_id = mlflow.active_run().info.run_id
        print(f"\nRun ID: {run_id}")
        print(f"Public val score: {best_val_score:.6f}")

        return run_id, best_val_score


DIRECTION_RATIONALE = (
    "Exploring deep learning for tabular classification. EDA shows weak individual "
    "feature correlations but potential for non-linear interactions. Neural networks "
    "with modern regularization can capture these from a different inductive bias than "
    "tree-based methods explored by other agents."
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="JSON config string or file path")
    parser.add_argument("--rationale", type=str, required=True, help="Run rationale")
    args = parser.parse_args()

    if args.config.endswith(".json"):
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = json.loads(args.config)

    run_experiment(config, args.rationale, DIRECTION_RATIONALE)
