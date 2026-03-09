# Deep Learning for Tabular Classification

**Branch:** exp/deep-learning-tabular
**Status:** in_progress
**Started:** 2026-03-09T18:00:00Z
**Rationale:** Gradient boosting models (CatBoost 0.7826, LightGBM 0.7738) dominate the leaderboard but may miss non-linear patterns that MLPs can capture. Neural networks with BatchNorm, Dropout, and proper feature standardization have shown competitive performance on tabular data (TabNet, FT-Transformer literature). Even if an MLP doesn't beat CatBoost alone, it provides diversity for future ensembling. This direction explores MLP architectures (width, depth, skip connections), optimizers (Adam, AdamW), learning rate schedules, and regularization.

## References
- Prior directions: catboost-tuning (best 0.7826), lightgbm-hyperparam-tuning (best 0.7738), xgboost-hyperparam-tuning (best 0.5664 public / 0.7496 private)
- "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021)
- PyTorch documentation: https://pytorch.org/docs/stable/

## Results
(filled on completion by submit-experiment)

## Learnings
(filled on completion by submit-experiment)
