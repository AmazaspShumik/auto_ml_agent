# GBDT Ensemble with Hyperparameter Tuning

**Branch:** exp/gbdt-ensemble-tuning
**Status:** in_progress
**Started:** 2026-03-09T00:00:00Z
**Rationale:** Gradient boosted decision trees (LightGBM, XGBoost, CatBoost) are the gold standard for tabular binary classification. EDA reveals nonlinear feature interactions (f02, f03 rank high in RF importance despite low linear correlation with target), which tree-based methods capture naturally. This direction will: (1) establish strong baselines with all three GBDT frameworks, (2) tune the best performer, (3) engineer interaction features for the top predictors, and (4) build an ensemble/blend of the strongest models to maximize ROC AUC.

## References
- No prior directions exist — this is the first exploration.

## Results
(filled on completion by submit-experiment)

## Learnings
(filled on completion by submit-experiment)
