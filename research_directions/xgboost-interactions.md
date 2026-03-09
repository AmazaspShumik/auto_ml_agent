# XGBoost with Feature Interaction Engineering

**Branch:** exp/xgboost-interactions
**Status:** in_progress
**Started:** 2026-03-09T12:00:00Z
**Rationale:** EDA reveals only 2 features (f00, f01) with notable individual correlation to the target (~0.16), while most of the 30 features have near-zero marginal signal. This pattern strongly suggests the classification boundary depends on feature interactions rather than individual features. XGBoost is well-suited for learning such interactions via tree splits, but explicit interaction terms can help by reducing the depth needed to capture key patterns and improving generalization. This direction combines XGBoost's native interaction-learning with engineered pairwise/polynomial features from the most informative features, plus systematic hyperparameter tuning.

## References
- XGBoost documentation: https://xgboost.readthedocs.io/
- Feature interaction constraints in XGBoost
- Prior EDA showing f00, f01 as dominant signals with interaction-driven classification

## Results
(filled on completion by submit-experiment)

## Learnings
(filled on completion by submit-experiment)
