# Feature Engineering for Gradient Boosting

**Branch:** exp/feature-engineering
**Status:** in_progress
**Started:** 2026-03-09T18:00:00Z
**Rationale:** All prior directions used raw features (f00-f29) as-is. Gradient boosting models can benefit from explicit feature interactions, statistical aggregations, and dimensionality reduction features. This direction systematically explores engineered features on top of the best model (CatBoost with depth=8, lr=0.05). Strategy: run feature importance first, then target engineering at high-importance features.

## References
- Prior directions: catboost-tuning (best 0.7826), lightgbm-hyperparam-tuning (best 0.7738)
- CatBoost feature importance: https://catboost.ai/en/docs/concepts/fstr

## Results
(filled on completion by submit-experiment)

## Learnings
(filled on completion by submit-experiment)
