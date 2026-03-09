# Stacking Ensemble with Diverse Base Models

**Branch:** exp/stacking-ensemble
**Status:** in_progress
**Started:** 2026-03-09T12:30:00Z
**Rationale:** All existing directions optimize individual model families or use simple seed averaging. The current best (0.7815) is multi-seed CatBoost averaging from the GBDT ensemble direction. A proper stacking ensemble with cross-validated out-of-fold predictions from diverse base learners (CatBoost, XGBoost, LightGBM, Ridge, Random Forest) fed to a regularized meta-learner can capture complementary model strengths. Each base model has a different inductive bias: CatBoost uses ordered boosting with symmetric trees, XGBoost uses exact greedy splits, LightGBM uses leaf-wise growth, and linear models capture global trends. Stacking learns the optimal weighting across these biases, which typically outperforms any single model or simple averaging.

## References
- Wolpert, "Stacked Generalization" (1992)
- Breiman, "Stacked Regressions" (1996)
- research_directions/gbdt-ensemble-tuning.md - best single-family result (0.7815 multi-seed CatBoost)
- research_directions/xgboost-interactions.md - best XGBoost result (0.7786)
- research_directions/deep_learning_tabular.md - MLP baseline (0.7494)

## Results
(filled on completion by submit-experiment)

## Learnings
(filled on completion by submit-experiment)
