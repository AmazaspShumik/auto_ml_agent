# Stacked Generalization with Diverse Base Learners

**Branch:** exp/stacked-generalization
**Status:** in_progress
**Started:** 2026-03-09T00:00:00Z
**Rationale:** All three existing directions optimize individual model families (GBDT ensemble, XGBoost interactions, MLP). Stacked generalization trains diverse base learners (LightGBM, XGBoost, CatBoost, Extra Trees, Logistic Regression) using K-fold CV to produce out-of-fold predictions, then trains a meta-learner on those predictions. This captures complementary inductive biases. Prior EDA shows weak individual correlations but strong interactions, suggesting ensemble diversity could unlock performance no single model achieves alone.

## References
- Wolpert, "Stacked Generalization" (1992)
- Breiman, "Stacked Regressions" (1996)
- research_directions/gbdt-ensemble-tuning.md
- research_directions/xgboost-interactions.md
- research_directions/deep_learning_tabular.md

## Results
(filled on completion by submit-experiment)

## Learnings
(filled on completion by submit-experiment)
