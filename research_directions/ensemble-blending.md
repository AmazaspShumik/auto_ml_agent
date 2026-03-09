# Ensemble / Blending / Stacking

**Branch:** exp/ensemble-blending
**Status:** in_progress
**Started:** 2026-03-09T16:00:00Z
**Rationale:** CatBoost (0.7826 public) and LightGBM (0.7738 public) capture different patterns via different boosting algorithms and tree structures. Combining their predictions — via averaging, optimized weighting, and stacking with a meta-learner — should reduce variance and improve generalization. XGBoost (anomalous 0.5664 public but 0.7496 private) may add diversity if handled carefully.

## References
- research_directions/catboost-tuning.md (best public 0.7826, run 410413c3)
- research_directions/lightgbm-hyperparam-tuning.md (best public 0.7738, run 2c316d17)
- research_directions/xgboost-hyperparam-tuning.md (best public 0.5664, run 0f3bf111)
- Ensemble methods: Wolpert (1992) Stacked Generalization, Breiman (1996) Stacked Regressions

## Results
(filled on completion by submit-experiment)

## Learnings
(filled on completion by submit-experiment)
