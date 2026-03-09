# CatBoost Hyperparameter Tuning

**Branch:** exp/catboost-tuning
**Status:** completed
**Started:** 2026-03-09T05:15:00Z
**Rationale:** CatBoost is a strong gradient boosting framework with ordered boosting and symmetric tree structure. No prior agent had explored CatBoost on this dataset. The focus was on systematic hyperparameter search: depth, learning_rate, l2_leaf_reg, random_strength, bagging_temperature, and border_count.

## References
- CatBoost documentation: https://catboost.ai/en/docs/
- Prior directions: gbdt-ensemble-tuning (best 0.7815), xgboost-interactions (best 0.7793), stacked-generalization (best 0.7820)

## Results
- **public_val_score:** 0.7826
- **private_val_score:** 0.7740
- **Best MLflow run ID:** 410413c3d8124f41925c1991a31b19b7
- **Number of runs:** 16
- **Best config:** depth=8, learning_rate=0.05, l2_leaf_reg=3.0, iterations=450, SymmetricTree

## Learnings

### What worked
- **Depth 8 with moderate learning rate (0.05) was the sweet spot** (run 410413c3): Deeper trees (8) captured feature interactions that shallower trees missed, while lr=0.05 was slow enough to avoid overshooting but fast enough to converge in reasonable iterations.
- **Training on full data with a fixed iteration count** outperformed using a holdout for early stopping. Early stopping on a 90/10 split suggested ~400 iterations; using 450 on the full dataset with all 50k samples gave the best score.
- **CatBoost's default l2_leaf_reg=3 was optimal.** Both lower (1.0) and higher (10.0) values hurt performance.

### What didn't work
- **Very deep trees (depth=10, run 42259e3b):** Overfit despite early stopping. 0.7784 vs 0.7826 for depth=8.
- **High regularization (l2_leaf_reg=10, runs f308c103 and f40ecf1a):** Over-regularized the model, scores dropped to 0.7779-0.7788.
- **CatBoost-specific randomization features** (random_strength, bagging_temperature): Adding random_strength=1-2 and bagging_temperature=0.5-0.8 didn't improve over the base config (runs 0894bfb6, 2b0673a6). CatBoost's ordered boosting already provides sufficient regularization.
- **Lossguide (leaf-wise) growth policy (run 4a5d7ed4):** Dramatically worse at 0.7678. CatBoost's native SymmetricTree is clearly better suited than trying to emulate LightGBM-style growth.
- **Lower learning rate (0.03) with more iterations (run d05e56d4):** Score of 0.7783, likely overfitting with 1200 iterations even at the slower rate.
- **Reduced border_count (128, run 45fdd164):** Coarser quantization hurt at 0.7792.

### Surprising findings
- The public-private gap was ~0.009 (0.7826 vs 0.7740), suggesting some degree of overfitting to the public split. This is within normal range for this dataset size but worth noting for ensemble/stacking approaches.
- CatBoost's baseline (depth=6, lr=0.1, no early stopping) at 0.7745 was competitive with many tuned XGBoost configs, confirming CatBoost's reputation for strong out-of-the-box performance.

### What to try next
- Ensemble this CatBoost model with the best XGBoost and LightGBM models from other directions — diversity between frameworks should reduce variance.
- Try CatBoost with ordered boosting mode explicitly set to `Ordered` vs `Plain` to see if the ordering matters.
- Feature engineering (interactions, PCA components) could help all gradient boosting models.
