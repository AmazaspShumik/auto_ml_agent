# Ensemble / Blending / Stacking

**Branch:** exp/ensemble-blending
**Status:** completed
**Started:** 2026-03-09T16:00:00Z
**Rationale:** CatBoost (0.7826 public) and LightGBM (0.7738 public) capture different patterns via different boosting algorithms and tree structures. Combining their predictions — via averaging, optimized weighting, and stacking with a meta-learner — should reduce variance and improve generalization. XGBoost (anomalous 0.5664 public but 0.7496 private) may add diversity if handled carefully.

## References
- research_directions/catboost-tuning.md (best public 0.7826, run 410413c3)
- research_directions/lightgbm-hyperparam-tuning.md (best public 0.7738, run 2c316d17)
- research_directions/xgboost-hyperparam-tuning.md (best public 0.5664, run 0f3bf111)
- Wolpert (1992) Stacked Generalization
- Breiman (1996) Stacked Regressions

## Results
- **public_val_score:** 0.7826 (best run; equivalent to solo CatBoost since ensemble weights converged to 100% CatBoost)
- **private_val_score:** 0.7740
- **Best MLflow run ID:** fb16536f8d534227a4688f535f024dcd
- **Best genuine ensemble score:** 0.7810 (3-model stacking, run 05a4ceda75ac49dca769559612789852)
- **Number of runs:** 12

## Learnings

### What worked
- **3-model stacking (CB+LGB+XGB) with LogReg meta-learner achieved 0.7810** (run `05a4ceda75ac`): Using 5-fold CV out-of-fold predictions as features with LogReg(C=100) gave the best genuine ensemble score. The meta-learner assigned coefficients [4.42, 0.39, 0.43] showing CatBoost dominates but XGBoost adds complementary signal.
- **2-model stacking (CB+LGB) with tuned LogReg achieved 0.7793** (run `552c32d3f178`): Even without XGBoost, stacking captured some complementary signal between CatBoost and LightGBM OOF predictions.
- **Out-of-fold training avoids the serialization issue**: Fresh CatBoost and LightGBM models trained per fold during stacking worked correctly, unlike the broken LightGBM model from the registry.

### What didn't work
- **Ensembling with the loaded LightGBM model failed**: The LightGBM model from the MLflow Model Registry (serialized with cloudpickle) produced AUC ~0.498 (random) when loaded and used for prediction. Simple averaging with this broken model dragged scores down to 0.690 (run `0c877a4312e6`).
- **Retraining LightGBM from scratch also underperformed**: Using the reported best hyperparameters (LR=0.01, 63 leaves, 3496 rounds, full data) only achieved AUC 0.558 — far below the reported 0.7738. The original LightGBM model was likely trained with branch-specific preprocessing or feature engineering that wasn't captured in the hyperparameters alone.
- **Weighted averaging converged to 100% CatBoost** (run `fb16536f8d53`): The optimizer found that neither LightGBM (0.558 AUC retrained) nor XGBoost (0.759 AUC) could improve on CatBoost (0.783) when combined via linear weights.
- **Rank averaging performed poorly** (run `da50711fd770`, 0.696): Converting to ranks before averaging didn't help — the broken LightGBM ranks actively harmed the ensemble.
- **Power-mean ensemble hurt** (run `4b98ef155b8f`, 0.727): Non-linear combination with broken base model was worse than simple averaging.

### Surprising findings
- **The LightGBM model cannot be reproduced from hyperparameters alone.** The gap between reported (0.7738) and reproduced (0.558) public val score is massive. This suggests the original training pipeline included feature engineering, data preprocessing, or other transformations that weren't logged as MLflow artifacts. Future agents should ensure all preprocessing is saved with the model.
- **XGBoost loaded from registry works fine at 0.7593** — dramatically higher than its reported public score of 0.5664. The original XGBoost agent likely had a bug in its evaluation pipeline (possibly column ordering), not an actual modeling issue. The model itself is well-trained.
- **The public-private gap (0.7826 vs 0.7740 = 0.009) is consistent** across CatBoost whether used solo or in an ensemble, reinforcing that it's a genuine distribution shift rather than overfitting.
- **No ensemble strategy beat solo CatBoost on public val.** This is unusual and suggests either (a) the base models aren't sufficiently diverse, or (b) the models that should provide diversity (LightGBM, XGBoost) couldn't be properly leveraged due to reproducibility issues.

### What to try next
- **Investigate LightGBM reproducibility**: Check the `exp/lightgbm-hyperparam-tuning` branch for preprocessing steps, feature engineering, or data transformations that aren't reflected in the saved hyperparameters.
- **Train diverse base models from scratch in a single pipeline**: Rather than loading pre-trained models, train CatBoost, LightGBM, and XGBoost together in a shared pipeline with consistent preprocessing, then ensemble.
- **Try different ensemble diversity strategies**: Use models with different feature subsets, different random seeds (seed averaging), or different hyperparameter regimes to maximize diversity.
- **Feature engineering**: PCA components, interaction features, or target encoding might help all models and create more useful ensemble diversity.
