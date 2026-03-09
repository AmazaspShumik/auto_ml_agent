# XGBoost Hyperparameter Tuning

**Branch:** exp/xgboost-hyperparam-tuning
**Status:** completed
**Started:** 2026-03-09T12:00:00Z
**Rationale:** XGBoost is a strong baseline for tabular binary classification. This direction systematically explores its hyperparameter space — learning rate, tree depth, regularization (L1/L2), column/row subsampling, and min_child_weight — to find a well-tuned configuration that maximizes ROC AUC without overfitting.

## References
- XGBoost documentation: https://xgboost.readthedocs.io/en/stable/parameter.html

## Results
- **public_val_score:** 0.566408
- **private_val_score:** 0.749578
- **Best MLflow run ID:** 0f3bf1112f4c49ed97a816e19d3aa924
- **Number of runs:** 12

## Learnings

### What worked (and why)
- **Multi-level column sampling was the single most impactful technique** (run `0f3bf111`). Setting colsample_bytree=0.7, colsample_bylevel=0.7, colsample_bynode=0.7 achieved the best public_val_score=0.5664. This forces strong decorrelation between trees by sampling features at every split decision level, not just at the tree level.
- **Aggressive row subsampling** (subsample=0.6) was the second best technique (run `00b9c819`, public_val=0.5644). Combined with column subsampling, it reduces overfitting by introducing randomness similar to Random Forests.
- **Shallow trees (depth=4)** consistently outperformed deeper trees (depth=6, 8). Run `60fe5fd7` (depth=4) scored 0.5612 vs run `116b2b3d` (depth=8) at 0.5583. This suggests the useful signal is in low-order feature interactions.
- **Lower learning rates (0.02-0.03) with more rounds** generally helped when combined with good regularization, but the improvement was modest compared to subsampling effects.

### What didn't work (and why)
- **Strong explicit regularization (L1/L2, gamma)** had modest effects. Run `5c282887` (lambda=5) scored 0.5603 and run `758d0c90` (alpha=1) scored 0.5611 — both below simpler subsampling approaches. The implicit regularization from subsampling was more effective than explicit weight penalties for this dataset.
- **Combined heavy regularization** (run `17229950`, lambda=3 + alpha=0.5 + mcw=3) scored only 0.5601, the worst of the regularized configs. Over-regularizing multiple axes simultaneously hurt more than it helped.
- **Deeper trees (depth=8)** (run `116b2b3d`) scored 0.5583 with high overfitting (train AUC=0.994). The dataset doesn't benefit from modeling high-order interactions.

### Surprising findings
- The **private_val_score (0.7496) is dramatically higher than public_val_score (0.5664)** but very close to CV AUC (0.7584). This suggests the public validation split has a systematically different distribution from the training data, while the private split is more representative. Future agents should note that CV metrics are more reliable than public_val_score for this dataset.
- The large **train-val gap** (~0.85 train AUC vs ~0.57 public val) persisted across all configurations, even heavily regularized ones. This is likely intrinsic to the public validation split rather than pure overfitting.

### What to try next
- Dart booster mode (dropout for gradient boosting) could further reduce correlation between trees
- Feature engineering: polynomial interactions of top features may help, given that shallow trees worked best
- Stacking XGBoost with other model families (LightGBM, CatBoost, linear models) for diversity
- Bayesian optimization for fine-tuning the best config found here
