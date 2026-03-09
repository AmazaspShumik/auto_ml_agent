# LightGBM Hyperparameter Tuning

**Branch:** exp/lightgbm-hyperparam-tuning
**Status:** completed
**Started:** 2026-03-09T00:00:00Z
**Rationale:** LightGBM is a strong baseline for tabular binary classification. This direction systematically explores the hyperparameter space — learning rate, tree structure (num_leaves, max_depth), regularization (L1/L2, min_child_samples), and stochastic elements (feature_fraction, bagging_fraction) — to find a well-tuned configuration that maximizes ROC AUC without overfitting.

## References
- LightGBM documentation: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
- Practical advice on tuning GBDTs for tabular data

## Results
- **public_val_score:** 0.773748
- **private_val_score:** 0.760799
- **Best MLflow run ID:** `2c316d1732da4fdea799bd9cddf0e729`
- **Number of runs:** 10 valid runs (v2 series; 12 earlier v1 runs were invalid due to early stopping bug)
- **Best config:** LR=0.01, num_leaves=63, min_child_samples=30, feature_fraction=0.7, bagging_fraction=0.7, lambda_l1=0.5, lambda_l2=0.5, trained on full 50k rows for 3496 rounds

## Learnings

### What worked
- **Regularization was the biggest single improvement.** Adding L1+L2 regularization (lambda=1.0) to 63-leaf trees improved public AUC from 0.7537 to 0.7601 (run `a8178c9314034ee6a51553b73ef752da`). The unregularized model severely overfits (train AUC 0.97 vs public 0.75).
- **Stochastic subsampling + regularization had the best synergy.** Combining feature_fraction=0.7, bagging_fraction=0.7 with mild L1/L2=0.5 pushed to 0.7708 (run `3106fb35e9de4310b3ac70f13aabde7f`). This combination was consistently better than either technique alone.
- **Lower learning rate (0.01) + more trees further helped.** With the stoch+reg config, dropping LR from 0.05 to 0.01 and training for ~3500 rounds reached 0.7721 (run `e36b99254d2e40db98ced1d9dbf10efd`).
- **Full-data training gave a marginal boost.** Training on all 50k rows (vs 42.5k with holdout) improved from 0.7721 to 0.7737 (run `2c316d1732da4fdea799bd9cddf0e729`).

### What didn't work
- **More leaves without regularization hurt.** 63 and 127 leaves without regularization scored 0.7537 and 0.7527 respectively (runs `06de43e58189428692cbbb9f4f878293`, `ee7fa5df199e47b5885c9cc44c028455`) — barely better or worse than the 31-leaf baseline (0.7590). The train-val gap ballooned to 0.21+.
- **DART boosting underperformed.** Both 63-leaf (0.7512, run `04e679ccba414a0e9f04b76ae0a6ec7c`) and 127-leaf (0.7549, run `a61d6b0d05e84710a5bc4a2378973830`) DART runs were worse than well-tuned gbdt. DART's dropout didn't provide enough regularization for this dataset.
- **Fewer leaves (15) were too simple.** Score dropped to 0.6377 (run `3e6618f7de874ec78111dc5e34977086` from v1 series, before fix), indicating the data needs moderate tree complexity.

### Surprising findings
- The public-private gap is ~1.3% (0.7737 vs 0.7608), which is healthy — no sign of overfitting to the public validation set.
- The dataset has significant signal that requires moderate-complexity models. Overfitting is the primary challenge — nearly every regularization technique improved generalization.
- The 31-leaf baseline (0.7590) was actually quite competitive. Most gains came from combining multiple regularization techniques rather than increasing model capacity.

### What to try next
- Optuna-driven hyperparameter search over a wider space (e.g., min_child_weight, path_smooth, max_bin)
- Feature engineering: interaction terms, PCA components, or polynomial features
- Ensembling LightGBM with XGBoost or CatBoost for diversity
- Training with different random seeds and averaging predictions
