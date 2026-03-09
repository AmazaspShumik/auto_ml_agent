# Feature Engineering for Gradient Boosting

**Branch:** exp/feature-engineering
**Status:** completed
**Started:** 2026-03-09T18:00:00Z
**Rationale:** All prior directions used raw features (f00-f29) as-is. Gradient boosting models can benefit from explicit feature interactions, statistical aggregations, and dimensionality reduction features. This direction systematically explores engineered features on top of the best model (CatBoost with depth=8, lr=0.05). Strategy: run feature importance first, then target engineering at high-importance features.

## References
- Prior directions: catboost-tuning (best 0.7826), lightgbm-hyperparam-tuning (best 0.7738)
- CatBoost feature importance: https://catboost.ai/en/docs/concepts/fstr

## Results
- **public_val_score:** 0.781646
- **private_val_score:** 0.770585
- **Best MLflow run ID:** 4bb87b156263443a8a6d550970b48575
- **Number of runs:** 12
- **Best config:** CatBoost depth=8, lr=0.05, l2_leaf_reg=3.0, 450 iterations + row-wise statistics (mean, std, min, max, range, skew, abssum) over top 12 features

## Learnings

### Feature importance analysis (pre-experiment)
- Feature importance is clearly tiered: f03 (11.6%), f02 (11.4%), f07 (8.4%), f06 (8.0%) dominate, with f00-f11 forming the important group and f12-f29 each contributing <1.6%.
- All 30 features are **completely uncorrelated** (max pairwise |r| < 0.01) and standardized (mean~0, std~1). This makes PCA useless and suggests the features may already be principal components or independent sources.
- **f06*f07 is the dominant interaction** (CatBoost interaction strength 5.46, 3x stronger than any other pair). f02*f03, f02*f01, f00*f03 form a secondary tier (1.5-1.7).

### What didn't work (and this is the main finding)
- **No feature engineering approach beat the Wave 1 CatBoost baseline of 0.7826.** All 12 runs scored between 0.7774 and 0.7816. This is a strong negative result.
- **Pairwise interaction products** (run `1e7932bf`, 0.7810): CatBoost depth-8 already models these via multi-level splits. Explicit products are redundant.
- **Polynomial features (squares)** (run `4725fc05`, 0.7807): Squared features of standardized inputs are trivially approximated by split thresholds on both sides of zero.
- **Ratio/difference features** (run `1c57f108`, 0.7792): Worst performer. Ratios are noisy when denominators approach zero, and differences are near-redundant with originals.
- **More iterations with FE features** (run `6184960a`, 0.7795): More training capacity led to overfitting, not better integration of new features.
- **Lower learning rate** (run `ee0e8968`, 0.7774): Worst overall. Consistent with CatBoost tuning finding that lr=0.03 overfits with many iterations.
- **Absolute value features** (run `da8d2ec9`, 0.7788): V-shaped relationship already capturable by tree splits.

### What came closest
- **Row-wise statistical aggregations** (run `4bb87b15`, 0.7816): Mean, std, min, max, range, skew, abssum across top 12 features. This was the only feature type that captures genuine cross-feature information (per-sample "profile shape") that single-feature splits cannot directly represent. Still fell short by 0.001.
- **Feature selection (dropping bottom 18)** (run `f73723b9`, 0.7813): Removing 60% of features barely hurt, confirming CatBoost already ignores them. But no improvement either.

### Why feature engineering doesn't help this dataset
1. **Features are already orthogonal and standardized** — typical FE targets (rescaling, decorrelation, normalization) are unnecessary.
2. **CatBoost depth-8 trees are powerful enough** — with 2^8 = 256 leaves per tree and 450 trees, the model can represent complex interaction patterns without explicit features.
3. **The signal-to-noise ratio is moderate** (AUC ~0.78) — adding features increases dimensionality without adding information, slightly diluting the signal.

### What to try next
- **Ensembling/stacking** across model families (CatBoost + LightGBM + XGBoost) — diversity between models, not within features, is the likely path to improvement.
- **Target encoding or frequency encoding** if any features have hidden categorical structure (though the continuous distributions suggest they don't).
- **Neural network approaches** (TabNet, FT-Transformer) that learn feature interactions via attention — these might discover patterns that GBDT misses entirely.
- **Noise injection / adversarial validation** to understand why the public-private gap persists across all models (~0.009-0.013).
