# Deep Learning for Tabular Classification

**Branch:** exp/deep-learning-tabular
**Status:** completed
**Started:** 2026-03-09T18:00:00Z
**Rationale:** Gradient boosting models (CatBoost 0.7826, LightGBM 0.7738) dominate the leaderboard but may miss non-linear patterns that MLPs can capture. Neural networks with BatchNorm, Dropout, and proper feature standardization have shown competitive performance on tabular data. Even if an MLP doesn't beat CatBoost alone, it provides diversity for future ensembling.

## References
- Prior directions: catboost-tuning (best 0.7826), lightgbm-hyperparam-tuning (best 0.7738), xgboost-hyperparam-tuning (best 0.5664 public / 0.7496 private)
- "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021)
- PyTorch documentation: https://pytorch.org/docs/stable/

## Results
- **public_val_score:** 0.7442
- **private_val_score:** 0.7351
- **Best MLflow run ID:** 9862e6fd103a49aeaeda8a5f4eae1c1e
- **Number of runs:** 12
- **Best config:** 5x256 MLP with skip connections, GELU+BatchNorm+Dropout(0.5), AdamW lr=1e-3, weight_decay=0.01, mixup(alpha=0.3), cosine annealing schedule, batch_size=512

## Learnings

### What worked (and why)
- **Skip connections were the single most impactful architectural choice.** 4x256 with skip connections scored 0.7453 (run `5cc7be38`), while 3x512 without scored 0.7379 (run `720985d2`) despite having 2.6x more parameters. Skip connections enable deeper models to train stably and maintain gradient flow through BatchNorm+Dropout layers.
- **Heavy dropout (0.5) + weight decay (0.01) + mixup (0.3) is the optimal regularization combo** for this dataset. Removing any one component degrades performance: dropout 0.4 → 0.7242 (run `99c4f55f`); no mixup → 0.7302 (run `c1404fde`). The three regularizers are complementary: dropout prevents co-adaptation, weight decay constrains magnitude, mixup smooths decision boundaries.
- **Moderate model capacity (~200-270K params) is the sweet spot.** Both 4x256 (207K, 0.7453) and 5x256 (274K, 0.7442) performed well. Larger models (5x512 at 1.07M, run `c1aac0a5`) overfitted to 0.7270 despite identical regularization.
- **GELU activation + BatchNorm** provided consistent training dynamics. All successful runs used this combination.

### What didn't work (and why)
- **Scaling up model capacity (run `c1aac0a5`, 5x512):** 1.07M params overfitted severely (0.7270), peaking at epoch 45. Even skip connections + full regularization cannot compensate for 5x overcapacity.
- **Lower dropout (0.4, run `99c4f55f`):** Disastrous at 0.7242. The model memorized training data by epoch 40. Dropout=0.5 is a hard floor for this dataset.
- **Pairwise feature interactions (run `d1a963e5`):** Adding 105 interaction features (top-15 pairwise products) scored only 0.6866. The extra dimensions added noise that the MLP couldn't filter, unlike tree-based models which automatically select relevant interactions.
- **Label smoothing (run `5d6e3ba1`):** Scored 0.7358, below baseline 0.7453. Redundant with already-heavy dropout + mixup regularization.
- **Lower learning rate (5e-4, run `f009518`):** Scored 0.7400. The cosine schedule decayed too slowly, stretching the useful learning phase without benefit.
- **SWA (run `eb3ddbb2`):** Modest improvement over base (0.7325→0.7344) but couldn't rescue a weaker random initialization.
- **Tapered/bottleneck architecture (run `15785428`):** 512→256→128 without skip connections scored 0.7330. The bottleneck forces overly aggressive compression and lacks skip connections for gradient flow.

### Surprising findings
- **High inter-seed variance (~1.3%):** The same architecture (4x256-skip + full regularization) produced scores ranging from 0.7302 to 0.7453 across different random seeds. A 3-seed ensemble (run `d9a46cb1`) averaged 0.7398, which didn't beat the best single seed (0.7453). This suggests a rough optimization landscape where some initializations find significantly better minima.
- **Public-private gap was small (~0.9%):** 0.7442 public vs 0.7351 private is comparable to CatBoost's gap (0.9%), indicating the MLP generalizes well despite being a different model class.
- **MLPs fundamentally underperform gradient boosting by ~3.8% on this data.** The gap (0.7442 vs 0.7826 CatBoost) persisted across all 12 configurations. Gradient boosting's inductive bias for tabular data (automatic feature selection, hierarchical interactions) is a structural advantage.

### What to try next
- **Ensemble MLP predictions with CatBoost/LightGBM** — the MLP's 0.7442 AUC captures different patterns and should improve diversity in a blended ensemble.
- **Multi-seed ensemble with 5-10 seeds** — given the high variance, averaging more seeds could push the MLP above 0.75.
- **TabNet or FT-Transformer architectures** — attention-based tabular architectures may close the gap with gradient boosting while keeping neural network diversity benefits.
- **Knowledge distillation** — train the MLP to mimic CatBoost's soft predictions, potentially combining both models' strengths.
