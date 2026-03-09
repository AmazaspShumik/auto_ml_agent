# Deep Learning for Tabular Classification

**Branch:** exp/deep-learning-tabular
**Status:** in_progress
**Started:** 2026-03-09T00:00:00Z
**Rationale:** EDA reveals weak individual feature correlations (only f00 and f01 have |r| > 0.1 with target) but nearly balanced classes and standardized features — ideal conditions for neural networks that can discover complex non-linear feature interactions. Both existing directions focus on tree-based methods (GBDT ensemble, XGBoost). This direction explores MLP architectures with modern regularization (batch normalization, dropout, weight decay, learning rate scheduling) to capture interaction effects from a fundamentally different inductive bias.

## References
- Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data" (2021)
- Kadra et al., "Well-tuned Simple Nets Excel on Tabular Datasets" (2021)

## Results
(filled on completion by submit-experiment)

## Learnings
(filled on completion by submit-experiment)
