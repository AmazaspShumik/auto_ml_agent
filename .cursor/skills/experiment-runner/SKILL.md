---
name: experiment-runner
description: Explore a research direction through one or more ML experiments — train, sweep, evaluate, and log to MLflow. Supports single runs and multi-run hyperparameter optimization. Use when training models, running experiments, tuning hyperparameters, or when the user mentions experiments, training, evaluation, or metrics.
---

# Experiment Runner

Explore a research direction through one or more MLflow runs. You have full freedom of experimentation, be creative and thorough. Follow the conventions below so the user can understand what happened, and never cross the safety lines. Your goal is to improve the metric on public validation set, without overfitting it (so that results generalize to private validation as well).

## Before You Start

1. **Pull latest and review prior work** — `git pull`, then read all files in `research_directions/` to see what other agents have explored, what's in progress, and what was learned. Pay attention to the Learnings sections — they contain insights you should build on, not repeat.
2. **Create your branch** — branch off main so all exploratory work stays off main.
3. **Choose a direction** that complements prior work — don't repeat what's been done. Use your judgment on what preparation you need: EDA, web research, reading papers — whatever helps you make a well-informed choice.
4. **Switch to main and register your direction** — create `research_directions/<your-direction-name>.md` with status `in_progress`:

```markdown
# <Direction Title>

**Branch:** exp/<branch-name>
**Status:** in_progress
**Started:** <ISO timestamp>
**Rationale:** <why this direction is worth exploring>

## References
- <links to papers, blog posts, or prior direction files that informed this choice>

## Results
(filled on completion by submit-experiment)

## Learnings
(filled on completion by submit-experiment)
```

5. Commit, push to main, then switch back to your branch and start experimenting.

Registering your direction on main before experiments begin is critical — it signals your intent to other concurrent agents.

## MLflow Tracking URI

All agents must use the SQLite backend. Set this **before any MLflow calls** in every script:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

The path is relative to the project root — make sure your working directory is the project root when running experiments. The orchestrator creates `mlflow.db` before launching subagents, so it will already exist.

The SQLite database is the source of truth for run metadata (metrics, params, tags). MLflow still writes model artifacts to `mlruns/` on disk — that directory must be preserved and merged to main via the submit-experiment workflow.

## Logging Conventions

Every MLflow run MUST have these tags:
- `git.branch` — current git branch
- `direction_rationale` — use the same rationale from your `research_directions/` file. Shared across all runs in the same direction.
- `run_rationale` — why this specific configuration within the direction. This covers the low-level choice: a particular hyperparameter setting, a regularization strength, a specific feature subset.
- `run_analysis` — written *after* the run completes. What happened, why you think it happened, and what it suggests for next steps. Not all improvements can be fully explained, but ground your analysis in evidence from this run, prior experiments, and your ML knowledge. Use any explainability techniques when they'd help — just be aware some can be slow, so probably not worth doing at every run. Searching the web for relevant research is encouraged.

For example, you decide to try a convolutional neural network because the data has spatial structure (inductive bias) — that reasoning goes in `direction_rationale`. Within that direction, you try kernel sizes of 3, 5, and 7 to find the right receptive field — each of those is a separate `run_rationale`.

Every MLflow run MUST have these metrics:
`public_val_score` - validation metric computed on the public validation set. See the Evaluation section below for details.

Every MLflow run MUST log:
- Hyperparameters as params
- The trained model as an artifact via the appropriate `mlflow.<framework>.log_model()` flavor

Every run SHOULD log (when applicable):
- Per-step train/validation metrics for iterative models

### Model Logging

Every run must persist its trained model so it can be loaded later for prediction, ensembling, or submission — never rely on retraining to reproduce results. Use the appropriate `mlflow.<framework>.log_model()` flavor and always use `"model"` as the artifact path.

For composite models (stacking, ensembles), save each component as a separate artifact in the run (base models, meta-learner, blend weights, etc.) and implement a loading/prediction function on your branch that reassembles them. The branch code is the recipe; the MLflow artifacts are the ingredients.


## Evaluation

Use `evaluate_public()` from `src/evaluate.py` to score on public validation. Log the result as `public_val_score` in the MLflow run.

```python
import sys
sys.path.insert(0, "src")
from evaluate import evaluate_public
score = evaluate_public(preds)
```


## Training and Validation data

`data/train.csv`  - Training data (features + target)
`data/val_public_X.csv` - Public validation features

Private validation files (`val_private_*`) exist in `data/` but are off-limits during experimentation. They are only used at submission time by `submit.py`.


## Knowing when to stop:

If large number of experiments don't lead to significant improvement it might be better to wrap current research direction and submit your results. It's better to explore a fundamentally different approach than to micro-optimize within a plateaued direction


## Submission

When finished experimenting, use the `submit-experiment` skill.


## Hard Safety Rules

It is vitally important to NOT jeopardize quality of experimentation.

**Scientific rigor:**
- Never leak validation data into training (features, targets, or statistics derived from validation set)
- Never tune decisions directly against `public_val_score` to the point of overfitting the public split — if a change only helps by a tiny margin, be skeptical
- Always fill in `run_analysis` after a run — this is where you reason about results before deciding what to try next.
- Log bad runs too - deleting or hiding failures corrupts the experimental record



**Data integrity:**
- Never read `*_y.csv` files directly
- Never read `val_private_X.csv` or any private validation file — these are only used at submission time
- Never call `evaluate_private()` directly — only `submit.py` may call it
- Never run `src/submit.py` directly — use the `submit-experiment` skill
- Never use `private_val_score` as an optimization target. Use it only to assess generalization — a large public-private gap is a red flag worth investigating, not a number to maximize.
- Predictions must be probabilities (floats), not class labels
