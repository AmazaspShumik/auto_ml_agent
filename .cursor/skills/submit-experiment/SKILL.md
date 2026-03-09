---
name: submit-experiment
description: Wrap up a research direction — run private evaluation, log results to MLflow, update your research direction file, and merge to main. Use when done experimenting on a branch and ready to finalize.
---

# Submit Experiment

Finalize a research direction: run private evaluation on your best candidate, log the result to MLflow, record learnings, and merge to main

## When to Use

- When you have finished exploring a research direction on this branch
- After selecting your best MLflow run by `public_val_score`
- This is a one-shot action per branch — do not call repeatedly

## Instructions

1. Pick your best run — query MLflow for the run with the highest `public_val_score` on this branch
2. Generate private predictions — load your best model and produce predictions for `data/val_private_X.csv`
3. Run private evaluation — `submit.py` evaluates against private targets and logs `private_val_score` to the best MLflow run:

```bash
python src/submit.py predictions_private.csv <mlflow_run_id>
```

4. Switch to main and merge results:

```bash
git checkout main
git pull
git checkout <branch> -- mlruns/
git commit -m "Merge MLflow runs from direction: <direction-name>"
```

5. **Update your research direction file** — edit `research_directions/<your-direction>.md` on main:
   - Set status to `completed`
   - Fill in **Results**: `public_val_score`, `private_val_score`, best MLflow run ID, number of runs tried
   - Fill in **Learnings** — this is the most valuable part for future agents:
     - What worked and why (link to specific MLflow run IDs as evidence)
     - What didn't work and why (link to those runs too)
     - Surprising findings or unexpected behavior
     - What you'd try next if continuing this direction
     - References to papers, blog posts, or techniques that were useful
   - Commit and push

6. **Do not delete the branch** — it stays as an archive of the code that produced these results.

## What Lands on Main

- All MLflow runs from this branch (including `private_val_score` on the best run)
- Updated direction file with results, learnings, and references
- Nothing else — training code stays on the branch

## Why This Matters

Main is the shared knowledge base for all agents. Other agents — potentially running concurrently — read `research_directions/` and query MLflow before choosing their own direction. Clean, conflict-free merges keep this working.

## Hard Safety Rules

- Never use `private_val_score` as an optimization target. Use it only to assess generalization — a large public-private gap is a red flag worth investigating, not a number to maximize.
- Never modify `src/submit.py`
- Never call this more than once per branch
- You must be on a feature branch, not main
- The best run must have a `public_val_score` logged in MLflow
