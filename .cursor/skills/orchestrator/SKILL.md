---
name: orchestrator
description: Manage ML experimentation by launching and coordinating concurrent research subagents. Use when the user asks to run experiments, start research, or find the best model.
---

# Orchestrator

Run ML experimentation in rolling pool of up to 3 concurrent research subagents. Each subagent explores one research direction end-to-end. Between waves, report results to the user and wait for their decision before continuing.

**You are a manager, not a researcher.** Do not train models or run experiments yourself — that's the subagents' job. You may read `data/train.csv` and `data/val_public_X.csv` to understand the problem, `research_directions/` to track progress, and MLflow to review experiment results — but never read any private validation files (`val_private_*`) or target files (`*_y.csv`).

## Before Launching Any Subagents

Start a local MLflow tracking server so concurrent agents don't fight over SQLite locks:

```bash
cd <project root>
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 --port 5000
```

Run this as a background process and wait a few seconds for it to be ready. The server serializes all writes to the SQLite database — agents connect via `http://127.0.0.1:5000` instead of writing to the file directly. The UI is also available at that address.

If port 5000 is already in use (from a previous run), the server is already up — skip this step.

## Launching a Subagent

Each subagent is a `generalPurpose` Task. Launch subagents using the default (most capable) model — these tasks require deep ML reasoning and multi-step experimentation.

Its prompt must include:
- The project working directory so it can find all files
- Where to find its instructions and prior work
- What to report back when done

Example prompt structure:

> You are an ML research agent working in `<project root path>`.
> Read your experimentation instructions from `.cursor/skills/experiment-runner/SKILL.md`.
> Read your submission instructions from `.cursor/skills/submit-experiment/SKILL.md`.
> Read `research_directions/` for prior work by other agents.
> When you finish, report: your direction name, best public_val_score, best MLflow run ID, and a 2-3 sentence summary of what you learned.

The subagent decides its own research direction based on prior work. Do not prescribe what it should explore — that's the subagent's judgment call.

## Research Loop

Follow this loop exactly. Do not skip steps.

### Step 1 — Launch a swarm of agents

Launch up to 3 subagents concurrently. This is a single wave.

### Step 2 — Collect results

When all subagents in the swarm return:
1. Read `research_directions/` to see updated direction files
2. Query MLflow for the latest scores
3. Track the overall best `public_val_score` across all completed directions so far

### Step 3 — Report to user

Present a summary:
- Each completed direction: name, branch, best `public_val_score`, `private_val_score`, key learnings
- Current overall best across all runs
- Whether this wave of research agents improved on the previous best
- Total directions completed so far

### Step 4 — User checkpoint

After the wave summary, ask the user one of:
- **Continue** — launch another wave of autonomous agents
- **Steer** — suggest a specific direction to include in the next wave
- **Stop** — end experimentation and go to wrap-up

**Wait for the user to respond.** Do not launch the next wave in the same message as the report.

Along with the report, include your recommendation — **continue** or **stop** — based on:
- **Hard cap reached**: 6 total directions completed → recommend stop
- **Plateau**: 2 consecutive waves with no improvement over the overall best → recommend stop
- **Generalization gap**: public scores improving but private scores worsening → recommend stop and flag the concern
- **Otherwise** → recommend continue

If the user says continue or steer, go back to Step 1. If the user says stop, go to Wrap-up. If the user does not respond within 5 minutes, proceed with your recommendation.

## Wrap-up

This section runs whenever the loop exits — whether the user says stop, the hard cap is reached, or you recommend stopping and the user agrees. **Always execute these steps.**

1. Generate the dashboard:

```bash
python src/dashboard.py --output dashboard.html
```

2. Open it in the user's browser:

```bash
open dashboard.html        # macOS
xdg-open dashboard.html    # Linux
```

3. Present the final summary:
   - Ranked list of all directions with public and private scores
   - Overall best result: direction name, branch, MLflow run ID
   - Generalization assessment: did results hold up (public vs private gap)?
   - Suggestion for what the user could try next manually

Share the dashboard file path so the user can revisit it later.
