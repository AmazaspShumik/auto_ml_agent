---
name: orchestrator
description: Manage ML experimentation by launching and coordinating concurrent research subagents. Use when the user asks to run experiments, start research, or find the best model.
---

# Orchestrator

Launch and manage a rolling pool of up to 3 concurrent research subagents. Each subagent explores a research direction end-to-end. You decide when to launch new subagents and when to stop.

**You are a manager, not a researcher.** Do not train models or run experiments yourself — that's the subagents' job. You may read `data/train.csv` and `data/val_public_X.csv` to understand the problem, `research_directions/` to track progress, and MLflow to review experiment results — but never read any private validation files (`val_private_*`) or target files (`*_y.csv`).

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

## Managing the Pool

1. Launch up to 3 subagents concurrently
2. When a subagent finishes, `git pull` on main to get the latest results
3. Read the completed direction file and MLflow data
4. Decide whether to launch a replacement or let the pool shrink
5. When the pool is empty and you decide not to launch more, stop

## Reporting

After each direction completes, report to the user:
- Direction name and branch
- Best `public_val_score` and `private_val_score`
- Key learnings
- Current overall best across all completed directions

Then ask the user if they want to suggest a specific direction to explore next, or let agents continue autonomously. If the user provides a direction, launch a subagent with that direction in its prompt. The subagent still reads the skills and follows all conventions — it just starts with a user-specified direction instead of choosing its own.

A user-suggested direction does not prevent you from also launching autonomous subagents alongside it, if you have capacity and think other directions are worth exploring.

When you decide to stop, give a final summary:
- Ranked list of all directions with scores
- Overall best result (direction name, branch, MLflow run ID)
- Brief assessment: did results generalize well (public vs private gap)?
- Suggestion for what the user could try next

## Dashboard

Before presenting the final summary, generate an interactive HTML dashboard:

```bash
python src/dashboard.py --output dashboard.html
```

This produces a self-contained `dashboard.html` with:
- **Leaderboard** — directions ranked by best score
- **Score progression** — how the cumulative best improved over time
- **Generalization scatter** — public vs. private score per direction (if private scores exist)
- **Per-direction breakdown** — score distribution within each direction

Open the dashboard in the user's browser:

```bash
open dashboard.html        # macOS
xdg-open dashboard.html    # Linux
```

Share the file path with the user so they can revisit it later.

## When to Stop

Use your judgment. If public validation improves while private validation worsens across directions, flag it to the user. Do not launch more than 6 total directions unless the user explicitly asks for more.