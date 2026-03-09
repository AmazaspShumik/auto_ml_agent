# ML Agent Orchestration System

A template for autonomous ML experimentation using Cursor AI agents. Small agent swarm explore different research directions concurrently, log everything to MLflow, and communicate findings through a shared knowledge base on `main`.

## How It Works

```
                         main branch (knowledge base)
                        ┌─────────────────────────────────┐
                        │  research_directions/            │
                        │    ├── deeper-trees.md           │
                        │    ├── feature-interactions.md   │
                        │    └── ...                       │
                        │  mlruns/  (model artifacts)      │
                        │  src/    (evaluation scripts)    │
                        └──────┬──────────┬────────────────┘
                               │          │
                    ┌──────────┘          └──────────┐
                    ▼                                ▼
            exp/deeper-trees              exp/feature-interactions
            (Subagent 1)                  (Subagent 2)
            - trains models               - trains models
            - logs to MLflow              - logs to MLflow
            - submits results             - submits results
```

### The orchestrator launches research agents

- The **orchestrator** (main agent) manages a rolling pool of up to 3 concurrent subagents (3 here is just arbitatry constraint)
- Each **subagent** picks a research direction, experiments freely, logs everything to MLflow, and submits results back to `main`
- Between completions, the orchestrator reports progress to the user and decides whether to launch more agents

### Agents communicate through files on main

- Before experimenting, each agent writes a `research_directions/<name>.md` file on `main` declaring its intent
- After finishing, the agent updates that file with results, learnings, and references
- The next agent reads all direction files to understand what's been tried and what was learned

### MLflow tracks all experiment data

- Every run logs hyperparameters, per-step training/validation metrics, and the trained model as an artifact
- Each run has a `direction_rationale` (why this approach), `run_rationale` (why this config), and `run_analysis` (what happened and why)
- Private evaluation scores are logged to the best run from each direction
- View all results at `localhost:5000` by running `mlflow ui --backend-store-uri sqlite:///mlflow.db`

### Human in the loop

The system is autonomous but not a black box. After each direction completes, the orchestrator reports results and asks if you want to steer:

- **Let it run** — agents continue choosing directions autonomously
- **Suggest a direction** — tell the orchestrator what to try next ("ensemble the top two models", "try a neural net approach") and it launches a subagent with your guidance
- **Stop** — halt experimentation and get a final summary

You can also inspect progress at any time via `mlflow ui` or by reading the `research_directions/` files. The agents work for you, not instead of you.

### Only results merge to main

- Training code, predictions, and model files stay on the branch — they are committed there as a complete archive
- Only `mlruns/` and `research_directions/` merge to `main`
- Branches are never deleted — they serve as archives of everything that produced the results

## Evaluation Design: Public/Private Split

The validation data is split into two sets with targets (`*_y.csv`) separated from features (`*_X.csv`):

- **Public validation** — agents use this freely during experimentation. `evaluate_public()` scores predictions against `val_public_y.csv` and the result (`public_val_score`) drives all experiment decisions.
- **Private validation** — scored only once per direction at submission time via `submit.py`. The agent never reads `val_private_y.csv` directly.

### Why separate targets from features?

Agents have file system access. If targets lived in the same CSV as features, nothing would prevent an agent from reading the answer column (and in some experiments, where user is emphasizing importance of the task and puts pressure - agents do that!). Isolating `*_y.csv` files and enforcing "never read them directly" as a safety rule in skills creates a clear boundary — the agent works with features, the evaluation scripts work with targets. In some way this setup is very similar to Kaggle.

### Why two validation sets?

This mirrors the Kaggle public/private leaderboard design:

- **Public scores are visible during research** — agents use them to compare approaches, tune hyperparameters, and decide what to try next
- **Private scores reveal generalization** — computed once at submission, they show whether the public score was trustworthy or the agent overfit the public split
- **A large gap between public and private scores is a red flag** — it signals overfitting to the public validation set, which is the primary failure mode when an autonomous agent optimizes iteratively

Without this split, there's no way to detect if the agent's improvements are real or if it's just memorizing the validation set through repeated evaluation.

## Project Structure

```
.cursor/skills/
  orchestrator/SKILL.md       # Launches and manages research subagents
  experiment-runner/SKILL.md  # How each subagent runs experiments
  submit-experiment/SKILL.md  # How each subagent finalizes and merges

src/
  evaluate.py                 # Public validation scoring
  submit.py                   # Private evaluation + MLflow logging
  dashboard.py                # Generates interactive HTML dashboard

data/
  train.csv                   # Training data (features + target)
  val_public_X.csv            # Public validation features
  val_public_y.csv            # Public validation targets
  val_private_X.csv           # Private validation features
  val_private_y.csv           # Private validation targets

research_directions/          # Inter-agent communication (one file per direction)
mlruns/                       # MLflow model artifacts (saved models, blend configs)
mlflow.db                     # MLflow run metadata: metrics, params, tags (SQLite, created at runtime)
```

## Getting Started

### 1. Clone and install

```bash
git clone <repo-url>
cd ml_agent_demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your data

Replace the files in `data/` with your own dataset:
- `train.csv` — training data with features and a `target` column
- `val_public_X.csv` / `val_public_y.csv` — public validation split
- `val_private_X.csv` / `val_private_y.csv` — private validation split (held out)

### 3. Start experimenting

Open the project in Cursor and tell the agent:

> "Run experiments on this data"

The orchestrator skill activates, launches subagents, and manages the research process. You can also steer it:

> "Try an ensemble of the top two directions"

Or run a single direction manually — the experiment-runner skill guides any agent through the full workflow.

### 4. View results

```bash
cd automated_ml_science_team  # mlflow.db lives here — the URI is a relative path
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `localhost:5000` to see all runs, compare metrics, and inspect training curves. Or generate a static dashboard:

```bash
python src/dashboard.py --output dashboard.html
open dashboard.html
```

## Safety Rules

- Agents never read `*_y.csv` target files directly
- Private evaluation runs only through `submit.py`
- `private_val_score` is visible but cannot be used as an optimization target — only as a generalization check
- Bad runs are always logged — deleting failures is prohibited
- Every run must save its trained model as an MLflow artifact — never rely on retraining to reproduce results
- Only `mlruns/` and `research_directions/` may be committed to `main` — everything else stays on the branch


## Future Work

### Concurrency at scale

The current design has each agent merge its own results to `main` with a `git pull --rebase` + retry. This works for 3 concurrent agents because each touches unique paths (`mlruns/<experiment_id>/` and `research_directions/<name>.md`), so rebases resolve cleanly.

Beyond ~5 agents this starts to break down — more push collisions, longer retry chains, and a higher chance of two agents trying to register directions at the same moment. The fix is to stop agents from writing to `main` entirely: agents finish on their branch and report back to the orchestrator, which then merges results one at a time in a sequential loop. This serializes all writes to `main` through a single actor, eliminating conflicts at any concurrency level. The trade-off is a small delay in knowledge sharing — agents won't see each other's results until the orchestrator gets around to merging them.

### Shared utilities across agents

Currently each agent works in isolation on its own branch. If one agent writes a useful feature engineering function or a custom data loader, the next agent can't use it — it only sees `research_directions/` (learnings in prose) and `mlruns/` (saved models), not code.

A `src/shared/` directory on `main` could solve this. Agents would be allowed to promote battle-tested utilities from their branch into `src/shared/` during the merge step, alongside `mlruns/` and `research_directions/`. Future agents would import from it. The challenge is quality control — bad code in `src/shared/` would break other agents. This would likely need the orchestrator to review and approve shared code before it lands on `main`.

---

P.S. WHile I was working on this Karpathy published autoresearch: https://x.com/karpathy/status/2030371219518931079 
