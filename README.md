# Safe Evaluation Harness

Minimal local harness focused on policy-enforced model evaluation.

## Scope
- Public evaluation path for agent use (`mode=public`, `dataset=val_public`)
- Private evaluation path for human/CI-only use (`mode=private`, `dataset=val_private`)
- Strict policy checks before model/data loading
- Deterministic scoring and JSON result artifacts

## Repository Layout
- `data/`: datasets (kept intact)
- `src/safe_eval/`: evaluation contracts, policy checks, runners, CLIs
- `configs/public_eval_request.example.json`: example public request payload
- `configs/private_eval_request.example.json`: example private request payload
- `skills/public-evaluator/`: skill definition for public-only evaluation
- `tests/`: unit tests for contracts/policy/gate/runner

## Setup
```bash
cd cursor_ml_harness
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Public Evaluation (Agent-Safe)
Edit `configs/public_eval_request.example.json` so `model_path` points to a real model artifact under `outputs/`, then run:
```bash
PYTHONPATH=src python -m safe_eval.run_public_eval \
  --request configs/public_eval_request.example.json
```

Result artifact:
- `outputs/public_eval/<candidate_id>/result.json`

## Private Evaluation (Human/CI-Gated)
Private evaluation requires a gate secret and matching token:
```bash
export PRIVATE_EVAL_GATE_TOKEN='replace-with-secret'
PYTHONPATH=src python -m safe_eval.run_private_eval \
  --request configs/private_eval_request.example.json \
  --gate-token "$PRIVATE_EVAL_GATE_TOKEN"
```

Result artifact:
- `outputs/private_eval/<candidate_id>/result.json`

## Policy Guarantees
- Public path only accepts `mode=public` and `dataset=val_public`.
- Private path only accepts `mode=private` and `dataset=val_private`.
- Model artifact path must resolve under `outputs/`.
- Gate token is required for private CLI.
- Classification evaluation is accuracy-only.

## Tests
```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v
```
