"""
Generate a self-contained HTML dashboard from MLflow experiment data.

Produces:
  - Ranked leaderboard of research directions
  - Score progression over time (cumulative best)
  - Public vs. private generalization scatter
  - Per-direction run breakdown

Usage:
    python src/dashboard.py [--output dashboard.html]
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import mlflow
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _query_runs() -> list[dict]:
    """Fetch all finished runs from the default MLflow experiment."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Default")
    if experiment is None:
        experiments = client.search_experiments()
        if not experiments:
            return []
        experiment = experiments[0]

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time ASC"],
    )

    results = []
    for r in runs:
        public = r.data.metrics.get("public_val_score")
        if public is None:
            continue
        results.append(
            {
                "run_id": r.info.run_id,
                "start_time": r.info.start_time,
                "branch": r.data.tags.get("git.branch", "unknown"),
                "direction_rationale": r.data.tags.get("direction_rationale", ""),
                "run_rationale": r.data.tags.get("run_rationale", ""),
                "public_val_score": public,
                "private_val_score": r.data.metrics.get("private_val_score"),
                "params": dict(r.data.params),
            }
        )
    return results


def _direction_name(branch: str) -> str:
    return branch.removeprefix("exp/") if branch.startswith("exp/") else branch


def _build_leaderboard(runs: list[dict]) -> go.Figure:
    """Ranked table of directions by best public score, with private if available."""
    directions: dict[str, dict] = {}
    for r in runs:
        name = _direction_name(r["branch"])
        prev = directions.get(name)
        if prev is None or r["public_val_score"] > prev["public_val_score"]:
            directions[name] = {
                "direction": name,
                "public_val_score": r["public_val_score"],
                "private_val_score": r.get("private_val_score"),
                "best_run_id": r["run_id"],
                "num_runs": directions[name]["num_runs"] + 1 if prev else 1,
            }
        elif prev:
            directions[name]["num_runs"] += 1

    ranked = sorted(directions.values(), key=lambda d: d["public_val_score"], reverse=True)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Rank", "Direction", "Public Score", "Private Score", "Runs", "Best Run ID"],
                    fill_color="#1f2937",
                    font=dict(color="white", size=13),
                    align="left",
                ),
                cells=dict(
                    values=[
                        [i + 1 for i in range(len(ranked))],
                        [d["direction"] for d in ranked],
                        [f'{d["public_val_score"]:.5f}' for d in ranked],
                        [f'{d["private_val_score"]:.5f}' if d["private_val_score"] is not None else "—" for d in ranked],
                        [d["num_runs"] for d in ranked],
                        [d["best_run_id"][:12] for d in ranked],
                    ],
                    fill_color=[["#f9fafb", "#f3f4f6"] * ((len(ranked) + 1) // 2)] * 6,
                    align="left",
                    font=dict(size=12),
                    height=28,
                ),
            )
        ]
    )
    fig.update_layout(title="Leaderboard — Best Score per Direction", margin=dict(t=40, b=10, l=10, r=10))
    return fig


def _build_progression(runs: list[dict]) -> go.Figure:
    """Cumulative best public_val_score over time."""
    from datetime import datetime, timezone

    sorted_runs = sorted(runs, key=lambda r: r["start_time"])

    timestamps = []
    scores = []
    best_so_far = []
    current_best = float("-inf")

    for r in sorted_runs:
        ts = datetime.fromtimestamp(r["start_time"] / 1000, tz=timezone.utc)
        timestamps.append(ts)
        scores.append(r["public_val_score"])
        current_best = max(current_best, r["public_val_score"])
        best_so_far.append(current_best)

    directions = [_direction_name(r["branch"]) for r in sorted_runs]

    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=scores,
            mode="markers",
            name="Individual runs",
            marker=dict(size=7, opacity=0.6),
            text=[f"{d}<br>score: {s:.5f}" for d, s in zip(directions, scores)],
            hoverinfo="text",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=best_so_far,
            mode="lines",
            name="Cumulative best",
            line=dict(width=2, color="#ef4444"),
        )
    )
    fig.update_layout(
        title="Score Progression Over Time",
        xaxis_title="Time",
        yaxis_title="public_val_score (ROC AUC)",
        hovermode="closest",
        margin=dict(t=40, b=40),
    )
    return fig


def _build_generalization_scatter(runs: list[dict]) -> go.Figure | None:
    """Public vs. private score scatter — one point per direction's best run."""
    points: dict[str, dict] = {}
    for r in runs:
        if r.get("private_val_score") is None:
            continue
        name = _direction_name(r["branch"])
        prev = points.get(name)
        if prev is None or r["public_val_score"] > prev["public_val_score"]:
            points[name] = r

    if not points:
        return None

    names = list(points.keys())
    public = [points[n]["public_val_score"] for n in names]
    private = [points[n]["private_val_score"] for n in names]

    all_vals = public + private
    lo, hi = min(all_vals) - 0.01, max(all_vals) + 0.01

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=public,
            y=private,
            mode="markers+text",
            text=names,
            textposition="top center",
            marker=dict(size=12, color="#3b82f6"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(dash="dash", color="#9ca3af"),
            name="y = x (perfect generalization)",
        )
    )
    fig.update_layout(
        title="Generalization — Public vs. Private Score",
        xaxis_title="public_val_score",
        yaxis_title="private_val_score",
        xaxis=dict(range=[lo, hi]),
        yaxis=dict(range=[lo, hi]),
        margin=dict(t=40, b=40),
    )
    return fig


def _build_direction_breakdown(runs: list[dict]) -> go.Figure:
    """Box/strip plot of scores within each direction."""
    by_direction: dict[str, list[float]] = {}
    for r in runs:
        name = _direction_name(r["branch"])
        by_direction.setdefault(name, []).append(r["public_val_score"])

    sorted_dirs = sorted(by_direction.keys(), key=lambda d: max(by_direction[d]), reverse=True)

    fig = go.Figure()
    for name in sorted_dirs:
        fig.add_trace(
            go.Box(
                y=by_direction[name],
                name=name,
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.5,
            )
        )
    fig.update_layout(
        title="Score Distribution per Direction",
        yaxis_title="public_val_score (ROC AUC)",
        showlegend=False,
        margin=dict(t=40, b=40),
    )
    return fig


def generate_dashboard(output_path: Path) -> Path:
    """Query MLflow and write a self-contained HTML dashboard."""
    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))

    runs = _query_runs()
    if not runs:
        output_path.write_text(
            "<html><body><h1>No MLflow runs found</h1>"
            "<p>Run some experiments first.</p></body></html>"
        )
        return output_path

    leaderboard = _build_leaderboard(runs)
    progression = _build_progression(runs)
    generalization = _build_generalization_scatter(runs)
    breakdown = _build_direction_breakdown(runs)

    sections = [
        leaderboard.to_html(full_html=False, include_plotlyjs=False),
        progression.to_html(full_html=False, include_plotlyjs=False),
    ]
    if generalization is not None:
        sections.append(generalization.to_html(full_html=False, include_plotlyjs=False))
    sections.append(breakdown.to_html(full_html=False, include_plotlyjs=False))

    total_runs = len(runs)
    directions = len({_direction_name(r["branch"]) for r in runs})
    best = max(runs, key=lambda r: r["public_val_score"])
    best_name = _direction_name(best["branch"])

    summary = (
        f"{total_runs} runs across {directions} directions — "
        f'best public score: <strong>{best["public_val_score"]:.5f}</strong> '
        f"({html.escape(best_name)})"
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ML Experiment Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 20px 40px; background: #f9fafb; color: #111827; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 4px; }}
  .summary {{ color: #6b7280; margin-bottom: 24px; font-size: 1.05rem; }}
  .chart {{ background: white; border-radius: 8px; padding: 16px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
</style>
</head>
<body>
<h1>ML Experiment Dashboard</h1>
<p class="summary">{summary}</p>
{"".join(f'<div class="chart">{s}</div>' for s in sections)}
</body>
</html>"""

    output_path.write_text(page)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ML experiment dashboard")
    parser.add_argument("--output", default="dashboard.html", help="Output HTML file path")
    args = parser.parse_args()

    path = generate_dashboard(Path(args.output))
    print(f"Dashboard written to {path}")


if __name__ == "__main__":
    main()
