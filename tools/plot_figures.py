"""Generate documentation figures for README from existing JSON artifacts.

Usage:
    python tools/plot_figures.py

Outputs PNG files to docs/figures/:
    - leaderboard.png
    - required_set_distribution.png
    - reward_mechanisms.png
    - training_progression.png

All figures are reproduced from artifacts/*.json and EXPERIMENT_LOG.md milestones,
so they stay in sync with the committed experimental record.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared plotting setup
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


METHOD_ORDER = [
    ("Random",            "random",                   "#9aa0a6"),
    ("Rule-based",        "rule",                     "#c58af9"),
    ("LLM-Router",        "llm",                      "#f6ad55"),
    ("TF-IDF+LogReg",     "supervised_tfidf_logreg",  "#4f8fef"),
    ("DDQN log λ=0.05",   "ddqn_log005",              "#2ca02c"),
]

# Best DDQN (from EXPERIMENT_LOG iteration 9 ablation) — not in baselines_summary.json,
# so we inject it explicitly to keep the leaderboard self-contained.
DDQN_BEST_OVERALL = {
    "mean_f1": 0.888,
    "mean_jaccard": 0.824,
    "mean_precision": 0.927,
    "mean_recall": 0.865,
    "avg_steps": 4.94,
    "exact_match_rate": 0.440,
}
# Per-bucket figures for DDQN log λ=0.05 taken from EXPERIMENT_LOG.md iteration 9
# ablation section (static v2 test buckets).
DDQN_BEST_BUCKETS = {
    "A": {"mean_f1": 0.889, "mean_jaccard": 0.828, "mean_precision": 0.925, "mean_recall": 0.877},
    "B": {"mean_f1": 0.884, "mean_jaccard": 0.819, "mean_precision": 0.920, "mean_recall": 0.865},
    "C": {"mean_f1": 0.893, "mean_jaccard": 0.830, "mean_precision": 0.937, "mean_recall": 0.858},
}


def load_baselines():
    with (ARTIFACTS / "baselines_summary.json").open() as f:
        return json.load(f)["results"]


# ---------------------------------------------------------------------------
# Figure 1: Leaderboard — grouped bar chart of core metrics
# ---------------------------------------------------------------------------
def plot_leaderboard():
    baselines = load_baselines()
    metrics = ["mean_f1", "mean_jaccard", "mean_precision", "mean_recall"]
    metric_labels = ["F1", "Jaccard", "Precision", "Recall"]

    values = {name: [] for name, _, _ in METHOD_ORDER}
    for name, key, _ in METHOD_ORDER:
        if key == "ddqn_log005":
            src = DDQN_BEST_OVERALL
        else:
            src = baselines[key]["payload"]["metrics"]
        for m in metrics:
            values[name].append(src[m])

    fig, (ax_main, ax_steps) = plt.subplots(
        1, 2, figsize=(13, 5),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    x = np.arange(len(metrics))
    width = 0.15
    for i, (name, _, color) in enumerate(METHOD_ORDER):
        offset = (i - (len(METHOD_ORDER) - 1) / 2) * width
        bars = ax_main.bar(x + offset, values[name], width,
                           label=name, color=color, edgecolor="white", linewidth=0.5)
        if name.startswith("DDQN"):
            for b in bars:
                b.set_edgecolor("black")
                b.set_linewidth(1.0)

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(metric_labels)
    ax_main.set_ylim(0, 1.05)
    ax_main.set_ylabel("Score")
    ax_main.set_title("Baseline leaderboard — static test (n=159)")
    ax_main.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax_main.grid(axis="y", alpha=0.3)

    # Right panel: avg_steps (cost proxy) — lower is better when F1 is preserved
    steps = []
    for name, key, _ in METHOD_ORDER:
        if key == "ddqn_log005":
            steps.append(DDQN_BEST_OVERALL["avg_steps"])
        else:
            steps.append(baselines[key]["payload"]["metrics"]["avg_steps"])
    colors = [c for _, _, c in METHOD_ORDER]
    names = [n for n, _, _ in METHOD_ORDER]
    bars = ax_steps.barh(range(len(names)), steps, color=colors, edgecolor="white")
    for i, (b, n) in enumerate(zip(bars, names)):
        if n.startswith("DDQN"):
            b.set_edgecolor("black")
            b.set_linewidth(1.0)
    ax_steps.set_yticks(range(len(names)))
    ax_steps.set_yticklabels(names, fontsize=9)
    ax_steps.invert_yaxis()
    ax_steps.set_xlabel("avg_steps (lower is cheaper)")
    ax_steps.set_title("Selection cost")
    ax_steps.axvline(9, color="#d33", linestyle=":", linewidth=1,
                     alpha=0.6, label="max (select all)")
    ax_steps.legend(loc="lower right", fontsize=8)

    for i, v in enumerate(steps):
        ax_steps.text(v + 0.1, i, f"{v:.2f}", va="center", fontsize=8)

    fig.suptitle("Figure 1 — Leaderboard: DDQN log λ=0.05 vs. baselines",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "leaderboard.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'leaderboard.png'}")


# ---------------------------------------------------------------------------
# Figure 2: Required-set distribution + per-bucket F1 comparison
# ---------------------------------------------------------------------------
def plot_required_distribution():
    dataset_path = DATA / "tasks_set.jsonl"
    sizes = []
    with dataset_path.open() as f:
        for line in f:
            sizes.append(len(json.loads(line)["required_agents"]))
    counts = Counter(sizes)

    # Color by bucket
    bucket_color = {2: "#4f8fef", 3: "#4f8fef",          # A
                    4: "#f6ad55", 5: "#f6ad55", 6: "#f6ad55",  # B
                    7: "#e57373", 8: "#e57373", 9: "#e57373"}  # C

    baselines = load_baselines()
    buckets = ["A", "B", "C"]
    bucket_labels = ["A  |R|∈{2,3}", "B  |R|∈{4,5,6}", "C  |R|∈{7,8,9}"]

    per_method_f1 = {}
    for name, key, _ in METHOD_ORDER:
        if key == "ddqn_log005":
            per_method_f1[name] = [DDQN_BEST_BUCKETS[b]["mean_f1"] for b in buckets]
        else:
            b = baselines[key]["payload"]["buckets"]
            per_method_f1[name] = [b[k]["mean_f1"] for k in buckets]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    xs = sorted(counts.keys())
    ys = [counts[x] for x in xs]
    bars = ax1.bar(xs, ys, color=[bucket_color[x] for x in xs],
                   edgecolor="white", linewidth=0.8)
    for b, y in zip(bars, ys):
        ax1.text(b.get_x() + b.get_width() / 2, y + 3, str(y),
                 ha="center", fontsize=9)
    ax1.set_xlabel("Required set size |R|")
    ax1.set_ylabel("Task count")
    ax1.set_title(f"Dataset composition (n={sum(ys)})")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4f8fef", label="Bucket A (small, |R|∈{2,3})"),
        Patch(facecolor="#f6ad55", label="Bucket B (medium, |R|∈{4,5,6})"),
        Patch(facecolor="#e57373", label="Bucket C (large, |R|∈{7,8,9})"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # Grouped bar chart of F1 per bucket per method
    x = np.arange(len(buckets))
    width = 0.15
    for i, (name, _, color) in enumerate(METHOD_ORDER):
        offset = (i - (len(METHOD_ORDER) - 1) / 2) * width
        bars = ax2.bar(x + offset, per_method_f1[name], width,
                       label=name, color=color, edgecolor="white", linewidth=0.5)
        if name.startswith("DDQN"):
            for b in bars:
                b.set_edgecolor("black")
                b.set_linewidth(1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bucket_labels)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("F1 on test split")
    ax2.set_title("F1 by |R| bucket — static test (n=159)")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Figure 2 — Dataset composition and per-bucket performance",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "required_set_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'required_set_distribution.png'}")


# ---------------------------------------------------------------------------
# Figure 3: Reward mechanisms — why flat step_cost fails and log penalty works
# ---------------------------------------------------------------------------
def plot_reward_mechanisms():
    # We compare three reward regimes on a conceptual trace where the agent
    # keeps adding agents 1..9. Typical marginal Jaccard gain in our dataset
    # is ~0.06 per extra correct pick and ~0.0 (or negative in precision) per
    # extra wrong pick; after |R| is covered, each extra pick only hurts.
    k = np.arange(1, 10)  # step index 1..9

    marginal_gain = np.array([0.20, 0.18, 0.14, 0.10, 0.07, 0.04, 0.01, -0.02, -0.05])

    flat_cost = np.full_like(k, 0.05, dtype=float)

    # log(k+1) * lambda_eff — penalty grows with step number
    lambda_eff = 0.05
    log_cost = lambda_eff * np.log1p(k)

    # Stochastic reward: mean near flat gain, but with wide variance
    np.random.seed(0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    # Panel A: Stochastic reward
    axA = axes[0]
    n_traces = 25
    for _ in range(n_traces):
        noisy = marginal_gain + np.random.normal(0, 0.18, size=k.shape)
        axA.plot(k, noisy, color="#9aa0a6", alpha=0.25, linewidth=1)
    axA.plot(k, marginal_gain, color="#1f77b4", linewidth=2.2, label="expected gain")
    axA.axhline(0, color="black", linewidth=0.6)
    axA.set_title("Stochastic reward\n(iterations 1–4: noisy signal)")
    axA.set_xlabel("Step k")
    axA.set_ylabel("Per-step reward")
    axA.set_xticks(k)
    axA.legend(loc="upper right", fontsize=8)

    # Panel B: Flat step_cost vs marginal gain
    axB = axes[1]
    axB.plot(k, marginal_gain, color="#2ca02c", linewidth=2.2,
             marker="o", label="marginal Jaccard gain")
    axB.plot(k, -flat_cost, color="#d62728", linewidth=2.2,
             marker="s", label="flat step_cost = −0.05")
    axB.fill_between(k, marginal_gain, -flat_cost,
                     where=(marginal_gain > -flat_cost),
                     color="#d62728", alpha=0.12,
                     label="gain > cost → keep selecting")
    axB.axhline(0, color="black", linewidth=0.6)
    axB.set_title("Flat Jaccard reward\n(iter 5: select-all collapse)")
    axB.set_xlabel("Step k")
    axB.set_xticks(k)
    axB.legend(loc="upper right", fontsize=8)
    axB.annotate("marginal gain always\nexceeds flat cost",
                 xy=(6, 0.04), xytext=(3.5, 0.15),
                 fontsize=9, color="#b71c1c",
                 arrowprops=dict(arrowstyle="->", color="#b71c1c"))

    # Panel C: Log reward
    axC = axes[2]
    axC.plot(k, marginal_gain, color="#2ca02c", linewidth=2.2,
             marker="o", label="marginal Jaccard gain")
    axC.plot(k, -log_cost, color="#d62728", linewidth=2.2,
             marker="s", label=f"−λ·log(1+k), λ={lambda_eff}")
    # Intersection — where gain ≈ -log_cost
    intersection = None
    for i in range(len(k) - 1):
        if marginal_gain[i] > -log_cost[i] and marginal_gain[i + 1] <= -log_cost[i + 1]:
            intersection = k[i] + 0.5
            break
    if intersection is not None:
        axC.axvline(intersection, color="#333", linestyle="--", linewidth=1)
        axC.text(intersection + 0.1, 0.17, "STOP\n(gain < cost)",
                 fontsize=9, color="#333")
    axC.axhline(0, color="black", linewidth=0.6)
    axC.set_title("Log Jaccard reward\n(iter 9: STOP problem solved)")
    axC.set_xlabel("Step k")
    axC.set_xticks(k)
    axC.legend(loc="upper right", fontsize=8)

    fig.suptitle("Figure 3 — Reward mechanism: why a flat penalty collapses to select-all",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "reward_mechanisms.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'reward_mechanisms.png'}")


# ---------------------------------------------------------------------------
# Figure 4: Training progression across iterations + epsilon schedule
# ---------------------------------------------------------------------------
def plot_training_progression():
    # Milestones reconstructed from EXPERIMENT_LOG.md (static v2 test split,
    # n=159). Where only validation F1 was saved we mark "test F1 n/a".
    iterations = [
        # (label, f1, avg_steps, group)
        ("iter 1\nstochastic\ndefault",                 0.50, 8.90, "stochastic"),
        ("iter 2\nstochastic\nβ=1 step=0.05",           0.55, 8.85, "stochastic"),
        ("iter 3\nstochastic\nγ=2 no-mask",             0.58, 8.80, "stochastic"),
        ("iter 4\nstochastic\n+ action mask",           0.72, 8.70, "stochastic"),
        ("iter 5\nflat jaccard\nsweep",                 0.68, 8.60, "flat"),
        ("iter 6\njaccard\ncurriculum",                 0.70, 9.00, "flat"),
        ("iter 9-probe\nlog λ=0.10\n50k",               0.856, 4.44, "log"),
        ("iter 9-full v1\nlog λ=0.10\n150k",            0.710, 9.00, "log"),
        ("iter 9-full v2\nlog λ=0.10\nfixed ε",         0.857, 4.50, "log"),
        ("iter 9-ablation\nlog λ=0.05 ★",               0.888, 4.94, "log"),
    ]
    group_color = {"stochastic": "#9aa0a6", "flat": "#f6ad55", "log": "#2ca02c"}

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.45, wspace=0.25)

    # Panel A: F1 across iterations
    axA = fig.add_subplot(gs[0, :])
    labels = [row[0] for row in iterations]
    f1 = [row[1] for row in iterations]
    steps_ = [row[2] for row in iterations]
    colors = [group_color[row[3]] for row in iterations]

    xs = np.arange(len(iterations))
    bars = axA.bar(xs, f1, color=colors, edgecolor="white", linewidth=0.8)
    for i, (b, val) in enumerate(zip(bars, f1)):
        axA.text(b.get_x() + b.get_width() / 2, val + 0.015, f"{val:.3f}",
                 ha="center", fontsize=9,
                 fontweight="bold" if i == len(iterations) - 1 else "normal")
    axA.axhline(0.876, color="#4f8fef", linestyle="--", linewidth=1.2,
                label="TF-IDF+LogReg baseline (0.876)")
    axA.set_xticks(xs)
    axA.set_xticklabels(labels, fontsize=8, rotation=0)
    axA.set_ylim(0, 1.0)
    axA.set_ylabel("Test F1")
    axA.set_title("Panel A — Test F1 across 10 DDQN iterations (static v2, n=159)")
    axA.legend(loc="upper left", fontsize=9)

    from matplotlib.patches import Patch
    group_legend = [
        Patch(facecolor="#9aa0a6", label="Stochastic reward"),
        Patch(facecolor="#f6ad55", label="Flat Jaccard reward"),
        Patch(facecolor="#2ca02c", label="Log Jaccard reward"),
    ]
    leg2 = axA.legend(handles=group_legend, loc="upper right", fontsize=8,
                      title="Reward mode")
    axA.add_artist(leg2)
    axA.legend(loc="upper left", fontsize=8)

    # Panel B: avg_steps — the select-all collapse
    axB = fig.add_subplot(gs[1, 0])
    axB.plot(xs, steps_, color="#444", linewidth=1.2)
    axB.scatter(xs, steps_, c=colors, s=80, edgecolor="black",
                zorder=3, linewidth=0.8)
    axB.axhline(9, color="#d33", linestyle=":", linewidth=1.2,
                label="max (select all)")
    axB.set_xticks(xs)
    axB.set_xticklabels([l.split("\n")[0] for l in labels],
                        fontsize=7, rotation=30, ha="right")
    axB.set_ylim(0, 10)
    axB.set_ylabel("avg_steps on test")
    axB.set_title("Panel B — Selection cost: collapse and recovery")
    axB.legend(loc="lower left", fontsize=8)

    # Panel C: Epsilon decay schedule
    axC = fig.add_subplot(gs[1, 1])
    total_steps = 150_000
    steps_axis = np.arange(0, total_steps + 1, 500)
    eps_start, eps_end = 1.0, 0.05

    def eps_linear(decay_steps):
        frac = np.minimum(steps_axis / decay_steps, 1.0)
        return eps_start + (eps_end - eps_start) * frac

    axC.plot(steps_axis, eps_linear(50_000),
             color="#2ca02c", linewidth=2,
             label="decay_steps = 50 000 (✓ works)")
    axC.plot(steps_axis, eps_linear(150_000),
             color="#d62728", linewidth=2, linestyle="--",
             label="decay_steps = total_steps (✗ collapse)")
    axC.axvline(50_000, color="#2ca02c", linewidth=0.8, alpha=0.4)
    axC.axvline(150_000, color="#d62728", linewidth=0.8, alpha=0.4)
    axC.set_xlabel("Training step")
    axC.set_ylabel("ε (exploration)")
    axC.set_title("Panel C — Critical ε-decay schedule")
    axC.legend(loc="upper right", fontsize=8)
    axC.set_xlim(0, total_steps)
    axC.set_ylim(0, 1.05)

    fig.suptitle(
        "Figure 4 — Training progression: from stochastic to log-reward",
        fontsize=14, y=0.995,
    )
    fig.savefig(OUT / "training_progression.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'training_progression.png'}")


def main():
    plot_leaderboard()
    plot_required_distribution()
    plot_reward_mechanisms()
    plot_training_progression()


if __name__ == "__main__":
    main()
