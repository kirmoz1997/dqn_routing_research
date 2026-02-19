"""Supervised baseline for multi-agent set routing.

TF-IDF (1,2)-gram + OneVsRest(LogisticRegression) → multi-hot → agent set.

Run:
    python -m multiagent_dqn_routing.experiments.run_supervised_set
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime, timezone
from typing import Any, List

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from multiagent_dqn_routing.agents import N_AGENTS
from multiagent_dqn_routing.data.dataset import load_jsonl_set
from multiagent_dqn_routing.eval.evaluator_set import (
    evaluate_set_router,
    print_bucket_metrics,
)
from multiagent_dqn_routing.experiments.snapshot_utils import (
    build_meta,
    normalize_metrics,
    parse_reward_args,
)
from multiagent_dqn_routing.sim.reward_set import RewardSetConfig, RewardSetModel

# ── constants ────────────────────────────────────────────────────────

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MIN_SET_SIZE = 2
MAX_SET_SIZE = 9

ARTIFACT_DIR = "artifacts"
ARTIFACT_NAME = "supervised_tfidf_ovr_logreg.joblib"
ARTIFACT_KIND = "supervised_tfidf_ovr_logreg_set_router"
ARTIFACT_VERSION = 1

TRAIN_SPLIT_PATH = "data/splits/train.jsonl"
VAL_SPLIT_PATH = "data/splits/val.jsonl"
TEST_SPLIT_PATH = "data/splits/test.jsonl"

# ── helpers ──────────────────────────────────────────────────────────

def _build_multi_hot(items: list[dict]) -> np.ndarray:
    """Convert ``required_agents`` lists to a multi-hot matrix (n, 9)."""
    n = len(items)
    Y = np.zeros((n, N_AGENTS), dtype=int)
    for i, item in enumerate(items):
        for agent_id in item["required_agents"]:
            Y[i, agent_id] = 1
    return Y


def _proba_to_agents(
    proba: np.ndarray,
    threshold: float,
    max_set_size: int,
) -> List[List[int]]:
    """Convert probability matrix (n, 9) → list of agent-id lists.

    Rules per row:
      1. Take all agents with p >= threshold.
      2. If fewer than 2 → take top-2 by probability.
      3. If more than 9 → keep top-9 by probability.
      4. Return ids sorted by descending probability.
    """
    results: list[list[int]] = []
    for row in proba:
        selected_mask = row >= threshold
        indices = np.where(selected_mask)[0]

        if len(indices) < MIN_SET_SIZE:
            # fallback: top-2 by probability
            indices = np.argsort(row)[::-1][:MIN_SET_SIZE]
        elif len(indices) > max_set_size:
            indices = np.argsort(row)[::-1][:max_set_size]

        # sort selected indices by descending probability
        indices = sorted(indices, key=lambda idx: row[idx], reverse=True)
        results.append([int(idx) for idx in indices])
    return results


def _make_supervised_router(
    pipeline: Pipeline,
    threshold: float,
    max_steps: int,
):
    """Return a router function ``text → list[int]``."""

    def router(text: str) -> List[int]:
        proba = pipeline.predict_proba(
            [text]
        )  # shape (1, 9) — OneVsRest guarantees this
        return _proba_to_agents(proba, threshold, max_set_size=max_steps)[0]

    return router


def _print_metrics(metrics: dict, label: str, threshold: float) -> None:
    print(f"\n{'═' * 50}")
    print(f"  {label}")
    print(f"{'═' * 50}")
    print(f"  threshold            = {threshold}")
    print(f"  n_items              = {metrics['n_items']}")
    print(f"  mean_episode_reward  = {metrics['mean_episode_reward']:.4f}")
    print(f"  success_rate         = {metrics['success_rate']:.4f}")
    print(f"  exact_match_rate     = {metrics['exact_match_rate']:.4f}")
    print(f"  mean_jaccard         = {metrics['mean_jaccard']:.4f}")
    print(f"  mean_precision       = {metrics['mean_precision']:.4f}")
    print(f"  mean_recall          = {metrics['mean_recall']:.4f}")
    print(f"  mean_f1              = {metrics['mean_f1']:.4f}")
    print(f"  avg_steps            = {metrics['avg_steps']:.4f}")
    print(f"  avg_coverage         = {metrics['avg_coverage']:.4f}")
    print(f"  avg_overselection    = {metrics['avg_overselection']:.4f}")
    print(f"  avg_underselection   = {metrics['avg_underselection']:.4f}")


def _major_minor(version: str) -> tuple[int, int] | None:
    """Parse version string to (major, minor), if possible."""
    parts = version.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _check_sklearn_compatibility(
    metadata: dict[str, Any],
    *,
    fail_on_mismatch: bool,
) -> None:
    """Check sklearn major/minor compatibility for serialized artifacts."""
    saved = metadata.get("sklearn_version")
    current = sklearn.__version__

    if not isinstance(saved, str) or not saved.strip():
        warnings.warn(
            "Artifact metadata has no sklearn_version; compatibility unknown.",
            RuntimeWarning,
        )
        return

    saved_mm = _major_minor(saved)
    current_mm = _major_minor(current)
    if saved_mm is None or current_mm is None:
        warnings.warn(
            f"Cannot parse sklearn version for compatibility check: "
            f"saved={saved!r}, current={current!r}",
            RuntimeWarning,
        )
        return

    if saved_mm != current_mm:
        msg = (
            "scikit-learn version mismatch for artifact load: "
            f"saved={saved}, current={current}. "
            "Rebuild the artifact in the current environment."
        )
        if fail_on_mismatch:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning)


def load_supervised_artifact(
    path: str,
    *,
    fail_on_version_mismatch: bool = False,
) -> dict[str, Any]:
    """Load supervised artifact and validate sklearn compatibility."""
    obj = joblib.load(path)
    if not isinstance(obj, dict):
        raise ValueError(f"Artifact must be dict, got {type(obj).__name__}")

    metadata = obj.get("metadata", {})
    if isinstance(metadata, dict):
        _check_sklearn_compatibility(
            metadata,
            fail_on_mismatch=fail_on_version_mismatch,
        )
    else:
        warnings.warn(
            "Artifact has no valid metadata section; compatibility unknown.",
            RuntimeWarning,
        )
    return obj


# ── threshold selection ──────────────────────────────────────────────

def _select_best_threshold(
    val_items: list[dict],
    pipeline: Pipeline,
    reward_model: RewardSetModel,
    thresholds: list[float],
    max_steps: int,
) -> tuple[float, dict]:
    """Sweep thresholds on val, return (best_threshold, best_metrics).

    Ranking (higher is better unless noted):
      1. max mean_f1
      2. max mean_jaccard
      3. max mean_episode_reward
      4. min avg_steps
    """
    print("\n── Threshold sweep on val ──")
    print(
        f"  {'thr':>5}  {'f1':>7}  {'jaccard':>8}  {'exact':>6}  "
        f"{'success':>8}  {'under':>6}  {'over':>6}  "
        f"{'steps':>6}  {'reward':>8}"
    )

    best_threshold: float = thresholds[0]
    best_metrics: dict = {}
    best_key = None

    for thr in thresholds:
        router = _make_supervised_router(pipeline, thr, max_steps=max_steps)
        m = evaluate_set_router(val_items, router, reward_model)

        print(
            f"  {thr:5.2f}  {m['mean_f1']:7.4f}  {m['mean_jaccard']:8.4f}  "
            f"{m['exact_match_rate']:6.4f}  {m['success_rate']:8.4f}  "
            f"{m['avg_underselection']:6.4f}  {m['avg_overselection']:6.4f}  "
            f"{m['avg_steps']:6.4f}  {m['mean_episode_reward']:8.4f}"
        )

        # comparison key: maximise f1, jaccard, reward; minimise steps
        key = (
            -m["mean_f1"],
            -m["mean_jaccard"],
            -m["mean_episode_reward"],
            m["avg_steps"],
        )

        if best_key is None or key < best_key:
            best_key = key
            best_threshold = thr
            best_metrics = m

    print(f"\n  → best threshold = {best_threshold}")
    return best_threshold, best_metrics


# ── main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supervised TF-IDF + OVR LogReg baseline (set routing)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed decision threshold (if omitted, sweep on val)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default 42)",
    )
    parser.add_argument("--split_path", default=TEST_SPLIT_PATH)
    parser.add_argument("--train_split_path", default=TRAIN_SPLIT_PATH)
    parser.add_argument("--val_split_path", default=VAL_SPLIT_PATH)
    parser.add_argument("--dataset_path", default="data/tasks_set.jsonl")
    parser.add_argument("--max_steps", type=int, default=9, help="Max selected agents")
    parser.add_argument("--json_out", default=None, help="Write snapshot JSON to path")
    parser.add_argument("--reward_config_json", default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--p_good", type=float, default=0.85)
    parser.add_argument("--p_bad", type=float, default=0.30)
    parser.add_argument(
        "--strict_artifact_load_check",
        action="store_true",
        help=(
            "Fail if loaded artifact has incompatible sklearn major/minor "
            "(default: warn only)"
        ),
    )
    args = parser.parse_args()
    if args.max_steps < 2:
        raise ValueError("--max_steps must be >= 2")
    max_steps = min(args.max_steps, MAX_SET_SIZE)
    reward = parse_reward_args(args)

    # ── load splits ──────────────────────────────────────────────────
    train_items = load_jsonl_set(args.train_split_path)
    val_items = load_jsonl_set(args.val_split_path)
    test_items = load_jsonl_set(args.split_path)

    print(f"Загружено:  train={len(train_items)}  val={len(val_items)}  "
          f"test={len(test_items)}")

    # ── prepare data ─────────────────────────────────────────────────
    X_train = [item["text"] for item in train_items]
    Y_train = _build_multi_hot(train_items)

    # ── build & fit pipeline ─────────────────────────────────────────
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
        )),
        ("clf", OneVsRestClassifier(
            LogisticRegression(max_iter=2000, random_state=args.seed),
        )),
    ])

    print("\nОбучение TF-IDF + OVR(LogReg) …")
    pipeline.fit(X_train, Y_train)
    print("  done.")

    # ── reward model (shared for val & test) ─────────────────────────
    reward_model = RewardSetModel(
        RewardSetConfig(
            alpha=reward["alpha"],
            beta=reward["beta"],
            gamma=reward["gamma"],
            p_good=reward["p_good"],
            p_bad=reward["p_bad"],
            seed=args.seed,
        )
    )

    # ── threshold ────────────────────────────────────────────────────
    if args.threshold is not None:
        best_threshold = args.threshold
        print(f"\nИспользуется фиксированный threshold = {best_threshold}")
    else:
        best_threshold, val_metrics = _select_best_threshold(
            val_items, pipeline, reward_model, THRESHOLDS, max_steps,
        )
        _print_metrics(val_metrics, "Val metrics (best threshold)", best_threshold)

    # ── final evaluation on test ─────────────────────────────────────
    # re-create reward model with fresh RNG for deterministic test eval
    reward_model_test = RewardSetModel(
        RewardSetConfig(
            alpha=reward["alpha"],
            beta=reward["beta"],
            gamma=reward["gamma"],
            p_good=reward["p_good"],
            p_bad=reward["p_bad"],
            seed=args.seed,
        )
    )

    router = _make_supervised_router(pipeline, best_threshold, max_steps=max_steps)
    raw_test_metrics = evaluate_set_router(test_items, router, reward_model_test)
    metrics, buckets = normalize_metrics(raw_test_metrics)
    _print_metrics(raw_test_metrics, "Test metrics (final)", best_threshold)
    print_bucket_metrics(raw_test_metrics)

    # ── save artifact ────────────────────────────────────────────────
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    artifact_path = os.path.join(ARTIFACT_DIR, ARTIFACT_NAME)

    created_at = (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    artifact_obj = {
        "vectorizer": pipeline.named_steps["tfidf"],
        "clf": pipeline.named_steps["clf"],
        "threshold": best_threshold,
        "metadata": {
            "artifact_kind": ARTIFACT_KIND,
            "artifact_version": ARTIFACT_VERSION,
            "created_at_utc": created_at,
            "sklearn_version": sklearn.__version__,
            "seed": args.seed,
            "split_paths": {
                "train": args.train_split_path,
                "val": args.val_split_path,
                "test": args.split_path,
            },
            "reward_params": {
                **reward,
                "seed": args.seed,
            },
        },
    }

    joblib.dump(artifact_obj, artifact_path)
    print(f"\nАртефакт сохранён: {artifact_path}")

    loaded = load_supervised_artifact(
        artifact_path,
        fail_on_version_mismatch=args.strict_artifact_load_check,
    )
    loaded_meta = loaded.get("metadata", {})
    print(
        "Проверка загрузки артефакта: ok "
        f"(saved_sklearn={loaded_meta.get('sklearn_version')}, "
        f"current_sklearn={sklearn.__version__})"
    )

    if args.json_out:
        out_dir = os.path.dirname(args.json_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "baseline": "supervised_tfidf_logreg",
            "split_path": args.split_path,
            "seed": args.seed,
            "reward": reward,
            "max_steps": args.max_steps,
            "metrics": metrics,
            "buckets": buckets,
            "meta": build_meta(
                dataset_path=args.dataset_path,
                split_path=args.split_path,
            ),
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\nJSON snapshot saved: {args.json_out}")


if __name__ == "__main__":
    main()
