from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any

from multiagent_dqn_routing.eval.evaluator_set import BUCKETS

METRIC_KEYS = [
    "n_items",
    "mean_episode_reward",
    "success_rate",
    "exact_match_rate",
    "mean_jaccard",
    "mean_precision",
    "mean_recall",
    "mean_f1",
    "avg_steps",
    "avg_overselection",
    "avg_underselection",
    "avg_coverage",
]

BUCKET_LABEL_TO_KEY = {
    "A (|R|∈{2,3})": "A",
    "B (|R|∈{4,5,6})": "B",
    "C (|R|∈{7,8,9})": "C",
}


def now_utc_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def sha256_file(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    commit = proc.stdout.strip()
    return commit or None


def parse_reward_args(args: Any) -> dict[str, float]:
    if getattr(args, "reward_config_json", None):
        raw = args.reward_config_json
        if os.path.exists(raw):
            with open(raw, encoding="utf-8") as fh:
                cfg = json.load(fh)
        else:
            cfg = json.loads(raw)
        return {
            "alpha": float(cfg["alpha"]),
            "beta": float(cfg["beta"]),
            "gamma": float(cfg["gamma"]),
            "p_good": float(cfg["p_good"]),
            "p_bad": float(cfg["p_bad"]),
        }
    return {
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "gamma": float(args.gamma),
        "p_good": float(args.p_good),
        "p_bad": float(args.p_bad),
    }


def normalize_metrics(raw_metrics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    metrics = {k: raw_metrics.get(k, 0.0) for k in METRIC_KEYS}

    raw_buckets = raw_metrics.get("buckets", {})
    buckets: dict[str, dict[str, Any]] = {}

    for eval_bucket_label in BUCKETS:
        bucket_key = BUCKET_LABEL_TO_KEY[eval_bucket_label]
        bucket_metrics = raw_buckets.get(eval_bucket_label, {})
        buckets[bucket_key] = {k: bucket_metrics.get(k, 0.0) for k in METRIC_KEYS}

    return metrics, buckets


def build_meta(
    *,
    dataset_path: str,
    split_path: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "dataset_sha256": sha256_file(dataset_path),
        "split_sha256": sha256_file(split_path),
        "git_commit": get_git_commit(),
        "timestamp_utc": now_utc_iso(),
    }
    if extra:
        meta.update(extra)
    return meta

