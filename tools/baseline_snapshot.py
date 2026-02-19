#!/usr/bin/env python3
"""Run baseline snapshot protocol and aggregate results.

Usage:
    python tools/baseline_snapshot.py --config configs/baseline_protocol.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASELINE_TO_MODULE = {
    "random": "multiagent_dqn_routing.experiments.run_random_set",
    "rule": "multiagent_dqn_routing.experiments.run_rule_set",
    "supervised_tfidf_logreg": "multiagent_dqn_routing.experiments.run_supervised_set",
    "llm": "multiagent_dqn_routing.experiments.run_llm_set",
}

OVERALL_COLUMNS = [
    "mean_f1",
    "mean_jaccard",
    "exact_match_rate",
    "success_rate",
    "avg_underselection",
    "avg_overselection",
    "avg_steps",
    "mean_episode_reward",
]

BUCKET_COLUMNS = [
    "mean_f1",
    "mean_jaccard",
    "avg_underselection",
    "avg_overselection",
]


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


def git_commit_or_none() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return proc.stdout.strip() or None


def _fmt_num(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _run_baseline(
    *,
    repo_root: Path,
    baseline: str,
    cfg: dict[str, Any],
    output_path: Path,
) -> tuple[str, dict[str, Any]]:
    module = BASELINE_TO_MODULE[baseline]
    reward_json = json.dumps(cfg["reward"], ensure_ascii=False)

    cmd = [
        sys.executable,
        "-m",
        module,
        "--split_path",
        cfg["test_split_path"],
        "--seed",
        str(cfg["seed"]),
        "--max_steps",
        str(cfg["max_steps"]),
        "--dataset_path",
        cfg["dataset_path"],
        "--reward_config_json",
        reward_json,
        "--json_out",
        str(output_path),
    ]

    if baseline == "supervised_tfidf_logreg":
        cmd.extend(
            [
                "--train_split_path",
                cfg["train_split_path"],
                "--val_split_path",
                cfg["val_split_path"],
            ]
        )

    if baseline == "llm":
        llm_cfg = cfg["llm"]
        if not llm_cfg.get("include", False):
            return "skipped", {"reason": "llm.include=false"}
        api_key_env = llm_cfg["api_key_env"]
        if not os.environ.get(api_key_env):
            return "skipped", {"reason": f"env {api_key_env} is not set"}
        cmd.extend(
            [
                "--model",
                llm_cfg["model"],
                "--base_url",
                llm_cfg["base_url"],
                "--api_key_env",
                llm_cfg["api_key_env"],
                "--temperature",
                str(llm_cfg["temperature"]),
                "--max_tokens",
                str(llm_cfg["max_tokens"]),
                "--max_retries",
                str(llm_cfg["max_retries"]),
                "--prompt_version",
                llm_cfg["prompt_version"],
                "--cache_path",
                llm_cfg["cache_path"],
            ]
        )

    env = os.environ.copy()
    src_path = str(repo_root / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"

    print(f"\n>>> Running baseline: {baseline}")
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.stdout.strip():
        print(proc.stdout)
    if proc.stderr.strip():
        print(proc.stderr, file=sys.stderr)

    if proc.returncode != 0:
        return "failed", {"returncode": proc.returncode}
    if not output_path.exists():
        return "failed", {"reason": f"missing output file {output_path}"}

    with output_path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return "ok", payload


def _write_markdown(summary: dict[str, Any], output_md: Path) -> None:
    lines: list[str] = []
    lines.append("# Baseline Snapshot Summary")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{summary['generated_at_utc']}`")
    lines.append(f"- Git commit: `{summary.get('git_commit')}`")
    lines.append(f"- Dataset path: `{summary['protocol']['dataset_path']}`")
    lines.append(f"- Dataset sha256: `{summary.get('dataset_sha256')}`")
    lines.append("")

    lines.append("## Overall Comparison")
    lines.append("")
    header = ["baseline"] + OVERALL_COLUMNS
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for baseline in summary["protocol"]["order"]:
        rec = summary["results"].get(baseline, {})
        if rec.get("status") != "ok":
            lines.append("| " + baseline + " | " + " | ".join(["n/a"] * len(OVERALL_COLUMNS)) + " |")
            continue
        metrics = rec["payload"]["metrics"]
        row = [baseline] + [_fmt_num(metrics.get(col, "n/a")) for col in OVERALL_COLUMNS]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Buckets A/B/C")
    lines.append("")

    for baseline in summary["protocol"]["order"]:
        rec = summary["results"].get(baseline, {})
        lines.append(f"### {baseline}")
        if rec.get("status") != "ok":
            lines.append(f"- status: `{rec.get('status')}`")
            if "reason" in rec:
                lines.append(f"- reason: `{rec['reason']}`")
            lines.append("")
            continue

        lines.append("")
        bheader = ["bucket"] + BUCKET_COLUMNS
        lines.append("| " + " | ".join(bheader) + " |")
        lines.append("|" + "|".join(["---"] * len(bheader)) + "|")
        buckets = rec["payload"]["buckets"]
        for bucket_name in ["A", "B", "C"]:
            bm = buckets.get(bucket_name, {})
            row = [bucket_name] + [_fmt_num(bm.get(col, "n/a")) for col in BUCKET_COLUMNS]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate official baseline snapshot")
    parser.add_argument(
        "--config",
        default="configs/baseline_protocol.json",
        help="Path to baseline protocol JSON config",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve() if not os.path.isabs(args.config) else Path(args.config)
    with config_path.open(encoding="utf-8") as fh:
        cfg = json.load(fh)

    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "generated_at_utc": now_utc_iso(),
        "git_commit": git_commit_or_none(),
        "dataset_sha256": sha256_file(str(repo_root / cfg["dataset_path"])),
        "protocol": {
            "config_path": str(config_path),
            "dataset_path": cfg["dataset_path"],
            "test_split_path": cfg["test_split_path"],
            "val_split_path": cfg["val_split_path"],
            "train_split_path": cfg["train_split_path"],
            "seed": cfg["seed"],
            "max_steps": cfg["max_steps"],
            "reward": cfg["reward"],
            "buckets": cfg["buckets"],
            "order": cfg["order"],
            "llm": cfg.get("llm", {}),
        },
        "results": {},
    }

    for baseline in cfg["order"]:
        output_path = artifacts_dir / f"baseline_{baseline}.json"
        status, payload = _run_baseline(
            repo_root=repo_root,
            baseline=baseline,
            cfg=cfg,
            output_path=output_path,
        )
        if status == "ok":
            summary["results"][baseline] = {
                "status": status,
                "path": str(output_path.relative_to(repo_root)),
                "payload": payload,
            }
        else:
            rec = {"status": status}
            rec.update(payload)
            summary["results"][baseline] = rec

    summary_json = artifacts_dir / "baselines_summary.json"
    summary_md = artifacts_dir / "baselines_summary.md"

    with summary_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    _write_markdown(summary, summary_md)

    print("\n=== Baseline snapshot complete ===")
    print(f"JSON: {summary_json}")
    print(f"MD  : {summary_md}")


if __name__ == "__main__":
    main()

