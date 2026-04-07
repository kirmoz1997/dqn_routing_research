# Repository Audit Report

Generated: 2026-04-07

## Summary
- Total files: 83
- Essential: 71
- Review needed: 7
- Documentation: 5
- Placeholders found: 2

Counted project files only. `.git/` and `.venv/` are excluded from the audit because they are environment/VCS internals rather than repository deliverables.

## Documentation Files
- 📝 `README.md`
- 📝 `EXPERIMENT_LOG.md`
- 📝 `Research_Plan_MultiAgent_Set_Routing_v1.0.0.md`
- 📝 `tools/README.md`
- 📝 `REPO_AUDIT.md`

## Essential Files (do not touch)
### Root
- ✅ `pyproject.toml`
- ✅ `requirements.txt`
- ✅ `.gitignore`
- ✅ `.env.example`

### Configs
- ✅ `configs/baseline_protocol.json`
- ✅ `configs/ddqn_set_default.json`
- ✅ `configs/ddqn_set_beta1_step005.json`
- ✅ `configs/ddqn_set_beta1_step005_gamma2_nomask.json`
- ✅ `configs/ddqn_set_beta1_step005_gamma2_actionmask.json`
- ✅ `configs/ddqn_jaccard_step001.json`
- ✅ `configs/ddqn_jaccard_step005.json`
- ✅ `configs/ddqn_jaccard_step010.json`
- ✅ `configs/ddqn_jaccard_step020.json`
- ✅ `configs/ddqn_jaccard_curriculum.json`
- ✅ `configs/ddqn_adaptive_jaccard.json`
- ✅ `configs/ddqn_adaptive_jaccard_v2.json`
- ✅ `configs/ddqn_log_lambda010.json`
- ✅ `configs/ddqn_log_lambda010_full.json`
- ✅ `configs/ddqn_log_lambda010_full_v2.json`
- ✅ `configs/ddqn_log_lambda005.json`
- ✅ `configs/ddqn_log_lambda005_full.json`
- ✅ `configs/ddqn_log_lambda003.json`
- ✅ `configs/ddqn_log_lambda015.json`
- ✅ `configs/ddqn_adaptive_log_lambda005.json`

### Source
- ✅ `src/multiagent_dqn_routing/agents.py`
- ✅ `src/multiagent_dqn_routing/data/dataset.py`
- ✅ `src/multiagent_dqn_routing/envs/set_routing_env.py`
- ✅ `src/multiagent_dqn_routing/envs/adaptive_routing_env.py`
- ✅ `src/multiagent_dqn_routing/eval/evaluator_set.py`
- ✅ `src/multiagent_dqn_routing/experiments/train_ddqn_set.py`
- ✅ `src/multiagent_dqn_routing/experiments/run_random_set.py`
- ✅ `src/multiagent_dqn_routing/experiments/run_rule_set.py`
- ✅ `src/multiagent_dqn_routing/experiments/run_supervised_set.py`
- ✅ `src/multiagent_dqn_routing/experiments/run_llm_set.py`
- ✅ `src/multiagent_dqn_routing/experiments/snapshot_utils.py`
- ✅ `src/multiagent_dqn_routing/rl/replay_buffer.py`
- ✅ `src/multiagent_dqn_routing/rl/q_network.py`
- ✅ `src/multiagent_dqn_routing/rl/state_encoder.py`
- ✅ `src/multiagent_dqn_routing/rl/ddqn_agent.py`
- ✅ `src/multiagent_dqn_routing/sim/reward_set.py`

### Tools
- ✅ `tools/tsv_to_jsonl.py`
- ✅ `tools/dataset_stats_set.py`
- ✅ `tools/fix_dataset.py`
- ✅ `tools/split_jsonl_set.py`
- ✅ `tools/generate_adaptive_dataset.py`
- ✅ `tools/split_adaptive_dataset.py`
- ✅ `tools/baseline_snapshot.py`

### Data
- ✅ `data/tasks_set.jsonl`
- ✅ `data/tasks_set_draft.tsv`
- ✅ `data/tasks_set_adaptive_full.jsonl`
- ✅ `data/splits/train.jsonl`
- ✅ `data/splits/val.jsonl`
- ✅ `data/splits/test.jsonl`
- ✅ `data/splits_adaptive/train.jsonl`
- ✅ `data/splits_adaptive/val.jsonl`
- ✅ `data/splits_adaptive/test.jsonl`

### Artifacts
- ✅ `artifacts/baselines_summary.json`
- ✅ `artifacts/baselines_summary.md`
- ✅ `artifacts/baseline_random.json`
- ✅ `artifacts/baseline_rule.json`
- ✅ `artifacts/baseline_supervised_tfidf_logreg.json`
- ✅ `artifacts/baseline_llm.json`
- ✅ `artifacts/baseline_adaptive/random_train.json`
- ✅ `artifacts/baseline_adaptive/random_val.json`
- ✅ `artifacts/baseline_adaptive/random_test.json`
- ✅ `artifacts/baseline_adaptive/rule_train.json`
- ✅ `artifacts/baseline_adaptive/rule_val.json`
- ✅ `artifacts/baseline_adaptive/rule_test.json`
- ✅ `artifacts/ddqn/config_used.json`
- ✅ `artifacts/ddqn/metrics_val_best.json`
- ✅ `artifacts/ddqn/metrics_test.json`

## Files for Human Review
- ⚠️ `configs/ddqn_jaccard_curriculum_smoke.json` — smoke-only config; likely replaceable by `--smoke_test`, but preserved for historical reproducibility.
- ⚠️ `configs/ddqn_adaptive_jaccard_smoke.json` — smoke-only config; likely redundant with `--smoke_test`.
- ⚠️ `configs/ddqn_adaptive_jaccard_v2_smoke.json` — smoke-only config; likely redundant with `--smoke_test`.
- ⚠️ `configs/ddqn_log_lambda010_smoke.json` — smoke-only config; likely redundant with `--smoke_test`.
- ⚠️ `data/tasks_set_adaptive.jsonl` — legacy/near-duplicate adaptive dataset file; not referenced in current docs.
- ⚠️ `data/tasks_set_adaptive_smoke.jsonl` — smoke dataset artifact; not referenced in current docs or primary reproduction flow.
- ⚠️ `.python-version` — local interpreter pin; useful on one machine, but not required once `pyproject.toml` and `requirements.txt` are authoritative.

## Placeholders Filled
- Replaced the unfinished experiment-result stub in iteration 5 of `EXPERIMENT_LOG.md` with the actual ablation-sweep outcome.
- Replaced the unfinished experiment-result stub in iteration 6 of `EXPERIMENT_LOG.md` with the recorded smoke/full-run outcome.
- Removed stale planning/status labels from the experiment narrative and checklist.
- Added missing final documentation for iteration 9 full v1, iteration 9 full v2, and iteration 10.

## Inconsistencies Found and Fixed
- `README.md` previously listed only part of `configs/`; it now includes all actual config files with descriptions.
- `README.md` did not document the best reproducible result; it now includes a dedicated reproduction section for `configs/ddqn_log_lambda005_full.json`.
- `README.md` experiment summary lagged behind the current results; it now reflects static and adaptive leaderboards.
- `EXPERIMENT_LOG.md` still described iterations 5 and 6 as planned; it now documents them as completed experiments.
- `EXPERIMENT_LOG.md` lacked the final narrative for iteration 9 full runs and iteration 10; those sections are now present.
- `Research_Plan_MultiAgent_Set_Routing_v1.0.0.md` did not explicitly mark `epsilon_decay_steps=50000` as a critical reproducibility trap; it now does.

## Recommended Future Cleanup
- Add a dedicated `CHANGELOG.md` if the repository will be reviewed outside the thesis context.
- Decide whether smoke configs should remain in `configs/` or be generated exclusively via `--smoke_test`.
- Decide whether `data/tasks_set_adaptive.jsonl` and `data/tasks_set_adaptive_smoke.jsonl` should be archived into a clearly named `legacy/` or `scratch/` area.
- If `.python-version` is kept, document whether it is normative or purely local.
- Consider separating “best/current” docs from the full historical log if the repo is meant for external readers beyond course evaluation.

## Reproduction Checklist
1. Create and activate a clean virtual environment: `python -m venv .venv && source .venv/bin/activate`.
2. Install dependencies: `pip install -e '.[dev]'`.
3. Recreate main splits if needed: `python tools/split_jsonl_set.py`.
4. Verify that `data/splits/{train,val,test}.jsonl` exist and match the v2 dataset.
5. Run the best DDQN config: `python -m multiagent_dqn_routing.experiments.train_ddqn_set --config configs/ddqn_log_lambda005_full.json`.
6. Confirm that `config_used.json` keeps `reward_mode=jaccard_log`, `lambda_eff=0.05`, and `epsilon_decay_steps=50000`.
7. Compare produced metrics against the documented target: test `mean_f1 ≈ 0.888`, `precision ≈ 0.927`, `avg_steps ≈ 4.94`.
8. For adaptive experiments, regenerate/verify `data/splits_adaptive/*.jsonl` separately before training `configs/ddqn_adaptive_log_lambda005.json`.
