# Experiment Log

Experiment log for the Multi-Agent Set Routing with DQN project.

---

## Dataset Version

> **Dataset language note:** the dataset was originally developed and annotated in Russian. All documentation has been translated into English, but the underlying request texts remain Russian in the source data files.

### v1 (323 records) - used in the experiments below

- **File:** `data/tasks_set.jsonl`
- **SHA-256:** `d6e8f4d0f50b950f4e3c168ddf82cb49ab5181ce02c7bcaece0034e64db6a718`
- **Records:** 323
- **Splits:** train=227, val=47, test=49 (70/15/15, stratified by `|R|`)
- **Split seed:** 42
- **Text language:** Russian
- **`|R|` range:** 2..9

### v2 (1054 records) - update from 2026-03-01

The dataset was expanded to **1054 records**.

- **Splits:** train=736, val=158, test=159 (70/15/15, stratified by `|R|`)
- **Split seed:** 42

Baseline results (section 1) use v1. Section 2 (baseline snapshot) and DDQN iteration 4 (section 6) use v2.

### Adaptive full (`data/splits_adaptive/` splits)

Dataset with `adaptive.trajectory` fields for `AdaptiveRoutingEnv`. The full generated pool is `data/tasks_set_adaptive_full.jsonl`; stratified splits are built by `tools/split_adaptive_dataset.py`.

- **File (full pool):** `data/tasks_set_adaptive_full.jsonl`
- **SHA-256:** `4fa22c63b15e1d2c71933c87ddd205c4d605a59de4791db7a0290bc7df04df15`
- **Splits:** train=609, val=131, test=131 (70/15/15, stratified by `|R|`)
- **Split seed:** 42

---

## Adaptive Dataset - Random Baseline (2026-04-05)

**Script:** `python -m multiagent_dqn_routing.experiments.run_random_set`

**Protocol:** the same stochastic reward as in `configs/baseline_protocol.json` (the `reward` field), passed via `--reward_config_json` (`p_bad = 0.35`).

**Run parameters:** `seed=42`, `max_steps=9` (default), `--dataset_path data/tasks_set_adaptive_full.jsonl` only for meta / SHA in the JSON snapshot.

**Artifacts (local, directory ignored by `.gitignore`):** `artifacts/baseline_adaptive/random_{train,val,test}.json`

### Train / val / test (overall)

| Split | n | mean_f1 | mean_jaccard | exact_match | success_rate | avg_steps | avg_over | avg_under | mean_episode_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 609 | 0.5119 | 0.3758 | 0.0082 | 0.2562 | 5.4680 | 2.5813 | 1.9146 | 0.0977 |
| val | 131 | 0.5155 | 0.3842 | 0.0000 | 0.2824 | 5.4427 | 2.5420 | 1.8779 | 0.1641 |
| test | 131 | 0.5120 | 0.3799 | 0.0076 | 0.2977 | 5.4427 | 2.5573 | 1.8931 | 0.1565 |

### Test (n=131) - by `|R|` bucket

| Bucket | n | mean_f1 | mean_jaccard | success_rate | avg_over | avg_under |
|---|---:|---:|---:|---:|---:|---:|
| A | 42 | 0.3585 | 0.2479 | 0.4762 | 4.0952 | 0.9048 |
| B | 62 | 0.5485 | 0.4018 | 0.2419 | 2.3871 | 1.9677 |
| C | 27 | 0.6668 | 0.5348 | 0.1481 | 0.5556 | 3.2593 |

Bucket boundaries match `evaluator_set.py`: A - required set size 2-3; B - 4-6; C - 7-9.

**Reproduction (all three splits):**

```bash
REWARD='{"alpha":1.0,"beta":0.5,"gamma":1.0,"p_good":0.85,"p_bad":0.35}'
for SPLIT in train val test; do
  python -m multiagent_dqn_routing.experiments.run_random_set \
    --split_path "data/splits_adaptive/${SPLIT}.jsonl" \
    --dataset_path data/tasks_set_adaptive_full.jsonl \
    --seed 42 \
    --reward_config_json "$REWARD" \
    --json_out "artifacts/baseline_adaptive/random_${SPLIT}.json"
done
```

---

## Adaptive Dataset - Rule-based Baseline (2026-04-05)

**Script:** `python -m multiagent_dqn_routing.experiments.run_rule_set`

**Protocol:** the same stochastic reward as in `configs/baseline_protocol.json` (the `reward` field), via `--reward_config_json` (`p_bad = 0.35`).

**Router logic:** heuristic based on keyword markers in the request text (`TRIGGERS` in `run_rule_set.py`); if the result is below `min_len=2`, the router pads with random unique agents (fixed `seed+81`).

**Run parameters:** `seed=42`, `max_steps=9`, `--dataset_path data/tasks_set_adaptive_full.jsonl` for meta in the JSON snapshot.

**Artifacts (local, `.gitignore`):** `artifacts/baseline_adaptive/rule_{train,val,test}.json`

### Train / val / test (overall)

| Split | n | mean_f1 | mean_jaccard | exact_match | success_rate | avg_steps | avg_over | avg_under | mean_episode_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 609 | 0.5137 | 0.3719 | 0.0312 | 0.0361 | 2.3218 | 0.4778 | 2.9573 | −1.4745 |
| val | 131 | 0.5473 | 0.4059 | 0.0458 | 0.0534 | 2.3053 | 0.3817 | 2.8550 | −1.2634 |
| test | 131 | 0.5284 | 0.3881 | 0.0382 | 0.0382 | 2.3664 | 0.4427 | 2.8550 | −1.2099 |

### Test (n=131) - by bucket

| Bucket | n | mean_f1 | mean_jaccard | success_rate | avg_over | avg_under |
|---|---:|---:|---:|---:|---:|---:|
| A | 42 | 0.5032 | 0.3869 | 0.0952 | 0.9048 | 1.3571 |
| B | 62 | 0.5597 | 0.4123 | 0.0161 | 0.3065 | 2.8226 |
| C | 27 | 0.4955 | 0.3344 | 0.0000 | 0.0370 | 5.2593 |

Bucket boundaries: A - 2-3; B - 4-6; C - 7-9 required agents (as in `evaluator_set.py`).

**Reproduction (all three splits):**

```bash
REWARD='{"alpha":1.0,"beta":0.5,"gamma":1.0,"p_good":0.85,"p_bad":0.35}'
for SPLIT in train val test; do
  python -m multiagent_dqn_routing.experiments.run_rule_set \
    --split_path "data/splits_adaptive/${SPLIT}.jsonl" \
    --dataset_path data/tasks_set_adaptive_full.jsonl \
    --seed 42 \
    --reward_config_json "$REWARD" \
    --json_out "artifacts/baseline_adaptive/rule_${SPLIT}.json"
done
```

---

## 1. Baseline Snapshot v1 (2026-02-11)

**Protocol:** `tools/baseline_snapshot.py --config configs/baseline_protocol.json`

**Reward parameters (baseline snapshot):**

| Parameter | Value |
|---|---|
| alpha | 1.0 |
| beta | 0.5 |
| gamma | 1.0 |
| p_good | 0.85 |
| p_bad | 0.35 |

### Overall (test, n=49)

| Method | mean_f1 | mean_jaccard | exact_match | success_rate | precision | recall | avg_steps | avg_over | avg_under | reward |
|---|---|---|---|---|---|---|---|---|---|---|
| Random | 0.517 | 0.384 | 0.000 | 0.286 | 0.521 | 0.611 | 5.71 | 2.71 | 1.86 | 0.10 |
| Rule-based | 0.483 | 0.342 | 0.020 | 0.020 | 0.765 | 0.368 | 2.20 | 0.47 | 3.12 | −1.62 |
| Supervised (TF-IDF+LogReg) | **0.880** | **0.802** | **0.286** | **0.898** | 0.826 | **0.967** | 5.69 | 0.96 | **0.12** | **3.80** |
| LLM-Router | *skipped* | — | — | — | — | — | — | — | — | — |

### Bucket A (`|R| ∈ {2, 3}`)

| Method | mean_f1 | mean_jaccard | avg_over | avg_under |
|---|---|---|---|---|
| Random | 0.297 | 0.190 | 4.14 | 1.29 |
| Rule-based | 0.493 | 0.375 | 0.86 | 1.43 |
| Supervised | **0.820** | **0.715** | 0.93 | **0.14** |

### Bucket B (`|R| ∈ {4, 5, 6}`)

| Method | mean_f1 | mean_jaccard | avg_over | avg_under |
|---|---|---|---|---|
| Random | 0.560 | 0.410 | 2.54 | 1.96 |
| Rule-based | 0.464 | 0.319 | 0.42 | 3.38 |
| Supervised | **0.884** | **0.803** | 1.12 | **0.15** |

### Bucket C (`|R| ∈ {7, 8, 9}`)

| Method | mean_f1 | mean_jaccard | avg_over | avg_under |
|---|---|---|---|---|
| Random | 0.732 | 0.608 | 1.00 | 2.44 |
| Rule-based | 0.521 | 0.356 | 0.00 | 5.00 |
| Supervised | **0.965** | **0.934** | 0.56 | **0.00** |

---

## 2. Baseline Snapshot v2 (2026-03-01)

**Protocol:** `tools/baseline_snapshot.py --config configs/baseline_protocol.json`

**Dataset:** v2 (test n=159)

**Reward parameters (baseline snapshot):**

| Parameter | Value |
|---|---|
| alpha | 1.0 |
| beta | 0.5 |
| gamma | 1.0 |
| p_good | 0.85 |
| p_bad | 0.35 |

### Overall (test, n=159)

| Method | mean_f1 | mean_jaccard | exact_match | success_rate | precision | recall | avg_steps | avg_over | avg_under | reward |
|---|---|---|---|---|---|---|---|---|---|---|
| Random | 0.533 | 0.402 | 0.031 | 0.251 | 0.580 | 0.593 | 5.38 | 2.21 | 2.14 | 0.20 |
| Rule-based | 0.523 | 0.379 | 0.025 | 0.025 | 0.819 | 0.406 | 2.44 | 0.38 | 3.25 | −1.58 |
| Supervised (TF-IDF+LogReg) | **0.876** | **0.797** | **0.252** | **0.836** | **0.836** | **0.952** | **6.16** | **1.04** | **0.189** | **4.03** |
| LLM-Router | 0.854 | 0.773 | 0.333 | 0.352 | 0.928 | 0.804 | 4.46 | 0.26 | 1.11 | 2.43 |

### Bucket A (`|R| ∈ {2, 3}`, n=42)

| Method | mean_f1 | mean_jaccard | avg_over | avg_under |
|---|---|---|---|---|
| Random | 0.359 | 0.245 | 3.52 | 1.07 |
| Rule-based | 0.503 | 0.383 | 0.90 | 1.36 |
| Supervised | **0.824** | **0.730** | **0.95** | **0.14** |
| LLM-Router | 0.863 | 0.810 | 0.29 | 0.38 |

### Bucket B (`|R| ∈ {4, 5, 6}`, n=66)

| Method | mean_f1 | mean_jaccard | avg_over | avg_under |
|---|---|---|---|---|
| Random | 0.527 | 0.377 | 2.71 | 2.03 |
| Rule-based | 0.540 | 0.391 | 0.32 | 3.00 |
| Supervised | **0.854** | **0.756** | **1.33** | **0.30** |
| LLM-Router | 0.850 | 0.767 | 0.36 | 0.95 |

### Bucket C (`|R| ∈ {7, 8, 9}`, n=51)

| Method | mean_f1 | mean_jaccard | avg_over | avg_under |
|---|---|---|---|---|
| Random | 0.682 | 0.563 | 0.49 | 3.18 |
| Rule-based | 0.517 | 0.359 | 0.04 | 5.14 |
| Supervised | **0.948** | **0.905** | **0.75** | **0.08** |
| LLM-Router | 0.850 | 0.751 | 0.12 | 1.92 |

---

## 3. DDQN - Iteration 1 (config `ddqn_set_default.json`)

**Date:** 2026-02-12

**Dataset:** v1, `data/tasks_set.jsonl` (323 records; test n=49)

**Scientific hypothesis:**  
We test whether Double DQN can learn to select the optimal set of 9 agents using a TF-IDF representation of the text and a binary mask of selected agents as the state.

**Changes relative to the previous iteration:**
- first RL run instead of one-shot baselines
- kept the original stochastic reward without `step_cost`
- repeated selections were still not masked

**Motivation:**  
This was the starting point of the study: first, we needed to verify whether DDQN could learn a sequential selection policy at all in the basic setup. We deliberately started with the minimum number of new heuristics to observe the agent's natural dynamics.

**Parameters:**

| Parameter | Value |
|---|---|
| reward_mode | stochastic |
| alpha / beta / gamma | 1.0 / 0.5 / 1.0 |
| step_cost | 0.0 |
| p_good / p_bad | 0.85 / 0.30 |
| total_steps | 30 000 |
| lr / discount | 0.001 / 0.99 |
| buffer_size / batch_size | 20 000 / 128 |
| epsilon | 1.0 → 0.05 |
| target_update_every | 1 000 |
| use_action_mask | false |

**Result:** no separate artifact was saved, but the training log shows that the agent did not learn to press `STOP`: `avg_steps ≈ 9`. The policy quickly drifted into selecting almost all agents.

**Conclusion:** the agent did not learn to press `STOP`: `avg_steps ≈ 9`. `E[penalty] = p_bad × beta = 0.3 × 0.5 = 0.15` is too small compared with the terminal penalty `gamma = 1.0` for missing an agent. The agent prefers to play it safe and select everyone.

---

## 4. DDQN - Iteration 2 (config `ddqn_set_beta1_step005.json`)

**Date:** 2026-02-12

**Dataset:** v1, `data/tasks_set.jsonl` (323 records; test n=49)

**Scientific hypothesis:**  
We test whether increasing `beta` (penalty for an unnecessary agent) and adding `step_cost` will create enough pressure for earlier `STOP`.

**Changes relative to the previous iteration:**
- `beta`: `0.5 → 1.0`
- `step_cost`: `0.0 → 0.05`
- the rest of the architecture and schedule remained unchanged

**Motivation:**  
After iteration 1, it became clear that the penalty for an unnecessary agent was too weak compared to the fear of under-selection. The next step was simple reward shaping: increase the punishment for over-selection and make each additional step noticeably costly.

**Parameters:**

| Parameter | Value |
|---|---|
| reward_mode | stochastic |
| alpha / beta / gamma | 1.0 / 1.0 / 1.0 |
| step_cost | 0.05 |
| p_good / p_bad | 0.85 / 0.30 |
| total_steps | 30 000 |
| lr / discount | 0.001 / 0.99 |
| buffer_size / batch_size | 20 000 / 128 |
| epsilon | 1.0 → 0.05 |
| target_update_every | 1 000 |
| use_action_mask | false |

**Result:** no separate artifact was saved; according to the log, the agent still selected too many agents and did not show stable early `STOP`.

**Conclusion:** the expected penalty for an unnecessary agent increased to `0.35`, but the agent still preferred to select more agents. The imbalance between the fear of missing (`gamma`) and the penalty for extras (`beta × p_bad`) remained.

---

## 5. DDQN - Iteration 3 (config `ddqn_set_beta1_step005_gamma2_nomask.json`)

**Date:** 2026-02-13

**Dataset:** v1, `data/tasks_set.jsonl` (323 records; test n=49)

**Scientific hypothesis:**  
We test whether doubling the terminal penalty `gamma` (`1.0 → 2.0`) will strengthen the signal about missed agents and force the agent to select the set more carefully.

**Changes relative to the previous iteration:**
- `gamma`: `1.0 → 2.0`
- reward remains stochastic
- action masking is still disabled

**Motivation:**  
Iteration 2 showed that a flat increase in the penalty for an extra step does not change behavior radically. The next natural hypothesis was to amplify the terminal under-selection signal and see whether the agent would balance precision and recall more sensibly.

**Parameters:**

| Parameter | Value |
|---|---|
| reward_mode | stochastic |
| alpha / beta / gamma | 1.0 / 1.0 / 2.0 |
| step_cost | 0.05 |
| p_good / p_bad | 0.85 / 0.30 |
| total_steps | 30 000 |
| lr / discount | 0.001 / 0.99 |
| buffer_size / batch_size | 20 000 / 128 |
| epsilon | 1.0 → 0.05 |
| target_update_every | 1 000 |
| use_action_mask | false |

**Result:** `mean_f1 = 0.732`, `mean_jaccard = 0.600`, `precision = 0.638`, `recall = 0.918`, `avg_steps = 8.43` on test `n=49`.

**Conclusion:** paradoxical effect: increasing `gamma` amplified the fear of missing an agent, which led to even stronger over-selection (`avg_steps = 8.43`). Recall improved (`0.918`), but precision dropped (`0.638`). This is a classic precision-recall trade-off under a poorly calibrated reward.

---

## 6. DDQN - Iteration 4 (config `ddqn_set_beta1_step005_gamma2_actionmask.json`)

**Date:** 2026-03-01

**Dataset:** v2, `data/tasks_set.jsonl` (1054 records; test n=159)

**Scientific hypothesis:**  
We test whether forbidding repeated agent selection (action masking) will eliminate logical errors and improve set selection quality.

**Changes relative to the previous iteration:**
- `use_action_mask`: `false → true`
- buffer increased to `100 000`
- `target_update_every`: `1000 → 500`

**Motivation:**  
After iteration 3, it was important to remove at least technical behavior artifacts unrelated to the scientific `STOP` hypothesis. Action masking was supposed to eliminate duplicates and show how much of the problem comes from environment logic versus reward shape.

**Parameters:**

| Parameter | Value |
|---|---|
| reward_mode | stochastic |
| alpha / beta / gamma | 1.0 / 1.0 / 2.0 |
| step_cost | 0.05 |
| p_good / p_bad | 0.85 / 0.30 |
| total_steps | 30 000 |
| lr / discount | 0.001 / 0.99 |
| buffer_size / batch_size | 100 000 / 128 |
| epsilon | 1.0 → 0.05 |
| target_update_every | 500 |
| use_action_mask | true |

**Result:** `mean_f1 = 0.721`, `mean_jaccard = 0.600`, `precision = 0.601`, `recall = 0.995`, `avg_steps = 8.70`, `exact_match = 0.101` on test `n=159`.

**Conclusion:** action masking removed repeated selections, but did not solve the `STOP` problem. `avg_steps = 8.70` means the agent still chooses almost everyone. In bucket C (`|R| = 7..9`), DDQN reaches the level of Supervised (`F1 = 0.938`), but Supervised remains consistently ahead on overall F1 and exact match.

---

## 7. DDQN - Iteration 5 (config `ddqn_jaccard_step005.json`)

**Date:** 2026-03-28

**Dataset:** v2, `data/tasks_set.jsonl` (1054 records; test n=159)

**Scientific hypothesis:**  
We test whether replacing the stochastic stepwise reward with a deterministic terminal Jaccard reward will eliminate the high-variance learning signal and allow the Q-network to evaluate the `STOP` action confidently.

**Changes relative to the previous iteration:**
- reward was fully replaced by `reward_mode = "jaccard"`
- an ablation sweep over `step_cost ∈ {0.01, 0.05, 0.10, 0.20}` was run
- short `50 000`-step runs were used for each config in the sweep

**Motivation:**  
Iterations 1-4 showed that the problem looks less like a missing coefficient and more like a consequence of a noisy and internally contradictory reward scheme. The next step was therefore to remove stochasticity and switch to a signal that directly matches the target set metric.

**Parameters:**

| Parameter | Value |
|---|---|
| reward_mode | jaccard |
| step_cost | sweep: 0.01 / 0.05 / 0.10 / 0.20 |
| total_steps | 50 000 per config |
| hidden_sizes | [256, 256] |
| lr / discount | 0.001 / 0.99 |
| buffer_size / batch_size | 100 000 / 128 |
| use_action_mask | true |

**Result:** the ablation sweep over `step_cost ∈ {0.01, 0.05, 0.10, 0.20}` showed that all variants yield `avg_steps ≥ 8.4`. Separate test artifacts for each sweep config were not saved, but the qualitative outcome was identical: the agent gets stuck in the same local optimum regardless of the flat `step_cost` value.

**Conclusion:** flat `step_cost` is not the solution: the shape of the penalty matters more than its absolute magnitude. The marginal Jaccard gain from adding one more agent (`~0.06`) always exceeds the flat `step_cost` (`0.05`). This is a structural local optimum, not a parametric issue.

---

## 8. DDQN - Iteration 6 (config `ddqn_jaccard_curriculum.json`)

**Date:** 2026-03-29

**Dataset:** v2, `data/tasks_set.jsonl` (1054 records; test n=159)

**Scientific hypothesis:**  
We test whether training from simple examples to harder ones (`|R| = 2..3 → 4..6 → all`) will let the agent first learn early `STOP` on clearly favorable tasks and then transfer that skill to harder examples.

**Changes relative to the previous iteration:**
- added a 3-phase curriculum by `|R|` difficulty
- reward remains `jaccard` with `step_cost = 0.05`
- the full run was extended to `150 000` steps

**Motivation:**  
After the flat-Jaccard failure, it became clear that the agent may be averaging conflicting signals from easy and hard examples. Curriculum learning was supposed to temporarily isolate tasks where early `STOP` is clearly beneficial and then transfer that skill to the full dataset.

**Parameters:**

| Parameter | Value |
|---|---|
| reward_mode | jaccard |
| step_cost | 0.05 |
| curriculum | 0-50k: `|R|≤3`; 50k-100k: `|R|≤6`; 100k-150k: all |
| total_steps | 150 000 |
| hidden_sizes | [256, 256] |
| lr / discount | 0.001 / 0.99 |
| buffer_size / batch_size | 100 000 / 128 |
| use_action_mask | true |

**Result:** the smoke test showed `avg_steps = 6.4` - the best intermediate result before iteration 9. However, the full `150k` run returned the policy to `avg_steps = 9.0`; an exact standalone test `mean_f1` for the full run was not saved separately.

**Conclusion:** curriculum produced a temporary effect, but not a stable one. Catastrophic forgetting in phase 3 brought back the "select all" strategy. The problem is fundamental: the reward shape has to change, not only the training order.

---

## 9. DDQN - Iteration 7 (config `ddqn_adaptive_jaccard.json`)

**Date:** 2026-04-05

**Dataset:** adaptive full, `data/tasks_set_adaptive_full.jsonl` (871 records; test n=131)

**Scientific hypothesis:**  
We test whether extending the state with a context vector of intermediate agent outputs (`AdaptiveRoutingEnv`) will provide information unavailable to the supervised classifier and allow better decisions about whether to continue selection.

**Changes relative to the previous iteration:**
- introduced `AdaptiveRoutingEnv`
- state replaced with `[text_vec | selected_mask | context_vec]`
- `context_vec` is built from `adaptive.trajectory[*].output`
- reward remains `jaccard` + `step_cost = 0.05`

**Motivation:**  
This is the first genuinely sequential formulation where RL receives an observation that appears only after an action. If the hypothesis is correct, the adaptive environment should give DDQN an advantage over the one-shot supervised router not only because of the reward, but because of the richer state.

**Parameters:**

| Parameter | Value |
|---|---|
| env_mode | adaptive |
| reward_mode | jaccard |
| step_cost | 0.05 |
| total_steps | 150 000 |
| hidden_sizes | [256, 256] |
| lr / discount | 0.001 / 0.99 |
| buffer_size / batch_size | 100 000 / 128 |
| use_action_mask | true |

**Result:** best val `mean_f1 = 0.6682`, test `mean_f1 = 0.6517`, test `mean_jaccard = 0.5169`, `precision = 0.5514`, `recall = 0.8923`, `avg_steps = 7.7557`.

**Conclusion:** plateau from step `~36k` at `F1 ≈ 0.665`. `context_vec` turned out to be almost useless: the TF-IDF encoder is trained mostly on request texts and does not know the vocabulary of agent outputs. The agent learns to ignore `context_vec` as an uninformative signal.

---

## 10. DDQN - Iteration 8 (config `ddqn_adaptive_jaccard_v2.json`)

**Date:** 2026-04-05

**Dataset:** adaptive full, `data/tasks_set_adaptive_full.jsonl` (871 records; test n=131)

**Scientific hypothesis:**  
We test whether expanding the TF-IDF encoder training corpus with agent outputs (`texts + trajectory outputs`) will eliminate the OOV problem and make `context_vec` informative.

**Changes relative to the previous iteration:**
- encoder is trained on the joint corpus `texts + adaptive.trajectory[*].output`
- added helper `_build_adaptive_corpus()`
- train encoder corpus grew from `609` to `3533` documents

**Motivation:**  
Iteration 7 showed that the adaptive state is theoretically interesting, but the context representation is too weak. The next hypothesis was therefore narrowly technical: first remove the OOV gap between request language and intermediate agent-output language, and only then evaluate the usefulness of `context_vec`.

**Parameters:**

| Parameter | Value |
|---|---|
| env_mode | adaptive |
| reward_mode | jaccard |
| step_cost | 0.05 |
| encoder_corpus | `609` query texts + `2924` outputs |
| total_steps | 150 000 |
| hidden_sizes | [256, 256] |
| lr / discount | 0.001 / 0.99 |
| buffer_size / batch_size | 100 000 / 128 |
| use_action_mask | true |

**Result:** best val `mean_f1 = 0.6653`, test `mean_f1 = 0.6653`, `mean_jaccard = 0.5310`, `precision = 0.5310`, `recall = 1.0000`, `avg_steps = 9.0000`.

**Conclusion:** the hypothesis was partially confirmed technically (encoder corpus grew from `609` to `3533` documents), but it did not improve quality. The agent collapsed into the "select all" strategy (`avg_steps = 9.0`). The main limitation is not encoder quality, but the structural local optimum of flat Jaccard reward.

---

## 11. DDQN - Iteration 9 probe (config `ddqn_log_lambda010.json`)

**Date:** 2026-04-06

**Dataset:** v2 static, `data/splits/` (1054 records; test n=159)

**Scientific hypothesis:**  
We test whether an increasing logarithmic penalty (adapted from Puppeteer, Dang et al., NeurIPS 2025) will break the structural local optimum of flat reward: early steps are cheap (the agent is not afraid to start), late steps are expensive (adding unnecessary agents becomes unprofitable).

**Changes relative to the previous iteration:**
- introduced `reward_mode = "jaccard_log"`
- `lambda_eff = 0.10`
- `discount`: `0.99 → 0.95`, `lr`: `1e-3 → 1e-4`, `target_update_every`: `500 → 200`
- probe limited to `50 000` steps

**Motivation:**  
Iterations 5-8 showed that flat penalties do not break the select-all collapse. The next hypothesis therefore changed not the penalty level, but its geometry over time: make early steps cheap and later steps substantially expensive.

**Parameters:**

| Parameter | Value |
|---|---|
| reward_mode | jaccard_log |
| lambda_eff | 0.10 |
| total_steps | 50 000 |
| epsilon_decay_steps | 50 000 |
| lr / discount | 0.0001 / 0.95 |
| buffer_size / batch_size | 100 000 / 128 |
| target_update_every | 200 |
| use_action_mask | true |

**Result:** `val_mean_f1 = 0.856`, `avg_steps = 4.44`, `precision = 0.940`. The `STOP` problem was solved for the first time in 8 iterations; the model kept improving up to step `50k` with no sign of plateau.

**Conclusion:** the logarithmic penalty broke the select-all collapse. However, `epsilon_decay_steps = total_steps = 50k` meant that epsilon fell to `0.05` quickly in the probe run; later this turned out to be a critical success factor.

---

## 12. DDQN - Iteration 9 full v1 (config `ddqn_log_lambda010_full.json`)

**Date:** 2026-04-06

**Dataset:** v2 static, `data/splits/` (1054 records; test n=159)

**Scientific hypothesis:**  
A full `150k` run of the probe configuration should further improve F1 thanks to longer training.

**Changes relative to the previous iteration:**
- same `reward_mode = "jaccard_log"` and `lambda_eff = 0.10`
- `total_steps`: `50 000 → 150 000`
- mistakenly kept `epsilon_decay_steps = total_steps = 150 000`

**Motivation:**  
After the strong probe result, it was natural to expect that longer training would further improve the policy. This run tested the simple hypothesis "more steps = better" without changing the rest of the configuration.

**Parameters:**

| Parameter | Value |
|---|---|
| reward_mode | jaccard_log |
| lambda_eff | 0.10 |
| total_steps | 150 000 |
| epsilon_decay_steps | 150 000 |
| lr / discount | 0.0001 / 0.95 |
| buffer_size / batch_size | 100 000 / 128 |
| target_update_every | 200 |
| use_action_mask | true |

**Result:** plateau from step `~62k` at `F1 ≈ 0.710`, `avg_steps = 9.0`. The select-all collapse returned.

**Conclusion:** diagnosis: `epsilon_decay_steps = total_steps = 150k`. Epsilon decayed too slowly, the replay buffer filled with long random episodes, and the Q-network learned from "garbage" trajectories. The fix is to keep `epsilon_decay_steps = 50000` independently of `total_steps`.

---

## 13. DDQN - Iteration 9 full v2 (config `ddqn_log_lambda010_full_v2.json`)

**Date:** 2026-04-07

**Dataset:** v2 static, `data/splits/` (1054 records; test n=159)

**Scientific hypothesis:**  
We test whether fixing `epsilon_decay_steps = 50000` (independently of `total_steps = 150000`) will let the agent first explore under the correct pressure and then spend 100k steps learning from high-quality episodes.

**Changes relative to the previous iteration:**
- kept `reward_mode = "jaccard_log"` and `lambda_eff = 0.10`
- `epsilon_decay_steps`: `150 000 → 50 000`
- all other hyperparameters unchanged

**Motivation:**  
The failure of full v1 showed that the problem was not the idea of log reward itself, but the exploration schedule. The next experiment was therefore a focused diagnostic fix: keep the reward unchanged and modify only epsilon decay speed.

**Parameters:**

| Parameter | Value |
|---|---|
| reward_mode | jaccard_log |
| lambda_eff | 0.10 |
| total_steps | 150 000 |
| epsilon_decay_steps | 50 000 |
| lr / discount | 0.0001 / 0.95 |
| buffer_size / batch_size | 100 000 / 128 |
| target_update_every | 200 |
| use_action_mask | true |

**Result:** `val_mean_f1 = 0.897`, `test_mean_f1 = 0.857`, `avg_steps = 4.50`, `precision = 0.941`, `cost_ratio = 0.847`. This config outperformed Supervised on precision and exact match, but not on F1 versus the best variant.

**Conclusion:** the epsilon schedule is critical for log reward. With fast decay, the agent starts exploiting the correct policy quickly. The `val/test = 0.897 / 0.857` gap indicates mild overfitting to the val distribution.

---

## 14. DDQN - Iteration 9 ablation λ=0.05 (config `ddqn_log_lambda005_full.json`)

**Date:** 2026-04-07

**Dataset:** v2 static, `data/splits/` (1054 records; test n=159)

**Scientific hypothesis:**  
We test whether a smaller `λ = 0.05` (less aggressive step penalty) will improve recall (the agent will take more steps and cover more required agents) while preserving the improved precision of log reward.

**Changes relative to the previous iteration:**
- `lambda_eff`: `0.10 → 0.05`
- kept fixed `epsilon_decay_steps = 50000`
- the rest of the log-reward stack remained unchanged

**Motivation:**  
After full v2, it became clear that log reward works, but `λ = 0.10` may be too aggressive and stop the policy too early. The next step was an ablation over log-penalty strength to find the best precision / recall balance.

**Parameters:**

| Parameter | Value |
|---|---|
| reward_mode | jaccard_log |
| lambda_eff | 0.05 |
| total_steps | 150 000 |
| epsilon_decay_steps | 50 000 |
| lr / discount | 0.0001 / 0.95 |
| buffer_size / batch_size | 100 000 / 128 |
| target_update_every | 200 |
| use_action_mask | true |

**Result:** best val `mean_f1 = 0.9015`, test `mean_f1 = 0.8877`, `mean_jaccard = 0.8236`, `precision = 0.9267`, `recall = 0.8652`, `avg_steps = 4.9434`, `cost_ratio = 0.9313`, `exact_match = 0.4403`.

**Conclusion:** `λ = 0.05` is the best configuration of the study. A moderate logarithmic penalty creates the right balance: `precision = 0.927` (better than Supervised `0.836`) with `recall = 0.865` (lower than Supervised `0.952`). DDQN outperformed Supervised on test F1 for the first time (`0.888 vs 0.876`).

---

## 15. DDQN - Iteration 10 (config `ddqn_adaptive_log_lambda005.json`)

**Date:** 2026-04-07

**Dataset:** adaptive full, `data/tasks_set_adaptive_full.jsonl` (871 records; test n=131)

**Scientific hypothesis:**  
We test whether combining log reward (which solved the `STOP` problem on the static dataset) with `AdaptiveRoutingEnv` (which gives access to intermediate agent outputs) will create a synergistic effect: the agent should use context to decide more accurately whether to continue selection.

**Changes relative to the previous iteration:**
- combined `env_mode = "adaptive"` and `reward_mode = "jaccard_log"`
- used the best static log penalty `lambda_eff = 0.05`
- added separate train / val / test paths and `epsilon_decay_steps = 50000` for adaptive mode

**Motivation:**  
After the breakthrough on the static dataset, it was natural to test whether the same mechanism transfers to the richer adaptive state. This was the main test of the synergy hypothesis between correct reward geometry and access to intermediate agent context.

**Parameters:**

| Parameter | Value |
|---|---|
| env_mode | adaptive |
| reward_mode | jaccard_log |
| lambda_eff | 0.05 |
| total_steps | 150 000 |
| epsilon_decay_steps | 50 000 |
| lr / discount | 0.0001 / 0.95 |
| buffer_size / batch_size | 100 000 / 128 |
| target_update_every | 200 |
| use_action_mask | true |

**Result:** `val_mean_f1 = 0.668`, test `mean_f1 = 0.668`, `mean_jaccard = 0.532`, `precision = 0.568`, `recall = 0.938`, `avg_steps = 8.05`, `cost_ratio = 1.68`. Catastrophic forgetting was observed in steps `56k-66k`: F1 temporarily dropped to about `0.16`.

**Conclusion:** the synergy did not appear. Log reward did not help in the adaptive environment: the select-all collapse returned almost completely. The expanded adaptive-state space (`4009` vs `2009` dimensions) requires either a stronger Q-network or a better state encoder; TF-IDF `context_vec` remains too weak.

---

## Next Steps

- [x] Increase dataset size (LLM augmentation)
- [x] Try `use_action_mask = true` (forbid repeated selection)
- [x] `AdaptiveRoutingEnv` (iteration 7)
- [x] Expanded encoder corpus (iteration 8)
- [x] 50k probe for log reward `lambda_eff = 0.10`
- [x] Full 150k for log reward `lambda_eff = 0.10`
- [x] Fix `epsilon_decay_steps` for the full run (`50000` instead of `total_steps`)
- [x] Full 150k log-reward ablation `lambda_eff = 0.05` (`ddqn_log_lambda005_full.json`)
- [x] Check adaptive + log reward (`ddqn_adaptive_log_lambda005.json`)
- [ ] Ablation sweep `lambda_eff in {0.03, 0.15}` if needed
- [ ] Dense embeddings for the adaptive env
- [ ] Stronger network / different RL algorithm for the adaptive env (e.g. PPO)
- [ ] Compare against a one-shot adaptive policy
- [ ] Run the LLM-Router baseline with a real API key
