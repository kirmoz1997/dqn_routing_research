# Research Plan — Multi-Agent Set Routing with DQN (v1.0.0)

## 1. Context and Goal
We study **routing a user request through a network of specialized agents**, where each request requires **not one agent**, but a **set of 2-9 agents**.  
The goal is to learn how to choose the **optimal set of agents** for each request while avoiding:
- **under-selection** (missing required agents),
- **over-selection** (choosing unnecessary agents).

The order in which agents are called is **not important**. The final set is what matters.

> **Dataset language note:** the dataset was originally created and annotated in Russian. The project documentation is now translated into English, but the underlying request texts remain Russian in the source dataset.

## 2. Agents (9 classes)
Each agent is a narrow specialization:

0. Code Agent (Python)  
1. SQL Agent  
2. Data Analysis Agent (Pandas)  
3. Math Formula Solver  
4. Structured Extraction Agent (JSON)  
5. Summarization & Formatting Agent  
6. Requirements / Spec Agent  
7. Rewrite / Style Constraints Agent  
8. Finance / Numeric Computation Agent  

## 3. Problem Formulation (multi-label routing)
For each request, the true set of required agents is `R` (`|R| ∈ [2, 9]`).  
The router selects a set of agents `S` (`|S| ∈ [2, 9]`) through a sequence of actions.

Error categories:
- **Coverage**: `|S ∩ R|`
- **Over-selection**: `|S \ R|`
- **Under-selection**: `|R \ S|`

## 4. MDP Formalization (for DQN)
The task is formalized as an MDP, where the routing agent sequentially chooses agents.

### 4.1 State
The project uses two state formulations.

**Static formulation**
- request text `text`,
- selected-agent mask `selected_mask` (length 9, 0/1),
- step indicator (`step_feature` in the encoder code).

This is the original formulation, where the agent sees only the original request and the fact of previous selections.

**Adaptive formulation**
- TF-IDF vector of the request `text_vec`,
- selected-agent mask `selected_mask`,
- context vector `context_vec`.

`context_vec` is built by concatenating the outputs of already selected agents from `adaptive.trajectory` and encoding them with the same `TfidfStateEncoder` as the request itself.
Therefore, the adaptive environment uses the state:

`state = [text_vec | selected_mask | context_vec]`

This is the key extension of the formulation: after each action, the policy receives new information that was absent from the original request text.

### 4.2 Actions
- choose one of the 9 agents that has not yet been selected,
- a special `STOP` action (finish set selection).

In the current implementation, repeated selection depends on the mode:
- when `use_action_mask = true`, repeated selection is genuinely blocked by the action mask;
- when `use_action_mask = false`, the environment and evaluator technically allow duplicates, but such actions provide no new useful information and usually worsen reward/metrics.

### 4.3 Step Limit
- `max_steps = 9` (no more than the number of unique agents).
- `STOP` can be taken at any step.

## 5. Agent Quality Model (simulation)
The main quality simulation does not call a real LLM.  
Instead, it assumes:
- if the selected agent belongs to `R`, it "fires successfully" with probability `p_good = 0.85`,
- if the selected agent does not belong to `R`, a penalty is applied with probability `p_bad = 0.30`.

In v1.0.0, a step is considered "useful" if the selected agent belongs to `R` and has not been covered before.

## 6. Reward — Set Optimization
The reward is defined so that it:
- rewards selecting required agents,
- penalizes extra agents,
- penalizes missing agents after termination.

Initial parameters:
- `alpha = +1.0` for covering a new required agent,
- `beta = 0.5` penalty for selecting an unnecessary agent,
- `gamma = 1.0` penalty for each missed required agent at the end,
- `step_cost = 0.0` penalty for each selection step (encourages earlier `STOP`).

### 6.1 Per-step Reward
At each step, when choosing agent `a` (if it is not `STOP`):
- `reward -= step_cost` (per-step penalty, encourages shorter selections)
- if `a ∈ R` and is not yet covered: `reward += alpha` (with probability `p_good`)
- if `a ∉ R`: `reward -= beta` (with probability `p_bad`)
- if `a ∈ R`, but was already selected: `reward += 0`

### 6.2 Reward at `STOP` / episode termination
After `STOP` or reaching `max_steps`:
- `missing = |R \ S|`
- `reward -= gamma * missing`

The final objective is to maximize total episode reward.

### 6.3 Alternative: Global Jaccard Reward (Iteration 5+)

**Motivation:** the stochastic stepwise reward (§6.1) creates high-variance Q-targets and conflicting signals. The agent fears missing a required agent (because `gamma` is large) while receiving only a weak penalty for an unnecessary one (`E[penalty] = p_bad × beta`). As a result, it does not learn to press `STOP` and selects about 9/9 agents in each episode (`avg_steps ≈ 8.70` in iterations 1-4).

**New scheme:** fixed `step_cost` for each step + terminal Jaccard reward at episode end.

Formulas:
- At each step (agent selection): `r_step = -step_cost`
- At `STOP` / `max_steps`: `r_terminal = |S ∩ R| / |S ∪ R|` (Jaccard index)

**Advantages:**
- the reward directly matches the evaluation metric (`mean_jaccard`)
- it removes `p_good`/`p_bad` stochasticity from the learning signal
- deterministic signal -> more stable Q-targets

**Trade-off:** the reward becomes sparser and requires more training steps (`150k` vs `30k`).

**Usage:** `reward_mode = "jaccard"` in config and environment. Implemented by `RewardSetJaccard` in `sim/reward_set.py`.

In adaptive experiments, this scheme is used as the base because it:
- does not require simulating the quality of each intermediate output,
- makes policy comparison transparent through set metrics,
- allows `context_vec` to be interpreted without extra stochasticity in the reward.

### 6.4 Logarithmic Step Penalty (Iteration 9+)

**Motivation:** a flat `step_cost` creates a structural local optimum:
the marginal Jaccard gain from adding one more agent (`~0.06`)
consistently exceeds the same per-step penalty (`0.05`). The agent therefore rationally never stops.

**Solution (adapted from Puppeteer, Dang et al., NeurIPS 2025):**
an increasing penalty for each additional step:

    r_step(t) = -lambda * log(1 + t / max_steps)
    r_terminal = Jaccard(S, R)

where `t` is the current step number (1-indexed), and `lambda` is the weighting coefficient.

**Key property:** early steps are cheap (the agent is not afraid to start selecting), while late steps are expensive (adding extra agents becomes unprofitable). This breaks the structural local optimum of the "select all agents" strategy.

**Implementation:** `RewardSetLogJaccard` in `sim/reward_set.py`.
Activated by `reward_mode = "jaccard_log"` in the config.

**Additional hyperparameters from the Puppeteer analysis:**
- `gamma = 0.95` (instead of `0.99`): discounts the future more aggressively, creating pressure toward earlier `STOP`
- `lr = 1e-4` (instead of `1e-3`): stabilizes training under the new reward shape

**Experiment status (v2 static):** see `EXPERIMENT_LOG.md`, §11. A full `150k` run with `lambda_eff=0.05`, `epsilon_decay_steps=50000` (`configs/ddqn_log_lambda005_full.json`) achieved test `mean_f1 ≈ 0.89`, `avg_steps ≈ 4.9`, `cost_ratio ≈ 0.93` - the policy no longer collapses into selecting all agents when `λ` is relaxed.

## 7. Dataset

### 7.1 Size
The initial target size was about 300 requests.  
The current size is **1054 requests** (`data/tasks_set.jsonl`). The dataset was expanded after the initial version (323 records).

An additional **adaptive dataset** was built for the sequential setting:
- full pool: **871 records** (`data/tasks_set_adaptive_full.jsonl`),
- splits: train=609, val=131, test=131 (`data/splits_adaptive/`),
- each record contains not only `required_agents`, but also a synthetic trajectory of intermediate agent outputs.

### 7.2 Balance
Examples are balanced by:
- the size `|R|` (requests requiring 2..9 agents),
- signal types in the text.

### 7.3 Data Format
Editing source format: `TSV` (convenient for manual editing).  
Final code-ready format: `JSONL`.

**TSV (draft):** `data/tasks_set_draft.tsv`  
Fields:
- `id`
- `required_agents` (string like `0,2,4` - unique IDs, 2..9 items)
- `difficulty` (historical column; **removed** during conversion to JSONL)
- `eval_hint`
- `text`
- `notes`

**JSONL (canonical):** `data/tasks_set.jsonl`  
One line = one JSON object. The `difficulty` field is **not included** in JSONL.

Example record:
```json
{"id":"ex_0001","required_agents":[0,2,4],"eval_hint":"code + JSON extraction","text":"...","notes":"..."}
```

Canonical format of `required_agents`:
- list of `int`, sorted ascending, without duplicates,
- each element in the range `0..8`,
- length from `2` to `9`.

**Adaptive JSONL:** `data/tasks_set_adaptive_full.jsonl`

The adaptive version extends the base record with the field `adaptive.trajectory`:

```json
{
  "id": "gen_9_0001",
  "required_agents": [1, 2, 5],
  "text": "…",
  "adaptive": {
    "trajectory": [
      {
        "agent_id": 1,
        "agent_name": "SQL Agent",
        "output": "SQL query prepared ...",
        "remaining_gap": "Retention analysis and a short summary are still needed",
        "is_last": false
      }
    ]
  }
}
```

Field interpretation:
- `output`: text that becomes the observable context after selecting the agent,
- `remaining_gap`: which part of the task is still not covered after this step,
- `is_last`: indicator of the final step in the generated trajectory.

### 7.4 Train / Val / Test Splits
The script `tools/split_jsonl_set.py` splits the dataset into train / val / test.

- It supports **stratified** splitting by `k = len(required_agents)` (default).
- It guarantees that each bucket `k` appears in val and test (if there are at least 3 records in the group).
- Default proportions: train=0.70, val=0.15, test=0.15.
- Output: `data/splits/{train,val,test}.jsonl`.

For the adaptive dataset, a separate script `tools/split_adaptive_dataset.py` is used.
It repeats the same stratification by `|R|` and writes the result to
`data/splits_adaptive/{train,val,test}.jsonl`.

### 7.5 Generating Adaptive Annotations

The adaptive dataset is not meant to redefine ground truth; it models sequential partial observability.
For this purpose, `tools/generate_adaptive_dataset.py` builds a trajectory of intermediate steps for each request based on its `required_agents`.

Scientific motivation for this generation:
- the policy receives information that appears only **after** an action,
- the supervised multi-label baseline cannot use this information,
- RL gets a fundamentally richer setting than one-shot prediction.

## 8. Baseline Models (4)
For all baselines, the result is a set of agents `S`, interpreted as the final sequence of selected agents before stopping / reaching the limit. In the current implementation, `Random`, `Rule-based`, and `Supervised` produce sets without duplicates; the evaluator remains more general and can handle duplicates if they appear.

1. **Random Routing** (`run_random_set.py`)  
Selects agents randomly until stopping / hitting the limit (without duplicates).

2. **Rule-Based Routing** (`run_rule_set.py`)  
Extracts agents via trigger words / patterns.  
If too few agents are selected, it pads with random ones.  
If too many are selected, it trims by priority.

3. **Supervised Router** (`run_supervised_set.py`)  
TF-IDF `(1,2)`-gram + `OneVsRest(LogisticRegression)` -> multi-hot -> agent set.  
The threshold is selected by a sweep on the val split, maximizing `mean_f1`.  
A minimum of 2 agents and a maximum of 9 are guaranteed.

4. **LLM-Router (API + prompt)** (`run_llm_set.py`)  
A real LLM, guided by a system prompt, returns a JSON list of `agent_id` values (`2..9`).  
The prompt is recall-biased: it prioritizes completeness (not missing required agents) and includes an internal checklist of subtasks.  
The prompt version is fixed by the constant `PROMPT_VERSION` (current: `v2-recall-biased`).

**Caching:**  
- the cache is stored in a JSONL file (default: `cache/llm_router_cache.jsonl`)  
- cache key: item `id`, filtered by `model` + `prompt_version`  
- if the model or prompt version changes, the old cache is automatically ignored and items are re-requested via API  
- each cache entry contains: `id`, `pred`, `raw`, `model`, `prompt_version`

**Error handling:**  
- up to `max_retries` repeated requests on API failure or invalid JSON  
- keyword fallback on complete parsing failure (guarantees at least 2 agents)

## 9. Main Method: DQN
DQN is trained to choose actions (an agent or `STOP`) that maximize expected cumulative reward.  
The state includes the text and the mask of already selected agents.

### 9.1 AdaptiveRoutingEnv and Scientific Motivation

In the static setting, RL has little advantage over the supervised router:
if the full observation is known in advance, the task can be solved as one-step multi-label classification.

`AdaptiveRoutingEnv` changes this property. After an agent is selected, the policy receives its `output`, which updates `context_vec` and can change the next decision.
This turns routing into a problem of **sequential information acquisition**:
choices affect not only reward, but also future observations.

This is exactly where RL gains a theoretical advantage:
- access to intermediate context,
- the ability to adapt policy after each step,
- a natural interpretation of `STOP` as a decision that enough information has been collected.

### 9.2 Curriculum Learning

Standard uniform sampling over the full dataset creates conflicting learning signals:
for examples with small `|R|`, early stopping is optimal; for large `|R|`, it is not.
The agent averages these signals and falls back to the strategy "select all agents" (`avg_steps ≈ 8.7`).

Curriculum learning addresses this by organizing training into three phases
with increasing difficulty in `|R|`. In phase 1, the agent sees only examples
with `|R| ∈ {2,3}` and receives an unambiguous signal: stopping early is always correct.
This learned skill is then transferred to more complex examples in phases 2 and 3.

Implementation: the config parameter `curriculum.enabled`;
the method `env.set_items()` switches the active example pool at predefined points during training.

### 9.3 Shared-vocabulary Hypothesis for `context_vec`

In the adaptive environment, `context_vec` is encoded with the same TF-IDF encoder as the request itself.
This creates an important methodological issue: if the encoder is trained only on query texts,
a large share of the vocabulary from agent outputs becomes out-of-vocabulary.
Then `context_vec` degenerates into an almost zero signal, and the policy effectively ignores the main difference of the adaptive setting.

This led to the iteration 8 hypothesis:
- train the encoder on the joint corpus `texts + adaptive.trajectory[*].output`,
- obtain a shared vocabulary for the query space and the context space,
- increase the informativeness of `context_vec`.

The experiment showed that this hypothesis is **not sufficient**:
vocabulary coverage improved, but the policy still collapsed into a maximum-recall strategy.
Therefore, the bottleneck lies not only in the encoder, but also in the optimization dynamics of `STOP` under Jaccard reward.

## 10. Evaluation Metrics

### 10.1 Main Metrics (global)
For each method, we compute the following on the test set:

| Metric | Description |
|---|---|
| `mean_episode_reward` | Average total episode reward |
| `success_rate` | Share of tasks where `missing = 0` (all required agents are covered) |
| `exact_match_rate` | Share of tasks where `set(S) == set(R)` (exact set match) |
| `mean_jaccard` | Average Jaccard: `|S ∩ R| / |S ∪ R|` |
| `mean_precision` | Average precision: `|S ∩ R| / |S|` |
| `mean_recall` | Average recall: `|S ∩ R| / |R|` |
| `mean_f1` | Average F1: `2 * precision * recall / (precision + recall)` |
| `avg_steps` | Average selected-set length |
| `avg_coverage` | Average `|S ∩ R|` |
| `avg_overselection` | Average `|S \ R|` |
| `avg_underselection` | Average `|R \ S|` |

### 10.2 Bucket Metrics
All metrics from §10.1 are also computed for three difficulty groups (by target-set size):

| Bucket | Sizes `|R|` | Characterization |
|---|---|---|
| **A** | {2, 3} | Small sets - easier to select, higher over-selection risk |
| **B** | {4, 5, 6} | Medium sets - the bulk of the data |
| **C** | {7, 8, 9} | Large sets - higher under-selection risk (early `STOP`) |

### 10.3 Threshold Selection (Supervised Router)
Threshold sweep on the val split with this ranking:
1. max `mean_f1`
2. max `mean_jaccard`
3. max `mean_episode_reward`
4. min `avg_steps`

## 11. Reproducibility
- Random seeds are fixed for all stochastic generators.
- Dataset versions and reward parameters are stored in the repository.
- All experiments are run via scripts in `src/multiagent_dqn_routing/experiments/`.
- The stratified split guarantees representation of all `|R|` sizes in each split.
- For `LLM-Router`, the prompt version (`PROMPT_VERSION`) is fixed in code and stored in cache. When the version changes, the cache is automatically invalidated (old entries are ignored when `prompt_version` does not match).
- For adaptive mode, separate artifacts are fixed: `data/tasks_set_adaptive_full.jsonl`, `data/splits_adaptive/*.jsonl`, and configs `ddqn_adaptive_jaccard*.json`.
- The smoke-check for iteration 8 must print the expanded encoder-corpus size, which serves as a sanity check that `adaptive.trajectory[*].output` was correctly included in TF-IDF training.

### 11.1 Critical Hyperparameters

When reproducing the iteration 9 results, pay special attention to:
- `epsilon_decay_steps` must be **equal to 50000**, not `total_steps`
- when `epsilon_decay_steps = total_steps`, you reproduce the plateau (iteration 9 full v1), not the best result
- this is a documented reproducibility trap

## 12. Experimental Outcomes

### 12.1 Iteration Timeline

| Iteration | Key change | avg_steps | F1 (test) | Conclusion |
|---|---|---|---|---|
| 1 | Base DDQN | ~9.0 | ~0.70 | `STOP` does not work |
| 2 | `beta↑`, `step_cost` | ~9.0 | ~0.71 | Did not help |
| 3 | `gamma↑` | 8.43 | 0.732 | Worse precision |
| 4 | Action masking | 8.70 | 0.721 | `STOP` still unresolved |
| 5 | Flat Jaccard reward | 8.4-8.9 | ~0.71 | Structural optimum |
| 6 | Curriculum | 9.0 | ~0.71 | Catastrophic forgetting |
| 7 | Adaptive env | 7.76 | 0.652 | OOV problem |
| 8 | Expanded encoder | 9.0 | 0.665 | Did not help |
| **9** | **Log reward λ=0.05** | **4.94** | **0.888** | **BREAKTHROUGH** |
| 10 | Adaptive + log reward | 8.05 | 0.668 | Dense embeddings are needed |

### 12.2 Main Conclusion

The logarithmic per-step penalty (adapted from Puppeteer, Dang et al., NeurIPS 2025) is the key mechanism for solving the `STOP` problem in sequential set selection with DDQN. A flat `step_cost` creates a structural local optimum; an increasing penalty makes late steps unprofitable.
