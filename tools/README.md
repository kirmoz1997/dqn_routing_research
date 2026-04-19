# Tools

Utility scripts for working with the dataset. Run them from the **project root**.

> **Dataset language note:** the source dataset was originally created in Russian. These tools preserve that dataset content; this file only translates the documentation into English.

---

## tsv_to_jsonl.py

Converts the TSV draft into the JSONL dataset.

| Input | Output |
|------|-------|
| `data/tasks_set_draft.tsv` | `data/tasks_set.jsonl` |

If the input file is missing, it is skipped with a warning.

### Run

```bash
python tools/tsv_to_jsonl.py
```

### `tasks_set_draft.tsv` -> `tasks_set.jsonl`

**Header:**

```
id	required_agents	difficulty	eval_hint	text	notes
```

**Transformations:**

- `required_agents` is stored as a string like `"0,2,4"` and converted into the sorted integer list `[0, 2, 4]`
- `difficulty` is present in the TSV for compatibility, but is **not included** in the output JSONL

**Validation:**

- all `id` values must be unique and non-empty
- `required_agents` must contain 2 to 9 elements, all unique, each in the range `0..8`
- every row must contain exactly 6 columns

### General Behavior

- empty lines are skipped
- JSON is written as UTF-8 without ASCII escaping (`ensure_ascii=False`)
- on error, the script prints the line number and stops

---

## dataset_stats_set.py

Validation and statistics for `data/tasks_set.jsonl`.

### Run

```bash
python tools/dataset_stats_set.py [path_to_file]
```

Default: `data/tasks_set.jsonl`.

### What It Checks

- required fields: `id`, `text`, `required_agents`
- `required_agents`: list of ints, unique, range `0..8`, length `2..9`
- uniqueness of `id` across the whole file

### What It Prints

- total number of records
- distribution by `len(required_agents)` (`2..9`)
- frequency of each agent (`0..8`)
- top-10 `required_agents` signatures
- number of texts shorter than 20 characters

If errors are found, it prints up to 20 of them and exits with `exit(1)`.

---

## fix_dataset.py

One-off script for canonicalizing `data/tasks_set.jsonl`.

### Run

```bash
python tools/fix_dataset.py
```

### What It Does

1. Removes the `difficulty` field from each record.
2. Converts `required_agents` to canonical form (`sorted`, `unique`, `int 0..8`, `len 2..9`).
3. Validates before saving (if validation fails, the file is not overwritten).
4. Creates a backup: `data/tasks_set.jsonl.bak`.
5. Prints a report: how many records were processed, how many had `difficulty` removed, and how many had `required_agents` changed.

---

## split_jsonl_set.py

Splits `data/tasks_set.jsonl` into train / val / test subsets.

### Run

```bash
python tools/split_jsonl_set.py [options]
```

### Options

| Flag | Default | Description |
|------|-------------|----------|
| `--in_path` | `data/tasks_set.jsonl` | Path to the input JSONL |
| `--out_dir` | `data/splits` | Output directory for the splits |
| `--seed` | `42` | Shuffle seed |
| `--train` | `0.70` | Train share |
| `--val` | `0.15` | Validation share |
| `--test` | `0.15` | Test share |
| `--stratify_by_set_size` | `1` | `1` = stratify by `|R|`, `0` = plain split |

### Stratification (default)

- groups records by `k = len(required_agents)` (`k` from 2 to 9)
- inside each group: shuffle with seed, then split by proportions
- guarantees: if `n >= 3` in a group -> `val >= 1`, `test >= 1`; if `n == 2` -> `train=1`, `test=1`; if `n == 1` -> `train=1` plus a warning
- each split is shuffled again at the end

### Output

- final train / val / test sizes
- counts table by `k` (`2..9`) for each split
- warnings about agent coverage in train

### Result

```
data/splits/
  train.jsonl
  val.jsonl
  test.jsonl
```

---

## generate_adaptive_dataset.py

Generates `adaptive.trajectory` for adaptive experiments through an OpenAI-compatible API.

### Secure Configuration

The secret key **must not** be stored in code or in git. The script reads it from the environment:

- `ADAPTIVE_LLM_API_KEY` - required
- `ADAPTIVE_LLM_BASE_URL` - optional
- `ADAPTIVE_LLM_MODEL` - optional, default `qwen3-32b`

### Run

```bash
cp .env.example .env
# fill in .env with your values
source .env

python tools/generate_adaptive_dataset.py
```

You can override non-secret parameters with flags:

```bash
python tools/generate_adaptive_dataset.py \
  --base_url https://your-openai-compatible-endpoint/v1 \
  --model qwen3-32b \
  --limit 10
```

### Options

| Flag | Default | Description |
|------|-------------|----------|
| `--input` | `data/tasks_set.jsonl` | Input JSONL |
| `--output` | `data/tasks_set_adaptive_full.jsonl` | Output JSONL |
| `--base_url` | `env: ADAPTIVE_LLM_BASE_URL` | Base URL of the OpenAI-compatible API |
| `--model` | `env: ADAPTIVE_LLM_MODEL` or `qwen3-32b` | Model name |
| `--api_key_env` | `ADAPTIVE_LLM_API_KEY` | Name of the env variable holding the key |
| `--limit` | `None` | Process only the first N records |
| `--dry_run` | `false` | Show the first 2 prompts without making API calls |

---

## split_adaptive_dataset.py

Performs a stratified split of `data/tasks_set_adaptive_full.jsonl` into train / val / test.

### Run

```bash
python tools/split_adaptive_dataset.py
```

### What It Does

- reads the adaptive JSONL
- stratifies by `|R| = len(required_agents)`
- writes `data/splits_adaptive/{train,val,test}.jsonl`
- prints the set-size distribution across the splits

---

## baseline_snapshot.py

Official protocol for comparing baselines in a single run.

### Run

```bash
python tools/baseline_snapshot.py --config configs/baseline_protocol.json
```

### What It Does

- reads `configs/baseline_protocol.json`
- launches baseline scripts in the order defined by `order`
- passes shared `seed`, `max_steps`, `reward`, and `split_path`
- skips the LLM baseline if `include=false` or `api_key_env` is not set
- collects results into:
  - `artifacts/baselines_summary.json`
  - `artifacts/baselines_summary.md`
