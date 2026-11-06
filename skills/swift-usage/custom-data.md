# Custom datasets: from raw data to `--dataset`

How to take a pile of business data (raw logs, exports, scraped tables) and make it trainable
with swift. Three channels, ordered by how little code they need. For the other plugin kinds
(reward / tool / callback / sampler) see [plugins.md](plugins.md); for CLI flag mechanics see
[cli-reference.md](cli-reference.md).

## Decision rules

| Channel | When | What you write |
| --- | --- | --- |
| Local file, zero registration | Data already looks like standard rows | Nothing — pass the path |
| `dataset_info` JSON | Many datasets, differences are only ids / column names / splits | One JSON file, no Python |
| `DatasetLoader` subclass | Rows need derivation logic (a column computed from another, an in-code corpus) | One Python plugin file |

**The core fact that makes this easy**: an *unregistered* dataset is not an error. Any hub id and
any local jsonl/csv is a legitimate dataset — the loader registry only adds knowledge (which
subsets exist, which preprocessor untangles the fields). Registration is an optimization, not a
gate (see `get_dataset_loader` in `swift/dev/dataset/loader/base.py`).

## Channel 1 — local file, zero registration

Write rows in the **messages-centric schema** the auto-detector understands (one conversation =
non-empty list of `{role, content}`):

```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Preference rows use `chosen`/`rejected` + `messages`; reranker rows use `query` +
`positive_messages`/`negative_messages`; embedding rows use `query` + `pos`/`neg` —
`swift/dev/dataset/schema.py` (`TaskKeySpec`) is the authoritative contract per task.

Then train directly:

```shell
USE_SWIFT_V5=1 swift sft --model ... --dataset /path/to/clean.jsonl ...
```

`--dataset` accepts the `#N` row-budget suffix (`/path/file.jsonl#2000`) and hub revisions.
This is where a preprocessing script should aim its output: a clean jsonl is the end of the
data story — no registration step exists to forget.

## Channel 2 — `dataset_info` JSON (data, not code)

When the only quirks are ids, column names, or splits: one JSON file, each entry becomes a real
`DatasetLoader` subclass (`register_dataset_info` in `swift/dev/dataset/loader/base.py`).

```json
{
  "my_math": {
    "ms_dataset_id": "my-org/my-math",
    "columns": {"question": "query", "answer": "response"},
    "tags": ["math"]
  },
  "my_local_qa": {
    "dataset_path": "/data/qa/train.jsonl",
    "columns": {"q": "query", "a": "response"}
  }
}
```

Entry keys (all optional except one id): `ms_dataset_id` / `hf_dataset_id` / `dataset_path`,
`dataset_type` (registry key; defaults to the trailing id component), `subsets`, `split`,
`columns` (raw name -> standard name, applied before auto-detection), `tags`, `help`,
`huge_dataset`. Load it:

```shell
USE_SWIFT_V5=1 swift sft --model ... --custom_dataset_info my_info.json --dataset my_math ...
```

`--custom_dataset_info` takes a path (or inline JSON string) and registers every entry at
startup; `--dataset my_math` then resolves through the registry like any built-in family.

## Channel 3 — `DatasetLoader` subclass (code)

Rows need real derivation logic. Declare everything on the class; the registry is keyed by
`dataset_type`:

```python
from swift.dev.dataset.loader import DatasetLoader, register_dataset

@register_dataset
class MyMathLoader(DatasetLoader):
    dataset_type = 'my_math'                      # registry key; required
    datasets = [('my-org/my-math', None)]         # (ms_id, hf_id); None = absent side
    subsets = ('default',)                        # most families leave this alone
    split = ('train',)
    # preprocessor = 'module:ClassName'           # only for genuinely odd field names
    columns = {'question': 'query'}               # renames beat preprocessor subclasses
    tags = ('math',)

# import path A: --external_plugins my_data.py   (CLI, before anything is built)
# import path B: drop the file in the webui workspace: workspace/plugins/my_data.py
```

Then `--external_plugins my_data.py --dataset my_math`. The full declaration surface is the
class docstring in `swift/dev/dataset/loader/base.py` (`datasets`, `dataset_paths`, `subsets`,
`split`, `preprocessor`, `columns`, `ms_revision`/`hf_revision`, `tags`, `help`,
`huge_dataset`). Same-name re-registration raises (pass `exist_ok=True` to override deliberately).

Rules of thumb inside the class:
- **Odd field names only** -> `columns` renames. No preprocessor subclass needed.
- **A column derived from another column** -> a `Preprocessor` (or build rows in code).
- Keep the plugin file self-contained: it is imported before anything is built.

## The webui workbench route (raw data lives on the user's machine)

The full preprocessing loop, as the workbench agent should run it:

1. **Ask the user** for the data format and the path. Paths outside the workspace are readable
   only after the user registers them (Settings -> 外部数据目录, read-only) — probe with
   `read_file` / `list_dir` with `root="external"`. Profile large files with the `dataset_profile`
   tool (streaming rows/columns/value-distribution/sample counts; no size limit, no shell needed);
   fall back to shell `head` / `wc` / `jq` only for details `dataset_profile` cannot answer.
2. **Write the preprocessing script** into the workspace (`write_file`), run it as a background
   task (`run_script`; shell has a 30s timeout). Completion wakes the conversation automatically —
   let the turn end after launching and wait for the wake-up; do NOT poll `read_script_log` for
   progress. Read the log only for mid-run diagnosis (suspected divergence, failure attribution),
   once, not in a loop. Edit scripts with `edit_file`, never `sed`.
3. **Aim the output at Channel 1**: a clean messages-format jsonl in the workspace. Pass its
   path in the train form's dataset field — no registration needed.
4. **Only if the data needs code**: write the plugin as `workspace/plugins/<name>.py`. Files
   there appear automatically in the Plugins page and every form's plugin picker (selected ->
   `--external_plugins`). A `dataset_info` JSON goes through the form's extra-args panel as
   `--custom_dataset_info /path/to/info.json`.

Contamination rule: the held-out evaluation split is frozen before any augmentation, and
nothing that generates training rows may read from it.

## Mixing external datasets

Your own data and hub datasets mix freely in one run: `--dataset own.jsonl 'org/id#N' ...`
(space-separated; `#N` caps the row budget per source — that is how you set the mixture
ratio). Reach for this when your own data lacks a capability signal that a public dataset
covers (search high quality candidates with the `list_catalog` tool or the ModelScope hub). Two guardrails:

- **Contamination check before mixing**: public code/QA collections often contain benchmark
  originals (coding collections regularly include HumanEval problems). Before adding a
  candidate, check it against your frozen eval set — same-source contamination silently
  invalidates the eval, which is worse than an eval that simply doesn't improve.
- **Keep the mixture traceable**: record every source and its `#N` in the experiment log.
  A mixture is one more variable; untracked ratios make regressions unattributable.
