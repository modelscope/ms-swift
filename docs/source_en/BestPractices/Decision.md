# Decision Model Training

SWIFT supports training and inference of **typed-decision (System-1)** scoring models via the `decision` task type. Unlike generative models that produce tokens, a decision model performs a **single forward pass** and outputs a calibrated probability for each option of each question, then softmaxes within each question's own option set.

This is useful for classification, yes/no, and ordered-scoring tasks where the model must choose among a fixed set of options per question rather than generate free-form text.

## Environment Setup

```shell
pip install "transformers>=5.2.0" "qwen_vl_utils>=0.0.14" decord
pip install "ms-swift>=4.6.0"
```

For OmniJev video support, `ffmpeg` and `ffprobe` must also be installed on the system.

## Supported Models

1. JEV-27B-VL: Qwen3-VL + PEFT adapter
   - [ModelScope](https://www.modelscope.cn/models/autotrust/JEV-27B-VL)
2. Clef: Full Qwen3-VL with LoRA already merged
   - [ModelScope](https://www.modelscope.cn/models/Cloudflare/clef)
3. OmniJev: Qwen3.5-4B (hybrid gated-deltanet + VL) + PEFT adapter
   - [ModelScope](https://modelscope.cn/models/AI-ModelScope/OmniJev)

| Model | `--model_type` | Base | Head Artifacts | `--loss_type` |
|-------|----------------|------|----------------|----------------|
| JEV-27B-VL | `jev` | Qwen3-VL + PEFT adapter | `decision_head.json` + `calibration.json` (24-slot verbalizer bias) | `distill_rps` |
| Clef | `clef` | Merged Qwen3-VL (LoRA already merged) | `joint_head.safetensors` + `joint_head_config.json` | `smooth_brier` |
| OmniJev | `omnijev` | Qwen3.5-4B hybrid + PEFT adapter | `head.pt` (FiLM) + `ord.pt` (ordinal regression) + `new_tok_emb.pt` (2 new tokens) + `head_meta.json` | `ce_rps` |

Developers can convert any CausalLM into a JEV-style decision model by specifying:

```shell
swift sft --task_type decision
```

## Loss Functions

All decision losses are **proper scoring rules** computed per-question over each question's own option set.

### Notation

- $Q$: Total questions in a micro-batch (after all records are flattened)
- $K_q$: Number of options for question q (`noul`=2, `score`=6, `choice`=dataset-defined)
- $z_q$: Per-option logits from the head (before softmax), shape $\mathbb{R}^{K_q}$
- $p_q = \mathrm{softmax}(z_q)$: Predicted probabilities (masked softmax, normalized over real options only), shape $\mathbb{R}^{K_q}$
- $t_q$: Target distribution (hard one-hot or soft distillation labels), shape $\Delta^{K_q}$
- $\hat{p}_q, \hat{t}_q$: Cumulative distributions (CDFs) of $p_q, t_q$, where $\hat{p}_q^{(k)} = \sum_{j \le k} p_q^{(j)}$

### Shared Term: Masked Soft Cross-Entropy

All losses share a base term — masked soft cross-entropy, which **is** the **log_score** (a proper scoring rule):

$$\mathcal{L}_{\mathrm{CE}} = \frac{1}{|Q|} \sum_{q \in Q} \left( -\sum_{k=1}^{K_q} t_q^{(k)} \log p_q^{(k)} \right)$$

When $t_q$ is a soft target, this equals $\mathrm{KL}(t_q \| p_q)$ (dropping the constant $H(t_q)$ which does not affect gradients); when $t_q$ is one-hot, it reduces to standard cross-entropy. Each model-specific loss adds a calibration term on top.

The shared cross-entropy is computed with FP32 masked `log_softmax`, without clipping probabilities. This preserves corrective gradients for confidently wrong predictions, including with BF16 logits. Padding and rows with no valid options contribute zero loss and gradient.

### Loss Details

**distill_rps** (JEV):

$$\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \lambda_r \cdot \mathcal{L}_{\mathrm{RPS}}^{\mathrm{score}}$$

- The base term uses soft-label distillation targets (teacher probability distributions), so $\mathcal{L}_{\mathrm{CE}} \equiv \mathrm{KL}(t \| p)$.
- The RPS term applies only to `score` questions (ordinal scoring), with weight $\lambda_r = 0.5$.

$$\mathcal{L}_{\mathrm{RPS}}^{\mathrm{score}} = \frac{1}{|Q|} \sum_{q \in Q_{\mathrm{score}}} \frac{1}{K_q - 1} \sum_{k=1}^{K_q-1} (\hat{p}_q^{(k)} - \hat{t}_q^{(k)})^2$$

**smooth_brier** (Clef):

$$\mathcal{L} = \mathcal{L}_{\mathrm{CE}}^{\mathrm{LS}} + \lambda_b \cdot \mathcal{L}_{\mathrm{Brier}}$$

- Label-smoothed CE: mixes the one-hot target with a uniform distribution, $\tilde{t}_q^{(k)} = (1-\varepsilon) \cdot t_q^{(k)} + \varepsilon / K_q$, with $\varepsilon$ = `CLEF_LABEL_SMOOTHING` (default 0.1).
- Brier score (all question types), with weight $\lambda_b$ = `CLEF_BRIER_WEIGHT` (default 1.0).

$$\mathcal{L}_{\mathrm{Brier}} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{K_q} \sum_{k=1}^{K_q} (p_q^{(k)} - \tilde{t}_q^{(k)})^2$$

**ce_rps** (OmniJev):

$$\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \lambda_r \cdot \mathcal{L}_{\mathrm{RPS}}^{\mathrm{score}}$$

- The base term is plain log_score with no label smoothing (`noul` 2-column binary log-score, `choice` $K{+}1$ columns including abstain, `score` $L$ columns ordinal).
- RPS applies only to `score` questions, with weight $\lambda_r$ = `OMNIJEV_RPS_WEIGHT` (default 0.5).
- Unlike `distill_rps`, the RPS here is **not** per-row normalized (no division by $K_q{-}1$):

$$\mathcal{L}_{\mathrm{RPS}} = \frac{1}{|Q|} \sum_{q \in Q_{\mathrm{score}}} \sum_{k=1}^{K_q} (\hat{p}_q^{(k)} - \hat{t}_q^{(k)})^2$$

**scoring**:

$$\mathcal{L} = \mathcal{L}_{\mathrm{CE}}$$

- Only the shared term, no extra calibration. Use when no model-specific loss applies.

## Accuracy

`ScoringTrainer` automatically computes two accuracy metrics during training and validation, both logged in the training log — no extra configuration needed:

- `acc` **(per-question)**: masked argmax over each question's option set gives the prediction, compared to the gold (argmax of `target_probs`) — each question is scored independently.
- `record_acc` **(per-record)**: record-level accuracy where all must be correct — a record is counted as correct only when all of its questions are predicted correctly, otherwise it is counted as incorrect. Especially meaningful for multi-question joint-decode models (e.g. Clef).

## Temperature Calibration

The `temperature_calibration` callback performs **post-training temperature scaling** on the **eval set** after training finishes and before the final save — making the model's confidence numbers honest (questions where the model claims 80% confidence should be right about 80% of the time).

### Why calibration is needed

Fine-tuned models are systematically **overconfident** (or underconfident): the softmax probabilities drift away from the empirical accuracy. Temperature scaling corrects the logits with a scalar $T$: $p = \mathrm{softmax}(z/T)$. Since $T$ does not change the relative order of logits, **the argmax (predicted option, accuracy) is unchanged** — only the "sharpness" of the distribution is reshaped: T > 1 flattens the distribution (lower confidence), T < 1 sharpens it (higher confidence). If your downstream consumes confidence for thresholding, ranking, or risk weighting, calibration is the prerequisite for those numbers to be meaningful.

### Activation and skip conditions

**Auto-activation**: the callback is attached automatically for `task_type=decision` — no manual flag needed. It can also be enabled explicitly:

```bash
--callbacks temperature_calibration
```

**Silently skipped** (temperatures stay at their loaded/factory values, no error):

- the head carries no temperature buffer (e.g. Clef's `ClefJointSchemaHead`);
- the trainer has no eval_dataset;
- fewer than 10 questions are collected from the eval set (tunable via `SWIFT_CALIBRATION_MIN_ITEMS`);
- an exception is raised during logits collection.

**Prerequisite**: an eval set. The example training scripts set `--split_dataset_ratio 0`; to calibrate, change it to `--split_dataset_ratio 0.1` (carves 10% of the training rows, whole records, as the eval set) or pass `--val_dataset` explicitly.

### How calibration works (two-stage fitting)

The temperature is not a single scalar but a **global × per-kind** decomposition:

1. **Stage 1**: fit a single global temperature `head_temperature` ($T_h$) on ALL questions pooled across kinds, capturing the model's overall "sharpness" bias;
2. **Stage 2**: scale logits by $T_h$, then fit a per-kind relative temperature $T_r^{\mathrm{kind}}$, capturing kind-specific residuals;
3. The served temperature is the product $T^{\mathrm{kind}} = T_h \cdot T_r^{\mathrm{kind}}$, clamped to the range 0.05~20.

The fitting objective (the quantity being minimised) is selected by `SWIFT_CALIBRATION_LOSS`. Let $p = \mathrm{softmax}(z/T)$ be the calibrated probability (masked, normalised over real options only) and $t$ be the target distribution; the two objectives are:

$$
\mathcal{L}_{\mathrm{Brier}} = \sum_k \left( p_k - t_k \right)^2 \qquad \mathcal{L}_{\mathrm{NLL}} = -\sum_k t_k \log p_k
$$

| Loss          | Solver                          | Notes                                                       |
| ------------- | ------------------------------- | ----------------------------------------------------------- |
| `brier` (default) | coarse grid (0.05–10, 50 steps) + golden section | No scipy/LBFGS dependency, bounded gradients, robust to noise |
| `nll`         | LBFGS + strong_wolfe line search | Stronger penalty on confident-but-wrong answers; more aggressive at suppressing overconfidence |

**Minimum sample threshold**: calibration does not start with fewer than 10 questions in total (tunable via `SWIFT_CALIBRATION_MIN_ITEMS`); once started, a kind with fewer than 10 questions keeps $T_r = 1$ (only sufficiently populated kinds are fitted). Fitting a temperature on a tiny sample fits noise — keeping the default T = 1 (no correction) is safer than a wrong correction.

**Reported metrics (2-fold cross-validation)**: to avoid the in-sample optimism of scoring the fitted data with the fitted temperature, the accuracy / ECE / Brier / NLL in the logs are **out-of-fold** — eval records are split into two halves by parity, each half's temperature is fitted on the opposite half, and used to score its own half. Note that **the temperature written into the checkpoint is still fitted on ALL eval data** (the production estimate); the CV only affects the reported numbers. All questions of one record always land in the same fold, preventing leakage.

### Reading the logs

A base baseline is logged at the start of training (`on_train_begin`, evaluated with the loaded/factory temperatures, report-only):

```
temperature_calibration: base metrics (pre-training, n=520): accuracy=0.6231, ece=0.1523, brier=0.2045, nll=1.1023
```

After training and calibration:

```
temperature_calibration done (loss=brier, head_T=1.412): noul: T=1.523 (Brier 0.1823 -> 0.1102, n=180); choice: T=1.298 (Brier 0.2151 -> 0.1534, n=220); score: T=1.611 (Brier 0.2401 -> 0.1620, n=120)
temperature_calibration CV (out-of-fold): accuracy=0.8077, ece=0.0311, brier=0.1420, nll=0.5621
temperature_calibration: base (pre-training, n=520) vs tuned (out-of-fold): accuracy 0.6231 -> 0.8077, ece 0.1523 -> 0.0311, brier 0.2045 -> 0.1420, nll 1.1023 -> 0.5621
temperature_calibration: written to ['output/scoring_head.json', ...]
```

- Line 1: per-kind final T, in-sample Brier before/after calibration, and sample counts;
- Line 2: out-of-fold metrics. **Accuracy is unchanged by calibration** (temperature does not move the argmax) — it is the control; **ECE is the most direct measure of whether calibration worked** (pure calibration error, not involved in the fitting);
- Line 3: base (pre-training) vs tuned comparison, quantifying what the whole fine-tune bought you.

### Persistence and serving

The fitted temperatures are written into the head's non-persistent buffers and serialised via `save_pretrained` to **`scoring_head.json`** in `output_dir` and the last checkpoint directory (not into safetensors — the JSON stays human-readable). Loading a checkpoint restores them automatically via `maybe_load_trained_scoring_head` → `_load_calibration_meta`; the probabilities served by `swift infer` are therefore already **calibrated**, with no extra deployment step.

### Multi-GPU (DDP) behavior

Under DDP training the eval set is sharded across ranks (each rank sees only about 1/world_size of the questions); the callback handles this in two layers:

- **Collection**: each rank forwards over its own shard to collect logits/targets/masks, then merges them into the **full eval set** via `all_gather_object` (record_ids are offset per rank to avoid collisions when splitting CV folds) — ensuring the temperature is fitted on the full data rather than on one shard;
- **Fitting and writing**: only the main process (rank 0) fits the temperature, writes `scoring_head.json` and logs; the other ranks return right after the gather — guaranteeing that the persisted values and the logged values agree, and that the result is deterministic.

### Environment variables

| Variable                      | Values         | Default | Description                                                   |
| ----------------------------- | -------------- | ------- | ------------------------------------------------------------- |
| `SWIFT_CALIBRATION_LOSS`      | `brier` / `nll` | `brier` | Fitting objective (invalid values fall back to brier with a warning) |
| `SWIFT_CALIBRATION_MIN_ITEMS` | positive int   | `10`    | Minimum sample threshold for starting calibration / fitting a kind |

Environment variables are process-scoped: in multi-node training, make sure they are exported on every node. Their values are not recorded in `args.json`, so preserve them separately when reproducing experiments.

```bash
# Example: switch to the NLL objective and lower the sample threshold
export SWIFT_CALIBRATION_LOSS=nll
export SWIFT_CALIBRATION_MIN_ITEMS=5
swift sft --task_type decision ... --split_dataset_ratio 0.1
```

## Dataset Format

Each model has its own raw JSONL schema. The preprocessor normalizes them into the unified internal contract.

### JEV

JEV scores **one question per forward**. A record with multiple questions is fanned out into one row per question.

```json lines
{"state": "The sky is blue.", "questions": [{"kind": "noul", "question": "Is the sky blue?", "gold": 1}]}
{"state": "A review: the product arrived late.", "questions": [{"kind": "choice", "question": "Classify sentiment.", "options": ["positive", "negative", "neutral"], "gold": 1}]}
{"state": "some text to rate", "questions": [{"kind": "score", "question": "Rate quality.", "gold": [0.05, 0.1, 0.15, 0.4, 0.2, 0.1]}]}
```

- `state`: string (text state to evaluate). Can also be a dict or JSON object.
- `questions`: list of question dicts. Each has `kind`, `question` (text), `options` (for `choice` only; `noul` and `score` use canonical fixed options), and `gold`.
- `gold`: for `noul` — int (0 = no, 1 = yes); for `choice` — int option index; for `score` — either int level index or a full soft distribution (list of floats for distillation).

> **Canonical options**: JEV forces fixed canonical options for `noul` (yes/no) and `score` (levels 0–5) and ignores caller-supplied option text for those kinds. Only `choice` uses caller-supplied options. The preprocessor follows this behavior.

### Clef

Clef decodes **all questions of a record jointly in a single forward** — the only multi-question-per-forward model of the three.

```json lines
{"id": "clef_noul", "state": {"text": "The sky is blue.", "lang": "en"}, "questions": {"is_blue": {"type": "noul", "instructions": "Is the sky blue?"}}, "answers": {"is_blue": true}}
{"id": "clef_choice", "state": "A review: the product arrived late.", "questions": {"sentiment": {"type": "choice", "instructions": "Classify sentiment.", "criteria": {"positive": "Good", "negative": "Bad", "neutral": "Mixed"}}}, "answers": {"sentiment": "negative"}}
{"id": "clef_score", "state": {"doc": "some text"}, "questions": {"quality": {"type": "score", "instructions": "Rate quality.", "criteria": ["terrible", "poor", "ok", "good", "great"]}}, "answers": {"quality": 3}}
```

- `state`: string or dict (rendered as SCHEMA FIELDS).
- `questions`: a dict keyed by question ID. Each value has `type` (`noul`/`choice`/`score`), `instructions`, and `criteria` (option definitions: dict for `choice`, list for `score`).
- `answers`: a dict keyed by question ID, giving the gold answer (bool for `noul`, key string for `choice`, int index for `score`).

### OmniJev

OmniJev scores **one question per forward** over a **multimodal** state (image, or a video pre-composed into a mosaic).

```json lines
{"state": "Look at the image.", "images": ["http://example.com/animal.png"], "questions": [{"type": "noul", "instructions": "Is there an animal?", "gold": true}]}
{"images": ["http://example.com/animal.png"], "questions": [{"type": "choice", "instructions": "What is in the image?", "options": [{"key": "cat", "text": "a cat"}, {"key": "dog", "text": "a dog"}], "gold": "cat"}]}
{"images": ["http://example.com/animal.png"], "questions": [{"type": "score", "instructions": "How clear?", "levels": ["blurry", "ok", "sharp"], "gold": "sharp"}]}
{"images": ["http://example.com/animal.png"], "questions": [{"type": "choice", "instructions": "Which region?", "options": [{"key": "left", "region": {"box": [0, 0, 100, 100]}}, {"key": "none", "abstain": true}], "gold": "left"}]}
```

- `state` (optional): text state. If omitted, the image is the state.
- `images`: list of image paths or URLs.
- `questions`: list of question dicts. Each has `type`, `instructions`, and type-specific fields:
  - `noul`: `gold` is a bool.
  - `choice`: `options` is a list of `{key, text}` or `{key, region}` or `{key, abstain: true}`. `gold` is the key string.
  - `score`: `levels` is a list of level labels. `gold` is the level label string.
- `videos`: list of video paths. **Note**: the training/inference template does NOT decode raw video files inline. You must first call `swift.model.omnijev_media.video_state(path)` offline to convert a raw video into a `{"images": ["mosaic.jpg"], "video": {...}}` state, then pass that state in your data. `video_state` invokes `ffmpeg`/`ffprobe` to uniformly sample 16 frames, compose a 4×4 mosaic image, and attach timestamp metadata. Passing a raw video path directly will raise an error.

## Training

Training scripts provided by ms-swift (using **local mini datasets** — no download needed):

- [JEV](https://github.com/modelscope/ms-swift/blob/main/examples/train/decision/jev.sh) — `--external_plugins examples/train/decision/dataset.py --dataset jev_decision`
- [Clef](https://github.com/modelscope/ms-swift/blob/main/examples/train/decision/clef.sh) — `--external_plugins examples/train/decision/dataset.py --dataset clef_decision`
- [OmniJev](https://github.com/modelscope/ms-swift/blob/main/examples/train/decision/omnijev.sh) — `--external_plugins examples/train/decision/dataset.py --dataset omnijev_decision`

It also supports **remote datasets** (e.g. `--dataset open_jev_jev` as shown below, requiring online download of the Open-Jev corpus). The two methods differ only in data source; all other parameters are identical. See the [Datasets](#datasets) section for details.

### JEV

JEV's decision ability lives in a **shipped adapter** (`adapter_vllm/`), so training continues that adapter:

```bash
SWIFT_PY="$(sed -n '1s/^#!//p' "$(command -v swift)")"
JEV_ADAPTER="$("$SWIFT_PY" -c "import os; from modelscope import snapshot_download; print(os.path.join(snapshot_download('autotrust/JEV-27B-VL'), 'adapter_vllm'))")"

USE_HF=0 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model autotrust/JEV-27B-VL \
    --model_type jev \
    --task_type decision \
    --adapters "$JEV_ADAPTER" \
    --loss_type distill_rps \
    --dataset open_jev_jev:all \
    --torch_dtype bfloat16 \
    --attn_impl eager \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --max_length 4096 \
    --split_dataset_ratio 0 \
    --output_dir output \
    --dataloader_num_workers 1
```

Key parameters:
- `--model_type jev` — resolves JEV's base model and head artifacts
- `--task_type decision` — routes to `ScoringTrainer` + decision data collator
- `--adapters "$JEV_ADAPTER"` — continue-trains the shipped adapter (is_trainable=True)
- `--loss_type distill_rps` — KL(target‖model) + 0.5 × RPS on `score`

### Clef

Clef ships a **merged backbone**, so a fresh LoRA is created:

```bash
USE_HF=0 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Cloudflare/clef \
    --model_type clef \
    --task_type decision \
    --loss_type smooth_brier \
    --dataset open_jev_clef:all \
    --tuner_type lora \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --attn_impl eager \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --max_length 4096 \
    --split_dataset_ratio 0 \
    --output_dir output \
    --dataloader_num_workers 1
```

Key parameters:
- `--model_type clef`
- `--loss_type smooth_brier` — label-smoothing CE + Brier
- `--tuner_type lora --target_modules all-linear` — fresh LoRA on the merged backbone

### OmniJev

OmniJev ships **adapter-only** (base weights are separate), with head artifacts that live in the adapter directory:

```bash
SWIFT_PY="$(sed -n '1s/^#!//p' "$(command -v swift)")"
OMNIJEV_CKPT="$("$SWIFT_PY" -c "from modelscope import snapshot_download; print(snapshot_download('AI-ModelScope/OmniJev'))")"
export OMNIJEV_HEAD="$OMNIJEV_CKPT/head.pt"
export OMNIJEV_ORD="$OMNIJEV_CKPT/ord.pt"
export OMNIJEV_HEAD_META="$OMNIJEV_CKPT/head_meta.json"
export OMNIJEV_NEW_TOK_EMB="$OMNIJEV_CKPT/new_tok_emb.pt"
export MSO_BRANCH_CKPT=1

USE_HF=0 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen3.5-4B \
    --model_type omnijev \
    --task_type decision \
    --adapters "$OMNIJEV_CKPT" \
    --loss_type ce_rps \
    --dataset open_jev_omnijev:all \
    --torch_dtype bfloat16 \
    --attn_impl eager \
    --freeze_vit true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing false \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --max_length 4096 \
    --split_dataset_ratio 0 \
    --output_dir output \
    --dataloader_num_workers 1
```

Key parameters:
- `--model Qwen/Qwen3.5-4B` — supplies the hybrid base
- `--model_type omnijev`
- `--adapters "$OMNIJEV_CKPT"` — continue-trains the shipped adapter
- `--loss_type ce_rps` — proper scoring rules only (log_score + RPS on `score`)
- `--gradient_checkpointing false` — see [OmniJev Notes](#omnijev-msol_branch_ckpt-and-gradient-checkpointing)

### Train Any CausalLM from Scratch

The `generic_decision` model type lets you convert **any text CausalLM** into a JEV-style decision model. Unlike the three model-specific paths, this builds a randomly-initialized `JevVerbalizerHead` from the tokenizer — no factory weights required:

> **How `JevVerbalizerHead` works**: This is a lightweight verbalizer scoring head that introduces no additional neural network layers — it directly reuses the base CausalLM's `lm_logits`. At the last token position of each question's prompt, it applies `log_softmax` to `lm_logits`, then gathers the log-probabilities of the verbalizer tokens corresponding to the question's `kind`. The three kinds map to **24 fixed verbalizer tokens**:
> - `noul` (2): `false`, `true`
> - `score` (6): `0`, `1`, `2`, `3`, `4`, `5`
> - `choice` (16): `A` … `P`
>
> Each verbalizer token has a trainable fp32 bias scalar (24 in total), plus 3 per-kind temperature parameters. The forward pass computes `z = (log_prob + bias) / temperature`, followed by a softmax over the question's own option set. The `from_tokenizer` construction path resolves these 24 token IDs from the tokenizer, zero-initializes the bias, and unit-initializes the temperatures, so the head can be jointly trained with a LoRA adapter starting from a random state.

```bash
USE_HF=0 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --model_type generic_decision \
    --task_type decision \
    --loss_type distill_rps \
    --dataset open_jev_jev:all \
    --tuner_type lora \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --max_length 4096 \
    --split_dataset_ratio 0 \
    --output_dir output \
    --dataloader_num_workers 1
```

Key parameters:
- `--model <any CausalLM>` — any model that `AutoModelForCausalLM.from_pretrained` can load
- `--model_type generic_decision` — picks `GenericDecisionLoader`, which builds the head via `JevVerbalizerHead.from_tokenizer` (resolves the 24 verbalizer token ids from the tokenizer, zero-init bias, unit temperatures)
- `--loss_type distill_rps` — KL(target‖model) + 0.5 × RPS on `score`; or use `--loss_type scoring` for plain soft CE without RPS term
- `--tuner_type lora --target_modules all-linear` — `lm_head` is included in `all-linear`, adapting the verbalizer rows read by the head

The head's 24 fp32 bias scalars are jointly trained with the LoRA adapter — `ScoringTrainer.__init__` automatically re-activates the `scoring_head` parameters (sets `requires_grad=True`, converts fp16→fp32). After training, the checkpoint contains `adapter_model.safetensors` + `scoring_head.safetensors` + `scoring_head.json`, deployable with:

```bash
swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --model_type generic_decision \
    --adapters output/checkpoint-xxx
```

The data format is the same as JEV's.

## Datasets

Decision model datasets come in two forms: **local datasets** (registered via `--external_plugins` with the appropriate `ScoringPreprocessor` subclass) and **remote datasets** (SWIFT's built-in Open-Jev corpus, loaded directly with `--dataset <name>`).

> **Important**: Decision data **cannot** be loaded via `--dataset /path/to/file.jsonl` directly. Swift's standard data-loading method (passing a file path, which uses `AutoPreprocessor`) does not work for decision data — `AutoPreprocessor` does not recognize the `{state, questions: [{kind, question, options, gold}]}` schema and will either silently take the wrong path or error out. You **must** register a Python plugin via `--external_plugins` that uses the appropriate `ScoringPreprocessor` subclass.

### Remote Datasets (Open-Jev)

SWIFT registers the [Open-Jev](https://www.modelscope.cn/datasets/ZefanCai/Open-Jev) and [Open-Jev-v1.1](https://modelscope.cn/datasets/ZefanCai/Open-Jev-v1.1) corpora as built-in remote datasets. No `--external_plugins` is needed — use `--dataset <name>` directly.

| Dataset Name | Source | Model Adapter |
|---|---|---|
| `open_jev_jev` | `ZefanCai/Open-Jev` (v1) | `JevPreprocessor` |
| `open_jev_clef` | `ZefanCai/Open-Jev` (v1) | `ClefPreprocessor` |
| `open_jev_omnijev` | `ZefanCai/Open-Jev` (v1) | `OmniJevPreprocessor` |
| `open_jev_jev_v11` | `ZefanCai/Open-Jev-v1.1` | `JevPreprocessor` |
| `open_jev_clef_v11` | `ZefanCai/Open-Jev-v1.1` | `ClefPreprocessor` |
| `open_jev_omnijev_v11` | `ZefanCai/Open-Jev-v1.1` | `OmniJevPreprocessor` |

To use the v1.1 corpus, replace the dataset name (e.g. `--dataset open_jev_jev_v11`).

The `--dataset` in the training commands shown in this document uses remote dataset names (e.g. `open_jev_jev`), which require downloading the `ZefanCai/Open-Jev` corpus online.

#### Open-Jev Raw Row Examples

A single row in the Open-Jev dataset contains one standalone question record with fields `kind`, `question`, `state`, `options`, `target`:

```json lines
{"kind": "choice", "question": "What is the capital of France?", "state": "France is a country in Western Europe. Its capital city is Paris.", "options": ["London", "Paris", "Berlin", "Madrid"], "target": [0.1, 0.7, 0.15, 0.05]}
{"kind": "noul", "question": "Is the sky blue during daytime?", "state": "The sky appears blue due to Rayleigh scattering.", "target": [0.1, 0.9]}
{"kind": "score", "question": "Rate the quality of this response.", "state": "The response is mostly accurate but lacks detail.", "target": [0.0, 0.0, 0.2, 0.5, 0.2, 0.1]}
```

- `kind`: question type (`noul` / `choice` / `score`).
- `question`: question text.
- `state`: the text state to evaluate (string or JSON object).
- `options`: list of options (only required for `choice`; `noul` and `score` use fixed canonical options and may be omitted).
- `target`: teacher soft distribution (float list) for distillation. Length aligns with option count.

#### Preprocessed Output Rows

`OpenJevPreprocessor(JevPreprocessor())` normalizes each row into the unified internal contract. `target` is remapped to align with the canonical option order:

choice (`options` preserve original order, `target` aligned):
```json
{"messages": [{"role": "user", "content": "France is a country in Western Europe. Its capital city is Paris."}], "questions": ["What is the capital of France?"], "kinds": ["choice"], "options": [["London", "Paris", "Berlin", "Madrid"]], "target_probs": [[0.1, 0.7, 0.15, 0.05]]}
```

noul (`options` forced to canonical `["false", "true"]`, `target` remapped):
```json
{"messages": [{"role": "user", "content": "The sky appears blue due to Rayleigh scattering."}], "questions": ["Is the sky blue during daytime?"], "kinds": ["noul"], "options": [["false", "true"]], "target_probs": [[0.1, 0.9]]}
```

score (`options` forced to canonical `["0", "1", "2", "3", "4", "5"]`):
```json
{"messages": [{"role": "user", "content": "The response is mostly accurate but lacks detail."}], "questions": ["Rate the quality of this response."], "kinds": ["score"], "options": [["0", "1", "2", "3", "4", "5"]], "target_probs": [[0.0, 0.0, 0.2, 0.5, 0.2, 0.1]]}
```

> **Note**: JEV renders one question per row (`JevPreprocessor` fans a record out into per-question rows), `noul`/`score` options are forced to the canonical verbalizer set, and `target` is remapped to align with canonical options.

### Local Datasets

Local datasets require a Python plugin file passed via `--external_plugins`. The plugin uses `register_dataset` with the appropriate `ScoringPreprocessor` subclass.

#### Ready-Made Examples

The repository ships a complete set of local mini datasets in `examples/train/decision/`:

| File | Description |
| --- | --- |
| `dataset.py` | Plugin registration file; registers `jev_decision`, `clef_decision`, `omnijev_decision` dataset names |
| `data/jev.jsonl` | JEV-style mini example (4 rows: noul / choice / score / multi-question fan-out) |
| `data/clef.jsonl` | Clef-style mini example (4 rows: noul / choice / score / multi-question joint decode) |
| `data/omnijev.jsonl` | OmniJev-style mini example (4 rows: noul / choice / score / region+abstain) |
| `jev.sh` / `clef.sh` / `omnijev.sh` | Training scripts that use the local datasets |

Run a training script directly:

```bash
sh examples/train/decision/jev.sh
```

The scripts load local data via `--external_plugins examples/train/decision/dataset.py --dataset jev_decision` — no remote download needed. The training commands in this document use remote datasets (e.g. `--dataset open_jev_jev`); the only difference is the data source — all other parameters are identical.

#### Writing a Custom Plugin

```python
# my_decision_dataset.py
import os
from swift.dataset import DatasetMeta, register_dataset
from swift.dataset.preprocessor.decision import JevPreprocessor, ClefPreprocessor, OmniJevPreprocessor

register_dataset(
    DatasetMeta(
        dataset_name='my_jev_decision',
        dataset_path='/path/to/my_jev_data.jsonl',
        preprocess_func=JevPreprocessor(),
    ),
    exist_ok=True,
)
```

Then use it in training:

```bash
swift sft --external_plugins my_decision_dataset.py --dataset my_jev_decision ...
```

Choose the preprocessor based on your data schema:
- `JevPreprocessor` — for JEV-style `{state, questions: [{kind, question, options, gold}]}` records (one question per row after fan-out)
- `ClefPreprocessor` — for Clef-style `{state, questions: {qid: {type, instructions, criteria}}, answers}` records (all questions per row, joint decode)
- `OmniJevPreprocessor` — for OmniJev-style multimodal records with `images`/`videos` and option blocks
- `GenericDecisionPreprocessor` — identical behavior to `JevPreprocessor`; pairs with the `generic_decision` model type

## Data Augmentation

Decision datasets support training-time schema augmentation via CLI parameters. All disabled by default — when all parameters are at their defaults, augmentation is a zero-cost passthrough.

| CLI Parameter | Type | Default | Description |
|---------------|------|---------|-------------|
| `--decision_drop_instructions` | `float` | `0.0` | Probability of dropping the question's instruction text (the layout is still rendered, but the text is emptied) |
| `--decision_subsample_options` | `int` | `0` | Subsample `choice` options to at most this many (keeps gold + N-1 distractors). 0 disables |
| `--decision_state_as_text` | `float` | `0.0` | Probability of flattening a dict `state` into human-readable text |
| `--decision_shuffle_fields` | `bool` | `False` | Shuffle fields in dict `state` and question order in multi-question records |

> Augmentation random seed reuses `--data_seed`; no separate flag needed.

Example — train JEV with 30% instruction dropout, option subsampling to 8, and field shuffling:

```bash
swift sft \
    --model_type jev \
    --task_type decision \
    --loss_type distill_rps \
    --dataset open_jev_jev:all \
    --adapters .../adapter_vllm \
    --decision_drop_instructions 0.3 \
    --decision_subsample_options 8 \
    --decision_shuffle_fields true \
    --torch_dtype bfloat16 \
    --output_dir output \
    ...
```

The augmentation is applied by `augment_decision_row` during preprocessing. It is injected automatically from the CLI parameters — no plugin code needed.

## Inference

Decision models use the transformers backend only (vLLM is not supported for the `decision` task type). Inference follows the standard `swift infer` path — the trained head is automatically loaded via the guarded hook.

### Command Line

```bash
# JEV
swift infer \
    --model autotrust/JEV-27B-VL \
    --model_type jev \
    --adapters output/checkpoint-xxx

# Clef
swift infer \
    --model Cloudflare/clef \
    --model_type clef \
    --adapters output/checkpoint-xxx

# OmniJev
SWIFT_PY="$(sed -n '1s/^#!//p' "$(command -v swift)")"
OMNIJEV_CKPT="$("$SWIFT_PY" -c "from modelscope import snapshot_download; print(snapshot_download('AI-ModelScope/OmniJev'))")"
export OMNIJEV_HEAD="$OMNIJEV_CKPT/head.pt"
export OMNIJEV_ORD="$OMNIJEV_CKPT/ord.pt"
export OMNIJEV_HEAD_META="$OMNIJEV_CKPT/head_meta.json"
export OMNIJEV_NEW_TOK_EMB="$OMNIJEV_CKPT/new_tok_emb.pt"
export MSO_BRANCH_CKPT=1

swift infer \
    --model Qwen/Qwen3.5-4B \
    --model_type omnijev \
    --adapters output/checkpoint-xxx

# Generic Decision
swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --model_type generic_decision \
    --adapters output/checkpoint-xxx
```

The model returns per-option probabilities (softmaxed within each question's option set) rather than generated text. Each record's inference output is a list with one entry per question:

```json
[{"index": 1, "probs": [0.1, 0.7, 0.2]}]
```

where `index` is the model's predicted option index (argmax of `probs`), and `probs` are the softmax probabilities over each option.

### Inference Evaluation (accuracy)

Pass `--metric acc` during inference to automatically compute per-question accuracy on a decision dataset:

```bash
swift infer \
    --model autotrust/JEV-27B-VL \
    --model_type jev \
    --adapters output/checkpoint-xxx \
    --metric acc
```

The computation takes the argmax of `target_probs` as the gold answer and the `index` field from each question's `response` as the prediction, compares them per-question, and outputs `{'acc': ...}`.

## Head Save / Load Mechanism

Decision models carry a **pluggable scoring head** (`ScoringHead`) that is trained jointly with the backbone (or LoRA adapter). The head is stored **separately** from the backbone weights to avoid the "config says generative model but weights contain a classifier" mismatch:

- **Save**: `ScoringTrainer._save` first saves the adapter (or full model) via the standard path, then writes the head artifacts beside it. For adapter (LoRA) training, it writes both `scoring_head.safetensors` (head parameters) and `scoring_head.json` (head type + calibration metadata). For full-finetune training, it writes only `scoring_head.json` — the head parameters are already inside the model's safetensors shards (ignored by `from_pretrained` because `scoring_head` was not yet attached).
- **Load**: `maybe_load_trained_scoring_head(model, adapter_dir)` is called automatically after `Swift.from_pretrained` on all three loading paths (`--adapters` continue-training, `--resume_from_checkpoint`, and inference). It handles two scenarios: when `scoring_head.safetensors` exists (adapter checkpoint), it loads weights from it; when only `scoring_head.json` exists (full-finetune checkpoint), it extracts head weights from the model's safetensors shards. It is a **guarded no-op** for non-decision models.

This means a trained decision checkpoint can be deployed immediately — `swift infer` or `--adapters <checkpoint>` will automatically reload the trained head.

## OmniJev: `MSO_BRANCH_CKPT` and Gradient Checkpointing

OmniJev's hybrid Qwen3.5-4B backbone uses linear attention layers that carry recurrent state across the sequence. A block-diagonal attention mask cannot isolate options because the recurrent state leaks information between option blocks. This is solved with a **two-stage cached branch** mechanism (`swift/model/omnijev_branch.py`):

1. The shared prefix (chat template + image tokens + instructions) is forwarded once, keeping KV/recurrent cache.
2. Each (question, option) pair is a row that reuses the prefix cache via `expand_cache`, so options are physically isolated.

**`--gradient_checkpointing` is incompatible with this mechanism.** HF forces `use_cache=False` when `gradient_checkpointing=True` and `training=True`, but the branch's prefix pass must return a non-None cache. The branch forward auto-disables gradient checkpointing internally, so the flag is effectively a **no-op** for OmniJev.

**Use `MSO_BRANCH_CKPT=1` instead** for memory savings (measured: 14.3 GiB vs 16.2 GiB without it).

```bash
export MSO_BRANCH_CKPT=1
# ...
swift sft --gradient_checkpointing false ...
```

The OmniJev adapter also requires four head artifacts pointed to via environment variables (they live in the adapter directory, not the base model directory):

| Env Var | File | Purpose |
|---------|------|---------|
| `OMNIJEV_HEAD` | `head.pt` | FiLM dual-MLP scoring head |
| `OMNIJEV_ORD` | `ord.pt` | Ordinal regression (CORAL/CORN) head |
| `OMNIJEV_HEAD_META` | `head_meta.json` | Head metadata (kind temperatures, etc.) |
| `OMNIJEV_NEW_TOK_EMB` | `new_tok_emb.pt` | Embeddings for 2 new tokens (`<|opt|>`/`<|/opt|>`) |

### JEV / Clef Head Artifact Environment Overrides

When head artifacts are not in the same directory as the base model (e.g. custom deployment paths), JEV and Clef each have two environment variables to override head config/weight loading paths:

| Env Var | Model | Overrides File | Purpose |
|---|---|---|---|
| `JEV_DECISION_HEAD` | JEV | `decision_head.json` | Override verbalizer bias / ids load path |
| `JEV_DECIDE_CALIBRATION` | JEV | `calibration.json` | Override per-kind temperature calibration path |
| `CLEF_JOINT_HEAD_CONFIG` | Clef | `joint_head_config.json` | Override head config load path |
| `CLEF_JOINT_HEAD_WEIGHTS` | Clef | `joint_head.safetensors` | Override head weights load path |

When unset, the loader searches `model_dir` (and `adapter_vllm/` subdirectory for JEV) automatically. On file-missing errors, the message instructs the user to set the corresponding env var.

## Known Limitations

- **Multi-GPU FSDP / DeepSpeed-zero3 head save**: Single-GPU works correctly; under sharding, rank 0 may only get a shard.
- **`--target_modules all-linear` injects LoRA into the head**: Clef's and OmniJev's heads contain `Linear` layers; `all-linear` may attach LoRA to the head.
- **JEV choice > 16 options**: Only `single` mode (≤16 options in one forward) is supported.
- **OmniJev audio spectrogram**: Not in the inference path. Users must provide their own converter.
- **`tie_word_embeddings=True` adapter merge**: Only affects "merge adapter / convert format"; training and inference are unaffected.

## Verification Report

> **Environment**: transformers 5.16.1 / ms-swift 4.6.0.dev0 / 8× L20X (143 GiB) / ffmpeg + ffprobe installed
>
> **Verification method**: 4 models each executing the full pipeline: pre-training inference → local dataset training → post-training inference, using the CLI examples above
>
> **Generic Decision**: uses Qwen2.5-0.5B as the base model

### Benchmark — Post-Training Inference Comparison

| # | Model | Before acc | After acc | Δ acc | Before top prob | After top prob | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | JEV | 0.6667 | **1.0000** | +0.3333 | 0.9855 | **1.0000** | ✅ +50% |
| 2 | Clef | 0.5000 | **0.8333** | +0.3333 | 0.9712 | 0.9311 | ✅ +67% |
| 3 | OmniJev | 0.5500 | 0.7500 | +0.2884 | 0.8465 | 0.8196 | ✅ +37% |
| 4 | Generic Decision | 0.1667 | **0.6667** | +0.5000 | 0.5682 | 0.9363 | ✅ +300% |

### Temperature Ablation: Same Checkpoint With/Without Temperature Post-Processing

Using the **same checkpoint** (identical weights and bias), a single forward pass is run over the 80 held-out rows actually reserved during training; the metrics are computed on the same raw logits with and without temperature scaling (temperature is a monotonic scaling before softmax, so the argmax is unchanged and acc must be identical — this doubles as the validity check of the experiment).

Setup: JEV-27B-VL + released adapter continued training (`open_jev_jev:all#800`, `split_dataset_ratio 0.1`, 1 epoch, checkpoint-23); script `examples/train/decision/ablation_jev_confidence.py`.

| Variant | T | acc | conf | gap(conf−acc) | ECE | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| **No temperature** | 1.000 | 0.7500 | 0.7023 | −0.048 | 0.1650 | 0.0882 |
| Checkpoint-persisted T | 0.903 | 0.7500 | 0.7204 | −0.030 | 0.1304 | 0.0860 |
| Training-log T (shard-fitted) | 0.528 | 0.7500 | 0.8132 | +0.063 | 0.0830 | 0.0829 |
| Full refit (in-sample) | 0.618 | 0.7500 | 0.7871 | +0.037 | 0.0961 | 0.0823 |
| **2-fold out-of-fold** | ~0.645 | 0.7500 | 0.7873 | +0.037 | 0.1210 | **0.0798** |

Per kind (no temperature → out-of-fold temperature):

| Kind | n | acc | conf change | ECE change | Notes |
| --- | --- | --- | --- | --- | --- |
| noul | 50 | 0.900 | 0.781 → 0.862 | 0.130 → 0.089 | Main driver of underconfidence; T<1 raising confidence is the right direction |
| choice | 22 | 0.636 | 0.613 → 0.705 | 0.185 → 0.228 | Neutral, slightly positive |
| score | 8 | 0.125 | 0.456 → 0.550 | 0.379 → 0.455 | **Counter-example**: few samples and very low acc; a uniform T<1 hurts — per-kind temperatures are needed |

Conclusions:

1. **Direction and magnitude are sound**: the model is underconfident (conf 0.702 < acc 0.750, noul gap −0.119), so the fitted T<1 is correct; out-of-fold (the half of the data that did not participate in fitting) Brier goes 0.0882→0.0798 (−9.5%) and ECE 0.1650→0.1210 — the improvement generalises rather than being an in-sample artifact;
2. **Per-kind temperatures are valuable**: the score kind has acc of only 0.125 with few samples (8 < 10), and a uniform temperature hurts it — the two-stage per-kind fitting design is necessary, and keeping T=1 for kinds without enough samples is the safe fallback;
3. **Fitting sample size determines noise**: 10-row shards fitted two divergent values (0.528/0.903), the full 80 rows fitted 0.618, and the 40-row subsets fitted 0.446/0.844 respectively — the more calibration data, the more stable the temperature.
