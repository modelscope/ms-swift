# Copyright (c) ModelScope Contributors. All rights reserved.
"""Data contract shared by Template / Head / Loss / Trainer:

A batch flattens the questions of all records into `total_q` rows. Because each question
scores a different number of options (noul=2, choice=n, score=6), the per-option logits are
ragged and are stored padded:

    model.forward(...) -> ScoringOutput
        logits:      FloatTensor [total_q, max_opt]   per-option logits, padded
        option_mask: BoolTensor  [total_q, max_opt]   True = a real option slot
        kinds:       LongTensor  [total_q]            0=noul, 1=choice, 2=score (per-kind temp)
        counts:      LongTensor  [num_records]        questions per record (regroup total_q at decode)

    meta (assembled by Template._post_encode, consumed by ScoringHead.forward):
        n_options:      List[int]                 per-question option count (len == total_q)
        question_spans: List[Tuple[int, int]]     per-question token span (Clef/OmniJev pool)
        option_spans:   List[List[Tuple[int,int]]] per-question per-option token span
        verbalizer_ids: List[List[int]]           per-option lm_head column to read (JEV)
        decide_idx:     List[int]                 per-question <decide> token position (Kev)
        opt_idx:        List[List[int]]           per-question </opt> token positions (Kev)
        bias / temperatures: calibration buffers (JEV/OmniJev), kept on the head

    labels (training), consumed by ScoringLoss and DecisionScoringMetric:
        target_probs: FloatTensor [total_q, max_opt]  padded target distribution
                  (one-hot for Clef/OmniJev CE, soft for JEV distillation; pad cols = 0)
"""

import math
import os
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput

QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
KIND_NAMES = ("noul", "choice", "score")


@dataclass
class ScoringOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    option_mask: Optional[torch.BoolTensor] = None
    kinds: Optional[torch.LongTensor] = None
    counts: Optional[torch.LongTensor] = None


def pad_option_logits(
    per_question: List[torch.Tensor], width: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_q = len(per_question)
    if width is None:
        width = max((int(t.shape[0]) for t in per_question), default=0)
    if total_q == 0 or width == 0:
        empty = torch.zeros((total_q, width))
        return empty, torch.zeros((total_q, width), dtype=torch.bool)
    device = per_question[0].device
    dtype = per_question[0].dtype
    logits = torch.full((total_q, width), float("-inf"), device=device, dtype=dtype)
    mask = torch.zeros((total_q, width), device=device, dtype=torch.bool)
    for i, t in enumerate(per_question):
        n = min(int(t.shape[0]), width)
        if n == 0:
            raise ValueError(
                f"question {i} carries zero options; the decision contract requires "
                ">= 1 real option per question (an all-masked row would silently "
                "zero out in masked_softmax and train/eval on nothing)."
            )
        logits[i, :n] = t[:n].to(dtype)
        mask[i, :n] = True
    return logits, mask


def masked_softmax(logits: torch.Tensor, option_mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits = logits.float().masked_fill(~option_mask, float("-inf"))
    probs = torch.softmax(logits, dim=dim)
    return torch.nan_to_num(probs, nan=0.0)


def masked_argmax(logits: torch.Tensor, option_mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Argmax restricted to real option slots (pad slots never win)."""
    return logits.float().masked_fill(~option_mask, float("-inf")).argmax(dim=dim)


def get_scoring_head(model: nn.Module) -> Optional[nn.Module]:
    # twinkle's attach registers one head under the dual names task_head / scoring_head; nn.Module
    # deduplicates named_modules by object identity, so the head shows up under its FIRST name
    # (task_head) only. Match either alias or the calibration / checkpoint callbacks silently
    # find no head and skip.
    for name, module in model.named_modules():
        if name == "scoring_head" or name.endswith(".scoring_head") or name == "task_head" or name.endswith(
            ".task_head"):
            return module
    return None


def maybe_load_trained_scoring_head(model: nn.Module, adapter_dir: Optional[str]) -> bool:
    """After an adapter/checkpoint is applied, overlay a TRAINED head if that dir carries one"""
    if not adapter_dir:
        return False
    head = get_scoring_head(model)
    if head is None:
        return False
    loaded = head.load_pretrained(adapter_dir)
    if loaded:
        from swift.utils import get_logger

        get_logger().info(f"decision: loaded trained scoring_head from {adapter_dir}.")
    return loaded


def clef_option_ids(kind: str, criteria: Any) -> List[str]:
    if kind == "noul":
        return ["true", "false"]
    if kind == "choice":
        return [str(key) for key, _ in sorted((str(k), v) for k, v in (criteria or {}).items())]
    return [str(index) for index in range(len(criteria or []))]


def omnijev_column_keys(kind: str, keys: Sequence[Any]) -> List[str]:
    if kind == "noul":
        return ["yes", "no"]
    if kind == "choice":
        return [str(k) for k in keys] + ["abstain"]
    return [str(k) for k in keys]


def thx02_option_ids(kind: str, criteria: Any) -> List[str]:
    """THX-02 answer keys in prompt order: `choice` keeps the criteria order, `noul` is false/true,
    `score` is the level index."""
    if kind == "noul":
        return ["false", "true"]
    if kind == "choice":
        return [str(key) for key in (criteria or {})]
    return [str(index) for index in range(len(criteria or []))]


class ScoringHead(nn.Module):
    """Abstract base"""

    # Set by subclasses that need the base model's output embeddings
    needs_output_embeddings: bool = False
    # Set by subclasses that read lm_head logits (JEV verbalizer columns). When False the loader
    # skips passing lm_logits
    needs_lm_logits: bool = False
    keep_fp32: bool = False
    # Set by subclasses over an encoder-style backbone (AutoModel on a BERT-family config, e.g.
    # LayaHead's ModernBERT): the backbone forward already returns `last_hidden_state` and computes
    # no lm logits, so the loader's forward wrapper reads `last_hidden_state` directly instead of
    # forcing `output_hidden_states=True` and slicing `hidden_states[-1]`.
    encoder_backbone: bool = False

    def set_output_embeddings(self, output_embeddings: nn.Module) -> None:
        self._output_embeddings = output_embeddings

    def has_kind_temperatures(self) -> bool:
        return hasattr(self, "temperatures") or hasattr(self, "temps")

    def get_temperature(self, kind: int) -> float:
        buf = getattr(self, "temperatures", None)
        if buf is None:
            buf = getattr(self, "temps", None)
        return float(buf[int(kind)])

    def set_temperature(self, kind: int, value: float) -> None:
        buf = getattr(self, "temperatures", None)
        if buf is None:
            buf = getattr(self, "temps", None)
        buf[int(kind)] = float(value)

    @staticmethod
    def _locate(model_dir: str, filename: str, env_var: str, subdir: Optional[str] = None) -> Optional[str]:
        """Locate a head artifact file"""
        env = os.environ.get(env_var)
        if env and os.path.exists(env):
            return env
        candidates = [os.path.join(model_dir, filename)]
        if subdir:
            candidates.append(os.path.join(model_dir, subdir, filename))
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    def forward(
        self, hidden_states: torch.Tensor, lm_logits: Optional[torch.Tensor], meta: Dict[str, Any]
    ) -> ScoringOutput:
        """Compute per-option logits.

        Args:
            hidden_states: `[batch, seq_len, hidden]` last-hidden-state of the base model.
            lm_logits: `[batch, seq_len, vocab]` lm_head logits (JEV verbalizer columns);
                may be None for heads that only use hidden_states.
            meta: the per-question locating/calibration dict described in the module contract
                (n_options / spans / verbalizer_ids / kinds ...), already moved to device.

        Returns:
            ScoringOutput with padded `logits` `[total_q, max_opt]` + `option_mask`.
        """
        raise NotImplementedError

    def extra_state_for_save(self) -> Dict[str, Any]:
        """Non-parameter meta (verbalizer_ids / bias / temperatures) to persist alongside the
        head weights, so a reloaded checkpoint can rebuild the head."""
        return {}

    @classmethod
    def from_pretrained(cls, model_dir: str, model=None, config=None) -> "ScoringHead":
        """Construct the head and STRICT-load its factory weights from `model_dir`"""
        raise NotImplementedError

    HEAD_WEIGHTS_NAME = "scoring_head.safetensors"
    HEAD_META_NAME = "scoring_head.json"

    def head_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if not k.startswith("_output_embeddings")}

    def save_pretrained(self, save_directory: str, save_weights: bool = True) -> None:
        """Persist the head beside the adapter"""
        from safetensors.torch import save_file

        os.makedirs(save_directory, exist_ok=True)
        if save_weights:
            save_file(self.head_state_dict(), os.path.join(save_directory, self.HEAD_WEIGHTS_NAME))
        with open(os.path.join(save_directory, self.HEAD_META_NAME), "w", encoding="utf-8") as f:
            json.dump(self.extra_state_for_save(), f, indent=2, ensure_ascii=False)

    def load_pretrained(self, load_directory: str) -> bool:
        """In adaper checkpoint or full-finetune cases, if ``HEAD_META_NAME`` carries
        non-parameter calibration state (temperatures, biases, etc. from
        ``extra_state_for_save``), it is applied to the head so a reloaded TRAINED checkpoint
        restores its fitted calibration, not the factory defaults.
        """
        from safetensors.torch import load_file

        weights_file = os.path.join(load_directory, self.HEAD_WEIGHTS_NAME)
        meta_file = os.path.join(load_directory, self.HEAD_META_NAME)
        has_weights = os.path.exists(weights_file)
        has_meta = os.path.exists(meta_file)
        if not has_weights and not has_meta:
            return False
        if has_weights:
            state = load_file(weights_file)
            self._validate_and_load(state, weights_file)
        elif has_meta:
            state = self._extract_head_state_from_model_dir(load_directory)
            if state is not None:
                self._validate_and_load(state, "model safetensors")
        if has_meta:
            self._load_calibration_meta(meta_file)
        return True

    def _validate_and_load(self, state: Dict[str, torch.Tensor], source: str) -> None:
        own_keys = {k for k in self.state_dict() if not k.startswith("_output_embeddings")}
        own_params = {n for n, _ in self.named_parameters() if not n.startswith("_output_embeddings")}
        unexpected = set(state) - own_keys
        missing_params = own_params - set(state)
        if unexpected or missing_params:
            raise ValueError(
                f"{source} does not match head {type(self).__name__}: "
                f"unexpected={sorted(unexpected)} missing_params={sorted(missing_params)}"
            )
        self.load_state_dict(state, strict=False)

    def _extract_head_state_from_model_dir(self, model_dir: str) -> Optional[Dict[str, torch.Tensor]]:
        """Used when loading a full-finetune checkpoint: ``from_pretrained`` ignored scoring_head keys
        (the submodule was not yet attached); this reads them back from the safetensors files.
        Returns ``None`` when the model dir carries no scoring_head weights (e.g. a factory model).
        """
        from safetensors.torch import load_file

        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            head_keys = {k for k in weight_map if "scoring_head" in k}
            if not head_keys:
                return None
            shards = {weight_map[k] for k in head_keys}
            result: Dict[str, torch.Tensor] = {}
            for shard in shards:
                data = load_file(os.path.join(model_dir, shard))
                for k, v in data.items():
                    if "scoring_head" in k:
                        result[k.split("scoring_head.", 1)[1] if "scoring_head." in k else k] = v
            return result if result else None
        single = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(single):
            data = load_file(single)
            result = {}
            for k, v in data.items():
                if "scoring_head" in k:
                    result[k.split("scoring_head.", 1)[1] if "scoring_head." in k else k] = v
            return result if result else None
        return None

    def _load_calibration_meta(self, meta_file: str) -> None:
        """Subclasses override to map the JSON dict onto their own buffers"""
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        self._apply_calibration_meta(meta)

    def _apply_calibration_meta(self, meta: Dict[str, Any]) -> None:
        """Apply non-parameter calibration meta (temperatures, biases, ...) to this head"""
        if "head_temperature" in meta and hasattr(self, "head_temperature"):
            self.head_temperature = torch.as_tensor(float(meta["head_temperature"]), dtype=torch.float32)
        temps = meta.get("temperatures")
        if temps and self.has_kind_temperatures():
            for k, name in enumerate(KIND_NAMES):
                if name in temps:
                    self.set_temperature(k, float(temps[name]))


class JevVerbalizerHead(ScoringHead):
    needs_lm_logits = True
    keep_fp32 = True

    DEFAULT_RANGES = {"noul": (0, 2), "choice": (8, 24), "score": (2, 8)}
    DEFAULT_TEMPERATURES = {"noul": 1.0, "choice": 1.0, "score": 1.0}

    def __init__(
        self,
        bias: Sequence[float],
        verbalizer_ids: Sequence[int],
        ranges: Dict[str, Sequence[int]],
        temperatures: Dict[str, float],
        head_temperature: float = 1.0,
    ):
        super().__init__()
        self.bias = nn.Parameter(torch.as_tensor(list(bias), dtype=torch.float32))
        self.register_buffer(
            "verbalizer_ids", torch.as_tensor(list(verbalizer_ids), dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "ranges",
            torch.as_tensor([list(ranges[KIND_NAMES[k]]) for k in range(len(KIND_NAMES))], dtype=torch.long),
            persistent=False,
        )
        # Two-stage temperature: head_temperature (global) × temperatures[kind] (per-kind residual).
        self.register_buffer(
            "head_temperature", torch.as_tensor(float(head_temperature), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "temperatures",
            torch.as_tensor([float(temperatures[KIND_NAMES[k]]) for k in range(len(KIND_NAMES))], dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self, hidden_states: torch.Tensor, lm_logits: Optional[torch.Tensor], meta: Dict[str, Any]
    ) -> ScoringOutput:
        if lm_logits is None:
            raise ValueError("JevVerbalizerHead requires lm_logits (needs_lm_logits must be True).")
        record_index = [int(r) for r in meta["record_index"]]
        rows, cols = self._readout_positions(record_index, meta, lm_logits.shape[1], lm_logits.device)
        read_logits = lm_logits[rows, cols, :].float()  # [total_q, vocab]
        return self.forward_readout(read_logits, meta)

    def forward_readout(self, readout_logits: torch.Tensor, meta: Dict[str, Any]) -> ScoringOutput:
        """Score pre-read full-vocab logits: ``[total_q, vocab]`` rows, one per question, each
        already sliced at its readout position by the caller.

        Two readout granularities share this scoring core: the transformers backend goes
        through ``forward`` (which reads the rows out of one normal forward's lm_logits),
        while the megatron backend hands pre-read rows directly -- its forward_step owns the
        position machinery (packed / CP / mask) and gathers the vocab-parallel shards first.
        A third granularity would be another entry method, not a change to this one.
        """
        kinds = [int(k) for k in meta["kinds"]]
        n_options = [int(n) for n in meta["n_options"]]
        total_q = len(n_options)
        device = readout_logits.device
        counts = meta.get("counts")
        counts_t = torch.as_tensor([int(c) for c in counts], dtype=torch.long) if counts else None
        if total_q == 0:
            max_opt = int(meta.get("max_opt") or 0)
            empty = torch.zeros((0, max_opt), device=device, dtype=torch.float32)
            return ScoringOutput(
                logits=empty,
                option_mask=torch.zeros((0, max_opt), dtype=torch.bool, device=device),
                kinds=torch.zeros((0,), dtype=torch.long, device=device),
                counts=counts_t,
            )

        log_probs = torch.log_softmax(readout_logits.float(), dim=-1)  # log P(token), matching vLLM's logprob

        verbalizer_ids = self.verbalizer_ids.to(device)
        bias = self.bias.to(device=device, dtype=torch.float32)
        ranges = self.ranges.to(device)
        temperatures = self.temperatures.to(device=device, dtype=torch.float32)
        head_t = self.head_temperature.to(device=device, dtype=torch.float32)
        per_question: List[torch.Tensor] = []
        for i in range(total_q):
            k = kinds[i]
            n = n_options[i]
            lo = int(ranges[k][0])
            hi = int(ranges[k][1])
            if lo + n > verbalizer_ids.shape[0] or n > hi - lo:
                raise ValueError(
                    f"JEV kind {KIND_NAMES[k]} supports at most {hi - lo} options (verbalizer slots), got {n}."
                )
            ids = verbalizer_ids[lo : lo + n]  # [n]
            lp = log_probs[i].gather(0, ids)  # [n]
            z = (lp + bias[lo : lo + n]) / (head_t * temperatures[k])
            per_question.append(z)

        max_opt = int(meta.get("max_opt") or max(n_options))
        logits, option_mask = pad_option_logits(per_question, width=max_opt)
        return ScoringOutput(
            logits=logits,
            option_mask=option_mask,
            kinds=torch.as_tensor(kinds, dtype=torch.long, device=device),
            counts=counts_t,
        )

    @staticmethod
    def _readout_positions(
        record_index: List[int], meta: Dict[str, Any], padded_len: int, device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        left_padding = bool(meta.get("left_padding", False))
        seq_lens = meta.get("seq_lens")
        cols = []
        for ri in record_index:
            if left_padding:
                cols.append(padded_len - 1)
            else:
                sl = int(seq_lens[ri]) if seq_lens is not None else padded_len
                cols.append(sl - 1)
        rows = torch.as_tensor(record_index, dtype=torch.long, device=device)
        cols_t = torch.as_tensor(cols, dtype=torch.long, device=device)
        return rows, cols_t

    def extra_state_for_save(self) -> Dict[str, Any]:
        return {
            "bias": self.bias.detach().float().tolist(),
            "verbalizer_ids": self.verbalizer_ids.tolist(),
            "ranges": {KIND_NAMES[k]: self.ranges[k].tolist() for k in range(len(KIND_NAMES))},
            "head_temperature": float(self.head_temperature),
            "temperatures": {KIND_NAMES[k]: float(self.temperatures[k]) for k in range(len(KIND_NAMES))},
        }

    @classmethod
    def from_pretrained(cls, model_dir: str, model=None, config=None) -> "JevVerbalizerHead":
        head_file = cls._locate(model_dir, "decision_head.json", "JEV_DECISION_HEAD", subdir="adapter_vllm")
        if head_file is None:
            raise FileNotFoundError(
                f"JEV decision_head.json not found under {model_dir!r} (or {model_dir}/adapter_vllm). "
                "Set JEV_DECISION_HEAD to its path."
            )
        with open(head_file, "r") as f:
            head = json.load(f)
        for key in ("bias", "verbalizer_ids"):
            if key not in head:
                raise KeyError(f"decision_head.json missing required key {key!r}: {head_file}")
        ranges = {k: tuple(v) for k, v in head.get("slots", {}).get("ranges", {}).items()}
        for kind, default in cls.DEFAULT_RANGES.items():
            ranges.setdefault(kind, default)

        temperatures = dict(cls.DEFAULT_TEMPERATURES)
        calib_file = os.environ.get("JEV_DECIDE_CALIBRATION")
        if not calib_file or not os.path.exists(calib_file):
            calib_file = cls._locate(
                os.path.dirname(os.path.dirname(os.path.abspath(head_file))),
                "calibration.json",
                "JEV_DECIDE_CALIBRATION",
            )
        if calib_file and os.path.exists(calib_file):
            with open(calib_file, "r") as f:
                calib = json.load(f)
            per_kind = calib.get("per_kind", {})
            temperatures.update({k: float(v) for k, v in per_kind.items() if k in temperatures})
        head_temperature = (
            float(calib.get("head_temperature", 1.0)) if calib_file and os.path.exists(calib_file) else 1.0
        )
        return cls(head["bias"], head["verbalizer_ids"], ranges, temperatures, head_temperature=head_temperature)

    @staticmethod
    def _resolve_verbalizer_token_id(tokenizer, text: str) -> int:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
        unk = getattr(tokenizer, "unk_token_id", None)
        if unk is None:
            raise ValueError(
                f"Tokenizer cannot resolve verbalizer token for {text!r} "
                f"(encodes to {len(ids)} tokens: {ids}) and has no unk_token_id."
            )
        from swift.utils import get_logger

        get_logger().warning(
            f"decision: verbalizer label {text!r} encodes to {len(ids)} tokens ({ids}); "
            f"falling back to unk_token_id={unk}. The head slot will read this unk column "
            f"instead of the intended token; a LoRA on lm_head can still learn the correct "
            f"distribution on that column, but multiple labels collapsing to the same unk "
            f"id makes them indistinguishable (degenerate). Consider using a tokenizer that "
            f"encodes the verbalizer labels as single tokens."
        )
        return unk

    @classmethod
    def from_tokenizer(cls, tokenizer, config=None) -> "JevVerbalizerHead":
        """Random-init path for training a decision model from scratch on any CausalLM.

        Args:
            tokenizer: the base model's tokenizer (any `PreTrainedTokenizerBase`).
            config: unused (accepted for signature parity with `from_pretrained`).

        Returns:
            A `JevVerbalizerHead` with zero-initialized bias and unit temperatures, ready for
            joint training with a LoRA adapter on the base CausalLM.
        """
        noul_words = ["false", "true"]
        score_words = ["0", "1", "2", "3", "4", "5"]
        choice_words = list(string.ascii_uppercase[:16])  # A..P

        verbalizer_ids = [
            cls._resolve_verbalizer_token_id(tokenizer, w) for w in noul_words + score_words + choice_words
        ]
        bias = [0.0] * len(verbalizer_ids)
        ranges = dict(cls.DEFAULT_RANGES)
        temperatures = dict(cls.DEFAULT_TEMPERATURES)
        return cls(bias, verbalizer_ids, ranges, temperatures, head_temperature=1.0)


class StartLuxLetterHead(JevVerbalizerHead):
    DEFAULT_RANGES = {"noul": (0, 2), "choice": (0, 26), "score": (0, 10)}

    @classmethod
    def from_pretrained(  # noqa: C901
        cls,
        model_dir: str,
        model=None,
        config=None,
        tokenizer=None,
    ) -> "StartLuxLetterHead":
        """model/config are accepted for signature parity with the other heads' loaders; `tokenizer`
        is used only to resolve `letter_token_ids` when decision_config.json omits them."""
        head_file = cls._locate(model_dir, "decision_config.json", "STARTLUX_DECISION_CONFIG")
        if head_file is None:
            raise FileNotFoundError(
                f"StartLux decision_config.json not found under {model_dir!r}. The letter token ids and the "
                "per-kind read-out temperatures ship in the model folder beside the weights; point --model at a "
                "StartLuxAI/StartLux-Decision-v1.1-*-AutoThink checkpoint (or set STARTLUX_DECISION_CONFIG)."
            )
        with open(head_file, "r") as f:
            head = json.load(f)
        config_file = os.path.join(os.path.dirname(head_file), "config.json")
        prompt_attention = head.get("prompt_attention")
        if prompt_attention is None and os.path.isfile(config_file):
            with open(config_file, "r") as f:
                prompt_attention = json.load(f).get("jev_prompt_attention")
        if prompt_attention == "bidirectional":
            raise ValueError(
                f"{model_dir!r} is an Instant (prompt-bidirectional) StartLux-Decision folder: its "
                "prompt is read with bidirectional attention in the full-attention layers, which this "
                "loader does not implement. Use an AutoThink checkpoint (causal) or the official "
                "startlux-decide package for Instant models."
            )
        temperatures = head.get("temperature_by_type") or {}
        if any(k not in KIND_NAMES or not isinstance(v, (int, float)) or v <= 0 for k, v in temperatures.items()):
            raise ValueError(f"{head_file}: temperature_by_type must map noul/choice/score to positive numbers.")
        temperatures = {k: float(temperatures[k]) for k in KIND_NAMES if k in temperatures}
        for kind, default in cls.DEFAULT_TEMPERATURES.items():
            temperatures.setdefault(kind, default)
        letters = head.get("letter_token_ids")
        if letters is None:
            if tokenizer is None:
                raise ValueError(f"{head_file} has no letter_token_ids and no tokenizer was passed to resolve them.")
            from swift.utils import get_logger

            get_logger().warning(f"{head_file}: letter_token_ids missing; resolving A..Z from the tokenizer.")
            letters = []
            for letter in string.ascii_uppercase:
                ids = tokenizer.encode(letter, add_special_tokens=False)
                if len(ids) != 1:
                    raise ValueError(
                        f"The letter {letter!r} is not a single token ({ids}); the letter read-out "
                        "needs the 26 ids from decision_config.json."
                    )
                letters.append(ids[0])
        if (
            not isinstance(letters, (list, tuple))
            or len(letters) != len(string.ascii_uppercase)
            or not all(isinstance(t, int) for t in letters)
            or len(set(letters)) != len(letters)
        ):
            raise ValueError(f"{head_file}: letter_token_ids must be 26 distinct token ids (A to Z).")
        bias = [0.0] * len(letters)
        return cls(bias, list(letters), dict(cls.DEFAULT_RANGES), temperatures, head_temperature=1.0)


class EvidenceRoutingLayer(nn.Module):
    """One cross-attention block of Clef's `JointSchemaHead`."""

    def __init__(self, width: int, heads: int, feedforward: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, width),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_queries = self.query_norm(queries)
        routed, _ = self.attention(
            normalized_queries, self.memory_norm(memory), self.memory_norm(memory), need_weights=False
        )
        queries = queries + self.attention_dropout(routed)
        return queries + self.feedforward(self.feedforward_norm(queries))


class ClefJointSchemaHead(ScoringHead):
    """Clef's joint schema head: a real parameterised decoder over span-pooled hidden states."""

    needs_output_embeddings = True
    needs_lm_logits = False
    keep_fp32 = False

    def __init__(
        self,
        hidden_size: int,
        width: int,
        routing_layers: int,
        layers: int,
        heads: int,
        feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(3, width)
        self.evidence_layers = nn.ModuleList(
            [
                EvidenceRoutingLayer(width=width, heads=heads, feedforward=feedforward, dropout=dropout)
                for _ in range(routing_layers)
            ]
        )
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=width,
                    nhead=heads,
                    dim_feedforward=feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.residual_scorer = nn.Sequential(
            nn.Linear(width * 4, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )
        self.prior_logit_scale = nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = nn.Parameter(torch.zeros(()))
        self.residual_gate = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _mean_span(values: torch.Tensor, span: Tuple[int, int]) -> torch.Tensor:
        start, end = span
        return values[start:end].mean(dim=0)

    def forward(
        self, hidden_states: torch.Tensor, lm_logits: Optional[torch.Tensor], meta: Dict[str, Any]
    ) -> ScoringOutput:
        input_ids = meta.get("input_ids")
        if input_ids is None:
            raise ValueError(
                'ClefJointSchemaHead requires meta["input_ids"] '
                "(stashed by ClefTemplate._post_encode before the base drops it)."
            )
        if getattr(self, "_output_embeddings", None) is None:
            raise RuntimeError(
                "ClefJointSchemaHead needs output embeddings; the loader must call "
                "set_output_embeddings (needs_output_embeddings=True)."
            )
        device = hidden_states.device
        n_options = [int(n) for n in meta["n_options"]]
        kinds = [int(k) for k in meta["kinds"]]
        total_q = len(n_options)
        counts = meta.get("counts")
        counts_t = torch.as_tensor([int(c) for c in counts], dtype=torch.long, device=device) if counts else None
        if total_q == 0:
            max_opt = int(meta.get("max_opt") or 0)
            empty = torch.zeros((0, max_opt), device=device, dtype=hidden_states.dtype)
            return ScoringOutput(
                logits=empty,
                option_mask=torch.zeros((0, max_opt), dtype=torch.bool, device=device),
                kinds=torch.zeros((0,), dtype=torch.long, device=device),
                counts=counts_t,
            )

        question_spans = meta["question_spans"]
        option_spans = meta["option_spans"]
        record_index = [int(r) for r in meta["record_index"]]
        num_records = int(hidden_states.shape[0])
        padded_len = int(hidden_states.shape[1])
        left_padding = bool(meta.get("left_padding", False))
        seq_lens = meta.get("seq_lens") or [padded_len] * num_records
        output_embedding_weight = self._output_embeddings.weight

        by_record: List[List[int]] = [[] for _ in range(num_records)]
        for qi in range(total_q):
            by_record[record_index[qi]].append(qi)

        normalized_hidden = self.hidden_norm(hidden_states)
        per_question: List[torch.Tensor] = []
        per_kind: List[int] = []
        for rec in range(num_records):
            qidxs = by_record[rec]
            if not qidxs:
                continue
            seq_len = int(seq_lens[rec])
            pad_offset = (padded_len - seq_len) if left_padding else 0
            sequence_hidden = normalized_hidden[rec, pad_offset : pad_offset + seq_len]
            record_logits = self._score_record(
                sequence_hidden,
                input_ids[rec],
                pad_offset,
                qidxs,
                question_spans,
                option_spans,
                kinds,
                output_embedding_weight,
            )
            per_question.extend(record_logits)
            per_kind.extend(kinds[qi] for qi in qidxs)

        max_opt = int(meta.get("max_opt") or max(n_options))
        logits, option_mask = pad_option_logits(per_question, width=max_opt)
        return ScoringOutput(
            logits=logits,
            option_mask=option_mask,
            kinds=torch.as_tensor(per_kind, dtype=torch.long, device=device),
            counts=counts_t,
        )

    def _score_record(
        self,
        sequence_hidden: torch.Tensor,
        input_ids_row: torch.Tensor,
        pad_offset: int,
        qidxs: List[int],
        question_spans: List[Any],
        option_spans: List[Any],
        kinds: List[int],
        output_embedding_weight: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Per-record scoring body for the joint schema head."""
        memory = self.memory_projection(sequence_hidden).unsqueeze(0)  # [1, seq_len, width]
        global_vector = sequence_hidden[-1]  # [H]
        question_vectors = torch.stack([self._mean_span(sequence_hidden, question_spans[qi]) for qi in qidxs])
        type_ids = torch.as_tensor([kinds[qi] for qi in qidxs], device=sequence_hidden.device, dtype=torch.long)

        option_contexts: List[torch.Tensor] = []
        lexical_options: List[torch.Tensor] = []
        option_counts: List[int] = []
        for qi in qidxs:
            spans = option_spans[qi]
            context_vectors = torch.stack([self._mean_span(sequence_hidden, span) for span in spans])
            lexical_vectors = []
            for start, end in spans:
                token_ids = input_ids_row[pad_offset + int(start) : pad_offset + int(end)]
                lexical_vectors.append(output_embedding_weight[token_ids].mean(dim=0))
            lexical_options.append(torch.stack(lexical_vectors))
            option_contexts.append(context_vectors)
            option_counts.append(len(spans))

        option_queries = []
        for question_index, (context_vectors, lexical) in enumerate(zip(option_contexts, lexical_options)):
            option_queries.append(
                self.option_context_projection(context_vectors)
                + self.option_lexical_projection(lexical)
                + self.option_question_projection(question_vectors[question_index]).unsqueeze(0)
            )
        routed_options = torch.cat(option_queries, dim=0).unsqueeze(0)
        for layer in self.evidence_layers:
            routed_options = layer(routed_options, memory)
        routed_options = routed_options[0]
        split_options = list(torch.split(routed_options, option_counts, dim=0))

        base_fields = self.question_projection(question_vectors)
        option_summaries = []
        for field, options in zip(base_fields, split_options):
            routing_weights = torch.softmax(torch.matmul(options, field) / math.sqrt(options.shape[-1]), dim=0)
            option_summaries.append(torch.sum(routing_weights.unsqueeze(-1) * options, dim=0))
        fields = (
            base_fields
            + self.option_summary_norm(torch.stack(option_summaries))
            + self.global_projection(global_vector).unsqueeze(0)
            + self.type_embedding(type_ids)
        )
        fields = fields.unsqueeze(0)
        for layer in self.layers:
            fields = layer(fields, memory)
        fields = self.field_norm(fields[0])

        record_logits: List[torch.Tensor] = []
        for field, lexical, routed in zip(fields, lexical_options, split_options):
            anchor = F.normalize(question_vectors[len(record_logits)] + global_vector, dim=-1)
            lexical_anchor = F.normalize(lexical, dim=-1)
            prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
            prior = prior_scale * torch.matmul(lexical_anchor, anchor)
            options = self.option_norm(routed)
            repeated_field = field.unsqueeze(0).expand_as(options)
            cosine = F.cosine_similarity(repeated_field, options, dim=-1)
            features = torch.cat(
                [repeated_field, options, repeated_field * options, torch.abs(repeated_field - options)], dim=-1
            )
            residual = self.residual_scorer(features).squeeze(-1)
            joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
            joint = joint_scale * cosine + residual
            record_logits.append(prior + torch.sigmoid(self.residual_gate) * joint)
        return record_logits

    @classmethod
    def from_pretrained(cls, model_dir: str, model=None, config=None) -> "ClefJointSchemaHead":
        from safetensors.torch import load_file

        config_file = cls._locate(model_dir, "joint_head_config.json", "CLEF_JOINT_HEAD_CONFIG")
        weights_file = cls._locate(model_dir, "joint_head.safetensors", "CLEF_JOINT_HEAD_WEIGHTS")
        if config_file is None:
            raise FileNotFoundError(
                f"Clef joint_head_config.json not found under {model_dir!r}. Set CLEF_JOINT_HEAD_CONFIG to its path."
            )
        if weights_file is None:
            raise FileNotFoundError(
                f"Clef joint_head.safetensors not found under {model_dir!r}. Set CLEF_JOINT_HEAD_WEIGHTS to its path."
            )
        with open(config_file, "r") as f:
            head_config = json.load(f)
        head = cls(**head_config)
        state = load_file(weights_file)
        missing, unexpected = head.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Clef joint head strict load failed: missing={missing}, unexpected={unexpected}")
        return head


THX02_DECISION_CONFIG_KEYS = ("attention_mode", "codes", "temperature")


def load_thx02_decision_config(model_dir: str) -> Dict[str, Any]:
    """Read and validate THX-02's `decision_config.json` (answer codes, temperature, attention mode, image
    pixel bound). `THX02_DECISION_CONFIG` overrides the path."""
    config_file = ScoringHead._locate(model_dir, "decision_config.json", "THX02_DECISION_CONFIG")
    if config_file is None:
        raise FileNotFoundError(f"THX-02 decision_config.json not found under {model_dir!r}.")
    with open(config_file, "r") as f:
        decision_config = json.load(f)
    missing = [key for key in THX02_DECISION_CONFIG_KEYS if key not in decision_config]
    if missing:
        raise ValueError(
            f"THX-02 {config_file!r} is missing required keys {missing}; expected {list(THX02_DECISION_CONFIG_KEYS)}."
        )
    if decision_config["attention_mode"] != "noncausal_full_attention":
        raise ValueError(
            f"THX-02 expects attention_mode=noncausal_full_attention in {config_file!r}, got "
            f"{decision_config['attention_mode']!r}."
        )
    if not decision_config["codes"]:
        raise ValueError(f"THX-02 {config_file!r} lists no answer codes.")
    return decision_config


class Thx02ReadoutHead(ScoringHead):
    """THX-02 inline readout"""

    needs_output_embeddings = False
    needs_lm_logits = False
    keep_fp32 = False

    def __init__(self, hidden_size: int, num_codes: int = 255, temperature: float = 1.0) -> None:
        super().__init__()
        self.readout = nn.Linear(hidden_size, num_codes, bias=False)
        self.temperature = float(temperature)
        self.register_buffer("head_temperature", torch.as_tensor(1.0, dtype=torch.float32), persistent=False)
        self.register_buffer(
            "temperatures", torch.as_tensor([1.0 for _ in KIND_NAMES], dtype=torch.float32), persistent=False
        )

    def forward(
        self, hidden_states: torch.Tensor, lm_logits: Optional[torch.Tensor], meta: Dict[str, Any]
    ) -> ScoringOutput:
        device = hidden_states.device
        n_options = [int(n) for n in meta["n_options"]]
        counts = meta.get("counts")
        counts_t = torch.as_tensor([int(c) for c in counts], dtype=torch.long, device=device) if counts else None
        max_opt = int(meta.get("max_opt") or max(n_options, default=0))
        if not n_options:
            return ScoringOutput(
                logits=torch.zeros((0, max_opt), device=device, dtype=hidden_states.dtype),
                option_mask=torch.zeros((0, max_opt), dtype=torch.bool, device=device),
                kinds=torch.zeros((0,), dtype=torch.long, device=device),
                counts=counts_t,
            )
        padded_len = int(hidden_states.shape[1])
        seq_lens = meta.get("seq_lens") or [padded_len] * int(hidden_states.shape[0])
        left_padding = bool(meta.get("left_padding", False))
        rows, cols = [], []
        for record, position in zip(meta["record_index"], meta["readout_positions"]):
            offset = padded_len - int(seq_lens[int(record)]) if left_padding else 0
            rows.append(int(record))
            cols.append(offset + int(position))
        rows_t = torch.as_tensor(rows, dtype=torch.long, device=device)
        cols_t = torch.as_tensor(cols, dtype=torch.long, device=device)
        kinds_t = torch.as_tensor([int(k) for k in meta["kinds"]], dtype=torch.long, device=device)
        calibration = (
            self.head_temperature.to(device=device, dtype=torch.float32)
            * self.temperatures.to(device=device, dtype=torch.float32)[kinds_t]
        )
        code_logits = self.readout(hidden_states[rows_t, cols_t]).float() / self.temperature / calibration[:, None]
        per_question = [code_logits[i, :n] for i, n in enumerate(n_options)]
        logits, option_mask = pad_option_logits(per_question, width=max_opt)
        return ScoringOutput(logits=logits, option_mask=option_mask, kinds=kinds_t, counts=counts_t)

    def extra_state_for_save(self) -> Dict[str, Any]:
        return {
            "hidden_size": self.readout.in_features,
            "num_codes": self.readout.out_features,
            "temperature": self.temperature,
            "head_temperature": float(self.head_temperature),
            "temperatures": {KIND_NAMES[k]: float(self.temperatures[k]) for k in range(len(KIND_NAMES))},
        }

    @classmethod
    def from_pretrained(cls, model_dir: str, model=None, config=None) -> "Thx02ReadoutHead":
        from safetensors.torch import load_file

        weights_file = cls._locate(model_dir, "readout.safetensors", "THX02_READOUT")
        if weights_file is None:
            raise FileNotFoundError(f"THX-02 readout.safetensors not found under {model_dir!r}.")
        decision_config = load_thx02_decision_config(model_dir)
        state = load_file(weights_file)
        num_codes, hidden_size = state["weight"].shape
        if num_codes != len(decision_config["codes"]):
            raise ValueError(
                f"THX-02 readout has {num_codes} codes, decision_config.json lists {len(decision_config['codes'])}."
            )
        head = cls(hidden_size=hidden_size, num_codes=num_codes, temperature=float(decision_config["temperature"]))
        missing, unexpected = head.readout.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"THX-02 readout strict load failed: missing={missing}, unexpected={unexpected}")
        return head


class OmniOptionScorer(nn.Module):
    """OmniJev's decision head"""

    def __init__(self, d_model: int, d_hidden: int = 1024, n_types: int = 3, norm: str = "sigmoid") -> None:
        super().__init__()
        self.opt = nn.Sequential(nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden))
        self.que = nn.Sequential(nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden))
        self.score = nn.Linear(d_hidden, 1)
        self.log_tau = nn.Parameter(torch.zeros(n_types))
        self.norm = norm
        self.abstain = nn.Linear(d_hidden, 1)
        self.feat = nn.Linear(6, d_hidden)
        self.feat_lin = nn.Linear(6, 1)

    def logits(self, u_opts: torch.Tensor, z_q: torch.Tensor, type_id: int = 0, feats=None) -> torch.Tensor:
        """The vector the probabilities come from, abstain last"""
        g = torch.tanh(self.que(z_q))
        h = self.opt(u_opts) * g[None, :]  # FiLM-style gating
        if feats is not None:
            h = h + self.feat(feats)
        tau = torch.exp(self.log_tau[type_id])
        logit = self.score(h).squeeze(-1)
        if feats is not None:
            logit = logit + self.feat_lin(feats).squeeze(-1)
        logit = logit / tau
        if self.norm == "softmax" and type_id != 0:
            a = self.abstain(g).squeeze(-1) / tau
            return torch.cat([logit, a[None]])
        return torch.cat([logit, torch.zeros_like(logit[:1])])


class OmniOrdinalHead(nn.Module):
    """OmniJev's ordinal (score) head. Strict-loads"""

    def __init__(self, d_model: int, d_hidden: int = 512) -> None:
        super().__init__()
        self.z = nn.Sequential(nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))
        self.cut = nn.Sequential(nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))

    def forward(self, z_q: torch.Tensor, u_levels: torch.Tensor) -> torch.Tensor:
        """z_q [D], u_levels [L, D] (levels in the caller's order) -> probs [L] (sum ~ 1)."""
        z = self.z(z_q).squeeze(-1)
        raw = self.cut(u_levels).squeeze(-1)
        theta = raw[0] + torch.cat([torch.zeros(1, device=raw.device), torch.cumsum(F.softplus(raw[1:]), 0)])
        cdf = torch.sigmoid(theta - z)
        cdf = torch.cat([cdf[:-1], torch.ones(1, device=cdf.device)])
        return torch.cat([cdf[:1], cdf[1:] - cdf[:-1]]).clamp(min=1e-8)


class OmniJevHead(ScoringHead):
    """OmniJev System-1 head (branch-only: the loader's branch forward supplies the per-question
    (u, zq, feats) features; scoring happens in ``forward_features`` / ``score_features``)."""

    needs_lm_logits = False
    keep_fp32 = True

    def __init__(
        self,
        hidden_size: int,
        head_hidden: int = 1024,
        ord_hidden: int = 512,
        n_types: int = 3,
        norm: str = "softmax",
        ordinal: bool = True,
        temperatures: Optional[Dict[str, float]] = None,
        noul_bias: float = 0.0,
        head_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.head = OmniOptionScorer(hidden_size, head_hidden, n_types, norm)
        self.ord = OmniOrdinalHead(hidden_size, ord_hidden)
        self.ordinal = bool(ordinal)
        temps = temperatures or {}
        self.register_buffer(
            "head_temperature", torch.as_tensor(float(head_temperature), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "temps",
            torch.as_tensor([float(temps.get(KIND_NAMES[k], 1.0)) for k in range(n_types)], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("noul_bias", torch.as_tensor(float(noul_bias), dtype=torch.float32), persistent=False)

    def forward_features(
        self,
        per_question_features: Sequence[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        meta: Dict[str, Any],
    ) -> ScoringOutput:
        """Branch entry point (decision G1). The loader runs the two-stage cached branch forward"""
        kinds = [int(k) for k in meta["kinds"]]
        n_options = [int(n) for n in meta["n_options"]]
        counts = meta.get("counts")
        if not n_options:
            return self._assemble([], kinds, counts, int(meta.get("max_opt") or 0), torch.device("cpu"))
        device = per_question_features[0][0].device
        per_question = [
            self.score_features(u, zq, feats, kinds[qi]) for qi, (u, zq, feats) in enumerate(per_question_features)
        ]
        max_opt = int(meta.get("max_opt") or max(n_options))
        return self._assemble(per_question, kinds, counts, max_opt, device)

    def _assemble(
        self,
        per_question: List[torch.Tensor],
        kinds: List[int],
        counts: Optional[Any],
        max_opt: int,
        device: torch.device,
    ) -> ScoringOutput:
        counts_t = torch.as_tensor([int(c) for c in counts], dtype=torch.long, device=device) if counts else None
        if not per_question:
            return ScoringOutput(
                logits=torch.zeros((0, max_opt), device=device, dtype=torch.float32),
                option_mask=torch.zeros((0, max_opt), dtype=torch.bool, device=device),
                kinds=torch.zeros((0,), dtype=torch.long, device=device),
                counts=counts_t,
            )
        logits, option_mask = pad_option_logits(per_question, width=max_opt)
        return ScoringOutput(
            logits=logits,
            option_mask=option_mask,
            kinds=torch.as_tensor(kinds, dtype=torch.long, device=device),
            counts=counts_t,
        )

    def score_features(
        self, u: torch.Tensor, zq: torch.Tensor, feats: Optional[torch.Tensor], kind: int
    ) -> torch.Tensor:
        if self.ordinal and kind == 2:
            probs = self.ord(zq, u)
            cols = torch.log(probs.clamp(min=1e-8))
        else:
            cols = self.head.logits(u, zq, kind, feats)  # noul -> [yes,0]; choice -> [opts..., abstain]
        if not self.training:  # serving-time post-hoc calibration (temperature + noul log-odds bias)
            head_t = self.head_temperature.to(cols.device)
            tau = (head_t * self.temps[kind]).to(cols.device)
            if kind == 0:
                bias = self.noul_bias.to(cols.device)
                cols = torch.cat([(cols[0:1] + bias) / tau, cols[1:] / tau])
            else:
                cols = cols / tau
        return cols

    def extra_state_for_save(self) -> Dict[str, Any]:
        """Persist the calibration + flags so a reloaded checkpoint rebuilds the head without head_meta.json."""
        return {
            "norm": self.head.norm,
            "ordinal": self.ordinal,
            "head_temperature": float(self.head_temperature),
            "temperatures": {KIND_NAMES[k]: float(self.temps[k]) for k in range(len(KIND_NAMES))},
            "noul_bias": float(self.noul_bias),
        }

    def _apply_calibration_meta(self, meta: Dict[str, Any]) -> None:
        """Restore fitted temperatures (shared base form, via the ``temps`` alias) + noul_bias."""
        super()._apply_calibration_meta(meta)
        if "noul_bias" in meta:
            self.noul_bias = torch.as_tensor(float(meta["noul_bias"]), dtype=torch.float32)

    @classmethod
    def from_pretrained(cls, model_dir: str, model=None, config=None) -> "OmniJevHead":
        meta_file = cls._locate(model_dir, "head_meta.json", "OMNIJEV_HEAD_META")
        head_meta: Dict[str, Any] = {}
        if meta_file is not None:
            with open(meta_file, "r") as f:
                head_meta = json.load(f)
        norm = head_meta.get("norm", "sigmoid")
        ordinal = bool(head_meta.get("ordinal"))
        temps = head_meta.get("temperatures") or {}
        biases = head_meta.get("biases") or {}
        stray = [k for k in biases if k != "noul"]
        if stray:
            raise ValueError(
                f"head_meta.json has biases for {sorted(stray)}, but only noul has a boundary a "
                "bias can move (a renormalised softmax over several options is unchanged by a "
                "constant)."
            )
        head_file = cls._locate(model_dir, "head.pt", "OMNIJEV_HEAD")
        if head_file is None:
            raise FileNotFoundError(f"OmniJev head.pt not found under {model_dir!r}. Set OMNIJEV_HEAD to its path.")
        head_sd = torch.load(head_file, map_location="cpu")
        opt0 = head_sd.get("opt.0.weight")
        hidden_size = int(opt0.shape[1]) if opt0 is not None else cls._resolve_hidden_size(model_dir, model, config)
        head = cls(
            hidden_size,
            head_hidden=min(1024, 2 * hidden_size),
            ord_hidden=min(512, hidden_size),
            norm=norm,
            ordinal=ordinal,
            temperatures={k: float(v) for k, v in temps.items()},
            noul_bias=float(biases.get("noul", 0.0)),
            head_temperature=float(head_meta.get("head_temperature", 1.0)),
        )
        missing, unexpected = head.head.load_state_dict(head_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"OmniJev head.pt strict load failed: missing={missing}, unexpected={unexpected}")
        if ordinal:
            ord_file = cls._locate(model_dir, "ord.pt", "OMNIJEV_ORD")
            if ord_file is None:
                raise FileNotFoundError(
                    f"OmniJev ord.pt not found under {model_dir!r} (ordinal=True). Set OMNIJEV_ORD to its path."
                )
            o_missing, o_unexpected = head.ord.load_state_dict(torch.load(ord_file, map_location="cpu"), strict=False)
            if o_missing or o_unexpected:
                raise RuntimeError(f"OmniJev ord.pt strict load failed: missing={o_missing}, unexpected={o_unexpected}")
        return head

    @staticmethod
    def _resolve_hidden_size(model_dir: str, model, config) -> int:
        """The backbone text hidden size (== head.pt's `opt.0.weight` in-features). Prefer the live
        model/config, else read the checkpoint's `config.json` text_config."""
        for cfg in (config, getattr(model, "config", None)):
            text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
            hidden = getattr(text_cfg, "hidden_size", None) or getattr(cfg, "hidden_size", None)
            if hidden:
                return int(hidden)
        cfg_path = os.path.join(model_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                raw = json.load(f)
            hidden = (raw.get("text_config") or {}).get("hidden_size") or raw.get("hidden_size")
            if hidden:
                return int(hidden)
        raise ValueError(f"OmniJevHead cannot resolve the backbone hidden_size from {model_dir!r} / model / config.")


class KevPointerHead(ScoringHead):
    needs_lm_logits = False
    keep_fp32 = True
    DEFAULT_POINTER_DIM = 256

    def __init__(self, hidden_size: int, pointer_dim: int = DEFAULT_POINTER_DIM) -> None:
        super().__init__()
        # Attribute names q/k match head.pt['head'] keys exactly (strict load, 0 missing / 0 unexpected).
        self.q = nn.Linear(hidden_size, pointer_dim)
        self.k = nn.Linear(hidden_size, pointer_dim)
        self.register_buffer("temperature", torch.tensor(1.0), persistent=False)
        self.scale = float(pointer_dim) ** -0.5

    def _pointer_logits(self, h_decide: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:
        """The pointer score: q(decide) · k(option) scaled; temperature (eval-time only) applied."""
        z = (self.k(h_opts) * self.q(h_decide)).sum(-1) * self.scale
        if not self.training and float(self.temperature) != 1.0:
            z = z / self.temperature
        return z

    def _assemble(
        self, per_question: List[torch.Tensor], kinds: List[int], counts, max_opt: int, device
    ) -> ScoringOutput:
        counts_t = torch.as_tensor([int(c) for c in counts], dtype=torch.long, device=device) if counts else None
        logits, option_mask = pad_option_logits(per_question, width=max_opt)
        return ScoringOutput(
            logits=logits,
            option_mask=option_mask,
            kinds=torch.as_tensor(kinds, dtype=torch.long, device=device),
            counts=counts_t,
        )

    def forward_pointer(
        self, features: Sequence[Tuple[torch.Tensor, torch.Tensor]], meta: Dict[str, Any]
    ) -> ScoringOutput:
        n_options = [int(n) for n in meta["n_options"]]
        kinds = [int(k) for k in meta.get("kinds") or []]
        total_q = len(n_options)
        max_opt = int(meta.get("max_opt") or (max(n_options) if n_options else 0))
        if total_q == 0 or len(features) != total_q:
            raise ValueError(f"Kev branch features ({len(features)}) do not match total_q ({total_q}).")
        for qi, (_, h_opts) in enumerate(features):
            if h_opts.shape[0] != n_options[qi]:
                raise ValueError(
                    f"Kev meta mismatch: question {qi} has {n_options[qi]} options but "
                    f"{h_opts.shape[0]} </opt> hidden vectors."
                )
        per_question = [self._pointer_logits(h_decide, h_opts) for h_decide, h_opts in features]
        return self._assemble(per_question, kinds, meta.get("counts"), max_opt, features[0][0].device)

    def forward(
        self, hidden_states: torch.Tensor, lm_logits: Optional[torch.Tensor], meta: Dict[str, Any]
    ) -> ScoringOutput:
        n_options = [int(n) for n in meta["n_options"]]
        kinds = [int(k) for k in meta.get("kinds") or []]
        counts = meta.get("counts")
        device = hidden_states.device
        total_q = len(n_options)
        max_opt = int(meta.get("max_opt") or (max(n_options) if n_options else 0))
        if total_q == 0:
            empty = torch.zeros((0, max_opt), device=device, dtype=torch.float32)
            return ScoringOutput(
                logits=empty,
                option_mask=torch.zeros((0, max_opt), dtype=torch.bool, device=device),
                kinds=torch.zeros((0,), dtype=torch.long, device=device),
                counts=torch.as_tensor([int(c) for c in counts], dtype=torch.long, device=device) if counts else None,
            )

        record_index = [int(r) for r in meta["record_index"]]
        decide_idx = meta["decide_idx"]
        opt_idx = meta["opt_idx"]
        padded_len = int(hidden_states.shape[1])
        num_records = int(hidden_states.shape[0])
        left_padding = bool(meta.get("left_padding", False))
        seq_lens = meta.get("seq_lens") or [padded_len] * num_records
        hidden = hidden_states.float()

        per_question: List[torch.Tensor] = []
        for qi in range(total_q):
            if len(opt_idx[qi]) != n_options[qi]:
                raise ValueError(
                    f"Kev meta mismatch: question {qi} has {n_options[qi]} options but "
                    f"{len(opt_idx[qi])} </opt> positions."
                )
            rec = record_index[qi]
            seq_len = int(seq_lens[rec])
            offset = (padded_len - seq_len) if left_padding else 0
            if offset + int(decide_idx[qi]) >= padded_len:
                raise ValueError(
                    f"Kev decide_idx {decide_idx[qi]} out of range for record {rec} "
                    f"(padded_len={padded_len}, left_padding={left_padding})."
                )
            h_decide = hidden[rec, offset + int(decide_idx[qi])]  # [d]
            h_opts = hidden[rec, [offset + int(o) for o in opt_idx[qi]]]  # [K, d]
            per_question.append(self._pointer_logits(h_decide, h_opts))

        return self._assemble(per_question, kinds, counts, max_opt, device)

    def extra_state_for_save(self) -> Dict[str, Any]:
        return {"temperature": float(self.temperature)}

    def _apply_calibration_meta(self, meta: Dict[str, Any]) -> None:
        if "temperature" in meta:
            self.temperature = torch.tensor(float(meta["temperature"]))

    @classmethod
    def from_pretrained(cls, model_dir: str, model=None, config=None) -> "KevPointerHead":
        head_file = cls._locate(model_dir, "head.pt", "KEV_HEAD")
        if head_file is None:
            raise FileNotFoundError(
                f"Kev head.pt not found under {model_dir!r}. The pointer head ships beside the LoRA "
                "adapter in the jaredpalmer/kev-{0.8b,4b,9b} repo, NOT in the base repo; set KEV_HEAD to "
                "its path (see examples/train/decision/kev.sh)."
            )
        state = torch.load(head_file, map_location="cpu")
        head_sd = state.get("head") if isinstance(state, dict) else None
        if not head_sd or "q.weight" not in head_sd or "k.weight" not in head_sd:
            raise ValueError(
                f'Kev head.pt {head_file!r} has no head state dict under the "head" key '
                f"(expected q.weight / q.bias / k.weight / k.bias); "
                f"got keys={list(state) if isinstance(state, dict) else type(state)}."
            )
        head = cls(hidden_size=int(head_sd["q.weight"].shape[1]), pointer_dim=int(head_sd["q.weight"].shape[0]))
        missing, unexpected = head.load_state_dict(head_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Kev head.pt strict load failed: missing={missing}, unexpected={unexpected}")
        head.temperature = torch.tensor(float(state.get("temperature", 1.0)))
        return head


class LayaHead(ScoringHead):
    needs_lm_logits = False
    keep_fp32 = True
    # ModernBERT encoder backbone: the loader's forward wrapper feeds us last_hidden_state directly.
    encoder_backbone = True

    # swift kind id -> Laya question-type id
    LAYA_QTYPE_OF_KIND = {0: 2, 1: 0, 2: 1}

    def __init__(
        self,
        hidden_size: int,
        head_layers: int = 2,
        dropout: float = 0.1,
        temperatures: Optional[Sequence[float]] = None,
        temperature_by_options: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        d = hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.register_buffer(
            "temperature", torch.as_tensor([float(t) for t in (temperatures or (1.0, 1.0, 1.0))], dtype=torch.float32)
        )
        self.temperature_by_options: Dict[str, float] = {
            str(k): float(v) for k, v in (temperature_by_options or {}).items()
        }

    def has_kind_temperatures(self) -> bool:
        return True

    def get_temperature(self, kind: int) -> float:
        """Per-kind temperature in swift kind order (noul/choice/score)"""
        return float(self.temperature[self.LAYA_QTYPE_OF_KIND[int(kind)]])

    def set_temperature(self, kind: int, value: float) -> None:
        """Overwrite one kind's temperature (swift kind order) and drop its stale option-count
        buckets, so the refit value wins over any loaded ``temperature_by_options`` entry."""
        kind = int(kind)
        self.temperature[self.LAYA_QTYPE_OF_KIND[kind]] = float(value)
        self._clear_buckets_for_kind(kind)

    def _clear_buckets_for_kind(self, kind: int) -> None:
        prefix = f"{KIND_NAMES[kind]}:"
        self.temperature_by_options = {k: v for k, v in self.temperature_by_options.items() if not k.startswith(prefix)}

    @staticmethod
    def _temp_bucket(n_options: int) -> str:
        if n_options <= 2:
            return "2"
        if n_options <= 5:
            return "3-5"
        if n_options <= 10:
            return "6-10"
        return "11+"

    def forward(  # noqa: C901
        self, hidden_states: torch.Tensor, lm_logits: Optional[torch.Tensor], meta: Dict[str, Any]
    ) -> ScoringOutput:
        n_options = [int(n) for n in meta["n_options"]]
        kinds = [int(k) for k in meta.get("kinds") or []]
        counts = meta.get("counts")
        device = hidden_states.device
        counts_t = torch.as_tensor([int(c) for c in counts], dtype=torch.long, device=device) if counts else None
        total_q = len(n_options)
        batch, padded_len, _ = hidden_states.shape
        if total_q == 0:
            max_opt = int(meta.get("max_opt") or 0)
            return ScoringOutput(
                logits=torch.zeros((0, max_opt), device=device, dtype=torch.float32),
                option_mask=torch.zeros((0, max_opt), dtype=torch.bool, device=device),
                kinds=torch.zeros((0,), dtype=torch.long, device=device),
                counts=counts_t,
            )
        if batch != total_q:
            raise ValueError(
                f"LayaHead requires one question per row (batch={batch}, total_q={total_q}); "
                "LayaTemplate fans records out into one row per question."
            )
        record_index = [int(r) for r in meta["record_index"]]
        if record_index != list(range(total_q)):
            raise ValueError(
                f"LayaHead got record_index={record_index}, expected identity "
                f"{list(range(total_q))} (one question per row)."
            )
        markers = meta["laya_markers"]  # per question, real-token-relative <mask> positions
        if len(markers) != total_q or any(len(m) != n for m, n in zip(markers, n_options)):
            raise ValueError(
                f"Laya meta mismatch: {len(markers)} marker lists vs {total_q} questions (n_options={n_options})."
            )

        left_padding = bool(meta.get("left_padding", False))
        seq_lens = [int(s) for s in (meta.get("seq_lens") or [padded_len] * batch)]

        h = hidden_states.float()
        qtype_ids = torch.as_tensor([self.LAYA_QTYPE_OF_KIND.get(k, 0) for k in kinds], device=device)
        h = h + self.type_emb(qtype_ids)[:, None, :]
        if self.head is not None:
            pad = torch.zeros((batch, padded_len), dtype=torch.bool, device=device)  # True = masked
            for b in range(batch):
                sl = min(seq_lens[b], padded_len)
                if left_padding:
                    pad[b, : padded_len - sl] = True
                else:
                    pad[b, sl:] = True
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)

        kmax = max(1, max(n_options))
        marker_pos = torch.zeros((total_q, kmax), dtype=torch.long, device=device)
        marker_mask = torch.zeros((total_q, kmax), dtype=torch.bool, device=device)
        for qi in range(total_q):
            offset = (padded_len - seq_lens[qi]) if left_padding else 0
            for j, m in enumerate(markers[qi]):
                pos = offset + int(m)
                if pos >= padded_len:
                    raise ValueError(
                        f"Laya marker {m} (padded pos {pos}) out of range "
                        f"(padded_len={padded_len}, left_padding={left_padding})."
                    )
                marker_pos[qi, j] = pos
                marker_mask[qi, j] = True
        idx = marker_pos[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        if not self.training:
            logits = self._calibrate(logits, kinds, n_options)
        return ScoringOutput(
            logits=logits,
            option_mask=marker_mask,
            kinds=torch.as_tensor(kinds, dtype=torch.long, device=device),
            counts=counts_t,
        )

    def _calibrate(self, logits: torch.Tensor, kinds: List[int], n_options: List[int]) -> torch.Tensor:
        temps = []
        for kind, n in zip(kinds, n_options):
            key = f"{KIND_NAMES[kind]}:{self._temp_bucket(n)}"
            fitted = self.temperature_by_options.get(key)
            if fitted is None:
                fitted = float(self.temperature[self.LAYA_QTYPE_OF_KIND[kind]])
            temps.append(float(fitted))
        t = torch.as_tensor(temps, device=logits.device, dtype=torch.float32)[:, None]
        return logits / t.clamp(min=1e-6)

    def extra_state_for_save(self) -> Dict[str, Any]:
        return {
            "hidden_size": int(self.scorer[0].normalized_shape[0]),
            "head_layers": 0 if self.head is None else len(self.head.layers),
            "temperature": self.temperature.detach().float().tolist(),
            "temperature_by_options": dict(self.temperature_by_options),
        }

    def _apply_calibration_meta(self, meta: Dict[str, Any]) -> None:
        temps = meta.get("temperature")
        if isinstance(temps, (list, tuple)) and len(temps) == 3:
            self.temperature = torch.as_tensor([float(t) for t in temps], dtype=torch.float32)
        buckets = meta.get("temperature_by_options")
        if isinstance(buckets, dict):
            self.temperature_by_options = {str(k): float(v) for k, v in buckets.items()}

    @classmethod
    def from_pretrained(cls, model_dir: str, model=None, config=None) -> "LayaHead":
        head_config: Dict[str, Any] = {}
        config_file = cls._locate(model_dir, "rl_agent_config.json", "LAYA_CONFIG")
        if config_file is None:
            for subdir in ("typed-decisions", "multilingual"):
                candidate = os.path.join(model_dir, subdir, "rl_agent_config.json")
                if os.path.exists(candidate):
                    config_file = candidate
                    break
        if config_file is None:
            meta_file = os.path.join(model_dir, cls.HEAD_META_NAME)
            if os.path.isfile(meta_file):
                with open(meta_file, "r") as f:
                    head_config = json.load(f)
                hidden_size = int(
                    head_config.get("hidden_size")
                    or getattr(config, "hidden_size", 0)
                    or getattr(getattr(model, "config", None), "hidden_size", 0)
                )
                if not hidden_size:
                    raise ValueError(
                        f"LayaHead cannot resolve hidden_size from {meta_file!r} / config {config!r} / model {model!r}."
                    )
                return cls(
                    hidden_size=hidden_size,
                    head_layers=int(head_config.get("head_layers", 2)),
                    temperatures=head_config.get("temperature") or (1.0, 1.0, 1.0),
                    temperature_by_options=head_config.get("temperature_by_options") or {},
                )
            raise FileNotFoundError(
                f"Laya rl_agent_config.json not found under {model_dir!r} (or its typed-decisions / "
                "multilingual subfolders). Set LAYA_CONFIG to its path."
            )
        with open(config_file, "r") as f:
            head_config = json.load(f)
        hidden_size = int(
            getattr(config, "hidden_size", None) or getattr(getattr(model, "config", None), "hidden_size", 0)
        )
        if not hidden_size:
            raise ValueError(f"LayaHead cannot resolve hidden_size from config {config!r} / model {model!r}.")
        return cls(
            hidden_size=hidden_size,
            head_layers=int(head_config.get("head_layers", 2)),
            temperatures=head_config.get("temperature") or (1.0, 1.0, 1.0),
            temperature_by_options=head_config.get("temperature_by_options") or {},
        )
