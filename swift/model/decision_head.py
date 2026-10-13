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
        loss:        Optional                         filled when labels are consumed
        ordinal_logits: Optional                      OmniJev CORAL/CORN cumulative logits

    meta (assembled by Template._post_encode, consumed by ScoringHead.forward):
        n_options:      List[int]                 per-question option count (len == total_q)
        question_spans: List[Tuple[int, int]]     per-question token span (Clef/OmniJev pool)
        option_spans:   List[List[Tuple[int,int]]] per-question per-option token span
        verbalizer_ids: List[List[int]]           per-option lm_head column to read (JEV)
        bias / temperatures: calibration buffers (JEV/OmniJev), kept on the head

    labels (training), consumed by ScoringLoss and _compute_acc:
        target_probs: FloatTensor [total_q, max_opt]  padded target distribution
                  (one-hot for Clef/OmniJev CE, soft for JEV distillation; pad cols = 0)
"""
import json
import math
import os
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from typing import Any, Dict, List, Optional, Sequence, Tuple

# QUESTION_TYPES maps the JEV/Clef encoding: noul -> 0, choice -> 1, score -> 2.
QUESTION_TYPES = {'noul': 0, 'choice': 1, 'score': 2}
KIND_NAMES = ('noul', 'choice', 'score')


@dataclass
class ScoringOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    option_mask: Optional[torch.BoolTensor] = None
    kinds: Optional[torch.LongTensor] = None
    counts: Optional[torch.LongTensor] = None
    loss: Optional[torch.FloatTensor] = None
    ordinal_logits: Optional[torch.FloatTensor] = None


def pad_option_logits(per_question: List[torch.Tensor],
                      pad_value: float = float('-inf'),
                      width: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    total_q = len(per_question)
    if width is None:
        width = max((int(t.shape[0]) for t in per_question), default=0)
    if total_q == 0 or width == 0:
        empty = torch.zeros((total_q, width))
        return empty, torch.zeros((total_q, width), dtype=torch.bool)
    device = per_question[0].device
    dtype = per_question[0].dtype
    logits = torch.full((total_q, width), pad_value, device=device, dtype=dtype)
    mask = torch.zeros((total_q, width), device=device, dtype=torch.bool)
    for i, t in enumerate(per_question):
        n = min(int(t.shape[0]), width)
        if n == 0:
            continue
        logits[i, :n] = t[:n].to(dtype)
        mask[i, :n] = True
    return logits, mask


def masked_softmax(logits: torch.Tensor, option_mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits = logits.float().masked_fill(~option_mask, float('-inf'))
    probs = torch.softmax(logits, dim=dim)
    return torch.nan_to_num(probs, nan=0.0)


def masked_argmax(logits: torch.Tensor, option_mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Argmax restricted to real option slots (pad slots never win)."""
    return logits.float().masked_fill(~option_mask, float('-inf')).argmax(dim=dim)


def get_scoring_head(model: nn.Module) -> Optional[nn.Module]:
    for name, module in model.named_modules():
        if name == 'scoring_head' or name.endswith('.scoring_head'):
            return module
    return None


def maybe_load_trained_scoring_head(model: nn.Module, adapter_dir: Optional[str]) -> bool:
    """After an adapter/checkpoint is applied, overlay a TRAINED head if that dir carries one.

    Two scenarios:
    - Adapter checkpoint (LoRA): ``scoring_head.safetensors`` exists → load weights from it.
    - Full-finetune checkpoint: ``scoring_head.safetensors`` is absent, ``scoring_head.json``
      present → the head's parameters are inside the model safetensors shards (ignored by
      ``from_pretrained`` because ``scoring_head`` was not yet attached). Extract them here.
    """
    if not adapter_dir:
        return False
    head = get_scoring_head(model)
    if head is None:
        return False
    loaded = head.load_pretrained(adapter_dir)
    if loaded:
        from swift.utils import get_logger
        get_logger().info(f'decision: loaded trained scoring_head from {adapter_dir}.')
    return loaded


def clef_option_ids(kind: str, criteria: Any) -> List[str]:
    if kind == 'noul':
        return ['true', 'false']
    if kind == 'choice':
        return [str(key) for key, _ in sorted((str(k), v) for k, v in (criteria or {}).items())]
    return [str(index) for index in range(len(criteria or []))]


def omnijev_column_keys(kind: str, keys: Sequence[Any]) -> List[str]:
    if kind == 'noul':
        return ['yes', 'no']
    if kind == 'choice':
        return [str(k) for k in keys] + ['abstain']
    return [str(k) for k in keys]


class ScoringHead(nn.Module):
    """Abstract base for the three decision heads (JEV / Clef / OmniJev).

    A head turns base-model representations into per-option logits over each question's own
    option set. `forward` receives BOTH the last hidden states and the lm_head logits so every
    head can pick what it needs:
      - JEV reads lm_logits at the verbalizer columns of the final token (no nn.Module params
        beyond bias/temperature buffers);
      - Clef pools hidden_states over question/option spans and runs the JointSchema decoder;
      - OmniJev dots option/question MLPs over pooled hidden states and adds an ordinal head.

    Subclasses MUST return a `ScoringOutput` whose `logits`/`option_mask` follow the contract
    above, and MUST load their factory head weights with `strict=True`.
    """

    # Set by subclasses that need the base model's output embeddings (Clef lexical prior).
    needs_output_embeddings: bool = False
    # Set by subclasses that read lm_head logits (JEV verbalizer columns). When False the loader
    # skips passing lm_logits, so a Clef/OmniJev forward never pays for the full-seq vocab projection.
    needs_lm_logits: bool = False
    keep_fp32: bool = False

    def set_output_embeddings(self, output_embeddings: nn.Module) -> None:
        self._output_embeddings = output_embeddings

    @staticmethod
    def _locate(model_dir: str, filename: str, env_var: str, subdir: Optional[str] = None) -> Optional[str]:
        """Locate a head artifact file: check an env-var override first, then model_dir/filename
        (and optionally model_dir/subdir/filename). Returns the path if found, else None.
        Shared by all three head subclasses (JEV passes ``subdir='adapter_vllm'``)."""
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

    def forward(self, hidden_states: torch.Tensor, lm_logits: Optional[torch.Tensor], meta: Dict[str,
                                                                                                 Any]) -> ScoringOutput:
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
    def from_pretrained(cls, model_dir: str, model=None, config=None) -> 'ScoringHead':
        """Construct the head and STRICT-load its factory weights from `model_dir`.

        Each model subclass reads its own artifact (JEV: decision_head.json + calibration.json;
        Clef: joint_head.safetensors; OmniJev: head.pt + ord.pt) and asserts a 0-missing /
        0-unexpected strict load. `model`/`config` are passed for heads that size themselves from
        the base
        """
        raise NotImplementedError

    HEAD_WEIGHTS_NAME = 'scoring_head.safetensors'
    HEAD_META_NAME = 'scoring_head.json'

    def head_state_dict(self) -> Dict[str, torch.Tensor]:
        """The head's OWN state (params + persistent buffers), EXCLUDING the injected live
        `_output_embeddings`. `set_output_embeddings` assigns an nn.Module, which `nn.Module.__setattr__`
        auto-registers as a child, so a naive `state_dict()` would drag in the ENTIRE base lm_head
        (vocab x hidden, possibly tied to embed_tokens). That reference is re-wired by the loader on
        rebuild, so it is never persisted here.
        """
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if not k.startswith('_output_embeddings')}

    def save_pretrained(self, save_directory: str, save_weights: bool = True) -> None:
        """Persist the head beside the adapter: weights (`head_state_dict`) + the non-parameter
        calibration meta (`extra_state_for_save`), so a reloaded checkpoint rebuilds the TRAINED head
        without the factory `decision_head.json` / `joint_head.safetensors` (plan decision F).

        When ``save_weights=False`` (full-finetune checkpoint), the head's parameters are already
        inside the model ``state_dict`` saved by ``super()._save``; only the meta json is written so
        a reloaded checkpoint still carries a human-readable head manifest.
        """
        from safetensors.torch import save_file
        os.makedirs(save_directory, exist_ok=True)
        if save_weights:
            save_file(self.head_state_dict(), os.path.join(save_directory, self.HEAD_WEIGHTS_NAME))
        with open(os.path.join(save_directory, self.HEAD_META_NAME), 'w', encoding='utf-8') as f:
            json.dump(self.extra_state_for_save(), f, indent=2, ensure_ascii=False)

    def load_pretrained(self, load_directory: str) -> bool:
        """Two scenarios:
        - Adapter checkpoint (LoRA): ``HEAD_WEIGHTS_NAME`` exists → load weights from it (the head
          is NOT inside the LoRA adapter).
        - Full-finetune checkpoint: ``HEAD_WEIGHTS_NAME`` is absent → the head's parameters are
          already restored from the model ``state_dict`` by ``from_pretrained``; only the meta
          json is present, and we return True so the caller knows the head was recognised.
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
                self._validate_and_load(state, 'model safetensors')
        return True

    def _validate_and_load(self, state: Dict[str, torch.Tensor], source: str) -> None:
        own_keys = {k for k in self.state_dict() if not k.startswith('_output_embeddings')}
        own_params = {n for n, _ in self.named_parameters() if not n.startswith('_output_embeddings')}
        unexpected = set(state) - own_keys
        missing_params = own_params - set(state)
        if unexpected or missing_params:
            raise ValueError(f'{source} does not match head {type(self).__name__}: '
                             f'unexpected={sorted(unexpected)} missing_params={sorted(missing_params)}')
        self.load_state_dict(state, strict=False)

    def _extract_head_state_from_model_dir(self, model_dir: str) -> Optional[Dict[str, torch.Tensor]]:
        """Used when loading a full-finetune checkpoint: ``from_pretrained`` ignored scoring_head keys
        (the submodule was not yet attached); this reads them back from the safetensors files.
        Returns ``None`` when the model dir carries no scoring_head weights (e.g. a factory model).
        """
        from safetensors.torch import load_file
        index_path = os.path.join(model_dir, 'model.safetensors.index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index = json.load(f)
            weight_map = index.get('weight_map', {})
            head_keys = {k for k in weight_map if 'scoring_head' in k}
            if not head_keys:
                return None
            shards = set(weight_map[k] for k in head_keys)
            result: Dict[str, torch.Tensor] = {}
            for shard in shards:
                data = load_file(os.path.join(model_dir, shard))
                for k, v in data.items():
                    if 'scoring_head' in k:
                        result[k.split('scoring_head.', 1)[1] if 'scoring_head.' in k else k] = v
            return result if result else None
        single = os.path.join(model_dir, 'model.safetensors')
        if os.path.exists(single):
            data = load_file(single)
            result = {}
            for k, v in data.items():
                if 'scoring_head' in k:
                    result[k.split('scoring_head.', 1)[1] if 'scoring_head.' in k else k] = v
            return result if result else None
        return None


class JevVerbalizerHead(ScoringHead):

    needs_lm_logits = True
    keep_fp32 = True

    DEFAULT_RANGES = {'noul': (0, 2), 'choice': (8, 24), 'score': (2, 8)}
    DEFAULT_TEMPERATURES = {'noul': 1.0, 'choice': 1.0, 'score': 1.0}

    def __init__(self, bias: Sequence[float], verbalizer_ids: Sequence[int], ranges: Dict[str, Sequence[int]],
                 temperatures: Dict[str, float]):
        super().__init__()
        self.bias = nn.Parameter(torch.as_tensor(list(bias), dtype=torch.float32))
        self.register_buffer(
            'verbalizer_ids', torch.as_tensor(list(verbalizer_ids), dtype=torch.long), persistent=False)
        self.register_buffer(
            'ranges',
            torch.as_tensor([list(ranges[KIND_NAMES[k]]) for k in range(len(KIND_NAMES))], dtype=torch.long),
            persistent=False)
        self.register_buffer(
            'temperatures',
            torch.as_tensor([float(temperatures[KIND_NAMES[k]]) for k in range(len(KIND_NAMES))], dtype=torch.float32),
            persistent=False)

    def forward(self, hidden_states: torch.Tensor, lm_logits: Optional[torch.Tensor], meta: Dict[str,
                                                                                                 Any]) -> ScoringOutput:
        if lm_logits is None:
            raise ValueError('JevVerbalizerHead requires lm_logits (needs_lm_logits must be True).')
        kinds = [int(k) for k in meta['kinds']]
        n_options = [int(n) for n in meta['n_options']]
        record_index = [int(r) for r in meta['record_index']]
        total_q = len(n_options)
        device = lm_logits.device
        counts = meta.get('counts')
        counts_t = torch.as_tensor([int(c) for c in counts], dtype=torch.long) if counts else None
        if total_q == 0:
            max_opt = int(meta.get('max_opt') or 0)
            empty = torch.zeros((0, max_opt), device=device, dtype=torch.float32)
            return ScoringOutput(
                logits=empty,
                option_mask=torch.zeros((0, max_opt), dtype=torch.bool, device=device),
                kinds=torch.zeros((0, ), dtype=torch.long, device=device),
                counts=counts_t)

        rows, cols = self._readout_positions(record_index, meta, lm_logits.shape[1], device)
        read_logits = lm_logits[rows, cols, :].float()  # [total_q, vocab]
        log_probs = torch.log_softmax(read_logits, dim=-1)  # log P(token), matching vLLM's logprob

        verbalizer_ids = self.verbalizer_ids.to(device)
        bias = self.bias.to(device=device, dtype=torch.float32)
        ranges = self.ranges.to(device)
        temperatures = self.temperatures.to(device=device, dtype=torch.float32)
        per_question: List[torch.Tensor] = []
        for i in range(total_q):
            k = kinds[i]
            n = n_options[i]
            lo = int(ranges[k][0])
            hi = int(ranges[k][1])
            if lo + n > verbalizer_ids.shape[0] or n > hi - lo:
                raise ValueError(
                    f'JEV kind {KIND_NAMES[k]} supports at most {hi - lo} options (verbalizer slots), got {n}.')
            ids = verbalizer_ids[lo:lo + n]  # [n]
            lp = log_probs[i].gather(0, ids)  # [n]
            z = (lp + bias[lo:lo + n]) / temperatures[k]
            per_question.append(z)

        max_opt = int(meta.get('max_opt') or max(n_options))
        logits, option_mask = pad_option_logits(per_question, width=max_opt)
        return ScoringOutput(
            logits=logits,
            option_mask=option_mask,
            kinds=torch.as_tensor(kinds, dtype=torch.long, device=device),
            counts=counts_t)

    @staticmethod
    def _readout_positions(record_index: List[int], meta: Dict[str, Any], padded_len: int,
                           device) -> Tuple[torch.Tensor, torch.Tensor]:
        left_padding = bool(meta.get('left_padding', False))
        seq_lens = meta.get('seq_lens')
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
            'bias': self.bias.detach().float().tolist(),
            'verbalizer_ids': self.verbalizer_ids.tolist(),
            'ranges': {
                KIND_NAMES[k]: self.ranges[k].tolist()
                for k in range(len(KIND_NAMES))
            },
            'temperatures': {
                KIND_NAMES[k]: float(self.temperatures[k])
                for k in range(len(KIND_NAMES))
            },
        }

    @classmethod
    def from_pretrained(cls, model_dir: str, model=None, config=None) -> 'JevVerbalizerHead':
        head_file = cls._locate(model_dir, 'decision_head.json', 'JEV_DECISION_HEAD', subdir='adapter_vllm')
        if head_file is None:
            raise FileNotFoundError(
                f'JEV decision_head.json not found under {model_dir!r} (or {model_dir}/adapter_vllm). '
                'Set JEV_DECISION_HEAD to its path.')
        with open(head_file, 'r') as f:
            head = json.load(f)
        for key in ('bias', 'verbalizer_ids'):
            if key not in head:
                raise KeyError(f'decision_head.json missing required key {key!r}: {head_file}')
        ranges = {k: tuple(v) for k, v in head.get('slots', {}).get('ranges', {}).items()}
        for kind, default in cls.DEFAULT_RANGES.items():
            ranges.setdefault(kind, default)

        temperatures = dict(cls.DEFAULT_TEMPERATURES)
        calib_file = os.environ.get('JEV_DECIDE_CALIBRATION')
        if not calib_file or not os.path.exists(calib_file):
            calib_file = cls._locate(
                os.path.dirname(os.path.dirname(os.path.abspath(head_file))), 'calibration.json',
                'JEV_DECIDE_CALIBRATION')
        if calib_file and os.path.exists(calib_file):
            with open(calib_file, 'r') as f:
                per_kind = json.load(f).get('per_kind', {})
            temperatures.update({k: float(v) for k, v in per_kind.items() if k in temperatures})
        return cls(head['bias'], head['verbalizer_ids'], ranges, temperatures)

    @staticmethod
    def _resolve_verbalizer_token_id(tokenizer, text: str) -> int:

        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
        unk = getattr(tokenizer, 'unk_token_id', None)
        if unk is None:
            raise ValueError(f'Tokenizer cannot resolve verbalizer token for {text!r} '
                             f'(encodes to {len(ids)} tokens: {ids}) and has no unk_token_id.')
        from swift.utils import get_logger
        get_logger().warning(f'decision: verbalizer label {text!r} encodes to {len(ids)} tokens ({ids}); '
                             f'falling back to unk_token_id={unk}. The head slot will read this unk column '
                             f'instead of the intended token; a LoRA on lm_head can still learn the correct '
                             f'distribution on that column, but multiple labels collapsing to the same unk '
                             f'id makes them indistinguishable (degenerate). Consider using a tokenizer that '
                             f'encodes the verbalizer labels as single tokens.')
        return unk

    @classmethod
    def from_tokenizer(cls, tokenizer, config=None) -> 'JevVerbalizerHead':
        """Random-init path for training a decision model from scratch on any CausalLM.

        Args:
            tokenizer: the base model's tokenizer (any `PreTrainedTokenizerBase`).
            config: unused (accepted for signature parity with `from_pretrained`).

        Returns:
            A `JevVerbalizerHead` with zero-initialized bias and unit temperatures, ready for
            joint training with a LoRA adapter on the base CausalLM.
        """
        noul_words = ['false', 'true']
        score_words = ['0', '1', '2', '3', '4', '5']
        choice_words = list(string.ascii_uppercase[:16])  # A..P

        verbalizer_ids = [
            cls._resolve_verbalizer_token_id(tokenizer, w) for w in noul_words + score_words + choice_words
        ]
        bias = [0.0] * len(verbalizer_ids)
        ranges = dict(cls.DEFAULT_RANGES)
        temperatures = dict(cls.DEFAULT_TEMPERATURES)
        return cls(bias, verbalizer_ids, ranges, temperatures)


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
            normalized_queries, self.memory_norm(memory), self.memory_norm(memory), need_weights=False)
        queries = queries + self.attention_dropout(routed)
        return queries + self.feedforward(self.feedforward_norm(queries))


class ClefJointSchemaHead(ScoringHead):
    """Clef's joint schema head: a real parameterised decoder over span-pooled hidden states."""
    needs_output_embeddings = True
    needs_lm_logits = False
    keep_fp32 = False

    def __init__(self,
                 hidden_size: int,
                 width: int,
                 routing_layers: int,
                 layers: int,
                 heads: int,
                 feedforward: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(3, width)
        self.evidence_layers = nn.ModuleList([
            EvidenceRoutingLayer(width=width, heads=heads, feedforward=feedforward, dropout=dropout)
            for _ in range(routing_layers)
        ])
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True) for _ in range(layers)
        ])
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

    def forward(self, hidden_states: torch.Tensor, lm_logits: Optional[torch.Tensor], meta: Dict[str,
                                                                                                 Any]) -> ScoringOutput:
        input_ids = meta.get('input_ids')
        if input_ids is None:
            raise ValueError('ClefJointSchemaHead requires meta["input_ids"] '
                             '(stashed by ClefTemplate._post_encode before the base drops it).')
        if getattr(self, '_output_embeddings', None) is None:
            raise RuntimeError('ClefJointSchemaHead needs output embeddings; the loader must call '
                               'set_output_embeddings (needs_output_embeddings=True).')
        device = hidden_states.device
        n_options = [int(n) for n in meta['n_options']]
        kinds = [int(k) for k in meta['kinds']]
        total_q = len(n_options)
        counts = meta.get('counts')
        counts_t = torch.as_tensor([int(c) for c in counts], dtype=torch.long, device=device) if counts else None
        if total_q == 0:
            max_opt = int(meta.get('max_opt') or 0)
            empty = torch.zeros((0, max_opt), device=device, dtype=hidden_states.dtype)
            return ScoringOutput(
                logits=empty,
                option_mask=torch.zeros((0, max_opt), dtype=torch.bool, device=device),
                kinds=torch.zeros((0, ), dtype=torch.long, device=device),
                counts=counts_t)

        question_spans = meta['question_spans']
        option_spans = meta['option_spans']
        record_index = [int(r) for r in meta['record_index']]
        num_records = int(hidden_states.shape[0])
        padded_len = int(hidden_states.shape[1])
        left_padding = bool(meta.get('left_padding', False))
        seq_lens = meta.get('seq_lens') or [padded_len] * num_records
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
            sequence_hidden = normalized_hidden[rec, pad_offset:pad_offset + seq_len]
            record_logits = self._score_record(sequence_hidden, input_ids[rec], pad_offset, qidxs, question_spans,
                                               option_spans, kinds, output_embedding_weight)
            per_question.extend(record_logits)
            per_kind.extend(kinds[qi] for qi in qidxs)

        max_opt = int(meta.get('max_opt') or max(n_options))
        logits, option_mask = pad_option_logits(per_question, width=max_opt)
        return ScoringOutput(
            logits=logits,
            option_mask=option_mask,
            kinds=torch.as_tensor(per_kind, dtype=torch.long, device=device),
            counts=counts_t)

    def _score_record(self, sequence_hidden: torch.Tensor, input_ids_row: torch.Tensor, pad_offset: int,
                      qidxs: List[int], question_spans: List[Any], option_spans: List[Any], kinds: List[int],
                      output_embedding_weight: torch.Tensor) -> List[torch.Tensor]:
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
                token_ids = input_ids_row[pad_offset + int(start):pad_offset + int(end)]
                lexical_vectors.append(output_embedding_weight[token_ids].mean(dim=0))
            lexical_options.append(torch.stack(lexical_vectors))
            option_contexts.append(context_vectors)
            option_counts.append(len(spans))

        option_queries = []
        for question_index, (context_vectors, lexical) in enumerate(zip(option_contexts, lexical_options)):
            option_queries.append(
                self.option_context_projection(context_vectors) + self.option_lexical_projection(lexical)
                + self.option_question_projection(question_vectors[question_index]).unsqueeze(0))
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
            base_fields + self.option_summary_norm(torch.stack(option_summaries))
            + self.global_projection(global_vector).unsqueeze(0) + self.type_embedding(type_ids))
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
                [repeated_field, options, repeated_field * options,
                 torch.abs(repeated_field - options)], dim=-1)
            residual = self.residual_scorer(features).squeeze(-1)
            joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
            joint = joint_scale * cosine + residual
            record_logits.append(prior + torch.sigmoid(self.residual_gate) * joint)
        return record_logits

    @classmethod
    def from_pretrained(cls, model_dir: str, model=None, config=None) -> 'ClefJointSchemaHead':
        from safetensors.torch import load_file
        config_file = cls._locate(model_dir, 'joint_head_config.json', 'CLEF_JOINT_HEAD_CONFIG')
        weights_file = cls._locate(model_dir, 'joint_head.safetensors', 'CLEF_JOINT_HEAD_WEIGHTS')
        if config_file is None:
            raise FileNotFoundError(f'Clef joint_head_config.json not found under {model_dir!r}. '
                                    'Set CLEF_JOINT_HEAD_CONFIG to its path.')
        if weights_file is None:
            raise FileNotFoundError(f'Clef joint_head.safetensors not found under {model_dir!r}. '
                                    'Set CLEF_JOINT_HEAD_WEIGHTS to its path.')
        with open(config_file, 'r') as f:
            head_config = json.load(f)
        head = cls(**head_config)
        state = load_file(weights_file)
        missing, unexpected = head.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f'Clef joint head strict load failed: missing={missing}, unexpected={unexpected}')
        return head


class OmniOptionScorer(nn.Module):
    """OmniJev's decision head. The parameter
    names/shapes match `head.pt` exactly and a strict load has 0 missing / 0 unexpected.
    """

    INIT_SCORE_BIAS = -3.0

    def __init__(self, d_model: int, d_hidden: int = 1024, n_types: int = 3, norm: str = 'sigmoid') -> None:
        super().__init__()
        self.opt = nn.Sequential(nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden))
        self.que = nn.Sequential(nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden))
        self.score = nn.Linear(d_hidden, 1)
        self.log_tau = nn.Parameter(torch.zeros(n_types))
        self.norm = norm
        self.abstain = nn.Linear(d_hidden, 1)
        self.feat = nn.Linear(6, d_hidden)
        self.feat_lin = nn.Linear(6, 1)

    def forward(self, u_opts: torch.Tensor, z_q: torch.Tensor, type_id: int = 0, feats=None) -> torch.Tensor:
        lg = self.logits(u_opts, z_q, type_id, feats)
        if self.norm == 'softmax' and type_id != 0:
            return torch.softmax(lg, dim=0)[:-1]
        return torch.sigmoid(lg[:-1])

    def logits(self, u_opts: torch.Tensor, z_q: torch.Tensor, type_id: int = 0, feats=None) -> torch.Tensor:
        """The vector the probabilities come from, abstain last: `[K option logits, abstain]` for a
        softmax choice/score, `[logit, 0]` for noul (softmax of [x, 0] == sigmoid(x)).
        """
        g = torch.tanh(self.que(z_q))
        h = self.opt(u_opts) * g[None, :]  # FiLM-style gating
        if feats is not None:
            h = h + self.feat(feats)
        tau = torch.exp(self.log_tau[type_id])
        logit = self.score(h).squeeze(-1)
        if feats is not None:
            logit = logit + self.feat_lin(feats).squeeze(-1)
        logit = logit / tau
        if self.norm == 'softmax' and type_id != 0:
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


def omnijev_lm_option_feats(lm_head: nn.Module, h: torch.Tensor, ids: torch.Tensor, opens: Sequence[int],
                            closes: Sequence[int], q_end: int, open_len: int, close_len: int) -> torch.Tensor:
    """Features: [sum log p(option tokens)/10, mean, n_tokens/10, log_softmax over K of the sums,
    first-token log p at the question end, validity flag].
    """
    n_feat = 6
    k_opt = min(len(opens), len(closes))
    dev = h.device
    feats = torch.zeros(k_opt, n_feat, device=dev, dtype=torch.float32)
    if k_opt == 0:
        return feats
    pred_pos: List[int] = []
    tgt_tok: List[int] = []
    owner: List[int] = []
    firsts: List[int] = []
    for k in range(k_opt):
        o, c = int(opens[k]), int(closes[k])
        inner = list(range(o + open_len, c - close_len + 1))  # option text token positions
        firsts.append(int(ids[inner[0]]) if inner else -1)
        for t in inner:
            pred_pos.append(t - 1)
            tgt_tok.append(int(ids[t]))
            owner.append(k)
    sums = torch.zeros(k_opt, device=dev)
    cnts = torch.zeros(k_opt, device=dev)
    if pred_pos:
        pos_t = torch.as_tensor(pred_pos, device=dev, dtype=torch.long)
        lg = lm_head(h[pos_t]).float()  # [P, vocab]
        lp = torch.log_softmax(lg, dim=-1)
        tok = torch.as_tensor(tgt_tok, device=dev, dtype=torch.long)
        own = torch.as_tensor(owner, device=dev, dtype=torch.long)
        picked = lp.gather(1, tok[:, None]).squeeze(1)
        sums = sums.index_add(0, own, picked)
        cnts = cnts.index_add(0, own, torch.ones_like(picked))
    valid = cnts > 0
    mean = torch.where(valid, sums / cnts.clamp(min=1), torch.zeros_like(sums))
    dist = torch.log_softmax(torch.where(valid, sums, torch.full_like(sums, -1e4)), dim=0)
    first = torch.zeros(k_opt, device=dev)
    if q_end is not None and q_end >= 0 and any(f >= 0 for f in firsts):
        lq = torch.log_softmax(lm_head(h[q_end:q_end + 1]).float(), dim=-1)[0]
        for k, f in enumerate(firsts):
            if f >= 0:
                first[k] = lq[f]
    feats[:, 0] = sums / 10.0
    feats[:, 1] = mean
    feats[:, 2] = cnts / 10.0
    feats[:, 3] = dist
    feats[:, 4] = first
    feats[:, 5] = valid.float()
    return feats


class OmniJevHead(ScoringHead):
    """OmniJev System-1 head: `OptionScorer` (choice/noul) + `OrdinalScoreHead` (score) over the hidden
    state at each option's `<|/opt|>` marker, gated by the question hidden at the token before the first
    `<|opt|>`, plus the 6-dim backbone-opinion features.

    Reads (per question, all positions real-token-relative, `h_row = hidden[rec, pad_offset:+seq_len]`):
      * `u_o` = `h_row[close_o]` -- the hidden at each option's OPT_CLOSE last token;
      * `z_q` = `h_row[opens[0]-1]` -- the hidden just before the first OPT_OPEN;
      * 6-dim feats via `omnijev_lm_option_feats` using the base lm_head (== output embeddings), so
        `needs_output_embeddings=True`; the head runs in float32 (`keep_fp32=True`).

    Emitted `logits` are laid out so the shared `masked_softmax` reproduces the expected probabilities:
      * noul -> 2 cols `[yes_logit, 0]` (softmax == sigmoid(yes_logit) == P(yes));
      * choice -> K+1 cols `[opt_0..opt_{K-1}, abstain]` (P(abstain) == 1 - sum(option probs));
      * score -> L cols `log(ordinal_probs)` (the cumulative-link head's per-level probabilities).
    `meta['n_options']` must carry these column counts (2 / K+1 / L), NOT the option-block count.

    Calibration: the in-head learned `exp(log_tau)` is ALWAYS applied (inside `OptionScorer.logits`).
    The post-hoc `head_meta` per-kind temperature and the noul log-odds bias are serving-time only, so
    they are folded into the emitted logits ONLY when `not self.training` (a temperature on probs
    `p^(1/T)` == dividing the underlying logits by T; the noul bias shifts col0's log-odds first).
    """
    needs_output_embeddings = True
    needs_lm_logits = False
    keep_fp32 = True

    def __init__(self,
                 hidden_size: int,
                 head_hidden: int = 1024,
                 ord_hidden: int = 512,
                 n_types: int = 3,
                 norm: str = 'softmax',
                 lm_feats: bool = True,
                 ordinal: bool = True,
                 temperatures: Optional[Dict[str, float]] = None,
                 noul_bias: float = 0.0) -> None:
        super().__init__()
        self.head = OmniOptionScorer(hidden_size, head_hidden, n_types, norm)
        self.ord = OmniOrdinalHead(hidden_size, ord_hidden)
        self.lm_feats = bool(lm_feats)
        self.ordinal = bool(ordinal)
        temps = temperatures or {}
        self.register_buffer(
            'temps',
            torch.as_tensor([float(temps.get(KIND_NAMES[k], 1.0)) for k in range(n_types)], dtype=torch.float32),
            persistent=False)
        self.register_buffer('noul_bias', torch.as_tensor(float(noul_bias), dtype=torch.float32), persistent=False)
        self.open_len = 1
        self.close_len = 1

    def set_marker_lengths(self, open_len: int, close_len: int) -> None:
        """Token lengths of `<|opt|>` / `<|/opt|>` (normally 1 each); the loader sets these from the
        tokenizer so the option-text inner range in `omnijev_lm_option_feats` is exact."""
        self.open_len = int(open_len)
        self.close_len = int(close_len)

    def forward(self, hidden_states: torch.Tensor, lm_logits: Optional[torch.Tensor], meta: Dict[str,
                                                                                                 Any]) -> ScoringOutput:
        device = hidden_states.device
        kinds = [int(k) for k in meta['kinds']]
        n_options = [int(n) for n in meta['n_options']]
        total_q = len(n_options)
        counts = meta.get('counts')
        if total_q == 0:
            return self._assemble([], kinds, counts, int(meta.get('max_opt') or 0), device)

        lm_head = getattr(self, '_output_embeddings', None)
        if self.lm_feats and lm_head is None:
            raise RuntimeError('OmniJevHead needs output embeddings for lm_feats; the loader must call '
                               'set_output_embeddings (needs_output_embeddings=True).')
        input_ids = meta.get('input_ids')
        if self.lm_feats and input_ids is None:
            raise ValueError('OmniJevHead requires meta["input_ids"] for lm_feats '
                             '(stashed by OmniJevTemplate._post_encode before the base drops it).')
        option_markers = meta['option_markers']  # per question: [(open_pos, close_pos), ...] real-token-relative
        record_index = [int(r) for r in meta['record_index']]
        padded_len = int(hidden_states.shape[1])
        left_padding = bool(meta.get('left_padding', False))
        num_records = int(hidden_states.shape[0])
        seq_lens = meta.get('seq_lens') or [padded_len] * num_records

        per_question: List[torch.Tensor] = []
        for qi in range(total_q):
            rec = record_index[qi]
            seq_len = int(seq_lens[rec])
            pad_offset = (padded_len - seq_len) if left_padding else 0
            h_row = hidden_states[rec, pad_offset:pad_offset + seq_len]
            ids_real = input_ids[rec, pad_offset:pad_offset + seq_len] if input_ids is not None else None
            cols = self._score_question(h_row, ids_real, option_markers[qi], kinds[qi], lm_head)
            per_question.append(cols)

        max_opt = int(meta.get('max_opt') or max(n_options))
        return self._assemble(per_question, kinds, counts, max_opt, device)

    def forward_features(self, per_question_features: Sequence[Tuple[torch.Tensor, torch.Tensor,
                                                                     Optional[torch.Tensor]]],
                         meta: Dict[str, Any]) -> ScoringOutput:
        """Branch entry point (decision G1). The loader runs the two-stage cached branch forward"""
        kinds = [int(k) for k in meta['kinds']]
        n_options = [int(n) for n in meta['n_options']]
        counts = meta.get('counts')
        if not n_options:
            return self._assemble([], kinds, counts, int(meta.get('max_opt') or 0), torch.device('cpu'))
        device = per_question_features[0][0].device
        per_question = [
            self.score_features(u, zq, feats, kinds[qi]) for qi, (u, zq, feats) in enumerate(per_question_features)
        ]
        max_opt = int(meta.get('max_opt') or max(n_options))
        return self._assemble(per_question, kinds, counts, max_opt, device)

    def _assemble(self, per_question: List[torch.Tensor], kinds: List[int], counts: Optional[Any], max_opt: int,
                  device: torch.device) -> ScoringOutput:
        counts_t = torch.as_tensor([int(c) for c in counts], dtype=torch.long, device=device) if counts else None
        if not per_question:
            return ScoringOutput(
                logits=torch.zeros((0, max_opt), device=device, dtype=torch.float32),
                option_mask=torch.zeros((0, max_opt), dtype=torch.bool, device=device),
                kinds=torch.zeros((0, ), dtype=torch.long, device=device),
                counts=counts_t)
        logits, option_mask = pad_option_logits(per_question, width=max_opt)
        return ScoringOutput(
            logits=logits,
            option_mask=option_mask,
            kinds=torch.as_tensor(kinds, dtype=torch.long, device=device),
            counts=counts_t)

    def _score_question(self, h_row: torch.Tensor, ids_real: Optional[torch.Tensor], markers: Sequence[Tuple[int, int]],
                        kind: int, lm_head: Optional[nn.Module]) -> torch.Tensor:
        opens = [int(o) for o, _ in markers]
        closes = [int(c) for _, c in markers]
        u = h_row[torch.as_tensor(closes, device=h_row.device, dtype=torch.long)].float()  # [k, H]
        q_end = max(0, opens[0] - 1) if opens else -1
        zq = h_row[q_end].float() if opens else u.mean(0)
        feats = None
        if self.lm_feats and lm_head is not None and ids_real is not None:
            feats = omnijev_lm_option_feats(lm_head, h_row, ids_real, opens, closes, q_end, self.open_len,
                                            self.close_len)
        return self.score_features(u, zq, feats, kind)

    def score_features(self, u: torch.Tensor, zq: torch.Tensor, feats: Optional[torch.Tensor],
                       kind: int) -> torch.Tensor:
        if self.ordinal and kind == 2:
            probs = self.ord(zq, u)
            cols = torch.log(probs.clamp(min=1e-8))
        else:
            cols = self.head.logits(u, zq, kind, feats)  # noul -> [yes,0]; choice -> [opts..., abstain]
        if not self.training:  # serving-time post-hoc calibration (temperature + noul log-odds bias)
            tau = self.temps[kind].to(cols.device)
            if kind == 0:
                bias = self.noul_bias.to(cols.device)
                cols = torch.cat([(cols[0:1] + bias) / tau, cols[1:] / tau])
            else:
                cols = cols / tau
        return cols

    def extra_state_for_save(self) -> Dict[str, Any]:
        """Persist the calibration + flags so a reloaded checkpoint rebuilds the head without head_meta.json."""
        return {
            'norm': self.head.norm,
            'lm_feats': self.lm_feats,
            'ordinal': self.ordinal,
            'temperatures': {
                KIND_NAMES[k]: float(self.temps[k])
                for k in range(len(KIND_NAMES))
            },
            'noul_bias': float(self.noul_bias),
            'open_len': self.open_len,
            'close_len': self.close_len,
        }

    @classmethod
    def from_pretrained(cls, model_dir: str, model=None, config=None) -> 'OmniJevHead':
        meta_file = cls._locate(model_dir, 'head_meta.json', 'OMNIJEV_HEAD_META')
        head_meta: Dict[str, Any] = {}
        if meta_file is not None:
            with open(meta_file, 'r') as f:
                head_meta = json.load(f)
        norm = head_meta.get('norm', 'sigmoid')
        lm_feats = bool(head_meta.get('lm_feats'))
        ordinal = bool(head_meta.get('ordinal'))
        temps = head_meta.get('temperatures') or {}
        biases = head_meta.get('biases') or {}
        stray = [k for k in biases if k != 'noul']
        if stray:
            raise ValueError(f'head_meta.json has biases for {sorted(stray)}, but only noul has a boundary a '
                             'bias can move (a renormalised softmax over several options is unchanged by a '
                             'constant).')
        head_file = cls._locate(model_dir, 'head.pt', 'OMNIJEV_HEAD')
        if head_file is None:
            raise FileNotFoundError(f'OmniJev head.pt not found under {model_dir!r}. Set OMNIJEV_HEAD to its path.')
        head_sd = torch.load(head_file, map_location='cpu')
        opt0 = head_sd.get('opt.0.weight')
        hidden_size = int(opt0.shape[1]) if opt0 is not None else cls._resolve_hidden_size(model_dir, model, config)
        head = cls(
            hidden_size,
            head_hidden=min(1024, 2 * hidden_size),
            ord_hidden=min(512, hidden_size),
            norm=norm,
            lm_feats=lm_feats,
            ordinal=ordinal,
            temperatures={
                k: float(v)
                for k, v in temps.items()
            },
            noul_bias=float(biases.get('noul', 0.0)))
        missing, unexpected = head.head.load_state_dict(head_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f'OmniJev head.pt strict load failed: missing={missing}, unexpected={unexpected}')
        if ordinal:
            ord_file = cls._locate(model_dir, 'ord.pt', 'OMNIJEV_ORD')
            if ord_file is None:
                raise FileNotFoundError(
                    f'OmniJev ord.pt not found under {model_dir!r} (ordinal=True). Set OMNIJEV_ORD to its path.')
            o_missing, o_unexpected = head.ord.load_state_dict(torch.load(ord_file, map_location='cpu'), strict=False)
            if o_missing or o_unexpected:
                raise RuntimeError(f'OmniJev ord.pt strict load failed: missing={o_missing}, unexpected={o_unexpected}')
        return head

    @staticmethod
    def _resolve_hidden_size(model_dir: str, model, config) -> int:
        """The backbone text hidden size (== head.pt's `opt.0.weight` in-features). Prefer the live
        model/config, else read the checkpoint's `config.json` text_config."""
        for cfg in (config, getattr(model, 'config', None)):
            text_cfg = getattr(cfg, 'text_config', None) if cfg is not None else None
            hidden = getattr(text_cfg, 'hidden_size', None) or getattr(cfg, 'hidden_size', None)
            if hidden:
                return int(hidden)
        cfg_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                raw = json.load(f)
            hidden = (raw.get('text_config') or {}).get('hidden_size') or raw.get('hidden_size')
            if hidden:
                return int(hidden)
        raise ValueError(f'OmniJevHead cannot resolve the backbone hidden_size from {model_dir!r} / model / config.')
