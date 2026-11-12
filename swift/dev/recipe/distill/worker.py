# Copyright (c) ModelScope Contributors. All rights reserved.
# Distillation's remote worker: the teacher forward and the per-update teacher signal, in ONE actor.
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from twinkle import remote_class, remote_function
from twinkle.data_format import ground_truth

from swift.dev.utils.scoring import encode_privileged_feature


def response_ids_from_feature(feature: Dict[str, Any]) -> List[int]:
    """The response token ids of an encoded feature, recovered from its (next-token shifted) labels.

    Labels hold the response token ids at trainable positions and ``-100`` elsewhere, so filtering
    ``-100`` yields exactly the sampled response -- the tokens the teacher must re-score.
    """
    labels = feature.get('labels')
    if labels is None:
        raise ValueError('OPSD/MOPD teacher scoring requires a labelled feature to recover the response token ids.')
    response_ids = [int(label) for label in labels if int(label) != -100]
    if not response_ids:
        raise ValueError('OPSD/MOPD feature contains no response tokens to score.')
    return response_ids


def _run_grouped(tasks: Sequence[Tuple[Any, Callable[[], Any]]]) -> List[Any]:
    """Run teacher forwards grouped by placement: groups overlap as threads, within a group serial.

    Teachers on disjoint device groups overlap wall-clock; one group stays strictly serial (two
    forwards on the same cards would only race for memory). A single group runs inline.
    """
    groups: Dict[Any, List[int]] = {}
    for index, (key, _task) in enumerate(tasks):
        groups.setdefault(key, []).append(index)
    results: List[Any] = [None] * len(tasks)
    if len(groups) <= 1:
        for index, (_key, task) in enumerate(tasks):
            results[index] = task()
        return results

    def _run_group(indices: List[int]) -> None:
        for index in indices:
            results[index] = tasks[index][1]()

    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        for future in [pool.submit(_run_group, indices) for indices in groups.values()]:
            future.result()
    return results


@remote_class(execute='first')
class DistillWorker:
    """Build one update's teacher signal where the teacher forward runs.

    The teacher forward runs HERE (a CPU actor) so its heavy output never materialises in the
    driver: GKD's full-vocab logits and OPSD / MOPD's response-only logps are declared ``stash``
    channels and ride the kwargs as refs Ray derefs at the training actor. The update's
    trajectories are rollout trajectories (on-policy) or encoded dataset rows (off-policy); both
    carry ``input_ids`` / ``labels`` / ``messages`` and the dataset columns. ``teachers`` is always
    a list of bare frozen models, described by the plain parameters below.
    """

    def __init__(self, *, teachers: Sequence[Any], signal: str, template: Any = None,
                 disable_lora: bool = False, offload: bool = False,
                 placements: Optional[Sequence[str]] = None, gkd_logits_topk: Optional[int] = None,
                 sft_alpha: float = 0.0, teacher_weights: Optional[Sequence[float]] = None,
                 train_batch_size: int = 1):
        if signal not in ('logits', 'logps', 'multi_logps'):
            raise ValueError(f"distillation signal must be one of 'logits'/'logps'/'multi_logps', got {signal!r}.")
        self.teachers = list(teachers)
        self.signal = signal
        self.template = template
        self.disable_lora = disable_lora
        self.offload = offload
        self.placements = None if placements is None else list(placements)
        self.gkd_logits_topk = gkd_logits_topk
        self.sft_alpha = sft_alpha
        self.teacher_weights = None if teacher_weights is None else list(teacher_weights)
        self.train_batch_size = max(1, train_batch_size)

    def _response_logps(self, scorer: Any, features: List[dict], *, disable_lora: bool = False) -> List[List[float]]:
        """Score ``features`` with ``forward_only`` -> one response-token logp list per feature.

        The features form one update, so one whole-batch call -- the dispatch splits it per rank --
        is the whole story.
        """
        from twinkle.signals import response_token_scores

        frame = scorer.forward_only(
            inputs=features,
            micro_batch_size=max(1, self.train_batch_size // scorer.device_mesh.data_world_size),
            **({'disable_lora': True} if disable_lora else {}),
        )
        return response_token_scores(frame, features)

    def _privileged_feature(self, traj: Dict[str, Any]) -> dict:
        """Encode this trajectory's privileged teacher view (``teacher_prompt`` + the shared response tokens)."""
        teacher_prompt = ground_truth(traj, 'teacher_prompt')
        if not teacher_prompt:
            raise ValueError('OPSD requires a per-sample `teacher_prompt` carrying the privileged context; '
                             'a row in this batch has none.')
        return encode_privileged_feature(
            self.template, list(traj.get('messages') or []), teacher_prompt, response_ids_from_feature(traj), traj)

    @remote_function(dispatch='all', collect='none', lazy_collect=False,
                     stash=('teacher_logits', 'teacher_logps'))
    def forward_kwargs(self, trajs: List[Dict[str, Any]], use_student: bool) -> Dict[str, Any]:
        """One update's teacher signal in the ``signal`` form (the caller passes ``inputs`` itself).

        ``use_student`` is the update's on/off-policy flag -- only GKD reads it (``apply_sft_loss``
        adds the supervised CE term to a dataset-sourced update); the logps forms score the same rows
        either way.
        """
        from swift.dev.utils.scoring import copy_inputs

        if self.signal == 'logits':
            teacher = self.teachers[0]
            if self.offload:
                teacher.reload_to_gpu()
            try:
                logits = teacher.forward_only(
                    inputs=copy_inputs(trajs), return_logits=True,
                    micro_batch_size=max(1, self.train_batch_size // teacher.device_mesh.data_world_size),
                    **({'disable_lora': True} if self.disable_lora else {}),
                )['logits']
            finally:
                if self.offload:
                    teacher.offload_to_cpu()
            return {
                'teacher_logits': logits,
                'topk': self.gkd_logits_topk,
                'apply_sft_loss': bool(self.sft_alpha > 0 and not use_student),
            }
        if self.signal == 'logps':
            privileged = [self._privileged_feature(traj) for traj in trajs]
            return {'teacher_logps': self._response_logps(self.teachers[0], privileged, disable_lora=self.disable_lora)}
        # K response-only channels, overlapped across disjoint placements, plus the mixture weights.
        # A dict keeps the teacher axis separate from DP/micro-batch slicing of each sample channel.
        tasks = [(self.placements[index], lambda index=index: self._response_logps(self.teachers[index], trajs))
                 for index in range(len(self.teachers))]
        signal_kwargs: Dict[str, Any] = {'teacher_logps': dict(enumerate(_run_grouped(tasks)))}
        if self.teacher_weights is not None:
            # Mixture weights are global scalars; tuples are broadcast by DP dispatch.
            signal_kwargs['teacher_weights'] = tuple(self.teacher_weights)
        return signal_kwargs
