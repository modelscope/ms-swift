# Copyright (c) ModelScope Contributors. All rights reserved.
"""Channel contracts for the workflow canvas: one schema per port kind.

This is the machine-readable half of the component documentation. The canvas renders
these tables on the channel-contract page; the adapter validates edges against
``kind``; the ``guarded`` flag on TRAJ fields is the declared playground for code
components (guarded fields belong to the token ledger and stay closed).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChannelField:
    """One field of the TRAJ contract. ``guarded`` marks token-ledger fields."""

    name: str
    doc: str
    guarded: bool = False


@dataclass(frozen=True)
class ChannelSchema:
    """Contract of one port kind. ``reference`` marks object-view channels (MODEL,
    SAMPLER): the edge carries a handle, not a data payload; base-view ports are
    calling-convention sugar over the same object, not a second object."""

    kind: str
    doc: str
    reference: bool = False
    fields: tuple[ChannelField, ...] = field(default=())

    def to_dict(self) -> dict:
        return {
            'kind': self.kind,
            'doc': self.doc,
            'reference': self.reference,
            'fields': [{'name': f.name, 'doc': f.doc, 'guarded': f.guarded} for f in self.fields],
        }


TRAJ_FIELDS: tuple[ChannelField, ...] = (
    # Skeleton: the token ledger. Written by the encode/rollout layer, read by every
    # forward consumer; code components may not rewrite these (audit invariants).
    ChannelField('encoded', 'prompt+completion feature with already-shifted labels; holds input_ids/labels/completion_mask', True),
    ChannelField('response_token_ids', 'response token ids in the response frame', True),
    ChannelField('response_loss_mask', 'trainable mask mirroring response_token_ids', True),
    ChannelField('rollout_logprobs', 'sampling-policy old logps, one per response token', True),
    ChannelField('prompt', 'original prompt messages, stamped on the streaming collect path'),
    ChannelField('messages', 'full conversation messages of the trajectory'),
    # Signal fields: written by scoring/advantage/code stages; free to extend.
    ChannelField('orm_reward', 'per-scorer outcome reward row; None until scored'),
    ChannelField('prm_process', 'per-token process reward in the response-token frame'),
    ChannelField('baseline', 'group-level baseline (e.g. ReMax greedy reward)'),
    ChannelField('extra', 'free side-data dict for custom signals'),
    ChannelField('rollout_infos', 'engine-side rollout statistics'),
    ChannelField('sampling_mask', 'per-token sampling support set (twinkle SamplingMask)'),
    ChannelField('policy_version', 'policy version this trajectory was generated under (admission pin)'),
    ChannelField('version_span', 'publishes between admission and collection; in_place only'),
    ChannelField('group_key', 'kernel group identity; positions are the training contract'),
    ChannelField('decoded', 'decoded response text'),
    ChannelField('truncated', 'whether the response hit the length limit'),
)

CHANNEL_SCHEMAS: dict[str, ChannelSchema] = {
    'MODEL': ChannelSchema(
        'MODEL', 'A trainable or frozen model as an object-view reference: the edge declares '
        'which view (policy / adapter-disabled base / frozen copy) a consumer reads.', reference=True),
    'SAMPLER': ChannelSchema(
        'SAMPLER', 'A resident inference-engine resource handle; weight-sync targets are declared '
        'by wiring this handle into the sync effect.', reference=True),
    'TRAJ': ChannelSchema(
        'TRAJ', 'A batch of RolloutSample rows: the single data contract of the RL data plane.',
        fields=TRAJ_FIELDS),
    'DATA': ChannelSchema(
        'DATA', 'Raw dataset rows / prompt batches: messages plus free columns; upstream of encoding.'),
    'ADV': ChannelSchema(
        'ADV', 'Per-token advantage signal aligned with the TRAJ rows of the same batch.'),
    'REWARD': ChannelSchema(
        'REWARD', 'Per-sample scalar scores, one column per scorer.'),
    'LOSS': ChannelSchema(
        'LOSS', 'A loss callable handle produced by loss nodes; consumed by the train step.'),
    'METRIC': ChannelSchema('METRIC', 'Metric rows destined for the logger.'),
    'CKPT': ChannelSchema('CKPT', 'A checkpoint artifact reference (path + metadata).'),
    'CTRL': ChannelSchema(
        'CTRL', 'Unit token for pure ordering: edges that carry no data, only happens-before.'),
    'ENV': ChannelSchema(
        'ENV', 'A sandbox environment resource handle (local subprocess / OpenEnv pool / '
        'microVM): multi-turn rollouts execute tool calls inside it.', reference=True),
    'TOOL': ChannelSchema(
        'TOOL', 'A toolset declaration (JSON-schema tools the policy can call); rolled into the '
        'chat template and the environment\'s tool surface.'),
    'ENDPOINT': ChannelSchema(
        'ENDPOINT', 'A live serving endpoint reference (gateway url + served model id).'),
}
