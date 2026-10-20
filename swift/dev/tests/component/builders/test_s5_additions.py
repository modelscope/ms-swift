# Copyright (c) ModelScope Contributors. All rights reserved.
"""S5 additions: builders surfaces the carried-over suites do not reach.

Covers the ut_plan S5 rows with no prior anchor: param-group spec composition (LoRA+ rule order,
per-tower lrs, plugin append order), encode-mode/loader-layout resolution, the prompt-side dataset
builders, trajectory splitting, frozen-auxiliary config shrinking, reference/reward model dispatch,
engine-arg shaping, sampling-param mapping and RL sampler placement.

Deferred elsewhere (cross-references, not gaps): S5-18 _drop_last lives in dataset/test_api.py,
S5-24 _concat_with_cached in dataset/test_s3_additions.py, S5-25 build_dp_mesh in
config/test_hf_sp_config.py, S5-30/31/32 in the model suite, S5-36..40 in infer/test_sandbox.py
and processor/test_template_build.py.
"""
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------- param groups (S5-09/10/11)


def test_param_groups_lora_plus_emits_three_rules_in_order():
    from swift.dev.builders.optimizer import build_param_groups_spec
    from swift.dev.config import TrainConfig, TunerConfig

    cfg = TrainConfig(learning_rate=1e-4)
    tuner = TunerConfig(tuner='lora', lorap_lr_ratio=16.0, lorap_emb_lr=2e-6)
    spec = build_param_groups_spec(cfg, tuner, None)
    # legacy's isinstance chain, reproduced as ordered first-match-wins rules: Embedding ->
    # lora_B -> any 1-D param; lora_A matches none of them and keeps the base lr
    assert [r['match'] for r in spec] == [{'module_type': 'Embedding'}, {'name': '*lora_B*'}, {'ndim': 1}]
    assert spec[0]['lr'] == 2e-6
    assert spec[1]['lr_mult'] == 16.0 and spec[2]['lr_mult'] == 16.0
    # no knob set at all -> None, leaving twinkle's default grouping untouched
    assert build_param_groups_spec(TrainConfig(), TunerConfig(tuner='lora'), None) is None


def test_param_groups_tower_rules_aligner_claims_first():
    from swift.dev.builders.optimizer import build_param_groups_spec
    from swift.dev.config import TrainConfig

    # only vit_lr set: the aligner rule is STILL emitted (at base lr) so the broader
    # vision_tower rule cannot swallow aligner params -- its prefix nests under the tower's
    spec = build_param_groups_spec(TrainConfig(learning_rate=1e-4, vit_lr=2e-5), None, None)
    assert [r['match'] for r in spec] == [{'tower': 'aligner'}, {'tower': 'vision_tower'}]
    assert spec[0]['lr'] == 1e-4 and spec[1]['lr'] == 2e-5

    # only aligner_lr set: no vision_tower rule at all
    spec = build_param_groups_spec(TrainConfig(learning_rate=1e-4, aligner_lr=3e-5), None, None)
    assert [r['match'] for r in spec] == [{'tower': 'aligner'}] and spec[0]['lr'] == 3e-5

    # both: each tower keeps its own lr, aligner first
    spec = build_param_groups_spec(TrainConfig(learning_rate=1e-4, vit_lr=2e-5, aligner_lr=3e-5), None, None)
    assert [(r['match']['tower'], r['lr']) for r in spec] == [('aligner', 3e-5), ('vision_tower', 2e-5)]


def test_param_groups_plugins_append_after_builtins():
    from swift.dev.builders.optimizer import build_param_groups_spec
    from swift.dev.builders.optimizer_plugins import PARAM_GROUPS
    from swift.dev.config import TrainConfig, TunerConfig
    from swift.dev.utils.plugin import PluginRegistry

    @PluginRegistry.register(PARAM_GROUPS, 's5_double')
    class _Double(PARAM_GROUPS.base):

        def __call__(self, *, train_config, tuner_config):
            return [{'match': {'name': '*a*'}, 'lr_mult': 2.0}, {'match': {'name': '*b*'}, 'lr_mult': 3.0}]

    cfg = TrainConfig(learning_rate=1e-4, vit_lr=2e-5, param_groups_plugins=['s5_double'])
    spec = build_param_groups_spec(cfg, TunerConfig(tuner='lora', lorap_lr_ratio=4.0), None)
    # builtin rules first (3 LoRA+ + aligner claim + vision_tower), plugin rules extend after
    assert len(spec) == 3 + 2 + 2
    assert spec[-2:] == [{'match': {'name': '*a*'}, 'lr_mult': 2.0}, {'match': {'name': '*b*'}, 'lr_mult': 3.0}]
    assert spec[0]['match'] == {'module_type': 'Embedding'}
    PluginRegistry.kind(PARAM_GROUPS.name).entries.pop('s5_double', None)


# ---------------------------------------------------------------- encode mode / loader layout (S5-17/19)


def test_encode_mode_three_states():
    from swift.dev.builders.dataset import _encode_mode
    from swift.dev.config import DatasetConfig

    mm_template = SimpleNamespace(model_meta=SimpleNamespace(is_multimodal=True))
    text_template = SimpleNamespace(model_meta=SimpleNamespace(is_multimodal=False))

    assert _encode_mode(DatasetConfig(streaming=True), text_template) == 'stream'
    # legacy default: multimodal without cache/packing/length-grouping tokenizes lazily
    assert _encode_mode(DatasetConfig(), mm_template) == 'lazy'
    assert _encode_mode(DatasetConfig(packing=True), mm_template) == 'eager'
    assert _encode_mode(DatasetConfig(cached_dataset=['x']), mm_template) == 'eager'
    assert _encode_mode(DatasetConfig(group_by_length=True), mm_template) == 'eager'
    assert _encode_mode(DatasetConfig(), text_template) == 'eager'
    # explicit knob wins over the derived default in both directions
    assert _encode_mode(DatasetConfig(lazy_tokenize=True), text_template) == 'lazy'
    assert _encode_mode(DatasetConfig(lazy_tokenize=False, streaming=False), mm_template) == 'eager'


def test_twinkle_loader_layout_scales_by_dp_width(monkeypatch):
    from swift.dev.builders import dataset as dataset_module
    from swift.dev.config import DistributedConfig

    # no mesh, no nproc_per_node: plain torchrun scales by the env world size, mesh stays None
    monkeypatch.setattr('twinkle.utils.Platform.get_world_size', staticmethod(lambda: 4))
    assert dataset_module._twinkle_loader_layout(DistributedConfig(), 2) == (8, None)

    # megatron + ray: the loader is a bare driver loader (DP scatter happens in forward_backward)
    fake_mesh = SimpleNamespace(data_world_size=3)
    monkeypatch.setattr(dataset_module, 'build_parallel_mesh', lambda dc: fake_mesh, raising=False)
    import swift.dev.builders.model as model_module
    monkeypatch.setattr(model_module, 'build_parallel_mesh', lambda dc: fake_mesh)
    dc = DistributedConfig(backend='megatron', nproc_per_node=8, mode='ray')
    assert dataset_module._twinkle_loader_layout(dc, 2) == (6, None)
    # non-ray megatron hands the mesh to the loader's sampler
    dc_local = DistributedConfig(backend='megatron', nproc_per_node=8, mode='local')
    assert dataset_module._twinkle_loader_layout(dc_local, 2) == (6, fake_mesh)

    # an HF Ulysses mesh scales by data_world_size (SP peers get identical samples)
    sp_mesh = SimpleNamespace(data_world_size=2)
    assert dataset_module._twinkle_loader_layout(DistributedConfig(), 4, device_mesh=sp_mesh) == (8, sp_mesh)


# ---------------------------------------------------------------- prompt-side dataset (S5-21/22)


def test_load_prompt_only_rows_val_wins_and_ratio_slices(tmp_path):
    from swift.dev.builders.dataset import load_prompt_only_rows
    from swift.dev.config import DatasetConfig
    from swift.dev.tests.tiny import TinyData

    train_path = TinyData.sft(tmp_path / 'train.jsonl', n=8)
    val_path = TinyData.sft(tmp_path / 'val.jsonl', n=3)

    # val_dataset wins outright and loads WHOLE (ratio forced to 1.0 internally)
    rows = load_prompt_only_rows(DatasetConfig(dataset=[train_path], val_dataset=[val_path]), None, 0.25)
    assert len(rows) == 3

    # no val_dataset + ratio>0: only the eval slice of --dataset
    rows = load_prompt_only_rows(DatasetConfig(dataset=[train_path]), None, 0.25)
    assert len(rows) == 2

    # ratio=0 runs over ALL of --dataset
    rows = load_prompt_only_rows(DatasetConfig(dataset=[train_path]), None, 0.0)
    assert len(rows) == 8

    # max_rows cuts after the split
    rows = load_prompt_only_rows(DatasetConfig(dataset=[train_path]), 3, 0.0)
    assert len(rows) == 3


def test_build_prompt_dataloader_guards_and_iterates():
    from swift.dev.builders.dataset import build_prompt_dataloader

    with pytest.raises(ValueError, match='at least one prompt'):
        build_prompt_dataloader([], None, seed=0)
    prompts = [[{'role': 'user', 'content': 'hi'}]]
    with pytest.raises(ValueError, match='batch_size must be > 0'):
        build_prompt_dataloader(prompts, None, seed=0, batch_size=0)
    with pytest.raises(ValueError, match='must align with prompts'):
        build_prompt_dataloader(prompts * 2, [{}], seed=0)

    loader = build_prompt_dataloader(
        [[{'role': 'user', 'content': 'a'}], [{'role': 'user', 'content': 'b'}]],
        [{'solution': '4'}, {'solution': '5'}],
        seed=42,
        batch_size=2)
    batch = next(iter(loader))
    # rows carry messages plus their flattened extras, and a partial batch is never dropped
    assert len(batch) == 2
    assert {row['solution'] for row in batch} == {'4', '5'}
    assert all(row['messages'][0]['role'] == 'user' for row in batch)


# ---------------------------------------------------------------- trajectory shaping (S5-23)


def test_to_trajectory_replaces_system_and_carries_mm_columns():
    from swift.dev.builders.dataset import to_trajectory
    from swift.dev.config import TemplateConfig

    messages = [{'role': 'system', 'content': 'old'}, {'role': 'user', 'content': 'hi'}]
    traj = to_trajectory({'images': ['x.png']}, messages, TemplateConfig(system='new'))
    # TemplateConfig.system REPLACES the row's system turn instead of stacking a second one
    assert [m['role'] for m in traj['messages']] == ['system', 'user']
    assert traj['messages'][0]['content'] == 'new'
    assert traj['images'] == ['x.png']

    # no system configured: messages pass through, empty mm columns are not invented
    traj = to_trajectory({}, messages, TemplateConfig())
    assert traj['messages'] == messages and 'images' not in traj


def test_split_prompt_and_reference_moves_trailing_assistant():
    from swift.dev.builders.dataset import split_prompt_and_reference
    from swift.dev.config import TemplateConfig

    rows = [
        {
            'messages': [
                {
                    'role': 'user',
                    'content': 'q1'
                },
                {
                    'role': 'assistant',
                    'content': 'a1'
                },
            ]
        },
        {
            'messages': [{
                'role': 'user',
                'content': 'q2'
            }]
        },
    ]
    trajectories, references = split_prompt_and_reference(rows, TemplateConfig())
    assert references == ['a1', None]
    # generation sees a prompt that stops BEFORE the answer
    assert [m['role'] for m in trajectories[0]['messages']] == ['user']
    assert trajectories[1]['messages'][0]['content'] == 'q2'
    # the input rows are not mutated
    assert rows[0]['messages'][-1]['role'] == 'assistant'


# ---------------------------------------------------------------- frozen auxiliary / ref model (S5-26/27)


def test_frozen_auxiliary_config_inherits_identity_only():
    from swift.dev.builders.model import frozen_auxiliary_distributed_config
    from swift.dev.config import DistributedConfig

    trainer = DistributedConfig(backend='megatron', mode='local', nproc_per_node=8, tensor_model_parallel_size=4)
    aux = frozen_auxiliary_distributed_config(trainer, 2, parallel_spec='tp=2')
    # backend identity is inherited; the trainer's parallel layout and process count are NOT
    assert aux.backend == 'megatron' and aux.bridge_backend == trainer.bridge_backend
    assert aux.nproc_per_node == 2
    assert aux.parallel_spec == 'tp=2'
    assert aux.tensor_model_parallel_size == DistributedConfig().tensor_model_parallel_size
    # RL never runs local: mode is forced to ray
    assert aux.mode == 'ray'


def test_build_ref_model_three_branches(monkeypatch):
    from swift.dev.builders import model as model_module
    from swift.dev.config import DistributedConfig, ModelConfig

    # disable_adapter without the policy model is a hard error (the shared-weights view IS the ref)
    with pytest.raises(ValueError, match='needs the policy model'):
        model_module.build_ref_model(
            name='ref', model_id=None, model_config=ModelConfig(model='m'), template=None,
            distributed_config=DistributedConfig(), disable_adapter=True)
    policy = object()
    assert model_module.build_ref_model(
        name='ref', model_id=None, model_config=ModelConfig(model='m'), template=None,
        distributed_config=DistributedConfig(), policy_model=policy, disable_adapter=True) is policy
    # an empty model_id means the channel is off
    assert model_module.build_ref_model(
        name='ref', model_id=None, model_config=ModelConfig(model='m'), template=None,
        distributed_config=DistributedConfig()) is None

    # the real build path: frozen dist config, its own remote group named after the channel,
    # and more than one frozen adapter is refused
    calls: Dict[str, Any] = {}

    class _Frozen:

        def add_adapter_to_model(self, name, path, is_trainable):
            calls['adapter'] = (name, path, is_trainable)

        def set_processor(self, *a, **k):
            calls['processor'] = k

        def set_template(self, *a, **k):
            calls['template'] = a

    def fake_build_model(config, dist, **kwargs):
        calls['config'] = config
        calls['dist'] = dist
        calls['kwargs'] = kwargs
        return _Frozen()

    monkeypatch.setattr(model_module, 'build_model', fake_build_model)
    frozen = model_module.build_ref_model(
        name='teacher', model_id='teacher-id', model_config=ModelConfig(model='policy-id'), template='T',
        distributed_config=DistributedConfig(backend='megatron'), adapters=['/tmp/a'], model_type='qwen2')
    assert isinstance(frozen, _Frozen)
    assert calls['config'].model == 'teacher-id' and calls['config'].model_type == 'qwen2'
    assert calls['dist'].mode == 'ray' and calls['dist'].nproc_per_node == 1
    assert calls['kwargs']['remote_group'] == 'teacher'
    assert calls['adapter'] == ('teacher_adapter', '/tmp/a', False)

    with pytest.raises(ValueError, match='exactly one frozen adapter'):
        model_module.build_ref_model(
            name='teacher', model_id='t', model_config=ModelConfig(model='m'), template=None,
            distributed_config=DistributedConfig(), adapters=['a', 'b'])


# ---------------------------------------------------------------- reward model dispatch (S5-28/29)


def test_build_reward_model_dispatches_by_kind(monkeypatch):
    from swift.dev.builders import reward as reward_module
    from swift.dev.config import DistributedConfig, ModelConfig, TemplateConfig
    from swift.dev.rewards.scorer import ApiJudgeReward, GenerativeJudgeReward, RewardModel, RewardSpec

    common = dict(
        model_config=ModelConfig(model='policy'), template_config=TemplateConfig(),
        distributed_config=DistributedConfig(), rollout_config=SimpleNamespace())

    # api judge: no engine at all, the endpoint id rides along (the OpenAI client it builds
    # insists on a key, so give the env a fake one)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    scorer = reward_module.build_reward_model(
        RewardSpec(kind='api', model_id='http://judge/v1'), engine_args={'model': 'gpt'}, **common)
    assert isinstance(scorer, ApiJudgeReward)

    # generative_reuse shares the caller's sampler and does NOT own it
    sampler = object()
    scorer = reward_module.build_reward_model(
        RewardSpec(kind='generative_reuse', model_id='judge'), sampler=sampler, **common)
    assert isinstance(scorer, GenerativeJudgeReward)

    # scalar: the frozen seq-cls model is wrapped, and spec.display_name stamps the metric column
    monkeypatch.setattr(reward_module, '_build_frozen_reward_model', lambda *a, **k: (SimpleNamespace(), 'T'))
    spec = RewardSpec(kind='scalar', model_id='rm-0.6b')
    scorer = reward_module.build_reward_model(spec, remote_group='reward_0', **common)
    assert isinstance(scorer, RewardModel)
    assert scorer.display_name == 'rm-0.6b'
    assert scorer.placement_key == 'reward_0'
    # display_name falls back through adapter -> 'judge'
    assert RewardSpec(kind='scalar', adapter='/tmp/a').display_name == '/tmp/a'
    assert RewardSpec(kind='api').display_name == 'judge'


# ---------------------------------------------------------------- engine args / sampling (S5-34/35/42)


def test_build_engine_args_prefix_stripping():
    from swift.dev.builders.sampler import build_engine_args
    from swift.dev.config import InferConfig, RolloutConfig

    # 'pt' is an alias; the transformers engine only takes the batch ceiling
    assert build_engine_args('pt', InferConfig(max_batch_size=8), RolloutConfig()) == {'max_batch_size': 8}

    rollout = RolloutConfig(
        vllm_gpu_memory_utilization=0.5,
        vllm_enforce_eager=True,
        vllm_engine_kwargs={'override_me': 1},
        sglang_mem_fraction_static=0.7)
    args = build_engine_args('vllm', None, rollout)
    # the backend prefix is stripped off every non-None vllm_* field
    assert args['gpu_memory_utilization'] == 0.5 and args['enforce_eager'] is True
    # the other backend's knobs never leak across
    assert 'mem_fraction_static' not in args and not any(k.startswith(('vllm_', 'sglang_')) for k in args)
    # engine_kwargs is merged verbatim and its own key is excluded from the prefix sweep
    assert args['override_me'] == 1 and 'engine_kwargs' not in args


def test_to_sampling_params_maps_and_rejects():
    from swift.dev.builders.sampler import to_sampling_params
    from swift.dev.config import GenerationConfig

    gen = GenerationConfig(
        max_new_tokens=64, temperature=0.7, top_k=20, top_p=0.9, repetition_penalty=1.1,
        stop_words=['<|end|>'], logprobs=True, top_logprobs=5)
    params = to_sampling_params(gen)
    assert params.max_tokens == 64 and params.temperature == 0.7
    assert params.top_k == 20 and params.top_p == 0.9 and params.repetition_penalty == 1.1
    assert params.stop == ['<|end|>']
    # logprobs + top_logprobs collapse into ONE int (vLLM semantics)
    assert params.logprobs == 5
    # overrides win over the generation config
    assert to_sampling_params(gen, temperature=0.0).temperature == 0.0
    # unset fields stay at twinkle's defaults
    to_sampling_params(None)  # no generation config: twinkle defaults stand, nothing raises

    with pytest.raises(ValueError, match='num_beams'):
        to_sampling_params(GenerationConfig(num_beams=2))
    with pytest.raises(ValueError, match='structured_outputs_regex'):
        to_sampling_params(GenerationConfig(structured_outputs_regex='[0-9]+'))


def test_sampled_texts_groups_per_input():
    from swift.dev.builders.sampler import sampled_texts

    responses = [
        SimpleNamespace(sequences=[SimpleNamespace(decoded='a1'), SimpleNamespace(decoded='a2')]),
        SimpleNamespace(sequences=[SimpleNamespace(decoded='b1')]),
    ]
    assert sampled_texts(responses) == [['a1', 'a2'], ['b1']]
    assert sampled_texts([]) == []


# ---------------------------------------------------------------- RL sampler placement (S5-41)


def test_build_rl_sampler_placement_and_replay_knob(monkeypatch):
    from swift.dev.config import ModelConfig, RolloutConfig, TunerConfig
    from swift.dev.recipe import utils as recipe_utils

    calls: List[Dict[str, Any]] = []

    def fake_build_sampler(model_config, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr('swift.dev.builders.build_sampler', fake_build_sampler)
    monkeypatch.setattr('swift.dev.builders.build_engine_args', lambda *a, **k: {})

    model_config = ModelConfig(model='m')
    template = SimpleNamespace()

    # colocate: sleep knob pinned, no dedicated mesh, and the sampler lives in the MODEL group
    rollout = RolloutConfig(rollout_sampler='vllm', vllm_data_parallel_size=2)
    recipe_utils.build_rl_sampler(
        model_config=model_config, rollout_config=rollout, template=template, tuner_config=None, colocate=True)
    kw = calls[-1]
    assert kw['engine_args']['enable_sleep_mode'] is True
    assert kw['device_mesh'] is None and kw['remote_group'] == 'model'

    # disaggregated: its own DP-sized mesh in the sampler group
    recipe_utils.build_rl_sampler(
        model_config=model_config, rollout_config=rollout, template=template, tuner_config=None)
    kw = calls[-1]
    assert kw['remote_group'] == 'sampler' and kw['device_mesh'].data_world_size == 2

    # sglang reads its own replica knob and pins the memory saver instead of sleep mode
    sg = RolloutConfig(rollout_sampler='sglang', sglang_dp_size=3)
    recipe_utils.build_rl_sampler(
        model_config=model_config, rollout_config=sg, template=template, tuner_config=None, colocate=True)
    kw = calls[-1]
    assert kw['engine_args']['enable_memory_saver'] is True and 'enable_sleep_mode' not in kw['engine_args']

    # LoRA + adapter_snapshot opens the engine's hot-swap surface
    tuner = TunerConfig(tuner='lora', lora_rank=8)
    recipe_utils.build_rl_sampler(
        model_config=model_config,
        rollout_config=RolloutConfig(weight_sync_strategy='adapter_snapshot'),
        template=template,
        tuner_config=tuner)
    kw = calls[-1]
    assert kw['engine_args']['enable_lora'] is True
    assert kw['engine_args']['max_lora_rank'] == 8 and kw['engine_args']['max_loras'] == 2

    # the replay knob rides the RLConfig value passed in -- regression anchor for the crash where
    # build_rl_sampler read rollout_config.enable_sampling_replay, a field RolloutConfig never had
    # (every online-RL assembly died with AttributeError before the sampler was built)
    recipe_utils.build_rl_sampler(
        model_config=model_config, rollout_config=RolloutConfig(), template=template, tuner_config=None,
        enable_sampling_replay=True)
    assert calls[-1]['engine_args']['enable_sampling_replay'] is True
    recipe_utils.build_rl_sampler(
        model_config=model_config, rollout_config=RolloutConfig(), template=template, tuner_config=None)
    assert 'enable_sampling_replay' not in calls[-1]['engine_args']

    # a resource plan sizes the replica pool and stamps the elastic instance id
    plan = SimpleNamespace(sampler=SimpleNamespace(replica_count=4))
    recipe_utils.build_rl_sampler(
        model_config=model_config, rollout_config=RolloutConfig(), template=template, tuner_config=None,
        resource_plan=plan)
    kw = calls[-1]
    assert kw['instance_id'] == 'adaptive-sampler' and kw['device_mesh'].data_world_size == 4


def test_build_model_torch_dtype_overrides_on_a_copy(monkeypatch, tmp_path):
    """build_model's torch_dtype override must land on a COPY of the caller's ModelConfig.

    Regression anchor for `model_config = copy(model_config)` -- with the copy MODULE imported as
    `copy`, that line died with "'module' object is not callable" on every torch_dtype build.
    """
    from swift.dev.builders import model as model_module
    from swift.dev.config import DistributedConfig, ModelConfig

    seen: Dict[str, Any] = {}

    def fake_build(cfg, *args, **kwargs):
        seen['cfg'] = cfg
        return SimpleNamespace()

    monkeypatch.setattr(model_module, '_build_transformers_model', fake_build)
    # a local dir short-circuits the hub download the build does first
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    mc = ModelConfig(model=str(model_dir), torch_dtype='bfloat16')
    model_module.build_model(mc, DistributedConfig(), torch_dtype='float16')
    assert seen['cfg'].torch_dtype == 'float16'
    # the caller's config is untouched -- the override rode a copy
    assert mc.torch_dtype == 'bfloat16'
