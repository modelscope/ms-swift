"""RL recipe unit tests (cheap tier).

Pure-function / pure-Python coverage that runs in the normal suite without Ray, vLLM or GPUs:
  - :func:`plan_rl_device_groups` colocate vs heterogeneous placement + its validation;
  - :class:`PreferenceLoop` chosen/rejected interleaving (the layout the DPO family's split relies on);
  - :class:`GKDLoop` prompt-window wrap-around;
  - :func:`configure_rlhf_loss` rlhf_type -> loss mapping (skipped when twinkle is unavailable).

The heavy weight-sync / colocate / e2e paths need Ray + multi-GPU + vLLM and live behind
``@pytest.mark.slow`` mirroring ``twinkle/tests/sampler/test_weight_sync.py``; they are not here.
"""
import pytest


# ----------------------------------------------------------------------
# plan_rl_device_groups (pure function, no twinkle)
# ----------------------------------------------------------------------
def test_plan_device_groups_colocate_shares_one_group():
    """colocate: trainer + sampler share ONE 'model' group over the trainer's GPUs; colocate=True."""
    from swift.dev.recipe.run_grpo import plan_rl_device_groups

    groups, sampler_group, colocate = plan_rl_device_groups(4, 'colocate', 2)
    assert groups == [('model', [0, 1, 2, 3])]
    assert sampler_group == 'model'  # sampler placed in the shared group
    assert colocate is True


def test_plan_device_groups_heterogeneous_disjoint_ranks():
    """server/None: a 'model' group then a DISJOINT 'sampler' group after it; colocate=False."""
    from swift.dev.recipe.run_grpo import plan_rl_device_groups

    groups, sampler_group, colocate = plan_rl_device_groups(4, 'server', 2)
    assert groups == [('model', [0, 1, 2, 3]), ('sampler', [4, 5])]
    assert sampler_group == 'sampler'
    assert colocate is False
    # ranks must not overlap (NCCL weight-sync requires distinct devices).
    model_ranks, sampler_ranks = groups[0][1], groups[1][1]
    assert set(model_ranks).isdisjoint(sampler_ranks)


def test_plan_device_groups_validation():
    """Colocate that cannot fit the sampler, and non-positive counts, are rejected."""
    from swift.dev.recipe.run_grpo import plan_rl_device_groups

    with pytest.raises(ValueError):
        plan_rl_device_groups(2, 'colocate', 4)  # sampler bigger than the shared trainer GPUs
    with pytest.raises(ValueError):
        plan_rl_device_groups(0, 'server', 1)  # no trainer GPUs
    with pytest.raises(ValueError):
        plan_rl_device_groups(2, 'server', 0)  # no sampler GPUs


# ----------------------------------------------------------------------
# PreferenceLoop interleaving (pure Python; stub template/model)
# ----------------------------------------------------------------------
class _StubTemplate:
    """A template whose encode returns a fixed preference-encoded dict (chosen_*/rejected_*)."""

    def __init__(self, encoded):
        self._encoded = encoded

    def encode(self, row):
        return dict(self._encoded)


def _preference_loop(rlhf_type, encoded):
    from swift.dev.recipe.run_dpo import PreferenceLoop

    return PreferenceLoop(model=None, dataloader=None, template=_StubTemplate(encoded), rlhf_type=rlhf_type)


def test_preference_interleave_order_and_prefix_strip():
    """Two rows -> [chosen_1, rejected_1, chosen_2, rejected_2] with prefixes stripped, `length` dropped."""
    loop = _preference_loop(
        'dpo', {
            'chosen_input_ids': [1, 2],
            'chosen_labels': [-100, 2],
            'chosen_length': 2,
            'rejected_input_ids': [3, 4],
            'rejected_labels': [-100, 4],
            'rejected_length': 2,
        })
    features = loop._interleave([{'x': 0}, {'x': 1}])
    assert len(features) == 4  # 2 pairs -> 4 interleaved features (even, as the DPO split needs)
    assert features[0] == {'input_ids': [1, 2], 'labels': [-100, 2]}  # chosen_1
    assert features[1] == {'input_ids': [3, 4], 'labels': [-100, 4]}  # rejected_1
    assert features[2] == features[0] and features[3] == features[1]
    assert 'length' not in features[0]  # bookkeeping stripped


def test_preference_reward_pair_has_no_labels():
    """RM (seq_cls) encodes without labels -> the pair features carry only input_ids (RewardLoss reads
    logits, not logps)."""
    loop = _preference_loop('rm', {'chosen_input_ids': [1, 2], 'rejected_input_ids': [3, 4]})
    features = loop._interleave([{'x': 0}])
    assert features == [{'input_ids': [1, 2]}, {'input_ids': [3, 4]}]
    assert loop._is_reward is True


def test_preference_missing_side_is_fatal():
    """A row that did not encode a rejected sequence must fail loudly (paired data is required)."""
    loop = _preference_loop('dpo', {'chosen_input_ids': [1, 2], 'chosen_labels': [-100, 2]})
    with pytest.raises(ValueError):
        loop._interleave([{'x': 0}])


# ----------------------------------------------------------------------
# GKDLoop prompt window (pure Python)
# ----------------------------------------------------------------------
def test_gkd_prompt_batch_wraps():
    """The per-step prompt window rolls forward and wraps around the prompt list."""
    from swift.dev.recipe.run_gkd import GKDLoop

    prompts = [[{'role': 'user', 'content': str(i)}] for i in range(3)]
    loop = GKDLoop(model=None, teacher=None, template=None, prompts=prompts, batch_size=2)
    assert loop._prompt_batch(0) == [prompts[0], prompts[1]]
    assert loop._prompt_batch(1) == [prompts[2], prompts[0]]  # wraps at 3


def test_gkd_dataset_mix_and_teacher_offload(tmp_path):
    """lmbda selects dataset completions and an offloaded teacher brackets every scoring call."""
    torch = pytest.importorskip('torch')
    from swift.dev.recipe.run_gkd import GKDLoop

    class Teacher:

        def __init__(self):
            self.events = []

        def offload_to_cpu(self):
            self.events.append('offload')

        def reload_to_gpu(self):
            self.events.append('reload')

        def forward_only(self, *, inputs, return_logits):
            assert return_logits is True
            self.events.append(('forward', inputs))
            return {'logits': torch.zeros(len(inputs), 2, 3)}

    teacher = Teacher()
    features = [{'input_ids': [1, 2], 'labels': [-100, 2]}, {'input_ids': [3, 4], 'labels': [-100, 4]}]
    prompts = [[{'role': 'user', 'content': str(i)}] for i in range(2)]
    loop = GKDLoop(
        model=None,
        teacher=teacher,
        template=None,
        prompts=prompts,
        dataset_features=features,
        lmbda=0.0,
        sft_alpha=0.5,
        batch_size=1,
        offload_teacher_model=True,
        output_dir=str(tmp_path))

    assert loop._uses_student_generation(0) is False
    batch = loop._dataset_batch(0)
    batch[0]['input_ids'][0] = 99
    assert features[0]['input_ids'][0] == 1
    logits = loop._teacher_logits(loop._dataset_batch(0))
    assert logits.shape == (1, 2, 3)
    assert teacher.events == ['offload', 'reload', ('forward', loop._dataset_batch(0)), 'offload']


def test_gkd_remote_teacher_routes_and_returns_aligned_topk(tmp_path):
    """Remote GKD preserves response token ids, routes by dataset tag, and returns sparse teacher tensors."""
    torch = pytest.importorskip('torch')
    from swift.dev.recipe.run_gkd import GKDLoop, _RemoteGKDTeacher

    class Response:

        def __init__(self, offset):
            self.prompt_logprobs = [None] + [{
                str(offset + position): {
                    'logprob': -0.1,
                },
                str(offset + position + 10): {
                    'logprob': -0.2,
                },
            } for position in range(3)]

    class Client:

        def __init__(self, offset):
            self.offset = offset
            self.calls = []

        def infer(self, requests, *, request_config, use_tqdm):
            self.calls.append((requests, request_config, use_tqdm))
            return [Response(self.offset) for _ in requests]

    clients = {'http://math': Client(100), 'http://code': Client(200)}
    teacher = _RemoteGKDTeacher(
        '[{"url":"http://math","tags":["math"]},{"url":"http://code","tags":["code"]}]',
        2,
        client_factory=clients.__getitem__)
    prompts = [[{'role': 'user', 'content': 'm'}], [{'role': 'user', 'content': 'c'}]]
    features = [
        {'input_ids': [1, 11, 12], 'labels': [-100, 11, 12]},
        {'input_ids': [2, 21, 22], 'labels': [-100, 21, 22]},
    ]
    extras = [{'dataset': 'math'}, {'dataset': 'code'}]
    loop = GKDLoop(
        model=None,
        teacher=teacher,
        template=None,
        prompts=prompts,
        dataset_features=features,
        prompt_extras=extras,
        dataset_messages=[p + [{'role': 'assistant', 'content': 'old'}] for p in prompts],
        lmbda=0.0,
        batch_size=2,
        teacher_tag_key='dataset',
        output_dir=str(tmp_path))

    requests = loop._teacher_requests(features, loop._messages_batch(0, dataset=True), extras)
    assert requests[0].messages[-1]['content'] == [11, 12]
    result = teacher.score(features, requests, extras, 'dataset')
    assert result['teacher_topk_logprobs'].shape == (2, 3, 2)
    assert result['teacher_topk_indices'].tolist()[0][0] == [100, 110]
    assert result['teacher_topk_indices'].tolist()[1][0] == [200, 210]
    assert len(clients['http://math'].calls[0][0]) == len(clients['http://code'].calls[0][0]) == 1
    for client in clients.values():
        _, request_config, use_tqdm = client.calls[0]
        assert request_config.prompt_logprobs == 2
        assert request_config.max_tokens == 1
        assert request_config.temperature == 0.0
        assert use_tqdm is False
    assert torch.isfinite(result['teacher_topk_logprobs']).all()


def test_grpo_remote_teacher_routes_preserves_tokens_and_returns_sampled_logps():
    """Remote GRPO routes by tag and returns teacher logps for the exact sampled token IDs."""
    from types import SimpleNamespace

    from swift.dev.recipe.grpo import _RemoteGRPOTeacher

    class Response:

        def __init__(self, response_ids, offset):
            token_ids = [900 + offset, *response_ids, 999 + offset]
            self.prompt_logprobs = [None] + [{
                str(token_id): {
                    'logprob': -(offset + position) / 100,
                }
            } for position, token_id in enumerate(token_ids)]

    class Client:

        def __init__(self, offset):
            self.offset = offset
            self.calls = []

        def infer(self, requests, *, request_config, use_tqdm):
            self.calls.append((requests, request_config, use_tqdm))
            return [Response(request.messages[-1]['content'], self.offset) for request in requests]

    clients = {'http://math': Client(100), 'http://code': Client(200)}
    teacher = _RemoteGRPOTeacher(
        '[{"url":"http://math","tags":["math"]},{"url":"http://code","tags":["code"]}]',
        client_factory=clients.__getitem__)
    prompts = [[{'role': 'user', 'content': 'm'}], [{'role': 'user', 'content': 'c'}]]
    samples = [
        SimpleNamespace(
            prompt_id='0', response_token_ids=[[11, 12]], extra={
                'dataset': 'math',
                'teacher_prompt': 'privileged',
            }),
        SimpleNamespace(prompt_id='1', response_token_ids=[[21, 22]], extra={'dataset': 'code'}),
    ]

    logps = teacher.score(samples, prompts, template=None, tag_key='dataset')

    assert logps[0] == pytest.approx([-1.01, -1.02])
    assert logps[1] == pytest.approx([-2.01, -2.02])
    assert clients['http://math'].calls[0][0][0].messages == [
        {'role': 'user', 'content': 'privileged'},
        {'role': 'assistant', 'content': [11, 12]},
    ]
    assert clients['http://code'].calls[0][0][0].messages[-1]['content'] == [21, 22]
    for client in clients.values():
        _, request_config, use_tqdm = client.calls[0]
        assert request_config.prompt_logprobs == 0
        assert request_config.max_tokens == 1
        assert request_config.temperature == 0.0
        assert use_tqdm is False


def test_configure_frozen_adapter_binds_non_trainable_group():
    """Auxiliary adapters are loaded frozen and receive their own processor/template group."""
    from swift.dev.recipe.assembly import configure_frozen_adapter

    class Model:

        def __init__(self):
            self.calls = []

        def add_adapter_to_model(self, *args, **kwargs):
            self.calls.append(('adapter', args, kwargs))

        def set_processor(self, *args, **kwargs):
            self.calls.append(('processor', args, kwargs))

        def set_template(self, *args, **kwargs):
            self.calls.append(('template', args, kwargs))

    model = Model()
    template = object()
    assert configure_frozen_adapter(model, template, ['adapter-path'], role='teacher') is model
    assert model.calls[0] == ('adapter', ('teacher_adapter', 'adapter-path'), {'is_trainable': False})
    assert model.calls[1][0] == 'processor'
    assert model.calls[1][2] == {'adapter_name': 'teacher_adapter'}
    assert model.calls[2] == ('template', (template, ), {'adapter_name': 'teacher_adapter'})


# ----------------------------------------------------------------------
# configure_rlhf_loss mapping (needs twinkle's loss registry)
# ----------------------------------------------------------------------
class _StubModel:
    """Captures the loss instance set via set_loss so the mapping can be asserted."""

    def __init__(self):
        self.loss = None

    def set_loss(self, loss):
        self.loss = loss


@pytest.mark.parametrize('rlhf_type, cls_name', [
    ('dpo', 'DPOLoss'),
    ('kto', 'DPOLoss'),
    ('cpo', 'CPOLoss'),
    ('orpo', 'ORPOLoss'),
    ('simpo', 'SimPOLoss'),
    ('grpo', 'GRPOLoss'),
    ('gkd', 'GKDLoss'),
    ('rm', 'RewardLoss'),
    ('ppo', 'GRPOLoss'),
])
def test_configure_rlhf_loss_maps_type_to_loss(rlhf_type, cls_name):
    """Each rlhf_type resolves to its twinkle loss class (ppo's POLICY loss is the shared GRPO clip)."""
    pytest.importorskip('twinkle.loss')
    from swift.dev.config import RLHFConfig
    from swift.dev.loss import configure_rlhf_loss

    model = _StubModel()
    configure_rlhf_loss(model, RLHFConfig(rlhf_type=rlhf_type))
    assert type(model.loss).__name__ == cls_name


def test_configure_ppo_value_loss_sets_value_loss():
    """PPO's critic gets the clipped value loss, carrying cliprange_value / vf_coef."""
    pytest.importorskip('twinkle.loss')
    from swift.dev.config import RLHFConfig
    from swift.dev.loss import configure_ppo_value_loss

    value_model = _StubModel()
    configure_ppo_value_loss(value_model, RLHFConfig(rlhf_type='ppo', cliprange_value=0.3, vf_coef=0.5))
    assert type(value_model.loss).__name__ == 'PPOValueLoss'
    assert value_model.loss.cliprange_value == 0.3
    assert value_model.loss.vf_coef == 0.5


def test_ppo_checkpoint_saves_policy_and_value_and_validates_resume(monkeypatch, tmp_path):
    """A PPO checkpoint is one atomic policy directory with a nested, step-aligned critic."""
    from swift.dev.recipe.run_ppo import PPOLoop

    calls = []

    def fake_save(model, name, **kwargs):
        calls.append((model, name, kwargs))

    monkeypatch.setattr('swift.dev.recipe.train_loop.save_training_checkpoint', fake_save)
    loop = object.__new__(PPOLoop)
    loop.model = 'policy'
    loop.value_model = 'value'
    loop.output_dir = str(tmp_path)
    loop.global_step = 7
    loop.no_save_optim = True
    loop.no_save_rng = True

    checkpoint_dir = loop.save('checkpoint-7')
    assert checkpoint_dir == str(tmp_path / 'checkpoint-7')
    assert calls == [
        ('policy', 'checkpoint-7', {
            'output_dir': str(tmp_path),
            'consumed_train_samples': 7,
            'no_save_optim': True,
            'no_save_rng': True,
        }),
        ('value', 'value_model', {
            'output_dir': checkpoint_dir,
            'consumed_train_samples': 7,
            'no_save_optim': True,
            'no_save_rng': True,
        }),
    ]

    loop.resume({'consumed_train_samples': 7}, value_state={'consumed_train_samples': 7})
    assert loop.global_step == 7
    with pytest.raises(ValueError, match='different completed-step counts'):
        loop.resume({'consumed_train_samples': 7}, value_state={'consumed_train_samples': 6})


def test_configure_rlhf_loss_refuses_unknown():
    """An unknown rlhf_type is a ValueError (ppo is now supported, so it is no longer refused)."""
    pytest.importorskip('twinkle.loss')
    from swift.dev.config import RLHFConfig
    from swift.dev.loss import configure_rlhf_loss

    cfg = RLHFConfig(rlhf_type='dpo')
    cfg.rlhf_type = 'no_such_type'
    with pytest.raises(ValueError):
        configure_rlhf_loss(_StubModel(), cfg)


def test_gkd_sft_alpha_only_applies_to_dataset_batches():
    """GKD's supervised term is gated by the selected data source."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('twinkle.loss')

    from swift.dev.config import RLHFConfig
    from swift.dev.loss import configure_rlhf_loss

    model = _StubModel()
    configure_rlhf_loss(model, RLHFConfig(rlhf_type='gkd', beta=0.5, sft_alpha=0.75))
    labels = torch.tensor([[-100, 1, 2]])
    student_logits = torch.randn(1, 3, 4, requires_grad=True)
    teacher_logits = torch.randn(1, 3, 4)
    base = model.loss(
        {'labels': labels}, {'logits': student_logits}, teacher_logits=teacher_logits, apply_sft_loss=False)['loss']
    mixed = model.loss(
        {'labels': labels}, {'logits': student_logits}, teacher_logits=teacher_logits, apply_sft_loss=True)['loss']
    expected_ce = torch.nn.functional.cross_entropy(
        student_logits.reshape(-1, 4), labels.reshape(-1), ignore_index=-100)
    assert mixed.detach().item() == pytest.approx((base + 0.75 * expected_ce).detach().item())


def test_advanced_grpo_loss_is_serializable_and_combines_objectives():
    """The CHORD/SDAR wrapper is a real, module-level Loss and accepts a mixed RL+SFT batch."""
    import pickle

    torch = pytest.importorskip('torch')
    pytest.importorskip('twinkle.loss')
    from twinkle.loss import Loss

    from swift.dev.config import RLHFConfig
    from swift.dev.loss import configure_rlhf_loss

    model = _StubModel()
    configure_rlhf_loss(
        model,
        RLHFConfig(rlhf_type='grpo', chord_sft_dataset=['dummy'], sdar_loss_coef=0.25, sdar_gate_beta=2.0))
    assert isinstance(model.loss, Loss)
    restored = pickle.loads(pickle.dumps(model.loss))
    assert type(restored) is type(model.loss)

    labels = torch.tensor([[-100, 1, 2], [-100, 2, 3]])
    logps = torch.tensor([[0.0, -0.7, -0.8], [0.0, -0.4, -0.5]], requires_grad=True)
    logits = torch.randn(2, 3, 4, requires_grad=True)
    result = restored(
        {'labels': labels},
        {'logps': logps, 'logits': logits},
        old_logps=[[-0.9, -1.0]],
        ref_logps=[[-0.8, -0.9]],
        teacher_logps=[[-0.6, -0.7]],
        advantages=[1.0],
        chord_count=1,
        chord_mu=0.2,
        chord_phi=True)
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    assert logps.grad is not None and logits.grad is not None


def test_grpo_tensor_alignment_and_reward_weights_keep_device():
    """Tensor labels never enter Python truth testing, and reward weights follow reward placement."""
    torch = pytest.importorskip('torch')
    from swift.dev.recipe.grpo import GRPOLoop

    labels = torch.tensor([-100, 7, 8, -100])
    assert GRPOLoop._response_positions({'labels': labels}) == [1, 2]

    loop = object.__new__(GRPOLoop)
    loop.reward_weights = [0.25, 2.0]
    rewards = torch.tensor([[4.0, 3.0], [8.0, float('nan')]])
    weighted = loop._weighted_rewards(rewards)
    assert weighted.device == rewards.device
    assert weighted.tolist() == pytest.approx([7.0, 2.0])


def test_reward_model_default_plugin_encodes_rows_and_scores_batch():
    """The default RM adapter uses its own template and the twinkle forward-only contract."""
    torch = pytest.importorskip('torch')
    from swift.dev.reward import build_reward_model_plugins, compute_reward_model_scores

    class Template:

        def __init__(self):
            self.rows = []

        def encode(self, row):
            self.rows.append(row)
            return {'input_ids': [len(row['messages'])]}

    class Model:

        def forward_only(self, *, inputs, return_logits):
            assert return_logits is True
            return {'logits': torch.tensor([[item['input_ids'][0]] for item in inputs])}

    template = Template()
    plugins, names = build_reward_model_plugins([Model()], [template])
    rows = [{'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]}]
    scores = compute_reward_model_scores(rows, plugins)

    assert scores.tolist() == [[2.0]]
    assert names == ['_DefaultRewardModelPlugin']
    assert template.rows == rows
    assert template.rows[0] is not rows[0]


def test_grpo_combines_rule_and_reward_model_columns():
    """Rule rewards precede RM rewards and RM inputs retain prompt, completion, and dataset columns."""
    from types import SimpleNamespace

    from swift.dev.recipe.grpo import GRPOLoop

    captured = []

    def rule(completions, solution):
        assert completions == ['answer']
        assert solution == ['42']
        return [1.5]

    def reward_model(*, inputs):
        captured.extend(inputs)
        return [2.5]

    loop = object.__new__(GRPOLoop)
    loop.prompts = [[{'role': 'user', 'content': 'question'}]]
    loop.reward_funcs = [rule]
    loop.reward_model_plugins = [reward_model]
    loop.reward_fn = lambda _sample: -1.0
    sample = SimpleNamespace(prompt_id='0', decoded='answer', extra={'solution': '42'})

    rewards = loop._score([sample])
    assert rewards.tolist() == [[1.5, 2.5]]
    assert captured == [{
        'solution': '42',
        'messages': [
            {'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': 'answer'},
        ],
    }]


def test_grpo_appends_gym_reward_and_weights_every_column():
    """Gym total_reward is the final reward column and participates in reward_weights."""
    from types import SimpleNamespace

    from swift.dev.recipe.grpo import GRPOLoop

    captured = []

    def rule(completions, solution):
        return [1.0]

    def reward_model(*, inputs):
        captured.extend(inputs)
        return [2.0]

    loop = object.__new__(GRPOLoop)
    loop.prompts = [[{'role': 'user', 'content': 'unused'}]]
    loop.reward_funcs = [rule]
    loop.reward_model_plugins = [reward_model]
    loop.reward_fn = lambda _sample: -1.0
    loop.reward_weights = [1.0, 2.0, 3.0]
    loop.rlhf_config = SimpleNamespace(use_gym_env=True)
    messages = [
        {'role': 'user', 'content': 'question'},
        {'role': 'assistant', 'content': 'first'},
        {'role': 'user', 'content': 'observation'},
        {'role': 'assistant', 'content': 'second'},
    ]
    sample = SimpleNamespace(
        prompt_id='0',
        decoded='second',
        extra={'solution': '42'},
        messages=messages,
        rollout_infos={'total_reward': 3.0})

    rewards = loop._score([sample])
    assert rewards.tolist() == [[1.0, 2.0, 3.0]]
    assert loop._weighted_rewards(rewards).tolist() == [14.0]
    assert captured == [{'solution': '42', 'messages': messages}]

    sample.rollout_infos = {}
    with pytest.raises(RuntimeError, match='no total_reward'):
        loop._score([sample])


def test_grpo_builds_independent_reward_template(monkeypatch):
    """Reward model metadata and --reward_template drive an independent frozen scorer assembly."""
    from types import SimpleNamespace

    import swift.dev.builders as builders
    import swift.dev.recipe.assembly as assembly_module
    import swift.model as swift_model
    from swift.dev.config import ModelConfig, RLHFConfig, TemplateConfig
    from swift.dev.recipe.run_grpo import _build_reward_model_scorers

    seen = {}
    reward_model = SimpleNamespace(model=object(), config=SimpleNamespace(_name_or_path='rm-a'))
    reward_template = SimpleNamespace(max_length=123, use_model=True)

    monkeypatch.setattr(
        swift_model, 'get_model_info_meta',
        lambda *args, **kwargs: (SimpleNamespace(model_type='rm-type', task_type='seq_cls', num_labels=1), None))
    monkeypatch.setattr(swift_model, 'get_model_processor', lambda *args, **kwargs: (None, 'rm-processor'))

    def build_template(config, processor, *, task_type):
        seen['template_config'] = config
        seen['processor'] = processor
        seen['task_type'] = task_type
        return reward_template

    def build_model(config, distributed):
        seen['model_config'] = config
        seen['distributed'] = distributed
        return reward_model

    def freeze(model, template, adapters, *, role):
        seen['freeze'] = (model, template, adapters, role)
        return model

    monkeypatch.setattr(builders, 'build_template', build_template)
    monkeypatch.setattr(builders, 'build_model', build_model)
    monkeypatch.setattr(assembly_module, 'configure_frozen_adapter', freeze)

    policy_template_config = TemplateConfig(template='policy', max_length=2048)
    plugins, names = _build_reward_model_scorers(
        ModelConfig(model='policy'), policy_template_config,
        RLHFConfig(
            rlhf_type='grpo',
            reward_model=['rm-a'],
            reward_model_type=['rm-type'],
            reward_model_revision=['rev'],
            reward_model_plugin=['default'],
            reward_template=['judge'],
            reward_adapters=['adapter']))

    assert len(plugins) == 1 and names == ['rm-a']
    assert seen['template_config'].template == 'judge'
    assert seen['template_config'].max_length is None
    assert policy_template_config.template == 'policy' and policy_template_config.max_length == 2048
    assert seen['processor'] == 'rm-processor' and seen['task_type'] == 'seq_cls'
    assert seen['model_config'].task_type == 'seq_cls' and seen['model_config'].num_labels == 1
    assert seen['freeze'] == (reward_model, reward_template, ['adapter'], 'reward')
    assert reward_template.model is reward_model.model


class _DynamicSample:

    def __init__(self, reward, marker):
        self.reward = reward
        self.marker = marker
        self.prompt_id = None


class _DynamicRollout:

    def __init__(self):
        self.calls = []
        self.sync_count = 0
        self.finish_count = 0

    def sync_weights(self):
        self.sync_count += 1

    def finish_generate(self):
        self.finish_count += 1

    def generate(self, prompts, *, num_samples, sampling_params, prompt_extras):
        self.calls.append((prompts, prompt_extras, num_samples, sampling_params))
        if len(self.calls) == 1:
            return [
                _DynamicSample(0.0, 'p0-old-a'),
                _DynamicSample(0.0, 'p0-old-b'),
                _DynamicSample(0.0, 'p1-a'),
                _DynamicSample(1.0, 'p1-b'),
            ]
        return [_DynamicSample(1.0, 'p0-new-a'), _DynamicSample(2.0, 'p0-new-b')]


def test_grpo_dynamic_sampling_preserves_extras_and_group_order(tmp_path):
    """DAPO retries only zero-variance groups and restores original prompt ordering."""
    from swift.dev.config import RLHFConfig
    from swift.dev.recipe.grpo import GRPOLoop

    prompts = [[{'role': 'user', 'content': 'a'}], [{'role': 'user', 'content': 'b'}]]
    extras = [{'solution': 'A'}, {'solution': 'B'}]
    rollout = _DynamicRollout()
    loop = GRPOLoop(
        model=None,
        rollout_engine=rollout,
        prompts=prompts,
        prompt_extras=extras,
        num_generations=2,
        rlhf_config=RLHFConfig(
            rlhf_type='grpo', dynamic_sample=True, max_resample_times=1, reward_funcs=['dummy']),
        reward_fn=lambda sample: sample.reward,
        output_dir=str(tmp_path))

    samples, rewards = loop._dynamic_rollout()
    assert [sample.marker for sample in samples] == ['p0-new-a', 'p0-new-b', 'p1-a', 'p1-b']
    assert [sample.prompt_id for sample in samples] == ['0', '0', '1', '1']
    assert rewards[:, 0].tolist() == pytest.approx([1.0, 2.0, 0.0, 1.0])
    assert rollout.calls[0][1] == extras
    assert rollout.calls[1][1] == [extras[0]]
    assert rollout.sync_count == rollout.finish_count == 2


# ----------------------------------------------------------------------
# Advanced GRPO controls (pure tensor tests)
# ----------------------------------------------------------------------
def _advanced_grpo_loss(**overrides):
    from swift.dev.config import RLHFConfig
    from swift.dev.loss import configure_rlhf_loss

    config = RLHFConfig(rlhf_type='grpo', **overrides)
    model = _StubModel()
    configure_rlhf_loss(model, config)
    return model.loss


def test_grpo_dual_clip_preserves_unclamped_ppo_branch():
    """delta clips coef_1 only; negative advantages still use PPO's raw-ratio clipped branch."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('twinkle.loss')

    loss = _advanced_grpo_loss(delta=0.5, epsilon=0.2)
    logps = torch.tensor([[torch.log(torch.tensor(2.0))]], requires_grad=True)
    result = loss(
        {'labels': torch.tensor([[1]])},
        {'logps': logps},
        old_logps=[[0.0]],
        advantages=[-1.0])
    assert result['loss'].item() == pytest.approx(1.2)


def test_grpo_entropy_quantile_masks_numerator_not_denominator():
    """Top-entropy filtering zeroes low-entropy token losses without renormalizing over survivors."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('twinkle.loss')

    loss = _advanced_grpo_loss(top_entropy_quantile=0.5, log_entropy=True)
    logps = torch.zeros(1, 2, requires_grad=True)
    result = loss(
        {'labels': torch.tensor([[1, 2]])},
        {'logps': logps, 'entropies': torch.tensor([[0.0, 1.0]])},
        old_logps=[[0.0, 0.0]],
        advantages=[1.0])
    assert result['loss'].item() == pytest.approx(-0.5)
    assert result['channel_loss']['entropy'].tolist() == pytest.approx([1.0, 2.0])


def test_grpo_sequence_token_is_has_sequence_values_and_token_gradients():
    """GSPO-token uses detached sequence ratios numerically while retaining per-token policy gradients."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('twinkle.loss')

    loss = _advanced_grpo_loss(importance_sampling_level='sequence_token', epsilon=10.0)
    logps = torch.tensor([[torch.log(torch.tensor(2.0)), 0.0]], requires_grad=True)
    result = loss(
        {'labels': torch.tensor([[1, 2]])},
        {'logps': logps},
        old_logps=[[0.0, 0.0]],
        advantages=[1.0])
    assert result['loss'].item() == pytest.approx(-(2.0**0.5))
    result['loss'].backward()
    assert logps.grad.tolist()[0] == pytest.approx([-(2.0**0.5) / 2] * 2)


def test_grpo_fipo_future_kl_decay_matches_reference_formula():
    """FIPO position t receives its own log-ratio plus geometrically decayed future ratios."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('twinkle.loss')

    loss = _advanced_grpo_loss(
        loss_type=['fipo'],
        fipo_decay_rate=1.0,
        fipo_clip_range=None,
        fipo_safety_threshold=None)
    log_ratio = torch.log(torch.tensor([[2.0, 0.5]]))
    weights = loss._fipo_weights(
        log_ratio,
        torch.exp(log_ratio),
        torch.ones_like(log_ratio),
        torch.ones_like(log_ratio, dtype=torch.bool))
    assert weights.tolist()[0] == pytest.approx([2.0**0.5, 0.5])


@pytest.mark.parametrize('mode, expected', [
    ('token_truncate', [2.0, 0.5]),
    ('token_mask', [0.0, 0.5]),
    ('sequence_truncate', [2.0**0.5, 2.0**0.5]),
    ('sequence_mask', [4.0, 0.5]),
])
def test_grpo_rollout_importance_sampling_modes(mode, expected):
    """All four rollout IS modes produce the documented token/sequence correction weights."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('twinkle.loss')

    loss = _advanced_grpo_loss(
        rollout_importance_sampling_mode=mode,
        rollout_importance_sampling_threshold=2.0)
    ratios = torch.tensor([[4.0, 0.5]])
    weights = loss._rollout_weights(torch.log(ratios), torch.ones_like(ratios, dtype=torch.bool))
    assert weights.tolist()[0] == pytest.approx(expected)


def test_grpo_truncation_alignment_and_offpolicy_mask_gradients():
    """Response-only tensors align before truncation; masked negative off-policy rows have zero gradient."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('twinkle.loss')

    loss = _advanced_grpo_loss(
        beta=0.1,
        overlong_filter=True,
        log_rollout_offpolicy_metrics=True,
        off_policy_sequence_mask_delta=1.0)
    logps = torch.tensor([[-9.0, -2.0, -2.0], [-9.0, 0.0, 0.0]], requires_grad=True)
    result = loss(
        {'labels': torch.tensor([[-100, 1, 2], [-100, 3, 4]])},
        {'logps': logps},
        old_logps=[[0.0, 0.0], [0.0, 0.0]],
        ref_logps=[[-0.1, -0.1], [-0.1, -0.1]],
        rollout_logps=[[0.0, 0.0], [0.0, 0.0]],
        advantages=[-1.0, 1.0],
        truncated=[True, False])
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    assert logps.grad[0].abs().sum().item() == 0.0
    assert logps.grad[1, 1:].abs().sum().item() > 0.0


def test_grpo_num_iterations_reuses_rollout_without_expanding_max_steps(tmp_path):
    """num_iterations replays one rollout, while max_steps remains the optimizer-step budget."""
    from types import SimpleNamespace

    from swift.dev.recipe.grpo import GRPOLoop

    class Model:

        def __init__(self):
            self.forward_calls = 0
            self.step_calls = 0

        def forward_backward(self, **kwargs):
            self.forward_calls += 1

        def clip_grad_and_step(self, **kwargs):
            self.step_calls += 1

    class Group:

        @staticmethod
        def do_grad_sync(ga):
            return True

        @staticmethod
        def calculate_metrics(reset):
            return {'loss': '1.0'}

    class Tracker:

        @staticmethod
        def log(data, step):
            return data

        @staticmethod
        def should_log(step):
            return False

        @staticmethod
        def close():
            return None

    rollout_calls = []
    model = Model()
    loop = object.__new__(GRPOLoop)
    loop.model = model
    loop.rlhf_config = SimpleNamespace(
        num_iterations=4,
        overlong_filter=False,
        chord_enable_phi_function=False,
        sdar_loss_coef=0.0)
    loop.gradient_accumulation_steps = 1
    loop.max_steps = 3
    loop.max_grad_norm = 1.0
    loop.global_step = 0
    loop.micro_step = 0
    loop.history = []
    loop.chord_features = []
    loop.tracker = Tracker()
    loop._active_group = lambda: Group()
    loop._sync_reference = lambda: None
    sample = SimpleNamespace(input_feature={'labels': [1]}, truncated=False)
    batch = [{
        'sample': sample,
        'advantage': 1.0,
        'old_logps': [0.0],
        'rollout_logps': [0.0],
        'ref_logps': None,
        'teacher_logps': None,
    }]

    def rollout_step():
        rollout_calls.append(True)
        return batch

    loop._rollout_step = rollout_step
    assert len(loop.fit()) == 3
    assert len(rollout_calls) == 1
    assert model.forward_calls == model.step_calls == 3


# ----------------------------------------------------------------------
# GAEAdvantage (twinkle.advantage) -- PPO's per-token advantage/return
# ----------------------------------------------------------------------
def test_gae_full_lambda_credits_all_tokens():
    """gamma=lam=1: the terminal reward propagates fully back, returns = advantage + value."""
    pytest.importorskip('twinkle.advantage')
    from twinkle.advantage import GAEAdvantage

    advantages, returns = GAEAdvantage()([0.0, 0.0, 1.0], [0.5, 0.5, 0.5], gamma=1.0, lam=1.0)
    assert advantages == pytest.approx([0.5, 0.5, 0.5])
    assert returns == pytest.approx([1.0, 1.0, 1.0])


def test_gae_zero_lambda_is_one_step_td():
    """lam=0 collapses GAE to the one-step TD error delta_t = r_t + gamma*V_{t+1} - V_t."""
    pytest.importorskip('twinkle.advantage')
    from twinkle.advantage import GAEAdvantage

    advantages, returns = GAEAdvantage()([0.0, 0.0, 1.0], [0.5, 0.5, 0.5], gamma=1.0, lam=0.0)
    assert advantages == pytest.approx([0.0, 0.0, 0.5])
    assert returns == pytest.approx([0.5, 0.5, 1.0])


def test_gae_length_matches_response_and_requires_values():
    """The advantage/return lists are one-per-token; missing values is a hard error."""
    pytest.importorskip('twinkle.advantage')
    from twinkle.advantage import GAEAdvantage

    advantages, returns = GAEAdvantage()([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], gamma=0.9, lam=0.95)
    assert len(advantages) == 4 and len(returns) == 4
    with pytest.raises(ValueError):
        GAEAdvantage()([1.0, 2.0])  # value function is mandatory for GAE


# ----------------------------------------------------------------------
# PPOValueLoss per-token (needs torch + twinkle) -- response-only targets scatter onto the mask
# ----------------------------------------------------------------------
def test_ppo_value_loss_per_token_masks_and_scatters():
    """Response-only returns land on the masked tokens; the clipped MSE is averaged over them only."""
    pytest.importorskip('twinkle.loss')
    torch = pytest.importorskip('torch')
    from twinkle.loss.value import PPOValueLoss

    loss = PPOValueLoss(vf_coef=1.0)
    # T=4, response tokens at positions 2 and 3 (labels != -100). The critic emits per-token values in
    # logits (task='value' skips pooling).
    inputs = {'labels': torch.tensor([[-100, -100, 5, 6]])}
    outputs = {'logits': torch.tensor([[0.0, 0.0, 1.0, 2.0]])}
    # returns arrive response-only (one per response token); old_values=None -> no clipping.
    out = loss(inputs, outputs, returns=[[3.0, 4.0]], old_values=None)
    # sq err on the two response tokens: (1-3)^2=4, (2-4)^2=4 -> mean 4 -> 0.5*vf_coef*4 = 2.0.
    assert float(out['loss']) == pytest.approx(2.0)


def test_ppo_value_loss_clips_value_move():
    """With old_values set, the value may not move more than cliprange_value from the rollout estimate."""
    pytest.importorskip('twinkle.loss')
    torch = pytest.importorskip('torch')
    from twinkle.loss.value import PPOValueLoss

    loss = PPOValueLoss(vf_coef=1.0, cliprange_value=0.2)
    inputs = {'labels': torch.tensor([[-100, 5]])}
    outputs = {'logits': torch.tensor([[0.0, 1.0]])}
    # old_value=0.0 -> clipped prediction is 0.2; unclipped err (1-3)^2=4, clipped (0.2-3)^2=7.84;
    # PPO takes the LARGER (pessimistic) -> 7.84 over the single response token -> 0.5*1.0*7.84.
    out = loss(inputs, outputs, returns=[[3.0]], old_values=[[0.0]])
    assert float(out['loss']) == pytest.approx(0.5 * 7.84, rel=1e-4)
