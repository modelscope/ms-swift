# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
from types import MethodType, SimpleNamespace

from swift.dataset.preprocessor.decision import GS1Preprocessor


def _record():
    return {
        'request': {
            'model': 'GS1-0.8B',
            'state': {
                'balance': 18,
                'reserve': 10
            },
            'questions': {
                'choice': {
                    'type': 'choice',
                    'instructions': 'Choose the lower-cost job.',
                    'criteria': {
                        'job_a': 'Cost 2',
                        'job_b': 'Cost 12'
                    },
                },
                'truth': {
                    'type': 'noul',
                    'instructions': 'Is the balance above reserve?'
                },
                'rating': {
                    'type': 'score',
                    'instructions': 'Rate the balance.',
                    'criteria': ['Low', 'High']
                },
            },
        },
        'targets': {
            'choice': {
                'candidate_ids': ['job_b', 'job_a'],
                'distribution': [0.0, 1.0]
            },
            'truth': {
                'candidate_ids': ['false', 'true'],
                'distribution': [0.0, 1.0]
            },
            'rating': {
                'candidate_ids': ['0', '1'],
                'distribution': [0.25, 0.75]
            },
        },
    }


def test_preprocessor_preserves_native_state_criteria_and_candidate_order():
    rows = GS1Preprocessor().preprocess(_record())
    assert len(rows) == 3
    assert rows[0]['messages'][0]['content'] == '{"balance":18,"reserve":10}'
    assert rows[0]['state'] == '{"balance":18,"reserve":10}'
    assert rows[0]['criteria'] == ['{"job_a":"Cost 2","job_b":"Cost 12"}']
    assert rows[0]['options'] == [['job_a: Cost 2', 'job_b: Cost 12']]
    assert rows[0]['target_probs'] == [[1.0, 0.0]]
    assert rows[1]['target_probs'] == [[0.0, 1.0]]
    assert rows[2]['options'] == [['Low', 'High']]
    assert rows[2]['target_probs'] == [[0.25, 0.75]]


def test_gs1_template_encodes_while_model_is_detached(monkeypatch):
    import swift.template.templates.decision as decision_template
    from swift.template.template_inputs import StdTemplateInputs

    class Branch:
        suffix = (4, 5)
        candidates = ('a', 'b')

    class Compiler:

        def __init__(self, tokenizer, **kwargs):
            self.tokenizer = tokenizer
            self.kwargs = kwargs
            tokenizer.chat_template = 'compiler-local'

        def compile(self, request):
            assert request['state'] == {'battery': 18}
            return SimpleNamespace(state=(1, 2, 3), branches=(Branch(), ))

    class NativeModel:

        def decision_compiler(self, tokenizer, *, max_context):
            return Compiler(tokenizer, max_context=max_context)

    class Request:

        @staticmethod
        def model_validate(value):
            return value

    template = object.__new__(decision_template.GS1Template)
    template.model = None
    template.model_info = SimpleNamespace(model_dir='gs1-checkpoint')
    template.config = SimpleNamespace(
        inference_limits={
            'max_context': 64,
            'max_questions': 8,
            'max_request_tokens': 128,
            'max_branches': 32,
        },
        decision={},
        _name_or_path='bigai/GS1-0.8B',
    )
    template.max_length = 48
    template.processor = SimpleNamespace(chat_template='shared-template')
    monkeypatch.setattr(decision_template, 'import_module', lambda _: SimpleNamespace(DecisionRequest=Request))
    template.model = NativeModel()
    template.model = None
    encoded = template._encode(
        StdTemplateInputs(
            messages=[{
                'role': 'user',
                'content': '{"battery":18}'
            }],
            extra_kwargs={
                'kinds': ['choice'],
                'questions': ['Choose a job.'],
                'options': [['a', 'b']],
                'state': ['{"battery":18}'],
                'criteria': ['{"a":"Job A","b":"Job B"}'],
            },
        ))
    assert encoded['input_ids'] == [1, 2, 3, 4, 5]
    assert encoded['decision_meta']['kinds'] == [1]
    assert template.processor.chat_template == 'shared-template'


@pytest.mark.parametrize('mode', ['train', 'transformers'])
def test_gs1_collator_aligns_labels_and_native_branches(mode):
    import swift.template.templates.decision as decision_template

    template = object.__new__(decision_template.GS1Template)
    template.mode = mode
    template.task_type = 'decision'
    template.packing = False
    template.remove_unused_columns = True
    template.padding_side = 'right'
    template.processor = SimpleNamespace(pad_token_id=0)
    template.use_megatron = False
    template.padding_free = False
    template.sequence_parallel_size = 1
    batch = [
        {
            'input_ids': [1, 2, 3],
            'decision_meta': {
                'n_options': [2],
                'kinds': [1],
                'gs1_branches': ['long']
            },
            'target_probs': [[0.75, 0.25]],
        },
        {
            'input_ids': [4, 5],
            'decision_meta': {
                'n_options': [3],
                'kinds': [2],
                'gs1_branches': ['short']
            },
            'target_probs': None,
        },
    ]
    result = template.data_collator(batch)
    assert result['input_ids'].tolist() == [[1, 2, 3], [0, 4, 5]]
    assert result['attention_mask'].tolist() == [[1, 1, 1], [0, 1, 1]]
    if mode == 'train':
        assert result['position_ids'].tolist() == [[0, 1, 2], [0, 0, 1]]
    assert result['decision_meta']['kinds'] == [1, 2]
    assert result['decision_meta']['gs1_branches'] == ['long', 'short']
    torch.testing.assert_close(result['labels'], torch.tensor([[0.75, 0.25, 0.0], [0.0, 0.0, 0.0]]))


def test_gs1_loader_preserves_native_inference_and_freezes_calibrator(monkeypatch):
    import swift.model.models.decision as decision_model

    class FakeModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.calibrator = torch.nn.Linear(1, 1)

        def forward(self, input_ids, labels=None, **kwargs):
            if kwargs.get('branches'):
                return SimpleNamespace(logits=torch.ones((len(kwargs['branches']), 2)))
            return {'native': input_ids}

    model = FakeModel()
    monkeypatch.setattr(decision_model.ModelLoader, 'get_model', lambda *args, **kwargs: model)
    monkeypatch.setattr(decision_model, 'get_original_forward', lambda instance: instance.forward)

    def patch_forward(instance, function):
        instance.forward = MethodType(function, instance)

    monkeypatch.setattr(decision_model, 'patch_module_forward', patch_forward)
    loader = object.__new__(decision_model.GS1ModelLoader)
    wrapped = loader.get_model('unused', None, None, {})
    assert not any(parameter.requires_grad for parameter in wrapped.calibrator.parameters())
    ids = torch.tensor([[1, 2]])
    assert wrapped(input_ids=ids)['native'] is ids
    output = wrapped(
        input_ids=ids,
        decision_meta={
            'gs1_branches': [{
                'branch': object(),
                'state_length': 1
            }],
            'n_options': [2],
            'kinds': [1],
            'counts': [1],
        },
    )
    assert output.logits.shape == (1, 2)


@pytest.mark.parametrize('distribution', [[0.1, 0.1], [-0.1, 1.1], [float('nan'), 0.0]])
def test_preprocessor_rejects_invalid_target_distribution(distribution):
    row = _record()
    row['targets']['choice']['distribution'] = distribution
    with pytest.raises(ValueError, match='target distribution'):
        GS1Preprocessor().preprocess(row)
