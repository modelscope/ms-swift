import torch

from swift.template.base import Template


class _RebuildingTemplate(Template):

    def _post_encode(self, model, inputs):
        # Multimodal post-encoding paths may replace the kwargs dictionary.
        return {'inputs_embeds': inputs['input_ids']}


class _Model:

    device = torch.device('cpu')

    def forward(self, input_ids=None, inputs_embeds=None, output_router_logits=False):
        pass


def test_pre_forward_hook_preserves_output_router_logits():
    template = _RebuildingTemplate.__new__(_RebuildingTemplate)
    model = _Model()
    kwargs = {'input_ids': torch.ones(1, 2, dtype=torch.long), 'output_router_logits': True}

    _, forwarded_kwargs = template.pre_forward_hook(model, (), kwargs)

    assert forwarded_kwargs['output_router_logits'] is True
