"""Native meta adapters must initialize and reload after base allocation."""
import torch
from torch import nn


def test_native_meta_adapter_initialization_and_checkpoint_reload(tmp_path):
    from peft import LoraConfig, get_peft_model

    from swift.fsdpturbo.loading import initialize_native_lora

    class Backbone(nn.Module):

        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(5, 3, bias=False, device='meta')

        def forward(self, inputs):
            return self.projection(inputs)

    model = get_peft_model(Backbone(), LoraConfig(r=2, lora_alpha=4, target_modules=['projection']))
    model.to_empty(device='cpu')
    with torch.no_grad():
        model.base_model.model.projection.base_layer.weight.fill_(0.25)
    initialize_native_lora(model, [])
    layer = model.base_model.model.projection
    assert layer.lora_A.default.weight.isfinite().all()
    assert layer.lora_A.default.weight.count_nonzero() > 0
    assert layer.lora_B.default.weight.count_nonzero() == 0
    assert torch.equal(layer.base_layer.weight, torch.full((3, 5), 0.25))
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model(torch.ones(2, 5)).square().mean().backward()
    optimizer.step()
    assert layer.lora_B.default.weight.count_nonzero() > 0
    assert layer.base_layer.weight.grad is None
    expected = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    model.save_pretrained(tmp_path, safe_serialization=True)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.fill_(float('nan'))
    initialize_native_lora(model, [str(tmp_path)])
    for name, p in model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p, expected[name], rtol=0, atol=0)
