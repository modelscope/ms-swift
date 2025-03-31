from contextlib import contextmanager
from types import MethodType
from typing import Any, Optional

import torch
from peft.tuners import lora
from peft.tuners.lora import LoraLayer


def round_robin(num_reqs, num_workers):
    """Distribute requests evenly across workers using round-robin algorithm.

    Args:
        num_reqs (int): Total number of requests to distribute
        num_workers (int): Number of available workers

    Returns:
        list: A list of lists where each sublist contains the request indices
                assigned to that particular node
    """
    distribution = [[] for _ in range(num_workers)]
    for idx in range(num_reqs):
        worker_id = idx % num_workers
        distribution[worker_id].append(idx)
    return distribution


@contextmanager
def patch_lora_merge(model, parameter_group=None):
    """Patch LoraLayer's merge and get_delta_weight methods for controlled merging.

    Args:
        model: The PEFT model to patch
        parameter_group: Optional list of parameter names to restrict merging

    Yields:
        The patched model (context manager ensures cleanup)
    """
    from peft.tuners.tuners_utils import check_adapters_to_merge

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        if parameter_group and all(self.name not in pg for pg in parameter_group):
            return  # Skip if not in target parameter group
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if self.use_dora.get(active_adapter, False):
                    self.lora_magnitude_vector[active_adapter].weight.data = \
                        self.lora_magnitude_vector[active_adapter].weight.data.to(base_layer.weight.device)

        return self.merge_origin(safe_merge, adapter_names)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        # Ensure tensors are on correct device
        if isinstance(self, lora.Embedding):
            self.lora_embedding_A[adapter].data = self.lora_embedding_A[adapter].data.to(self.base_layer.weight.device)
            self.lora_embedding_B[adapter].data = self.lora_embedding_B[adapter].data.to(self.base_layer.weight.device)
        else:
            self.lora_A[adapter].weight.data = self.lora_A[adapter].weight.data.to(self.base_layer.weight.device)
            self.lora_B[adapter].weight.data = self.lora_B[adapter].weight.data.to(self.base_layer.weight.device)
        return self.get_delta_weight_origin(adapter).to(self.base_layer.weight.device)

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key).to(self.base_layer.weight.device)
        return value

    # Patch all LoraLayer instances
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module.name = name
            if not hasattr(module, 'merge_origin') and hasattr(module, 'base_layer'):
                module.merge_origin = module.merge
                module.merge = MethodType(merge, module)
                module.get_delta_weight_origin = module.get_delta_weight
                module.get_delta_weight = MethodType(get_delta_weight, module)
                module._cache_pop_origin = module._cache_pop
                module._cache_pop = MethodType(_cache_pop, module)

    try:
        yield model
    finally:
        # Cleanup: restore original methods
        for module in model.modules():
            if isinstance(module, LoraLayer):
                if hasattr(module, 'merge_origin'):
                    module.merge = module.merge_origin
                    del module.merge_origin
                    module.get_delta_weight = module.get_delta_weight_origin
                    del module.get_delta_weight_origin
                    module._cache_pop = module._cache_pop_origin
                    del module._cache_pop_origin


@contextmanager
def patch_lora_unmerge(model):
    """Patch the unmerge method to ensure proper device handling."""

    def _cache_pop_patched(self, key: str) -> Any:
        value = self._caches.pop(key).to(self.base_layer.weight.device)
        return value

    def unmerge_patched(self):
        if not self.merged:
            return
        # Move magnitude vectors to correct device first
        for adapter in list(self.merged_adapters):
            if self.use_dora.get(adapter, False):
                self.lora_magnitude_vector[adapter].weight.data = \
                    self.lora_magnitude_vector[adapter].weight.data.to(self.base_layer.weight.device)

        return self.unmerge_origin()

    for module in model.modules():
        if isinstance(module, LoraLayer) and not hasattr(module, 'unmerge_origin'):
            module.unmerge_origin = module.unmerge
            module.unmerge = MethodType(unmerge_patched, module)
            module._cache_pop_origin = module._cache_pop
            module._cache_pop = MethodType(_cache_pop_patched, module)

    try:
        yield model
    finally:
        for module in model.modules():
            if isinstance(module, LoraLayer) and hasattr(module, 'unmerge_origin'):
                module.unmerge = module.unmerge_origin
                del module.unmerge_origin
                module._cache_pop = module._cache_pop_origin
                del module._cache_pop_origin
