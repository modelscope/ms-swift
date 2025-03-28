from contextlib import contextmanager
from types import MethodType
from peft.tuners import lora
from peft.tuners.lora import LoraLayer
from typing import Optional, Any
import torch

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
    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        if parameter_group and all(self.name not in pg for pg in parameter_group):
            return  # Skip if not in target parameter group
        return self.merge_origin(safe_merge, adapter_names)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        # Ensure tensors are on correct device
        if isinstance(self, lora.Embedding):
            self.lora_embedding_A[adapter].data = self.lora_embedding_A[adapter].data.to(
                self.base_layer.weight.device)
            self.lora_embedding_B[adapter].data = self.lora_embedding_B[adapter].data.to(
                self.base_layer.weight.device)
        else:
            self.lora_A[adapter].weight.data = self.lora_A[adapter].weight.data.to(
                self.base_layer.weight.device)
            self.lora_B[adapter].weight.data = self.lora_B[adapter].weight.data.to(
                self.base_layer.weight.device)
        return self.get_delta_weight_origin(adapter).to(self.base_layer.weight.device)

    # Patch all LoraLayer instances
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module.name = name
            if not hasattr(module, 'merge_origin') and hasattr(module, 'base_layer'):
                module.merge_origin = module.merge
                module.merge = MethodType(merge, module)
                module.get_delta_weight_origin = module.get_delta_weight
                module.get_delta_weight = MethodType(get_delta_weight, module)

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



@contextmanager
def patch_lora_unmerge(model):
    """Patch LoraLayer's _cache_pop method to ensure proper device placement.
    
    Args:
        model: The PEFT model to patch
        
    Yields:
        The patched model (context manager ensures cleanup)
    """    
    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key).to(self.base_layer.weight.device)
        return value

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module.name = name
            if not hasattr(module, '_cache_pop_origin') and hasattr(module, 'base_layer'):
                module._cache_pop_origin = module._cache_pop
                module._cache_pop = MethodType(_cache_pop, module)
            yield
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    if hasattr(module, '_cache_pop_origin'):
                        module._cache_pop = module._cache_pop_origin
                        del module._cache_pop_origin