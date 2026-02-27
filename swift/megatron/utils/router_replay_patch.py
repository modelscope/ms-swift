import warnings
from enum import Enum
from typing import List, Optional, Callable

import torch

try:
    from megatron.core.transformer.moe.moe_utils import (
        apply_router_token_dropping,
        compute_routing_scores_for_aux_loss,
        group_limited_topk,
    )
except ImportError:
    warnings.warn("NPU not support router replay for now.", stacklevel=2)
    pass
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args


class RouterReplayAction(Enum):
    RECORD = "record"  # Record the topk indices for replay
    REPLAY_FORWARD = "replay_forward"  # Replay the recorded topk indices for forward pass
    REPLAY_BACKWARD = "replay_backward"  # Replay topk indices for re-compute during backward pass


class RouterReplay:
    """
    A class to manage the recording and replaying of MoE routing decisions.
    It holds all router instances and provides static methods to globally
    control recording and replaying.
    """

    # Static variable to hold all router instances, one per MoE layer.
    global_router_replay_instances: List['RouterReplay'] = []

    @staticmethod
    def set_replay_data(all_layers_topk_indices: List[torch.Tensor]):
        """
        Distributes the topk indices for all layers to their respective RouterReplay instances.
        :param all_layers_topk_indices: A list of tensors, where each tensor contains the
                                        topk indices for a specific layer. The order
                                        must match the instantiation order of the routers.
        """
        if len(all_layers_topk_indices) != len(RouterReplay.global_router_replay_instances):
            raise ValueError(
                f"The number of replay tensors ({len(all_layers_topk_indices)}) "
                f"does not match router instances ({len(RouterReplay.global_router_replay_instances)})."
            )
        for i, router_instance in enumerate(RouterReplay.global_router_replay_instances):
            router_instance.set_target_indices(all_layers_topk_indices[i])

    @staticmethod
    def get_recorded_data() -> List[torch.Tensor]:
        """
        Collects the recorded topk indices from all RouterReplay instances.
        :return: A list of tensors, each containing the recorded topk indices for a layer.
        """
        return [
            router.get_recorded_indices() for router in RouterReplay.global_router_replay_instances
        ]

    @staticmethod
    def clear_global_indices():
        """Clears the recorded and target topk indices in all instances."""
        for router in RouterReplay.global_router_replay_instances:
            router.clear_indices()

    @staticmethod
    def set_global_router_replay_action(router_replay_action: RouterReplayAction):
        """Sets the router replay action for all router instances."""
        for router in RouterReplay.global_router_replay_instances:
            router.set_router_replay_action(router_replay_action)

    @staticmethod
    def clear_global_router_replay_action():
        """Clears the router replay action for all router instances."""
        for router in RouterReplay.global_router_replay_instances:
            router.clear_router_replay_action()

    def __init__(self):
        """Initializes a RouterReplay instance for a specific layer."""
        self.target_topk_idx: Optional[torch.Tensor] = None  # Target topk indices for replay
        self.recorded_topk_idx: Optional[torch.Tensor] = None  # Recorded topk indices for replay
        self.router_replay_action: Optional[RouterReplayAction] = (
            None  # Router replay action for this layer
        )
        self.replay_backward_list: List[torch.Tensor] = (
            []
        )  # List of tensors for backward pass replay
        RouterReplay.global_router_replay_instances.append(self)

    def set_target_indices(self, topk_indices: torch.Tensor):
        """Sets the target topk indices for replay."""
        self.target_topk_idx = topk_indices
        self.replay_backward_list.append(topk_indices)

    def get_recorded_indices(self) -> Optional[torch.Tensor]:
        """Returns the recorded topk indices."""
        return self.recorded_topk_idx

    def record_indices(self, topk_indices: torch.Tensor):
        """Records the topk indices."""
        self.recorded_topk_idx = topk_indices

    def clear_indices(self):
        """Clears the recorded and target topk indices."""
        self.recorded_topk_idx = None
        self.target_topk_idx = None
        self.replay_backward_list = []

    def set_router_replay_action(self, router_replay_action: RouterReplayAction):
        """Sets the router replay action for this layer."""
        self.router_replay_action = router_replay_action

    def clear_router_replay_action(self):
        """Clears the router replay action for this layer."""
        self.router_replay_action = None

    def get_replay_topk(
        self,
        scores: torch.Tensor,
        topk: int,
        num_groups: Optional[int] = None,
        group_topk: Optional[int] = None,
        default_compute_topk: Callable[
            [torch.Tensor, int, Optional[int], Optional[int]], torch.Tensor
        ] = None,
    ) -> torch.Tensor:
        """Returns the target topk indices for replay."""
        if self.router_replay_action == RouterReplayAction.RECORD:
            probs, top_indices = default_compute_topk(
                scores, topk, num_groups=num_groups, group_topk=group_topk
            )
            self.record_indices(top_indices)
            return probs, top_indices
        elif self.router_replay_action == RouterReplayAction.REPLAY_FORWARD:
            top_indices = self.target_topk_idx
            # Ensure indices are on the correct device
            top_indices = top_indices.to(scores.device)
            # Gather the scores for the replayed indices to get the probabilities
            probs = scores.gather(1, top_indices)
            return probs, top_indices
        elif self.router_replay_action == RouterReplayAction.REPLAY_BACKWARD:
            top_indices = self.replay_backward_list.pop(0)
            # Ensure indices are on the correct device
            top_indices = top_indices.to(scores.device)
            # Gather the scores for the replayed indices to get the probabilities
            probs = scores.gather(1, top_indices)
            return probs, top_indices
        else:
            return default_compute_topk(scores, topk, num_groups, group_topk)


def _patched_topk_routing_with_score_function(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool = False,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    scaling_factor: Optional[float] = None,
    score_function: str = "softmax",
    expert_bias: Optional[torch.Tensor] = None,
    fused: bool = False,
    router_replay: Optional['RouterReplay'] = None,
):
    """
    Patched version of topk_routing_with_score_function that supports router replay.
    """
    assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."
    num_tokens, num_experts = logits.shape

    def _compute_topk(scores, topk, num_groups=None, group_topk=None):
        if group_topk:
            return group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                group_topk=group_topk,
            )
        else:
            return torch.topk(scores, k=topk, dim=1)

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        # Default behavior if no replay is active

        if router_replay is None:
            return _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
        else:
            return router_replay.get_replay_topk(
                scores, topk, num_groups, group_topk, _compute_topk
            )

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        else:
            scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float()).type_as(logits)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
            scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    if torch.are_deterministic_algorithms_enabled():
        # build [num_tokens, num_experts] from [num_tokens, topk]
        routing_probs = torch.zeros_like(logits)
        rows = torch.arange(num_tokens, device=logits.device).unsqueeze(1)
        routing_probs.index_put_((rows, top_indices), probs, accumulate=False)

        routing_map = torch.zeros_like(logits, dtype=logits.dtype)
        routing_map.index_put_(
            (rows, top_indices), torch.ones_like(probs, dtype=routing_map.dtype), accumulate=False
        )
        routing_map = routing_map.bool()
    else:
        # TODO Try using element-wise operations instead of scatter?
        routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()

    return routing_probs, routing_map


def patched_routing(self, logits: torch.Tensor):
    """Top-k routing function

    Args:
        logits (torch.Tensor): Logits tensor after gating.

    Returns:
        probs (torch.Tensor): The probabilities of token to experts assignment.
        routing_map (torch.Tensor): The mapping of token to experts assignment,
            with shape [num_tokens, num_experts].
    """
    seq_length, bsz = logits.shape[:2]
    logits = logits.view(-1, self.config.num_moe_experts)

    # Apply Z-Loss
    logits = self.apply_z_loss(logits)

    # Calculate probs and routing_map for token dispatching
    if self.routing_type == "sinkhorn":
        probs, routing_map = self.sinkhorn_load_balancing(logits)
    else:
        probs, routing_map = _patched_topk_routing_with_score_function(
            logits,
            self.topk,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
            fused=self.config.moe_router_fusion,
            router_replay=self.router_replay,
        )

    # Apply token dropping to probs and routing_map.
    if self.config.moe_expert_capacity_factor is not None:
        probs, routing_map = apply_router_token_dropping(
            probs,
            routing_map,
            router_topk=self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            drop_policy=self.config.moe_token_drop_policy,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
        )

    # Apply each aux loss type and attach aux loss autograd function to probs
    if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
        # Calculate scores and routing_map for aux loss
        routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
            logits, self.topk, self.score_function, fused=self.config.moe_router_fusion
        )
        probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
        probs = self._apply_seq_aux_loss(
            probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz
        )
        probs = self._apply_global_aux_loss(
            probs, scores_for_aux_loss, routing_map_for_aux_loss
        )

    # Optionally apply expert bias
    # Update expert bias and tokens_per_expert
    # Prevent extra local tokens accumulation on evaluation or activation recomputation
    if self.enable_expert_bias and torch.is_grad_enabled():
        with torch.no_grad():
            self.local_tokens_per_expert += routing_map.sum(dim=0)

    return probs, routing_map


def apply_router_replay_patch():
    """
    Applies the monkey patch for MoE Router Replay functionality.
    This patch dynamically adds the 'enable_routing_replay' attribute to TransformerConfig
    and modifies the TopKRouter to support recording and replaying of routing decisions.
    """
    print("Applying Router Replay Patch...")
    # Clear router instances to avoid state leakage between model initializations.
    RouterReplay.global_router_replay_instances.clear()
    # Step 1: Patch TransformerConfig to include the feature flag
    if not hasattr(TransformerConfig, "enable_routing_replay"):
        # Add class attribute with default value
        TransformerConfig.enable_routing_replay = False

        # Store original __init__ method
        original_tf_config_init = TransformerConfig.__init__

        # Define new __init__ method that safely handles enable_routing_replay parameter
        def patched_tf_config_init(self, *args, **kwargs):
            # Call original constructor with remaining kwargs
            original_tf_config_init(self, *args, **kwargs)

            mg_args = get_args()
            # Set the instance attribute
            self.enable_routing_replay = mg_args.enable_routing_replay or TransformerConfig.enable_routing_replay

        # Apply the patch
        TransformerConfig.__init__ = patched_tf_config_init

    # Step 2: Patch TopKRouter only once to ensure idempotency.
    if hasattr(TopKRouter, "_router_replay_patched"):
        return

    original_init = TopKRouter.__init__

    # Step 3: Define the new __init__ method
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.router_replay = None
        if self.config.enable_routing_replay:
            self.router_replay = RouterReplay()

    # Step 4: Apply the patches
    TopKRouter.__init__ = patched_init
    TopKRouter.routing = patched_routing
    TopKRouter._router_replay_patched = True
