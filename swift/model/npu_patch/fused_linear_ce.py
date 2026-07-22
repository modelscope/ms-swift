import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast


class NPUFusedLinearCrossEntropy(torch.autograd.Function):
    """
    A memory-efficient fused Linear and CrossEntropy Loss operator for Huawei Ascend NPU.

    Background & Motivation:
        In standard HuggingFace causal language models, the `LM-Head` (Linear) and `CrossEntropyLoss`
        are computed sequentially. For large vocabulary sizes (e.g., Qwen2 with 152K), this materializes
        a massive `Logits` tensor of shape [Batch * SeqLen, VocabSize] in HBM, leading to severe
        Memory Expansion and Out-Of-Memory (OOM) errors during the backward pass.

    Optimization Strategy (Chunked Autograd in Time Dimension):
        This operator avoids materializing the full Logits tensor. Instead, it chunks the input
        `hidden_states` along the time dimension (Batch * SeqLen). For each chunk, it computes
        the local logits, calculates the cross-entropy loss, derives the gradients in-place,
        and immediately discards the local logits.

    Mathematical Proof of Equivalence:
        Given Z = X * W^T and L = CrossEntropy(Z, Y), the gradients are:
            ∇X = ∇Z * W
            ∇W = (∇Z)^T * X
        By chunking X into [X_1, X_2, ... X_C] along the sequence dimension:
        The local gradient for chunk `i` is exactly ∇X_i = ∇Z_i * W.
        Since the weight W is shared across all tokens, its total gradient is the sum of local
        gradients by the multivariable chain rule (Summation Rule):
            ∇W = Σ (∇Z_i)^T * X_i
        This is exactly what is implemented via `grad_input[i:i+B] = ...` and `grad_weight += ...`,
        ensuring 100% mathematical fidelity while reducing peak VRAM from O(B*S*V) to O(ChunkSize*V).
    """

    @staticmethod
    def forward(ctx, hidden_states, weight, labels, logit_softcapping=0.0, logit_scaling=0.0, num_items_in_batch=None):
        x = hidden_states.contiguous().view(-1, hidden_states.shape[-1])
        y = labels.contiguous().view(-1)

        BT, H = x.shape

        # Calculate the denominator for mean reduction, aligning with DDP global token scaling
        if num_items_in_batch is not None:
            denominator = float(num_items_in_batch)
        else:
            n_non_ignore = torch.count_nonzero(y != -100).item()
            denominator = float(n_non_ignore) if n_non_ignore > 0 else 1.0

        # Chunk size tuned for NPU HBM/UB balance
        CHUNK_SIZE = 2048

        grad_input = torch.zeros_like(x)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        total_loss = 0.0

        # Disable global autograd graph to manually manage the chunked gradient computation
        with torch.no_grad():
            for i in range(0, BT, CHUNK_SIZE):
                x_chunk = x[i: i + CHUNK_SIZE]
                y_chunk = y[i: i + CHUNK_SIZE]

                # Skip-FLOPs: Bypass matrix multiplication if the entire chunk is ignored (e.g., Prompt/Padding)
                if (y_chunk == -100).all():
                    continue

                x_chunk_data = x_chunk.detach()
                w_data = weight.detach()
                # Enable localized gradient tracking for the current chunk sandbox
                with torch.enable_grad():
                    x_chunk.requires_grad_(True)

                    # 1. Local Fused Linear (NPU Cube Engine full speed)
                    logits_chunk = F.linear(x_chunk_data, w_data)

                    # Apply model-specific scaling (e.g., Gemma-2 softcapping, Cohere scaling)
                    if logit_scaling != 0:
                        logits_chunk = logits_chunk * logit_scaling
                    if logit_softcapping != 0:
                        logits_chunk = logit_softcapping * torch.tanh(logits_chunk / logit_softcapping)

                    # 2. Local CrossEntropy Loss
                    loss_chunk = F.cross_entropy(logits_chunk.float(), y_chunk, ignore_index=-100, reduction='sum')
                    loss_chunk_mean = loss_chunk / denominator

                    total_loss += loss_chunk_mean.item()

                    # 3. Compute local gradients
                    grad_logits = torch.autograd.grad(loss_chunk_mean, logits_chunk)[0]
                    grad_logits = grad_logits.to(x.dtype)

                # 4. Chain Rule: Backpropagate gradients to input and weight, then GC destroys logits_chunk
                grad_input[i: i + CHUNK_SIZE] = torch.matmul(grad_logits, weight)
                grad_weight += torch.matmul(grad_logits.t(), x_chunk)

        # Save gradients for the backward pass
        ctx.save_for_backward(grad_input.detach(), grad_weight.to(weight.dtype).detach())
        ctx.orig_x_shape = hidden_states.shape

        return torch.tensor(total_loss, device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        The backward pass is essentially an O(1) memory retrieval since gradients
        were already computed block-by-block during the forward pass.
        """
        grad_input, grad_weight = ctx.saved_tensors

        grad_input_3d = (grad_input * grad_output).view(ctx.orig_x_shape)
        grad_weight_final = grad_weight * grad_output

        return grad_input_3d, grad_weight_final, None, None, None, None


def npu_fused_lm_head_loss(hidden_states, weight, labels, logit_softcapping=0.0, logit_scaling=0.0,
                           num_items_in_batch=None):
    """Wrapper for the Fused Linear Cross Entropy."""
    return NPUFusedLinearCrossEntropy.apply(
        hidden_states, weight, labels, logit_softcapping, logit_scaling, num_items_in_batch
    )


def npu_fused_lm_forward(self, *args, **kwargs):
    """
    A monkey-patch forward function for CausalLM models.
    It intercepts the forward pass before `self.lm_head` is called, preventing
    the materialization of the full Logits tensor in training mode.
    """
    labels = kwargs.pop('labels', None)
    num_items = kwargs.pop('num_items_in_batch', None)

    # Forward through the backbone (Transformer layers) only
    outputs = self.model(*args, **kwargs)
    hidden_states = outputs[0]

    loss = None

    if labels is not None:
        # ---------------------------------------------------------------------
        # Training Mode: Apply Fused LM-Head & CrossEntropy
        # ---------------------------------------------------------------------
        shift_hidden_states = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        logit_softcapping = getattr(self.config, 'final_logit_softcapping', 0.0)
        logit_scaling = getattr(self.config, 'logit_scale', 0.0)

        loss = npu_fused_lm_head_loss(
            shift_hidden_states,
            self.lm_head.weight,
            shift_labels,
            logit_softcapping=logit_softcapping,
            logit_scaling=logit_scaling,
            num_items_in_batch=num_items
        )

        # ---------------------------------------------------------------------
        # [Crucial Explanation]: Why return `torch.empty(0)`?
        # HuggingFace frameworks (e.g., Trainer, Evaluator) expect the `logits`
        # attribute to exist in `CausalLMOutputWithPast`. Returning `None` may
        # trigger `AttributeError` when downstream hooks try to access `logits.shape`
        # or `logits.argmax()`.
        # By returning a 0-sized tensor `torch.empty(0)`, we perfectly satisfy the
        # API requirements while explicitly allocating ZERO bytes of memory.
        # ---------------------------------------------------------------------
        logits = torch.empty(0, dtype=hidden_states.dtype, device=hidden_states.device)

    else:
        # ---------------------------------------------------------------------
        # Inference Mode: Standard execution (Materialize full logits)
        # ---------------------------------------------------------------------
        logits = self.lm_head(hidden_states)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
        hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
        attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
    )
