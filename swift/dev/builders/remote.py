# Copyright (c) ModelScope Contributors. All rights reserved.
"""mode='remote' object sources: local builders yield in-process objects, these yield
twinkle_client handles over HTTP.

The remote shape (e2e-verified by examples/server/client.py): the dataset and template stay
in THIS process (build_dataset/build_template are reused; the InputFeature rows they emit
cross the wire as forward inputs), only the model is a MultiLoraTransformersModel handle.
That handle exposes the same method face as the local TrainableModel, with arguments
narrowed to "class name + serializable kwargs" -- the str branch of the local
``Union[instance, type, str]`` signatures. Loss and tuner therefore need remote variants
(the local ones pass peft/Loss instances); configure_optimizer's transformers branch is
already a description ("set_optimizer(name, scalar kwargs)") and is reused as is.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.config import DistributedConfig, ModelConfig, TrainConfig, TunerConfig

logger = get_logger()


def build_remote_model(model_config: 'ModelConfig', distributed_config: 'DistributedConfig') -> Any:
    """Connect to the twinkle-server and return the client-side trainable model handle.

    ``model`` names the SERVER-side served model id (e.g. 'ms://Qwen/Qwen2.5-7B-Instruct');
    the handle strips the scheme itself. The server must already run it as a multi-LoRA host
    (``swift deploy`` / a server_config.yaml with a model app); this function never starts one.
    """
    from twinkle import init_twinkle_client
    from twinkle_client.model import MultiLoraTransformersModel

    init_twinkle_client(base_url=distributed_config.server_base_url)
    model = MultiLoraTransformersModel(model_config.model or '')
    logger.info(f"[remote] driving server {distributed_config.server_base_url!r} model {model_config.model!r}")
    return model


def apply_remote_tuner(model: Any, tuner_config: Optional['TunerConfig'],
                       gradient_accumulation_steps: int) -> None:
    """Attach the tuner on the server: LoRA becomes a named adapter on the resident base model.

    trainable_tokens has no server-side counterpart yet, and 'full' needs no adapter call (the
    server model trains all parameters by default) -- both fail loudly rather than degrade.
    """
    from peft import LoraConfig

    if tuner_config is None:
        return
    if tuner_config.tuner == 'full':
        return
    if tuner_config.tuner != 'lora':
        raise ValueError(
            f"--mode remote does not support tuner={tuner_config.tuner!r}: the server-side adapter "
            "mechanism is LoRA (a named adapter on the resident base). Use --tuner lora, or drop "
            "--tuner for full-parameter remote training.")
    lora_config = LoraConfig(
        r=tuner_config.lora_rank,
        lora_alpha=tuner_config.lora_alpha,
        lora_dropout=tuner_config.lora_dropout,
        target_modules=list(tuner_config.target_modules),
        use_rslora=tuner_config.use_rslora,
    )
    # No save_dir: the server owns the checkpoint store (a client-side dir would have to exist
    # on the server already). GA rides the adapter because the server's grad-sync gate is
    # per-adapter state, not a forward kwarg.
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=gradient_accumulation_steps)


def configure_remote_loss(model: Any, train_config: 'TrainConfig') -> None:
    """Set the SFT/PT loss on the server model by NAME (the instance the local path passes
    cannot cross HTTP).

    Covers the supervised causal_lm main line (cross entropy, channel-grouped, DFT-weighted,
    liger fused). The channel/DFT/fused knobs forward verbatim; a server without the class
    registered (e.g. liger not installed there) fails at set_loss time, loudly.
    """
    from swift.dev.utils.naming import resolve_loss

    name = 'cross_entropy'
    if _liger_fused_ce_enabled(train_config):
        name = 'liger_fused_linear_cross_entropy'
    elif train_config.enable_channel_loss:
        name = 'channel'
    loss_cls = resolve_loss(name)
    model.set_loss(
        loss_cls.__name__,
        reduction='sum',
        dft=bool(train_config.enable_dft_loss),
        label_smoothing=train_config.label_smoothing_factor or 0.0,
    )


def _liger_fused_ce_enabled(train_config: 'TrainConfig') -> bool:
    """Same predicate the local recipe uses to pair the loss with loop_task='fused_lm_ce'."""
    from swift.dev.loss import liger_fused_ce_enabled

    return liger_fused_ce_enabled(train_config)


#: The loss families the remote path does not configure: embedding/seq_cls/reranker route
#: through configure_embedding_loss / configure_seq_cls_loss / configure_reranker_loss, which
#: pass loss instances; their remote story is untested. Refused here so the failure happens
#: at assembly time, not deep inside a server request.
REMOTE_UNSUPPORTED_TASKS = ('embedding', 'seq_cls', 'reranker', 'generative_reranker')
