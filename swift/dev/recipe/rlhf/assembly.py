# Copyright (c) ModelScope Contributors. All rights reserved.
# The offline RLHF family's assembly: the preference config -> object stages + the reference.
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from swift.dev.config.adapter_config import has_adapter
from swift.dev.recipe.utils import REF_GROUP
from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.config import (
        DistributedConfig,
        ModelConfig,
        RLConfig,
        TemplateConfig,
        TunerConfig,
    )
    from swift.dev.recipe.configs import RecipeConfigs
    from swift.dev.recipe.preparation import TrainingModel
    from swift.dev.recipe.resources import OfflineResources
    from swift.dev.template import Template

logger = get_logger()

#: rl_types that consult a frozen reference model (the rest build their own baseline in the loss).
_REF_TYPES = frozenset({"dpo", "kto"})


def _reference_needs_group(rl_config: "RLConfig", tuner_config: Optional["TunerConfig"]) -> bool:
    """Whether the reference build needs its own DeviceGroup (plan and build share this predicate).

    The reference-free types (cpo/orpo/simpo/rm), an explicit ``reference_free`` dpo/kto, and a LoRA
    run without a reference adapter (the shared-weights view) need NO group; only a full-parameter
    frozen copy -- full fine-tuning, or an explicit reference adapter -- is a separate Ray actor.
    """
    if rl_config.rl_type not in _REF_TYPES or rl_config.reference_free:
        return False
    return not (has_adapter(tuner_config) and not rl_config.ref_adapters)


def _build_reference(
    template: "Template",
    model_config: "ModelConfig",
    rl_config: "RLConfig",
    tuner_config: Optional["TunerConfig"],
    distributed_config: "DistributedConfig",
    model: TrainingModel,
    *,
    remote_group: Optional[str],
    template_config: Optional["TemplateConfig"] = None,
) -> Optional[TrainingModel]:
    """The frozen reference the encoder probes for ref-logps, or None for the reference-free types.

    LoRA without a reference adapter returns the POLICY itself -- the encoder derives
    ``disable_lora`` from that identity (the same shared-weights view the online recipes use). Full
    fine-tuning, or an explicit reference adapter, builds a separate frozen copy on the
    ``remote_group`` DeviceGroup the caller planned via the SAME predicate, so the two cannot drift.
    """
    if rl_config.rl_type not in _REF_TYPES or rl_config.reference_free:
        # Reference-free dpo/kto: DPOLoss.reference_free anchors on the policy alone; wired in
        # lockstep with configure_rl_loss forwarding ``reference_free`` to the loss.
        return None
    from swift.dev.builder import build_ref_model

    if has_adapter(tuner_config) and not rl_config.ref_adapters:
        return model
    if rl_config.ref_model is None:
        raise ValueError(
            "dpo/kto full fine-tuning needs a reference model, but RLConfig.ref_model is None. It is "
            "normally defaulted to the policy model by process.py::_derive_rl_ref_model; pass "
            "--ref_model explicitly if you bypassed config processing."
        )
    if remote_group is None and distributed_config.mode != "remote":
        raise ValueError(
            "A full-parameter frozen reference must be built on its own DeviceGroup, but "
            "remote_group is None: _reference_needs_group planned no group for it -- the "
            "plan and the build have drifted."
        )
    # mode='remote': the reference is another served-model handle (no DeviceGroup to plan).
    return build_ref_model(
        name=remote_group,
        model_id=rl_config.ref_model,
        model_config=model_config,
        template=template,
        template_config=template_config,
        distributed_config=distributed_config,
        adapters=tuple(rl_config.ref_adapters or ()),
        model_type=rl_config.ref_model_type,
        model_revision=rl_config.ref_model_revision,
    )


def build_preference_objects(
    configs: RecipeConfigs,
    *,
    recipe: str,
    output_dir: str = "output",
) -> tuple[OfflineResources, Optional[TrainingModel]]:
    """Run the preference config -> object stages in the locked order, up to (not including) the loop.

    Same stage order as the supervised family, with the preference differences inline: the full-
    parameter reference gets its own DeviceGroup at twinkle init, the template rides the preference
    encode mode, the dataset stays RAW (``encode=False``; the encoder encodes each batch), and the
    loss / metric are the rlhf ones. The reference build shares the group-planning predicate.
    """
    from swift.dev.loss import configure_rl_loss, configure_rl_metrics
    from swift.dev.recipe.offline import build_offline_objects

    model_config, template_config, distributed_config = configs.model, configs.template, configs.distributed
    rl_config, tuner_config = configs.rl, configs.tuner
    assert rl_config is not None
    ref_remote_group = REF_GROUP if _reference_needs_group(rl_config, tuner_config) else None
    cluster = build_offline_objects(
        configs,
        recipe=recipe,
        output_dir=output_dir,
        encode=False,
        template_mode=("kto" if rl_config.rl_type == "kto" else "rlhf"),
        auxiliary_groups=(() if ref_remote_group is None else ((ref_remote_group, 1),)),
        forward_task="causal_lm",
        configure_loss=lambda model: configure_rl_loss(model, rl_config),
        configure_metric=lambda model: configure_rl_metrics(model, rl_config),
    )
    reference = _build_reference(
        cluster.template,
        model_config,
        rl_config,
        tuner_config,
        distributed_config,
        cluster.model,
        remote_group=ref_remote_group,
        template_config=template_config,
    )
    return cluster, reference
