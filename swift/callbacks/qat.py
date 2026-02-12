# Copyright (c) ModelScope Contributors. All rights reserved.
import os

from swift.utils import get_logger
from .base import TrainerCallback
from transformers.utils import is_torchao_available

logger = get_logger()


class QatCallback(TrainerCallback):
    """An callback for QAT training support implementation"""

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        # check dependency
        assert is_torchao_available("0.15.0"), 'Version of torchao should be 0.15.0 or higher'

        # defined torchao quantization config e.g. Int4WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig
        from torchao.quantization import Int4WeightOnlyConfig
        self.quant_config = Int4WeightOnlyConfig()

    def on_train_begin(self, args, state, control, **kwargs):
        from torchao.quantization import quantize_
        from torchao.quantization.qat import QATConfig
        # insert fake quantizer
        logger.info(f'Initializing QAT with config: [{self.quant_config}]')
        train_model = self.trainer.model
        quantize_(train_model.model, QATConfig(self.quant_config, step="prepare"))
        logger.info(f"QAT Model: {train_model}")

    def on_train_end(self, args, state, control, **kwargs):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        from transformers import AutoModel, TorchAoConfig
        checkpoint_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.max_steps}")
        quantization_config = TorchAoConfig(quant_type=self.quant_config)
        pqt_model = AutoModel.from_pretrained(checkpoint_dir, dtype="auto",
                                              quantization_config=quantization_config)
        quantized_output_dir = os.path.join(args.output_dir, "quantized_model")
        pqt_model.save_pretrained(quantized_output_dir)
        self.trainer.tokenizer.save_pretrained(quantized_output_dir)
        logger.info(f"Quantized model saved to: {quantized_output_dir}")
