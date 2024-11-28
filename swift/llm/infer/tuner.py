from swift.plugin import extra_tuners
from swift.tuners import Swift
from ..argument import InferArguments
from ..model.register import load_by_unsloth


def prepare_infer_engine(args: InferArguments, infer_engine):
    if args.train_type in extra_tuners:
        extra_tuners[args.train_type].from_pretrained(infer_engine.model, args.ckpt_dir, inference_mode=True)
    else:
        if args.tuner_backend == 'unsloth':
            model, processor = load_by_unsloth(args.ckpt_dir, args.torch_dtype, args.max_length, args.quant_bits == 4,
                                               args.model_meta.is_multimodal)
            model_info = infer_engine.processor.model_info
            model_meta = infer_engine.processor.model_meta
            processor.model_info = model_info
            processor.model_meta = model_meta
            model.model_info = model_info
            model.model_meta = model_meta

            if args.model_meta.is_multimodal:
                from unsloth import FastVisionModel as UnslothModel
            else:
                from unsloth import FastLanguageModel as UnslothModel
            UnslothModel.for_inference(model)

            infer_engine.model = model
            infer_engine.generation_config = model.generation_config
            infer_engine.processor = processor
        else:
            # TODO: vllm lora
            infer_engine.model = Swift.from_pretrained(infer_engine.model, args.ckpt_dir, inference_mode=True)
