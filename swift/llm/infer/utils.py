import inspect

from swift.llm import InferArguments
from .infer_engine import InferEngine


def get_infer_engine(args: InferArguments) -> InferEngine:
    if args.infer_backend == 'pt':
        from .infer_engine import PtEngine
        infer_engine_cls = PtEngine
    elif args.infer_backend == 'vllm':
        from .infer_engine import VllmEngine
        infer_engine_cls = VllmEngine
    else:
        from .infer_engine import LmdeployEngine
        infer_engine_cls = LmdeployEngine

    parameters = inspect.signature(infer_engine_cls.__init__).parameters
    kwargs = {}
    for k, v in args.__dict__.items():
        if k in parameters:
            kwargs[k] = v

    return infer_engine_cls(**kwargs)
