import os
import sys

from swift.utils import git_clone_github
from .base import OptimizerCallback


class MuonOptimizerCallback(OptimizerCallback):

    def create_optimizer(self):
        args = self.args
        model = self.trainer.model
        if not args.local_repo_path:
            args.local_repo_path = git_clone_github('https://github.com/MoonshotAI/Moonlight.git')
        sys.path.append(os.path.join(args.local_repo_path, 'examples'))
        from toy_train import Muon

        # parse args.optim_args
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(' ', '').split(','):
                key, value = mapping.split('=')
                optim_args[key] = value

        model_arch = model.model_meta.model_arch
        embed_key = getattr(model_arch, 'embedding', None) or 'embed_tokens'
        lm_head_key = getattr(model_arch, 'lm_head', None) or 'lm_head'
        muon_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and p.ndim >= 2 and embed_key not in n and lm_head_key not in n
        ]
        adamw_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and not (p.ndim >= 2 and embed_key not in n and lm_head_key not in n)
        ]

        return Muon(
            lr=args.learning_rate,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            adamw_betas=(args.adam_beta1, args.adam_beta2),
            adamw_eps=args.adam_epsilon,
            **optim_args,
        )
