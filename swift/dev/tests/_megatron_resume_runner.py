"""Standalone torchrun entry: one phase of the Megatron mode='local' save/resume closed-loop test.

Launched by test_run_sft_e2e.py::test_run_sft_megatron_local_save_resume as SEPARATE torchrun
subprocesses (one per phase, each a clean process). Phases:
  - A            : seed0, train N steps, save (weights + optimizer).  [reference to resume from]
  - B            : seed999, resume A, train 0 steps, save.            [main criterion vs A]
  - cont         : seed0, train 2N steps continuous, save + dump per-step loss. [trajectory ref]
  - resume_cont  : seed0, resume A (already N steps), train N more, save + dump loss. [interrupt path]

Writes {"phase", "losses", "weights_path", "optim_dir"} to --out on rank 0.

Why this shape (per the review):
  - MAIN (weights bit-exact): B inits with a DIFFERENT seed (999) then resumes A; if resume fully
    restores, B's saved weights == A's bit-for-bit. Any non-zero diff means load_state_dict(
    strict=False) (twinkle megatron.py:1204/1209) silently dropped/misplaced a shard.
  - SECONDARY (optimizer existence): the optimizer dist-ckpt is checked only for presence + key/
    shape/step completeness (catches whole-shard loss/misplacement). NOTE: optimizer momentum
    ROUND-TRIP NUMERIC accuracy is NOT directly gated here -- with a 0-step resume there is no
    cheap direct criterion, and the last-bit region has known upstream mcore round-trip noise
    (~1e-4, see doc P2-4c-1), so pinning a baseline-less numeric band would risk false alarms.
  - MUST (trajectory): interrupt-resume (A+B_cont) vs continuous (cont) loss trajectories must
    match within an OBSERVED band -- this is the ONLY end-to-end signal that optimizer momentum was
    actually restored to the SAVED values (not silently reloaded as zero), which neither the weight
    nor the existence criterion can catch.
"""
import argparse
import json
import os
import torch

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'

# Fixed single-sample microbatch list (GA == len(inputs)); identical every step so the run is
# deterministic and the interrupt-vs-continuous comparison is apples-to-apples.
_IDS = list(range(10, 26))
_LABELS = ([-100] * 8 + _IDS[8:])[1:] + [-100]
_POS = list(range(len(_IDS)))
_MICRO = [{
    'input_ids': _IDS,
    'labels': _LABELS,
    'position_ids': _POS
}, {
    'input_ids': _IDS,
    'labels': _LABELS,
    'position_ids': _POS
}]


def _build(seed):
    import twinkle
    from modelscope import snapshot_download
    from twinkle import DeviceMesh

    from swift.dev.model.megatron.bridge import MCoreBridgeBackend
    from swift.dev.model.megatron.model import MegatronModel

    twinkle.initialize(mode='local')
    ws = int(os.environ.get('WORLD_SIZE', '2'))
    mesh = DeviceMesh.from_sizes(world_size=ws, dp_size=ws)
    model_path = snapshot_download(MODEL)
    torch.manual_seed(seed)
    # fp32 (mixed_precision='no'): the bit-exact 0-diff weight gate needs full precision; bf16 save
    # rounding would blur the complete-vs-incomplete-restore signal.
    m = MegatronModel(
        model_id=model_path,
        device_mesh=mesh,
        mixed_precision='no',
        backend=MCoreBridgeBackend(),
        use_distributed_optimizer=True)
    # lr=1e-6: small enough that single-sample training loss decreases smoothly/monotonically. A
    # larger lr (1e-4) makes the toy single-sample loss diverge/oscillate (0.63->9.38->...), and on
    # a chaotic trajectory a step's fp32 last-bit noise is amplified exponentially -- making the
    # interrupt-vs-continuous diff impossible to attribute to optimizer round-trip vs butterfly
    # effect. A smooth trajectory lets the trajectory diff actually measure momentum-restore fidelity.
    m.set_optimizer('Adam', lr=1e-6)
    return m


def _train(m, n_steps):
    losses = []
    for _ in range(n_steps):
        m.forward_backward(inputs=_MICRO, micro_batch_size=1)
        m.clip_grad_and_step(max_grad_norm=1.0)
        metrics = m.calculate_metric(is_training=True)
        losses.append(float(metrics['loss']) if metrics.get('loss') is not None else float('nan'))
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['A', 'B', 'cont', 'resume_cont'], required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--resume_from', default=None)
    parser.add_argument('--steps', type=int, default=2)
    args = parser.parse_args()

    if args.phase == 'A':
        m = _build(seed=0)
        losses = _train(m, args.steps)
    elif args.phase == 'B':
        m = _build(seed=999)  # different init: a complete resume must overwrite it entirely
        m.resume_from_checkpoint(args.resume_from)
        losses = _train(m, 0)  # 0 steps: weights are a pure restore-copy -> must be bit-exact to A
    elif args.phase == 'cont':
        m = _build(seed=0)
        losses = _train(m, 2 * args.steps)  # continuous reference trajectory
    else:  # resume_cont
        m = _build(seed=0)
        m.resume_from_checkpoint(args.resume_from)
        losses = _train(m, args.steps)  # resume A (already steps) then steps more -> interrupt path

    name = f'ckpt-{args.phase}'
    m.save(name=name, output_dir=args.out_dir, save_optimizer=True)
    ckpt_dir = os.path.join(args.out_dir, name)

    if int(os.environ.get('RANK', '0')) == 0:
        # locate the optimizer dist-ckpt iter dir (iter_XXXXXXX/) written by _save_mcore_optimizer.
        optim_dir = None
        for d in sorted(os.listdir(ckpt_dir)):
            if d.startswith('iter_') and os.path.isdir(os.path.join(ckpt_dir, d)):
                optim_dir = os.path.join(ckpt_dir, d)
        weights_path = os.path.join(ckpt_dir, 'model.safetensors')
        with open(args.out, 'w') as f:
            json.dump({'phase': args.phase, 'losses': losses, 'weights_path': weights_path, 'optim_dir': optim_dir}, f)
        print(f'RESUME_RUNNER_DONE phase={args.phase} losses={losses}', flush=True)


if __name__ == '__main__':
    main()
