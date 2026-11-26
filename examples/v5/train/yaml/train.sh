# v5 YAML-driven SFT: same run as train/lora/train.sh, but every argument lives in sft.yaml.
#
# The YAML file is passed as the FIRST positional arg after the subcommand; swift/cli/main.py expands it into
# the equivalent --flags before dispatch (see the header in sft.yaml for the scalar/list/dict rules).
# CUDA_VISIBLE_DEVICES is set HERE, in the shell, not in the yaml's ENV block: the CLI version is decided
# by the environment the process starts in (v5 is the default), before the yaml is parsed.
#
# Run from the repository root (the yaml path, like every examples/ path, is repo-root-relative).
# MEASURED: peak VRAM ~3 GiB/card (Qwen3.5-0.8B, 4-step smoke, 1 card(s), 2026-11-24)
CUDA_VISIBLE_DEVICES=0 \
swift sft examples/v5/train/yaml/sft.yaml
