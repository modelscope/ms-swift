# v5 YAML-driven SFT: same run as train/lora/train.sh, but every argument lives in sft.yaml.
#
# The YAML file is passed as the FIRST positional arg after the subcommand; swift/cli/main.py expands it into
# the equivalent --flags before dispatch (see the header in sft.yaml for the scalar/list/dict rules). USE_SWIFT_V5
# and CUDA_VISIBLE_DEVICES are set HERE, in the shell, not in the yaml's ENV block: resolve_route() reads
# USE_SWIFT_V5 before the yaml is parsed, so v5 routing must be decided by the environment the process starts in.
#
# Run from the repository root (the yaml path, like every examples/ path, is repo-root-relative).
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft examples/train/yaml/sft.yaml
