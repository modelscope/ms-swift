#!/bin/bash
set -e

set_env() {
    grep -q "^$1=" .deepspeed_env \
        && sed -i "s|^$1=.*|$1=$2|" .deepspeed_env \
        || echo "$1=$2" >> .deepspeed_env
}

pdsh_run() {
    hosts=$(grep -v '^\s*#' randy/hostfile | awk 'NF {print $1}' | paste -sd,)
    pdsh -S -R ssh -w "$hosts" "$@"
}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR" || exit 1

outputs=$(python randy/train.py "$@")
command=$(echo "$outputs" | awk -F'<randy>|</randy>' '{print $2}')
command=$(echo "$command" | sed 's| --| \\\n  --|g')

TMP_SCRIPT=$(mktemp --suffix=".randy")
cat > "$TMP_SCRIPT" << EOF
#!/bin/bash
cd "$SCRIPT_DIR" || exit 1
set -a && source .deepspeed_env && set +a
. "$WORK_DIR/miniforge3/etc/profile.d/conda.sh"
conda activate swift
EOF

# set_env OMP_NUM_THREADS 1
set_env SWIFT_CONFIG_FILE "$TMP_SCRIPT"

# command="python $command"
# command="deepspeed $command"
command="deepspeed --hostfile randy/hostfile $command"

echo "$command" >> "$TMP_SCRIPT"
echo "$TMP_SCRIPT" && cat "$TMP_SCRIPT"

pdsh_run "bash $(realpath randy/killer.sh)"
chmod +x "$TMP_SCRIPT" && bash "$TMP_SCRIPT"
