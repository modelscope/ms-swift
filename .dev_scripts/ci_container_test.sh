if [ "$MODELSCOPE_SDK_DEBUG" == "True" ]; then
    pip install uv
    uv pip install -r requirements/tests.txt
    git config --global --add safe.directory /ms-swift
    git config --global user.email tmp
    git config --global user.name tmp.com

    # linter test
    # use internal project for pre-commit due to the network problem
    if [ `git remote -v | grep alibaba  | wc -l` -gt 1 ]; then
        pre-commit run -c .pre-commit-config_local.yaml --all-files
        if [ $? -ne 0 ]; then
            echo "linter test failed, please run 'pre-commit run --all-files' to check"
            echo "From the repository folder"
            echo "Run 'pip install -r requirements/tests.txt' install test dependencies."
            echo "Run 'pre-commit install' install pre-commit hooks."
            echo "Finally run linter with command: 'pre-commit run --all-files' to check."
            echo "Ensure there is no failure!!!!!!!!"
            exit -1
        fi
    fi

    uv pip install -r requirements/framework.txt -U
    uv pip install decord einops -U
    pip uninstall autoawq -y
    uv pip install optimum
    uv pip install diffusers
    uv pip install "transformers<5.0"
    # pip install autoawq -U --no-deps

    # test with install
    uv pip install .
    uv pip install auto_gptq bitsandbytes deepspeed -U
else
    echo "Running case in release image, run case directly!"
fi
# remove torch_extensions folder to avoid ci hang.
rm -rf ~/.cache/torch_extensions
if [ $# -eq 0 ]; then
    ci_command="python tests/run.py --subprocess"
else
    ci_command="$@"
fi
echo "Running case with command: $ci_command"
$ci_command
