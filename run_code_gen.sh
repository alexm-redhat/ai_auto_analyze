#!/bin/bash

source auto_profile/utils.sh

# Clear outputs
clean_dir_contents "auto_code_gen/prs" "vllm" "sglang"

# Run
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=120000
python -m auto_code_gen.run_code_gen