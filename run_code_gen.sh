#!/bin/bash

source auto_profile/utils.sh

# Clear outputs
# clean_dir_contents "auto_code_gen/prs" "vllm" "sglang"

# Run
source "$(dirname "$0")/env.sh"
python -m auto_code_gen.run_code_gen