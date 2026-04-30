#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <run_config>"
    echo "Example: $0 ./auto_profile/test_configs/run_deepseek_r1_nvfp4.json"
    exit 1
fi

# Run
python -m auto_profile.run_profile_summary \
    --run-config "$1" \
    --override-output-dir \

