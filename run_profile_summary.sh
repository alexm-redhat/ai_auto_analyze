#!/bin/bash

test_name="deepseek_r1_nvfp4"

# Run
python -m auto_profile.run_summary \
    --results-dir ./auto_profile/results \
    --output-dir ./auto_analyze/results/results_analyze_${test_name} \
    --override-output-dir \

