#!/bin/bash

source auto_profile/utils.sh

# Run
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=120000
python -m auto_code_gen.run_work_items \
    --code-gen-dir /home/alexm-redhat/code/ai_auto_perf_analysis/auto_code_gen/prs_step_2 \
    --work-items-file /home/alexm-redhat/code/ai_auto_perf_analysis/auto_code_gen/prs_step_2/work_item_error_1.txt \

