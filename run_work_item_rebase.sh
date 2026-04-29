#!/bin/bash

source auto_profile/utils.sh

# Run
source "$(dirname "$0")/env.sh"
python -m auto_code_gen.run_work_items \
    --code-gen-dir /home/alexm-redhat/code/ai_auto_perf_analysis/auto_code_gen/prs_step_1 \
    --work-items-file /home/alexm-redhat/code/ai_auto_perf_analysis/auto_code_gen/prs_step_1/work_item_rebase.txt \

