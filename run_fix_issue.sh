#!/bin/bash

source auto_profile/utils.sh

PRS_DIR="/home/alexm-redhat/code/ai_auto_perf_analysis/auto_code_gen/prs"
ISSUE_DIR="issue_1"

# Run
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=120000
python -m auto_code_gen.run_fix_issue \
    --high_level_code_plan_file ${PRS_DIR}/high_level_code_plan.txt \
    --prs_dir ${PRS_DIR} \
    --issue_to_fix_file ${PRS_DIR}/$ISSUE_DIR/issue_to_fix_run_log.txt \
    --issue_cwd ${PRS_DIR}/$ISSUE_DIR \
