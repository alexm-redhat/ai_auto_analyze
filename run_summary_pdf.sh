#!/bin/bash

RESULTS_DIR="/home/alexm-redhat/code/ai_auto_perf_analysis/auto_analyze/results"
# Run
python -m auto_analyze.run_summary_pdf \
    --transformer-blocks vllm_median_block.txt sglang_median_block.txt \
    --cmp-file ${RESULTS_DIR}/vllm_sglang__perf_compare_blocks.txt \
    --plan-file ${RESULTS_DIR}/vllm_sglang__plan.txt \
    --output-pdf-file ${RESULTS_DIR}/vllm_sglang_cmp_and_plan_summary_kimi_nvfp4_b200.pdf \
