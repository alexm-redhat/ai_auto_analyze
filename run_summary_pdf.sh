#!/bin/bash

RESULTS_DIR="/home/alexm-redhat/code/ai_auto_perf_analysis/auto_analyze/results"
# Run
python -m auto_analyze.run_summary_pdf \
    --cmp-file ${RESULTS_DIR}/vllm_sglang_trt__perf_compare_blocks.txt \
    --plan-file ${RESULTS_DIR}/vllm_sglang_trt__plan.txt \
    --output-pdf-file ${RESULTS_DIR}/vllm_sglang_trt_cmp_and_plan_summary.pdf \
