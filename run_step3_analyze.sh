#!/bin/bash
#
# Step 3: Run the analysis pipeline for each test case — single-trace analysis
#          for each framework, cross-trace comparison, PDF report, Chrome trace,
#          and JIRA task creation.
#
# Usage: ./run_step3_analyze.sh <run_config>
# Example: ./run_step3_analyze.sh ./auto_profile/test_configs/run_deepseek_r1_nvfp4.json

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <run_config>"
    echo "Example: $0 ./auto_profile/test_configs/run_deepseek_r1_nvfp4.json"
    exit 1
fi

source "$(dirname "$0")/env.sh"

run_config="$1"
output_dir=$(python -c "from common.utils import output_dir_from_run_config; print(output_dir_from_run_config('${run_config}'))")
test_results_dir="${output_dir}/test_results"

echo "========================================"
echo "  STEP 3: Run analysis pipeline"
echo "========================================"
echo "  Run config:   ${run_config}"
echo "  Output dir:   ${output_dir}"
echo "  Test results: ${test_results_dir}"
echo "========================================"

if [ ! -d "${test_results_dir}" ]; then
    echo "No test_results directory found at ${test_results_dir}"
    echo "Run step 2 first: ./run_step2_parse.sh ${run_config}"
    exit 1
fi

for test_dir in "${test_results_dir}"/*/; do
    test_dir="${test_dir%/}"
    test_name="$(basename "${test_dir}")"

    single_configs=( "${test_dir}"/single_trace_config_*.json )
    if [ ! -f "${single_configs[0]}" ]; then
        continue
    fi

    cross_config="${test_dir}/cross_trace_config.json"
    if [ ! -f "${cross_config}" ]; then
        echo "SKIP ${test_name}: no cross_trace_config.json"
        continue
    fi

    echo ""
    echo "========================================"
    echo "  Analyzing: ${test_name}"
    echo "========================================"

    # 3a: Run single-trace analysis for each framework
    for st_config in "${single_configs[@]}"; do
        fw_name="$(basename "${st_config}" .json | sed 's/single_trace_config_//')"
        echo ""
        echo "--- Single-trace analysis: ${fw_name} ---"
        python -m auto_analyze.run_single_trace \
            --config "${st_config}"
    done

    # 3b: Run cross-trace analysis
    echo ""
    echo "--- Cross-trace analysis ---"
    python -m auto_analyze.run_cross_trace \
        --config "${cross_config}"

    cross_output_dir=$(python -c "import json; print(json.load(open('${cross_config}'))['output_dir'])")
    echo ""
    echo "  Cross-trace results: ${cross_output_dir}"

    # 3c: Generate summary PDF report
    echo ""
    echo "--- Summary PDF ---"
    python -m auto_analyze.run_summary_pdf \
        --cross-config "${cross_config}"

    # 3d: Build combined trace visualization
    echo ""
    echo "--- Cross trace JSON ---"
    python -m auto_analyze.run_chrome_trace \
        --cross-config "${cross_config}"

    # 3e: Create JIRA tasks for identified issues
    echo ""
    echo "--- Create JIRA tasks ---"
    python -m auto_analyze.run_jiras \
        --cross-config "${cross_config}"
done

echo ""
echo "  Step 3 complete."
