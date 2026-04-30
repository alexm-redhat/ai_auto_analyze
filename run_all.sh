#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <run_config>"
    echo "Example: $0 ./auto_profile/test_configs/run_deepseek_r1_nvfp4.json"
    exit 1
fi

source "$(dirname "$0")/env.sh"

run_config="$1"
output_dir=$(python -c "from common.utils import output_dir_from_run_config; print(output_dir_from_run_config('${run_config}'))")

# Step 1: Run benchmarks and profiling across frameworks (vllm, sgl, trt)
./auto_profile/run_profile_core.sh "${run_config}"

# Step 2: Parse results and generate per-test-case analysis configs
python -m auto_profile.run_profile_summary \
    --run-config "${run_config}" \
    --override-output-dir \

# Step 3: For each test case where vLLM is slower, run the analysis pipeline
for config_file in ${output_dir}/analyze_*.json; do
    if [ ! -f "$config_file" ]; then
        echo "No analyze_*.json files found in ${output_dir}"
        break
    fi
    echo "=== Running pipeline: $(basename "$config_file") ==="

    # 3a: Analyze performance bottlenecks
    python -m auto_analyze.run_analyze --config "$config_file"

    # 3b: Generate summary PDF report
    python -m auto_analyze.run_summary_pdf --config "$config_file"

    # 3c: Build combined trace visualization
    python -m auto_analyze.run_combined_trace --config "$config_file"

    # 3d: Create JIRA tasks for identified issues
    python -m auto_analyze.run_create_jiras --config "$config_file"
done
