#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <config.sh>"
    exit 1
fi

source "$1"
source "$(dirname "$0")/env.sh"

./run_profile_core.sh \
    ./auto_profile/test_configs/infra_config.json \
    ./auto_profile/test_configs/run_${test_name}.json \

python -m auto_profile.run_summary \
    --results-dir ./auto_profile/results \
    --output-dir ${output_dir} \
    --override-output-dir \

# Analyze
for config_file in ${output_dir}/analyze_*.json; do
    if [ ! -f "$config_file" ]; then
        echo "No analyze_*.json files found in ${output_dir}"
        break
    fi
    echo "=== Running pipeline: $(basename "$config_file") ==="

    # Step 1: Analyze
    python -m auto_analyze.run_analyze --config "$config_file"

    # Step 2: Summary PDF
    python -m auto_analyze.run_summary_pdf --config "$config_file"

    # Step 3: Combined Trace
    python -m auto_analyze.run_combined_trace --config "$config_file"

    # Step 4: Create JIRA Tasks
    python -m auto_analyze.run_create_jiras --config "$config_file"
done
