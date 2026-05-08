#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <cross_trace_config.json> [--clean]"
    echo ""
    echo "Run cross-trace analysis comparing multiple single-trace results."
    echo ""
    echo "Arguments:"
    echo "  config.json   Path to cross-trace config JSON"
    echo "  --clean       Clean output directory before running"
    echo ""
    echo "Supported analysis types:"
    echo "  cross-framework   Compare different frameworks (vLLM vs SGLang)"
    echo "  regression        Compare different versions of the same framework"
    echo ""
    echo "Example config: auto_analyze/cross_trace_config_example.json"
    exit 1
fi

source "$(dirname "$0")/env.sh"
python -m auto_analyze.run_cross_trace --config "$@"
