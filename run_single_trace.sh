#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: $0 <single_trace_config.json> <claude_config.json> [--no-clean]"
    echo ""
    echo "Run single-trace analysis on a framework execution trace."
    echo "Output directory is cleaned before running by default."
    echo ""
    echo "Arguments:"
    echo "  single_trace_config.json   Path to single-trace config JSON"
    echo "  claude_config.json         Path to Claude config JSON"
    echo "  --no-clean                 Do not clean output directory before running"
    echo ""
    echo "Example configs:"
    echo "  auto_analyze/single_trace_config_example.json"
    echo "  auto_analyze/claude_config.json"
    exit 1
fi

source "$(dirname "$0")/env.sh"
python -m auto_analyze.run_single_trace --config "$1" --claude-config "$2" "${@:3}"
