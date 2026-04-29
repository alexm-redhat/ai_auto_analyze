#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <config.json>"
    exit 1
fi

# Run combined trace step
source "$(dirname "$0")/env.sh"
python -m auto_analyze.run_combined_trace --config "$1"
