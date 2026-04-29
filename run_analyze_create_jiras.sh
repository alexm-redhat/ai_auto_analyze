#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <config.json>"
    exit 1
fi

# Run JIRA task creation step
source "$(dirname "$0")/env.sh"
python -m auto_analyze.run_create_jiras --config "$1"
