#!/bin/bash

# Clear outputs
./run_clear_analyze.sh

# Run
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=120000
python -m auto_analyze.run_analyze