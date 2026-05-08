"""
Cross-Trace Analysis Entry Point
=================================

Compares multiple single-trace analysis results to find performance
differences and generate improvement plans.

Supports two analysis modes:
  - cross-framework:  Compare different frameworks (e.g., vLLM vs SGLang)
  - regression:       Compare different versions/commits of the same framework

Usage:
    python -m auto_analyze.run_cross_trace --config cross_trace_config.json

Inputs (via config JSON):
    - trace_results:     List of single-trace result references, each with:
        - trace_id:              Unique identifier for this trace
        - framework_name:        Framework name
        - framework_source_code: Path to framework source code
        - result_dir:            Path to single-trace analysis result directory
    - analysis_type:     "cross-framework" or "regression"
    - target_trace_id:   Which trace to analyze/optimize
    - model:             Model name
    - gpu_type:          GPU type
    - output_dir:        Output directory for results

Outputs (in output_dir):
    1. perf_compare_blocks.txt  - Per-operation comparison of median blocks
    2. perf_diff_analysis.txt   - Full explanation of target's perf differences
    3. improvement_plan.txt     - Improvement/fix proposals for target trace
"""

import os
import sys
import time
import asyncio
import argparse

from common.utils import setup_logging, safe_clean_dir
from common.claude_utils import claude_run, ClaudeConfig

from auto_analyze.cross_trace_config import (
    CrossTraceConfig,
    ANALYSIS_TYPE_CROSS_FRAMEWORK,
)
from auto_analyze.cross_trace_prompts import gen_cross_trace_prompts


def build_cross_trace_claude_config(output_dir):
    return ClaudeConfig(
        model="claude-opus-4-6[1m]",
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd=output_dir,
    )


def log_config(config: CrossTraceConfig):
    mode_label = (
        "Cross-Framework Comparison"
        if config.analysis_type == ANALYSIS_TYPE_CROSS_FRAMEWORK
        else "Same-Framework Regression Analysis"
    )

    print("=" * 70)
    print("  CROSS-TRACE ANALYSIS")
    print("=" * 70)
    print(f"  Mode:          {mode_label}")
    print(f"  Model:         {config.model}")
    print(f"  GPU:           {config.gpu_type}")
    print(f"  Target trace:  {config.target_trace_id}")
    print(f"  Output dir:    {config.output_dir}")
    print()
    print("  Traces:")
    for i, tr in enumerate(config.trace_results):
        marker = " (*)" if tr.trace_id == config.target_trace_id else ""
        print(f"    [{i+1}] {tr.trace_id}{marker}")
        print(f"        Framework:   {tr.framework_name}")
        print(f"        Source code: {tr.framework_source_code}")
        if tr.result_dir:
            print(f"        Result dir:  {tr.result_dir}")
        else:
            print(f"        Median:      {tr.median_block_file}")
            print(f"        High-level:  {tr.high_level_ops_file}")
    print("=" * 70)


def log_outputs(output_dir, output_files):
    print()
    print("=" * 70)
    print("  OUTPUT FILES")
    print("=" * 70)
    for label, filename in output_files.items():
        path = os.path.join(output_dir, filename)
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {label}: {filename}")
    print("=" * 70)


if __name__ == "__main__":
    setup_logging("cross_trace")

    parser = argparse.ArgumentParser(
        description="Run cross-trace analysis comparing multiple single-trace results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to cross-trace config JSON file",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean output directory before running",
    )
    args = parser.parse_args()

    config = CrossTraceConfig.from_json(args.config)

    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    if args.clean and os.path.exists(config.output_dir):
        safe_clean_dir(config.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)

    log_config(config)

    claude_config = build_cross_trace_claude_config(config.output_dir)
    prompts, output_files = gen_cross_trace_prompts(config)

    start_time = time.time()
    print(f"\nStarting analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Running {len([p for p in prompts if isinstance(p, str)])} analysis steps...\n")

    asyncio.run(claude_run(claude_config, prompts))

    duration = time.time() - start_time
    print(f"\nAnalysis completed in {duration:.1f}s")

    log_outputs(config.output_dir, output_files)
