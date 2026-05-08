"""
Single-Trace Analysis Entry Point
==================================

Analyzes a single framework trace to extract transformer block structure,
correlate GPU operations to high-level operations, and produce performance
analysis artifacts.

Usage:
    python -m auto_analyze.run_single_trace \\
        --config single_trace_config.json \\
        --claude-config claude_config.json

Inputs (via config JSON):
    - framework_name:        Name of the framework (e.g., "vllm", "sglang")
    - model:                 HuggingFace model tag (e.g., "nvidia/DeepSeek-R1-NVFP4")
    - gpu_type:              GPU type (e.g., "B200")
    - batch_size_range:      Batch size range (e.g., "1" or "1,4,16")
    - prefill_size_range:    Input/prefill size range (e.g., "4")
    - output_size_range:     Output size range (e.g., "1024")
    - trace_file:            Path to trace file (.nsys-rep, .sqlite, .json, or .json.gz)
    - framework_source_code: Path to framework source code
    - trace_gpu_focus:       "ALL" or a GPU ID like "0" (default: "0")
    - run_command:           Command used to run the framework
    - run_log:               (optional) Path to run log file
    - high_level_focus:      (optional) Focus guidance for high-level block analysis
    - perf_analysis_focus:   (optional) Focus areas for performance analysis
    - output_dir:            Output directory for results
    - max_gpu_ops:           (optional) Max GPU ops to extract (default: 2000)

Claude config JSON:
    - model:                 Claude model to use (e.g., "claude-opus-4-6[1m]")
    - allowed_tools:         List of tools (e.g., ["Read", "Write", "Bash"])

Outputs (in output_dir):
    1. transformer_block_high_level_ops.txt/.json - High-level transformer block ops
    2. gpu_ops.txt/.json                          - GPU operations extracted from trace
    3. gpu_ops_to_blocks.txt                      - GPU ops correlated to blocks
    4. median_block.txt                           - Median transformer block
    5. perf_analysis.txt                          - Performance analysis
    6. transformer_block_trace.json/.txt           - Trace visualization + summary
"""

import json
import os
import sys
import time
import asyncio
import argparse

from common.utils import setup_logging, safe_clean_dir
from common.claude_utils import claude_run, ClaudeConfig

from auto_analyze.single_trace_config import SingleTraceConfig
from auto_analyze.single_trace_prompts import gen_single_trace_prompts

import shutil
from dataclasses import asdict


def load_claude_config(claude_config_path, output_dir):
    with open(claude_config_path) as f:
        data = json.load(f)
    return ClaudeConfig(
        model=data["model"],
        allowed_tools=data["allowed_tools"],
        perm_mode="acceptEdits",
        cwd=output_dir,
    )


def log_config(config: SingleTraceConfig):
    print("=" * 70)
    print("  SINGLE-TRACE ANALYSIS")
    print("=" * 70)
    print(f"  Framework:     {config.framework_name}")
    print(f"  Model:         {config.model}")
    print(f"  HF URL:        https://huggingface.co/{config.model}")
    print(f"  GPU:           {config.gpu_type}")
    print(f"  Batch size:    {config.batch_size_range}")
    print(f"  Prefill size:  {config.prefill_size_range}")
    print(f"  Output size:   {config.output_size_range}")
    print(f"  Trace file:    {config.trace_file}")
    print(f"  Trace type:    {config.trace_file_type}")
    print(f"  GPU focus:     {config.trace_gpu_focus}")
    print(f"  Source code:   {config.framework_source_code}")
    if config.commit_id:
        print(f"  Commit ID:     {config.commit_id}")
    if config.run_log:
        print(f"  Run log:       {config.run_log}")
    if config.run_command:
        print(f"  Run command:   {config.run_command}")
    if config.high_level_focus:
        print(f"  Focus:         {config.high_level_focus}")
    print(f"  Output dir:    {config.output_dir}")
    print("=" * 70)


def log_claude_config(claude_config: ClaudeConfig):
    print(f"  Claude model:  {claude_config.model}")
    print(f"  Tools:         {claude_config.allowed_tools}")
    print(f"  Perm mode:     {claude_config.perm_mode}")
    print(f"  CWD:           {claude_config.cwd}")


def save_run_artifacts(config: SingleTraceConfig):
    output_dir = config.output_dir

    # Save run_params.txt
    params_path = os.path.join(output_dir, "run_params.txt")
    with open(params_path, "w") as f:
        f.write("SINGLE TRACE ANALYSIS PARAMETERS\n")
        f.write("=" * 50 + "\n")
        for key, value in asdict(config).items():
            f.write(f"{key}: {value}\n")
    print(f"  Saved run parameters: {params_path}")

    # Create run_originals subdirectory
    originals_dir = os.path.join(output_dir, "run_originals")
    os.makedirs(originals_dir, exist_ok=True)

    # Copy trace file
    if config.trace_file and os.path.exists(config.trace_file):
        dst = os.path.join(originals_dir, os.path.basename(config.trace_file))
        if os.path.abspath(config.trace_file) != os.path.abspath(dst):
            shutil.copy2(config.trace_file, dst)
            print(f"  Copied trace file: {dst}")
        else:
            print(f"  Trace file already in output dir, skipping copy")

    # Save run_command.txt
    if config.run_command:
        cmd_path = os.path.join(originals_dir, "run_command.txt")
        with open(cmd_path, "w") as f:
            f.write(config.run_command + "\n")
        print(f"  Saved run command: {cmd_path}")

    # Copy run log
    if config.run_log and os.path.exists(config.run_log):
        dst = os.path.join(originals_dir, "run_log.txt")
        shutil.copy2(config.run_log, dst)
        print(f"  Copied run log: {dst}")


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
    setup_logging("single_trace")

    parser = argparse.ArgumentParser(
        description="Run single-trace analysis on a framework execution trace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to single-trace config JSON file",
    )
    parser.add_argument(
        "--claude-config",
        required=True,
        type=str,
        help="Path to Claude config JSON file (model, allowed_tools, perm_mode)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clean output directory before running (default: clean)",
    )
    args = parser.parse_args()

    config = SingleTraceConfig.from_json(args.config)

    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    if not args.no_clean and os.path.exists(config.output_dir):
        safe_clean_dir(config.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)

    config.prepare_source_code()
    config.prepare_trace_file()

    log_config(config)

    print("\nSaving run artifacts...")
    save_run_artifacts(config)

    claude_config = load_claude_config(args.claude_config, config.output_dir)
    log_claude_config(claude_config)

    prompts, output_files = gen_single_trace_prompts(config)

    start_time = time.time()
    print(f"\nStarting analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Running {len([p for p in prompts if isinstance(p, str)])} analysis steps...\n")

    asyncio.run(claude_run(claude_config, prompts))

    duration = time.time() - start_time
    print(f"\nAnalysis completed in {duration:.1f}s")

    config.save_result_metadata()

    log_outputs(config.output_dir, output_files)
    print(f"\nResult metadata saved to: {config.output_dir}/result_metadata.json")
