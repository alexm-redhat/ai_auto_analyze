"""
Combined Analysis Pipeline (Backward Compatible)
=================================================

Runs single-trace analysis for each framework and then cross-trace
analysis, using the old multi-framework config JSON format.

This maintains backward compatibility with run_all.sh and existing
analyze_*.json config files.

Usage:
    python -m auto_analyze.run_analyze --config analyze_config.json
"""

import os
import glob
import time
import asyncio
import argparse

from common.utils import setup_logging, safe_clean_dir
from common.claude_utils import claude_run, ClaudeConfig

from auto_analyze.analyze_configs import (
    add_analyze_args,
    load_config,
)
from auto_analyze.single_trace_config import (
    SingleTraceConfig,
    MEDIAN_BLOCK_FILE,
    HIGH_LEVEL_OPS_FILE,
)
from auto_analyze.single_trace_prompts import gen_single_trace_prompts
from auto_analyze.cross_trace_config import (
    CrossTraceConfig,
    TraceResult,
    ANALYSIS_TYPE_CROSS_FRAMEWORK,
)
from auto_analyze.cross_trace_prompts import gen_cross_trace_prompts


def _find_trace_file(test_dir):
    patterns = ["trace-*.nsys-rep", "trace-*.sqlite", "trace-*.json", "trace-*.json.gz"]
    for pattern in patterns:
        matches = glob.glob(os.path.join(test_dir, pattern))
        if matches:
            return matches[0]
    return ""


def _find_run_log(test_dir):
    matches = glob.glob(os.path.join(test_dir, "run-log-*.txt"))
    non_profile = [m for m in matches if "profile" not in os.path.basename(m)]
    if non_profile:
        return non_profile[0]
    return matches[0] if matches else ""


def build_single_trace_configs(config):
    configs = []
    for fw in config["frameworks"]:
        trace_file = _find_trace_file(fw["test_dir"])
        run_log = _find_run_log(fw["test_dir"])

        configs.append(
            SingleTraceConfig(
                framework_name=fw["name"],
                model=config["model"],
                gpu_type=config["gpu_type"],
                batch_size_range=str(config.get("batch_size", "")),
                prefill_size_range=str(config.get("isl", "")),
                output_size_range=str(config.get("osl", "")),
                trace_file=trace_file,
                framework_source_code=fw["source_code"],
                trace_gpu_focus=str(fw.get("trace_gpu_focus", "0")),
                run_command=fw.get("run_command", ""),
                run_log=run_log,
                output_dir=config["claude_output_dir"],
            )
        )
    return configs


def build_cross_trace_config(config, single_configs):
    output_dir = config["claude_output_dir"]
    trace_results = []
    for sc in single_configs:
        trace_results.append(
            TraceResult(
                trace_id=sc.framework_name,
                framework_name=sc.framework_name,
                framework_source_code=sc.framework_source_code,
                median_block_file=f"{sc.framework_name}_{MEDIAN_BLOCK_FILE}",
                high_level_ops_file=f"{sc.framework_name}_{HIGH_LEVEL_OPS_FILE}",
            )
        )

    return CrossTraceConfig(
        trace_results=trace_results,
        analysis_type=ANALYSIS_TYPE_CROSS_FRAMEWORK,
        target_trace_id=config["target_framework"],
        model=config["model"],
        gpu_type=config["gpu_type"],
        output_dir=output_dir,
    )


def gen_analyze_step_prompts(config):
    single_configs = build_single_trace_configs(config)
    cross_config = build_cross_trace_config(config, single_configs)

    prompts = []

    for sc in single_configs:
        prefix = f"{sc.framework_name}_"
        st_prompts, _ = gen_single_trace_prompts(sc, file_prefix=prefix)
        prompts.extend(st_prompts)

    fw_names = [sc.framework_name for sc in single_configs]
    cross_prefix = f"{'_'.join(fw_names)}__"
    ct_prompts, _ = gen_cross_trace_prompts(cross_config, file_prefix=cross_prefix)
    prompts.extend(ct_prompts)

    return prompts


def build_claude_config(config):
    return ClaudeConfig(
        model="claude-opus-4-6[1m]",
        allowed_tools=["Read", "Write", "Bash"],
        perm_mode="acceptEdits",
        cwd=config["claude_output_dir"],
    )


if __name__ == "__main__":
    setup_logging("analyze")

    parser = argparse.ArgumentParser(description="Run analysis step")
    add_analyze_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    claude_config = build_claude_config(config)

    output_dir = config["claude_output_dir"]
    if os.path.exists(output_dir):
        safe_clean_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    print("=== Analyze ===")
    asyncio.run(
        claude_run(
            claude_config,
            gen_analyze_step_prompts(config),
        )
    )

    duration_time = time.time() - start_time
    print("FINISHED: duration = {:.1f}s".format(duration_time))
