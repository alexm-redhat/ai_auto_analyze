"""Run runtime iterations for a specific model configuration.

Standalone entry point for runtime iterations after the code gen pipeline
has produced a patch. By default, runs with the original benchmarked model.
Can also run with a smaller auto-detected model, or a specific model from
additional_benchmark_configs.

Each model's results (logs, success file) go into a model-specific
subdirectory (runtime_<model_slug>/) inside output_dir. Patches are
shared across models in the main output_dir.

Usage:

    # Default: run with original model
    python -m auto_code_gen.run_runtime_iters --config <config.json>

    # Run with auto-detected smaller model
    python -m auto_code_gen.run_runtime_iters --config <config.json> --use-smaller-model

    # Run with a specific additional model by index (0-based)
    python -m auto_code_gen.run_runtime_iters --config <config.json> --model-index 0

    # Run ALL models (original + smaller if enabled + all additional)
    python -m auto_code_gen.run_runtime_iters --config <config.json> --all-models

    # Resume from last incomplete iteration
    python -m auto_code_gen.run_runtime_iters --config <config.json> --resume
"""

import os as _os
import sys as _sys

_project_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import os
import sys
import time
import argparse
import asyncio

from common.utils import setup_logging
from common.claude_utils import claude_run, PipelineStep

from auto_code_gen.code_gen_configs import CodeGenConfig
from auto_code_gen.use_cases.llm_framework import LLMFrameworkUseCase

from auto_code_gen.code_gen_prompts import (
    code_trace_filename,
    RUNTIME_SUCCESS_FILE,
    RUNTIME_RESULTS_MANIFEST,
)

from auto_code_gen.run_code_gen import (
    _find_latest_patch,
    _check_runtime_success,
    _make_model_runtime_dir,
    _format_duration,
    _format_tokens,
    _print_detailed_table,
    _print_phase_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run runtime iterations (standalone mode). "
        "Runs after run_code_gen.py has produced a code patch."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the code generation JSON config file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last incomplete runtime iteration.",
    )

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--use-smaller-model",
        action="store_true",
        help="Run with an auto-detected smaller model instead of the original.",
    )
    model_group.add_argument(
        "--model-index",
        type=int,
        default=None,
        help="Run a specific additional model by index (0-based from additional_benchmark_configs).",
    )
    model_group.add_argument(
        "--all-models",
        action="store_true",
        help="Run ALL models: original, smaller (if enabled), and all additional.",
    )
    return parser.parse_args()


def _build_model_list(args, config, use_case):
    """Build the list of model configs to run based on CLI args."""
    models = []

    if args.all_models:
        # Original model
        models.append({
            "model": config.model,
            "label": "original",
            "execution_command": config.target_run_command,
            "smaller_model_file": None,
            "disable_new_feature": config.disable_new_feature_for_runtime,
        })
        # Smaller model (if enabled in config)
        if config.use_smaller_model_for_runtime:
            models.append({
                "model": "smaller",
                "label": "smaller",
                "execution_command": config.target_run_command,
                "smaller_model_file": "__pending__",
                "disable_new_feature": config.disable_new_feature_for_runtime,
            })
        # All additional models
        for bench_cfg in config.additional_benchmark_configs:
            model = bench_cfg.get("model", "")
            num_gpus = bench_cfg.get("num_gpus", 8)
            label = bench_cfg.get("label", model)
            exec_cmd = bench_cfg.get("run_command", "")
            if not exec_cmd:
                exec_cmd = use_case._derive_run_command(
                    config.target_run_command, model, num_gpus
                )
            models.append({
                "model": model,
                "label": label,
                "execution_command": exec_cmd,
                "smaller_model_file": None,
                "disable_new_feature": config.disable_new_feature_for_runtime,
            })

    elif args.use_smaller_model:
        models.append({
            "model": "smaller",
            "label": "smaller",
            "execution_command": config.target_run_command,
            "smaller_model_file": "__pending__",
            "disable_new_feature": config.disable_new_feature_for_runtime,
        })

    elif args.model_index is not None:
        idx = args.model_index
        if idx < 0 or idx >= len(config.additional_benchmark_configs):
            print("ERROR: model-index {} out of range (0-{})".format(
                idx, len(config.additional_benchmark_configs) - 1
            ))
            sys.exit(1)
        bench_cfg = config.additional_benchmark_configs[idx]
        model = bench_cfg.get("model", "")
        num_gpus = bench_cfg.get("num_gpus", 8)
        label = bench_cfg.get("label", model)
        exec_cmd = bench_cfg.get("run_command", "")
        if not exec_cmd:
            exec_cmd = use_case._derive_run_command(
                config.target_run_command, model, num_gpus
            )
        models.append({
            "model": model,
            "label": label,
            "execution_command": exec_cmd,
            "smaller_model_file": None,
            "disable_new_feature": config.disable_new_feature_for_runtime,
        })

    else:
        # Default: original model
        models.append({
            "model": config.model,
            "label": "original",
            "execution_command": config.target_run_command,
            "smaller_model_file": None,
            "disable_new_feature": config.disable_new_feature_for_runtime,
        })

    return models


def _find_code_trace_files(output_dir, code_gen_config):
    source_fw = code_gen_config.source_framework
    target_fw = code_gen_config.target_framework
    files = [
        os.path.join(output_dir, code_trace_filename(source_fw)),
        os.path.join(output_dir, code_trace_filename(target_fw)),
    ]
    for f in files:
        if not os.path.isfile(f):
            raise FileNotFoundError(
                "Code trace file not found: {}\n"
                "  Run the code generation pipeline (run_code_gen.py) first.".format(f)
            )
    return files


def _find_latest_plan_file(output_dir, prefix):
    import re as _re
    pattern = _re.compile(r'^{}_review_V(\d+)\.txt$'.format(_re.escape(prefix)))
    latest = None
    latest_ver = 0
    for f in os.listdir(output_dir):
        m = pattern.match(f)
        if m:
            ver = int(m.group(1))
            if ver > latest_ver:
                latest_ver = ver
                latest = f
    if latest is None:
        pattern2 = _re.compile(r'^{}_V(\d+)\.txt$'.format(_re.escape(prefix)))
        for f in os.listdir(output_dir):
            m = pattern2.match(f)
            if m:
                ver = int(m.group(1))
                if ver > latest_ver:
                    latest_ver = ver
                    latest = f
    return latest


async def run_standalone_runtime_iterations(code_gen_config, claude_config,
                                            resume=False, run_models=None):
    use_case = LLMFrameworkUseCase()
    context = use_case.create_context_str(claude_config, code_gen_config)
    output_dir = code_gen_config.output_dir

    # Verify target branch
    print("Verifying target framework branch...")
    if not code_gen_config.verify_target_branch():
        print("\nERROR: Target repo is not on the expected branch.")
        print("  Expected branch: {}".format(code_gen_config.get_target_branch_name()))
        print("  Please clean the repo and retry.")
        sys.exit(1)

    # Find code trace files from previous pipeline run
    code_trace_files = _find_code_trace_files(output_dir, code_gen_config)

    # Find latest plan files
    code_port_plan_file = _find_latest_plan_file(output_dir, "code_port_plan") or ""
    test_plan_file = _find_latest_plan_file(output_dir, "test_plan") or ""

    phase_results, all_step_timings = await use_case.run_runtime_iterations(
        context, code_gen_config, claude_config,
        code_trace_files=code_trace_files,
        code_port_plan_file=code_port_plan_file,
        test_plan_file=test_plan_file,
        resume=resume,
        run_models=run_models,
    )

    return phase_results, all_step_timings


if __name__ == "__main__":
    args = parse_args()

    code_gen_config = CodeGenConfig.from_json(args.config)
    claude_config = code_gen_config.make_claude_config()
    use_case = LLMFrameworkUseCase()

    setup_logging("runtime_iters")

    os.makedirs(code_gen_config.output_dir, exist_ok=True)

    run_models = _build_model_list(args, code_gen_config, use_case)

    print("\n" + "=" * 80)
    print("STANDALONE RUNTIME ITERATIONS")
    print("=" * 80)
    print("  Output directory: {}".format(code_gen_config.output_dir))
    print("  Models to run: {}".format(len(run_models)))
    for m in run_models:
        print("    - {} ({})".format(m["label"], m["model"]))
    print("=" * 80 + "\n")

    start_time = time.time()
    phase_results, all_step_timings = asyncio.run(
        run_standalone_runtime_iterations(
            code_gen_config, claude_config,
            resume=args.resume,
            run_models=run_models,
        )
    )
    total_duration = time.time() - start_time

    if all_step_timings:
        _print_detailed_table(all_step_timings, total_duration)
    if phase_results:
        _print_phase_summary(phase_results, all_step_timings, total_duration)

    # Check results per model
    for m in run_models:
        model_dir = _make_model_runtime_dir(code_gen_config.output_dir, m["label"])
        if _check_runtime_success(model_dir):
            print("\n  {} — SUCCEEDED".format(m["label"]))
        else:
            print("\n  {} — DID NOT SUCCEED".format(m["label"]))

    manifest_path = os.path.join(code_gen_config.output_dir, RUNTIME_RESULTS_MANIFEST)
    if os.path.isfile(manifest_path):
        print("\nResults manifest: {}".format(manifest_path))
