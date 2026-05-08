"""
Generate a single_trace_config JSON from a test directory and framework name.

Usage:
    python -m auto_analyze.gen_single_trace_config \\
        --test-dir /path/to/test_results/nvidia__Model-tp_8-isl_4-osl_1024-b_1 \\
        --framework vllm \\
        --framework-source /path/to/vllm \\
        --trace-gpu-focus 0

The test directory is expected to have:
    <test_dir>/<framework>/
        run-log-<framework>-*.txt        (run log, first line has RUN-CMD: ...)
        trace-<framework>-*.sqlite       (NSYS trace) or trace-*.json (PyTorch trace)
        run_metadata.txt                 (GPU type, docker image, etc.)
"""

import os
import re
import sys
import json
import glob
import argparse


def parse_test_dir_name(test_dir):
    basename = os.path.basename(test_dir.rstrip("/"))
    result = {"model": "", "tp": "", "isl": "", "osl": "", "batch_size": ""}

    tp_match = re.search(r"-tp_(\d+)", basename)
    isl_match = re.search(r"-isl_(\d+)", basename)
    osl_match = re.search(r"-osl_(\d+)", basename)
    bs_match = re.search(r"-b_(\d+)", basename)

    if tp_match:
        result["tp"] = tp_match.group(1)
    if isl_match:
        result["isl"] = isl_match.group(1)
    if osl_match:
        result["osl"] = osl_match.group(1)
    if bs_match:
        result["batch_size"] = bs_match.group(1)

    model_part = basename
    for suffix in [f"-tp_{result['tp']}", f"-isl_{result['isl']}",
                   f"-osl_{result['osl']}", f"-b_{result['batch_size']}"]:
        if suffix and suffix != "-":
            model_part = model_part.split(suffix)[0]
    result["model"] = model_part.replace("__", "/")

    return result


def find_run_log(fw_dir, framework):
    matches = glob.glob(os.path.join(fw_dir, f"run-log-{framework}-*.txt"))
    non_profile = [m for m in matches if "profile" not in os.path.basename(m)]
    if non_profile:
        return non_profile[0]
    return matches[0] if matches else ""


def find_trace_file(fw_dir, framework):
    for pattern in [f"trace-{framework}-*.nsys-rep", f"trace-{framework}-*.sqlite",
                    f"trace-{framework}-*.json", f"trace-{framework}-*.json.gz"]:
        matches = glob.glob(os.path.join(fw_dir, pattern))
        if matches:
            return matches[0]
    return ""


def extract_run_command(run_log_path):
    if not run_log_path or not os.path.exists(run_log_path):
        return ""
    with open(run_log_path) as f:
        for line in f:
            if line.startswith("RUN-CMD:"):
                return line[len("RUN-CMD:"):].strip()
    return ""


def extract_gpu_type(fw_dir, test_dir):
    for metadata_path in [os.path.join(fw_dir, "run_metadata.txt"),
                          os.path.join(test_dir, "run_metadata_vllm.txt"),
                          os.path.join(test_dir, "run_metadata_sgl.txt")]:
        if not os.path.exists(metadata_path):
            continue
        with open(metadata_path) as f:
            for line in f:
                if line.strip().startswith("0,"):
                    gpu_name = line.strip().split(",", 1)[1].strip()
                    return gpu_name.replace("NVIDIA ", "")
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Generate single_trace_config JSON from a test directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--test-dir", required=True,
                        help="Path to test directory (e.g., .../nvidia__Model-tp_8-isl_4-osl_1024-b_1)")
    parser.add_argument("--framework", required=True,
                        help="Framework name (e.g., vllm, sgl, sglang, trt)")
    parser.add_argument("--framework-source", required=True,
                        help="Path to framework source code")
    parser.add_argument("--trace-gpu-focus", required=True,
                        help="GPU focus: 'ALL' or a GPU ID (e.g., '0')")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for analysis results (default: <test_dir>/analyze_<framework>)")
    parser.add_argument("--high-level-focus", default="",
                        help="Focus guidance for high-level block analysis (optional)")
    parser.add_argument("--perf-analysis-focus", default=None,
                        help="Focus areas for performance analysis (optional)")
    parser.add_argument("--analyze-non-block-ops", action="store_true", default=False,
                        help="Also analyze non-block ops (LM head, sampling, etc.)")
    parser.add_argument("--max-gpu-ops", type=int, default=2000,
                        help="Max GPU ops to extract (default: 2000)")
    args = parser.parse_args()

    test_dir = os.path.abspath(args.test_dir)
    fw_dir = os.path.join(test_dir, args.framework)

    if not os.path.isdir(test_dir):
        print(f"Error: test directory not found: {test_dir}")
        sys.exit(1)
    if not os.path.isdir(fw_dir):
        print(f"Error: framework directory not found: {fw_dir}")
        print(f"  Available: {[d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]}")
        sys.exit(1)

    parsed = parse_test_dir_name(test_dir)
    run_log = find_run_log(fw_dir, args.framework)
    trace_file = find_trace_file(fw_dir, args.framework)
    run_command = extract_run_command(run_log)
    gpu_type = extract_gpu_type(fw_dir, test_dir)

    if not trace_file:
        print(f"Error: no trace file found in {fw_dir}")
        sys.exit(1)
    if not run_command:
        print(f"Warning: no RUN-CMD found in run log, run_command will be empty")

    output_dir = args.output_dir or os.path.join(test_dir, f"analyze_{args.framework}")
    framework_source = os.path.abspath(args.framework_source)

    config = {
        "framework_name": args.framework,
        "model": parsed["model"],
        "gpu_type": gpu_type,
        "batch_size_range": parsed["batch_size"],
        "prefill_size_range": parsed["isl"],
        "output_size_range": parsed["osl"],
        "trace_file": trace_file,
        "framework_source_code": framework_source,
        "trace_gpu_focus": args.trace_gpu_focus,
        "run_command": run_command,
        "run_log": run_log,
        "high_level_focus": args.high_level_focus or None,
        "perf_analysis_focus": args.perf_analysis_focus,
        "analyze_non_block_ops": args.analyze_non_block_ops,
        "output_dir": output_dir,
        "max_gpu_ops": args.max_gpu_ops,
    }

    test_name = os.path.basename(test_dir)
    out_filename = f"single_trace_config_{test_name}_{args.framework}.json"
    out_path = os.path.join(os.path.dirname(test_dir), out_filename)

    with open(out_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Generated: {out_path}")
    print()
    print(f"  Model:          {config['model']}")
    print(f"  Framework:      {config['framework_name']}")
    print(f"  GPU:            {config['gpu_type']}")
    print(f"  Trace file:     {config['trace_file']}")
    print(f"  GPU focus:      {config['trace_gpu_focus']}")
    print(f"  Run command:    {config['run_command'][:80]}{'...' if len(config['run_command']) > 80 else ''}")
    print(f"  Run log:        {config['run_log']}")
    print(f"  Output dir:     {config['output_dir']}")
    print()
    print(f"To run: python -m auto_analyze.run_single_trace --config {out_path}")


if __name__ == "__main__":
    main()
