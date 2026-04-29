import os
import time
import asyncio
import argparse

from common.utils import setup_logging, safe_clean_dir
from common.claude_utils import claude_run

from auto_analyze.analyze_configs import (
    add_analyze_args,
    load_config,
    build_analyze_configs,
    build_claude_config,
)
from auto_analyze.analyze_prompts import (
    gen_analyze_prompts,
    gen_perf_compare_prompt,
    gen_plan_prompt,
)

def gen_analyze_step_prompts(analyze_configs, target_framework):
    prompts = []
    block_files = []
    for analyze_config in analyze_configs:
        cur_prompts, block_file, _ = gen_analyze_prompts(analyze_config)
        prompts.extend(cur_prompts)
        block_files.append(block_file)

    perf_cmp_prompt, perf_cmp_file = gen_perf_compare_prompt(
        analyze_configs, block_files
    )

    plan_prompt, plan_file = gen_plan_prompt(
        analyze_configs,
        block_files,
        perf_cmp_file,
        target_framework,
    )

    prompts.append(perf_cmp_prompt)
    prompts.append(plan_prompt)

    return prompts


if __name__ == "__main__":
    setup_logging("analyze")

    parser = argparse.ArgumentParser(description="Run analysis step")
    add_analyze_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    analyze_configs = build_analyze_configs(config)
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
            gen_analyze_step_prompts(
                analyze_configs, config["target_framework"]
            ),
        )
    )

    duration_time = time.time() - start_time
    print("FINISHED: duration = {:.1f}s".format(duration_time))
