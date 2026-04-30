import time
import asyncio
import argparse

from common.utils import setup_logging
from common.claude_utils import claude_run

from auto_analyze.analyze_configs import (
    add_analyze_args,
    load_config,
    build_analyze_configs,
    build_claude_config,
)
from auto_analyze.analyze_prompts import gen_jira_tasks_prompt

def gen_create_jiras_step_prompts(config, analyze_configs):
    jira_tasks_prompt = gen_jira_tasks_prompt(
        model=config["model"],
        precision=config["precision"],
        gpu_type=config["gpu_type"],
        isl=config["isl"],
        osl=config["osl"],
        batch_size=config["batch_size"],
        configs=analyze_configs,
        target_framework=config["target_framework"],
    )

    return [jira_tasks_prompt]


if __name__ == "__main__":
    setup_logging("create_jiras")

    parser = argparse.ArgumentParser(description="Create JIRA tasks")
    add_analyze_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    analyze_configs = build_analyze_configs(config)
    claude_config = build_claude_config(config)
    claude_config.allowed_tools.append("mcp__mcp-atlassian__*")

    start_time = time.time()

    print("=== Create JIRA Tasks ===")
    asyncio.run(
        claude_run(
            claude_config,
            gen_create_jiras_step_prompts(config, analyze_configs),
        )
    )

    duration_time = time.time() - start_time
    print("FINISHED: duration = {:.1f}s".format(duration_time))
