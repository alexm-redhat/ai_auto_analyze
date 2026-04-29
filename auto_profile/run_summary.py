import os
import sys
import time
import asyncio
import argparse

from common.utils import setup_logging, safe_clean_dir
from common.claude_utils import claude_run

from common.claude_utils import ClaudeConfig
from auto_profile.parse_prompts import ParseResultsPrompt

claude_config = ClaudeConfig(
    model="claude-opus-4-6",
    # model="claude-opus-4-5",
    allowed_tools=["Read", "Write", "Bash"],
    perm_mode="acceptEdits",  # "bypassPermissions",
    cwd=None,
)


def gen_parse_results_prompts(args):
    parse_results_prompt = ParseResultsPrompt(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )

    prompts = [parse_results_prompt.prompt()]
    return prompts


if __name__ == "__main__":
    setup_logging("parse")

    # Parse args
    parser = argparse.ArgumentParser(description="Parse auto_profile results")

    parser.add_argument(
        "--results-dir",
        required=True,
        type=str,
        help="Path to auto_profile results directory",
    )
    parser.add_argument(
        "--output-dir", required=True, type=str, help="Path to outputs directory"
    )
    parser.add_argument(
        "--override-output-dir",
        action="store_true",
        help="If set, allow output directory to exist and clean it before proceeding",
    )
    args = parser.parse_args()

    # Check that output_dir does not already exist (unless override is set)
    output_dir = args.output_dir
    if os.path.exists(output_dir):
        if args.override_output_dir:
            safe_clean_dir(output_dir)
        else:
            print(f"Error: output directory already exists: {output_dir}")
            sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Set CWD
    claude_config.cwd = os.getcwd()

    # Run
    start_time = time.time()
    asyncio.run(claude_run(claude_config, gen_parse_results_prompts(args)))
    duration_time = time.time() - start_time
    print("FINISHED ALL: total_duration = {}".format(duration_time))
