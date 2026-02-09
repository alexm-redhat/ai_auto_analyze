import os
import sys
import time
import asyncio
import argparse

from utils import Tee
from claude_utils import claude_run

from claude_utils import ClaudeConfig
from auto_profile.parse_prompts import ParseResultsPrompt

LOG_FILE = "__run_log_parse.txt"

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
        output_dir=os.path.join(args.results_dir, args.output_dir),
    )

    prompts = [parse_results_prompt.prompt()]
    return prompts


if __name__ == "__main__":
    # Set log file
    log_file = open(LOG_FILE, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

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
    args = parser.parse_args()

    # Set CWD
    claude_config.cwd = args.results_dir
    
    # Run 
    start_time = time.time()
    asyncio.run(claude_run(claude_config, gen_parse_results_prompts(args)))
    duration_time = time.time() - start_time
    print("FINISHED ALL: total_duration = {}".format(duration_time))
