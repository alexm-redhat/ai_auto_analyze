import sys
import time
import argparse
import asyncio

from pathlib import Path
from common.utils import Tee
from common.claude_utils import claude_run

from auto_code_gen.code_gen_configs import claude_config, code_gen_config
from auto_code_gen.code_gen_prompts import (
    create_context_str,
    gen_WorkItemsPrompt,
)

LOG_FILE = "__run_log_work_items.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute work items")

    parser.add_argument(
        "--code-gen-dir",
        required=True,
        type=Path,
        help="Path to the code generation results directory.",
    )

    parser.add_argument(
        "--work-items-file",
        required=True,
        type=Path,
        help="Path to the code generation results directory.",
    )

    return parser.parse_args()


def gen_prompts(args):
    context = create_context_str(claude_config, code_gen_config)

    work_items_prompt = gen_WorkItemsPrompt(
        context=context,
        code_gen_dir=args.code_gen_dir,
        work_items_file=args.work_items_file,
    )

    return [work_items_prompt.prompt()]


if __name__ == "__main__":
    # Redirect output to file as well
    log_file = open(LOG_FILE, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    args = parse_args()

    claude_config.cwd = args.code_gen_dir

    start_time = time.time()
    asyncio.run(claude_run(claude_config, gen_prompts(args)))
    duration_time = time.time() - start_time

    print("FINISHED ALL: total_duration = {}".format(duration_time))
