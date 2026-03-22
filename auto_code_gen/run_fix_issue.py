import sys
import time
import argparse
import asyncio

from pathlib import Path
from utils import Tee
from claude_utils import claude_run

from auto_code_gen.code_gen_configs import claude_config, code_gen_config
from auto_code_gen.code_gen_prompts import gen_FixIssuePrompt

LOG_FILE = "__run_log_code_gen.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix issue in PR"
    )
    parser.add_argument(
        "--high_level_code_plan_file",
        required=True,
        type=Path,
        help="Path to the high-level code plan file.",
    )
    parser.add_argument(
        "--prs_dir",
        required=True,
        type=Path,
        help="Path to the directory containing PR files.",
    )
    parser.add_argument(
        "--issue_to_fix_file",
        required=True,
        type=Path,
        help="Path to the issue-to-fix file.",
    )

    parser.add_argument(
        "--issue_cwd",
        required=True,
        type=Path,
        help="Path to the issue CWD.",
    )
    return parser.parse_args()


def gen_prompts(args):
    fix_issue_prompt = gen_FixIssuePrompt(
        claude_config=claude_config, 
        code_gen_config=code_gen_config, 
        high_level_code_plan_file=args.high_level_code_plan_file,
        prs_dir=args.prs_dir,
        issue_to_fix_file=args.issue_to_fix_file,
        issue_cwd=args.issue_cwd,
    )

    prompts = [fix_issue_prompt.prompt()]
    return prompts


if __name__ == "__main__":
    # Redirect output to file as well
    log_file = open(LOG_FILE, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    args = parse_args()

    start_time = time.time()
    asyncio.run(claude_run(claude_config, gen_prompts(args)))
    duration_time = time.time() - start_time

    print("FINISHED ALL: total_duration = {}".format(duration_time))
