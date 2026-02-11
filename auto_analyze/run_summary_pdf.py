import os
import sys
import time
import argparse
import asyncio

from utils import Tee
from claude_utils import claude_run

from auto_analyze.analyze_configs import claude_config
from auto_analyze.analyze_prompts import SummaryPDFPrompt

LOG_FILE = "__run_log_gen_pdf.txt"


def gen_prompts(args):
    summary_pdf_prompt = SummaryPDFPrompt(
        cmp_file=args.cmp_file,
        plan_file=args.plan_file,
        output_file=args.output_pdf_file,
    )

    prompts = [summary_pdf_prompt.prompt()]
    return prompts


if __name__ == "__main__":
    # Redirect output to file as well
    log_file = open(LOG_FILE, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    # Parse args
    parser = argparse.ArgumentParser(description="Generate PDF summary")

    parser.add_argument(
        "--cmp-file",
        required=True,
        type=str,
        help="Path to cmp file",
    )
    parser.add_argument(
        "--plan-file",
        required=True,
        type=str,
        help="Path to plan file",
    )
    parser.add_argument(
        "--output-pdf-file", required=True, type=str, help="Output PDF path"
    )
    args = parser.parse_args()

    # Set CWD
    claude_config.cwd = os.path.dirname(args.output_pdf_file)

    start_time = time.time()
    asyncio.run(claude_run(claude_config, gen_prompts(args)))
    duration_time = time.time() - start_time

    print("FINISHED ALL: total_duration = {}".format(duration_time))
