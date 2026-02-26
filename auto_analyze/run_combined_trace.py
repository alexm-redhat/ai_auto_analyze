import os
import sys
import time
import argparse
import asyncio

from utils import Tee
from claude_utils import claude_run

from auto_analyze.analyze_configs import (
    MODEL,
    PRECISION,
    GPU_TYPE,
    ISL,
    OSL,
    BATCH_SIZE,
    claude_config,
    analyze_configs,
)
from auto_analyze.analyze_prompts import gen_combined_trace_prompt

LOG_FILE = "__run_log_gen_combined_nsys.txt"


def gen_prompts(args):
    combined_trace_prompt = gen_combined_trace_prompt(
        model=MODEL,
        precision=PRECISION,
        gpu_type=GPU_TYPE,
        isl=ISL,
        osl=OSL,
        batch_size=BATCH_SIZE,
        configs=analyze_configs,
    )

    prompts = [combined_trace_prompt]
    return prompts


if __name__ == "__main__":
    # Redirect output to file as well
    log_file = open(LOG_FILE, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    # Parse args
    parser = argparse.ArgumentParser(description="Generate PDF summary")
    args = parser.parse_args()

    start_time = time.time()
    asyncio.run(claude_run(claude_config, gen_prompts(args)))
    duration_time = time.time() - start_time

    print("FINISHED ALL: total_duration = {}".format(duration_time))
