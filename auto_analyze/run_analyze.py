import sys
import time
import asyncio

from utils import Tee
from claude_utils import claude_run

from auto_analyze.analyze_configs import claude_config, analyze_configs
from auto_analyze.analyze_prompts import gen_analyze_prompts, gen_perf_compare_prompt, gen_plan_prompt

LOG_FILE = "__run_log_analyze.txt"

def gen_prompts():
    prompts = []
    block_files = []
    for analyze_config in analyze_configs:
        cur_prompts, block_file = gen_analyze_prompts(analyze_config)
        prompts.extend(cur_prompts)
        block_files.append(block_file)

    perf_cmp_prompt, perf_cmp_file = gen_perf_compare_prompt(analyze_configs, block_files)
    plan_prompt = gen_plan_prompt(analyze_configs, block_files, perf_cmp_file)
    
    prompts.append(perf_cmp_prompt)
    prompts.append(plan_prompt)

    return prompts


if __name__ == "__main__":
    # Redirect output to file as well
    log_file = open(LOG_FILE, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    start_time = time.time()
    asyncio.run(claude_run(claude_config, gen_prompts()))
    duration_time = time.time() - start_time

    print("FINISHED ALL: total_duration = {}".format(duration_time))
