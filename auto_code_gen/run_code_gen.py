import sys
import time
import asyncio

from utils import Tee
from claude_utils import claude_run

from auto_code_gen.code_gen_configs import claude_config, code_gen_config
from auto_code_gen.code_gen_prompts import (
    gen_HighLevelCodePlanPrompt,
    gen_SmallPRsPrompt,
)

LOG_FILE = "__run_log_code_gen.txt"


def gen_prompts():
    high_level_code_plan_prompt = gen_HighLevelCodePlanPrompt(
        claude_config=claude_config, code_gen_config=code_gen_config
    )

    small_prs_prompt = gen_SmallPRsPrompt(
        claude_config=claude_config,
        code_gen_config=code_gen_config,
        high_level_code_plan_file=high_level_code_plan_prompt.output_file,
    )

    prompts = [high_level_code_plan_prompt.prompt(), small_prs_prompt.prompt()]
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
