import sys
import time
import asyncio

from utils import Tee
from claude_utils import claude_run

from auto_code_gen.code_gen_configs import claude_config, code_gen_config, VLLM, SGLANG
# from auto_code_gen.code_gen_prompts import (
#     gen_HighLevelCodePlanPrompt,
#     gen_SmallPRsPrompt,
# )

from auto_code_gen.code_gen_prompts import (
    create_context_str,
    gen_CodeTracePrompt,
    gen_CodePortPlanPrompt,
    gen_ReviewCodePortPlanPrompt,
)

LOG_FILE = "__run_log_code_gen.txt"


def gen_prompts():
    context = create_context_str(claude_config, code_gen_config)
    frameworks = [SGLANG, VLLM]

    vllm_code_trace_prompt = gen_CodeTracePrompt(context=context, framework=VLLM)
    sglang_code_trace_prompt = gen_CodeTracePrompt(context=context, framework=SGLANG)

    framework_code_trace_files = [
        sglang_code_trace_prompt.output_file,
        vllm_code_trace_prompt.output_file,
    ]

    code_port_plan_prompt = gen_CodePortPlanPrompt(
        context=context,
        frameworks=frameworks,
        framework_code_trace_files=framework_code_trace_files,
    )

    review_code_port_plan_prompt = gen_ReviewCodePortPlanPrompt(
        context=context,
        frameworks=frameworks,
        framework_code_trace_files=framework_code_trace_files,
        code_port_plan_file=code_port_plan_prompt.output_file,
    )

    prompts = [
        sglang_code_trace_prompt.prompt(),
        vllm_code_trace_prompt.prompt(),
        code_port_plan_prompt.prompt(),
        review_code_port_plan_prompt.prompt(),
    ]
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
