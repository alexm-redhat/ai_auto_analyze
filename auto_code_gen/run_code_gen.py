import sys
import time
import asyncio

from utils import Tee
from claude_utils import claude_run

from auto_code_gen.code_gen_configs import (
    claude_config,
    code_gen_config,
    VLLM,
    SGLANG,
    DISALLOWED_MODULES,
)

from auto_code_gen.code_gen_prompts import (
    create_context_str,
    gen_CodeTracePrompt,
    gen_CodePortPlanPrompt,
    gen_ReviewCodePortPlanPrompt,
    gen_TestPlanPrompt,
    gen_CodeGenPrompt,
    gen_ReviewCodeGenPrompt,
    CODE_PORT_PLAN_FILE_PREFIX,
    CODE_GEN_FILE_PREFIX,
)

LOG_FILE = "__run_log_code_gen.txt"


def gen_prompts():
    clear_vllm_dir_cmd = {
        "cmd": "utils.clear_vllm_source_tree(\"{}\")".format(
            code_gen_config.framework_source_codes[0]
        )
    }

    context = create_context_str(claude_config, code_gen_config)
    frameworks = [SGLANG, VLLM]

    vllm_code_trace_prompt = gen_CodeTracePrompt(context=context, framework=VLLM)
    sglang_code_trace_prompt = gen_CodeTracePrompt(context=context, framework=SGLANG)

    framework_code_trace_files = [
        sglang_code_trace_prompt.output_file,
        vllm_code_trace_prompt.output_file,
    ]

    num_plan_reviews = 4
    code_port_plan_and_review_prompts = []
    previous_code_port_plan_attempt_file = ""
    for i in range(num_plan_reviews):
        code_port_plan_file = (
            "{}_V{}_from_{}_to_{}.txt".format(
                CODE_PORT_PLAN_FILE_PREFIX, i + 1, frameworks[0], frameworks[1]
            ),
        )

        code_port_plan_prompt = gen_CodePortPlanPrompt(
            context=context,
            frameworks=frameworks,
            framework_code_trace_files=framework_code_trace_files,
            disallowed_modules=DISALLOWED_MODULES,
            previous_code_port_plan_attempt_file=previous_code_port_plan_attempt_file,
            output_file=code_port_plan_file,
        )

        code_port_plan_review_file = (
            "{}_V{}_review_from_{}_to_{}.txt".format(
                CODE_PORT_PLAN_FILE_PREFIX, i + 1, frameworks[0], frameworks[1]
            ),
        )
        code_port_plan_fixed_file = (
            "{}_V{}_fixed_from_{}_to_{}.txt".format(
                CODE_PORT_PLAN_FILE_PREFIX, i + 1, frameworks[0], frameworks[1]
            ),
        )

        output_total_review_summary_file = (
            "{}_V{}_total_review_evolution_from_{}_to_{}.txt".format(
                CODE_PORT_PLAN_FILE_PREFIX, i + 1, frameworks[0], frameworks[1]
            ),
        )
        review_code_port_plan_prompt = gen_ReviewCodePortPlanPrompt(
            context=context,
            frameworks=frameworks,
            framework_code_trace_files=framework_code_trace_files,
            code_port_plan_file=code_port_plan_file,
            output_review_file=code_port_plan_review_file,
            output_fixed_file=code_port_plan_fixed_file,
            output_total_review_summary_file=output_total_review_summary_file,
        )
        previous_code_port_plan_attempt_file = code_port_plan_fixed_file

        code_port_plan_and_review_prompts.append(code_port_plan_prompt.prompt())
        code_port_plan_and_review_prompts.append(review_code_port_plan_prompt.prompt())

    test_plan_prompt = gen_TestPlanPrompt(
        context=context,
        frameworks=frameworks,
        framework_code_trace_files=framework_code_trace_files,
        code_port_plan_file=previous_code_port_plan_attempt_file,
    )

    final_code_port_plan_file = previous_code_port_plan_attempt_file

    num_code_reviews = 3
    code_gen_and_review_prompts = []
    previous_code_gen_attempt_file = ""

    for i in range(num_code_reviews):
        code_pr_info_file = (
            "{}_V{}_PR_INFO_from_{}_to_{}.txt".format(
                CODE_GEN_FILE_PREFIX, i + 1, frameworks[0], frameworks[1]
            ),
        )
        code_pr_file = "{}_V{}_PR_from_{}_to_{}.patch".format(
            CODE_GEN_FILE_PREFIX, i + 1, frameworks[0], frameworks[1]
        )

        code_gen_prompt = gen_CodeGenPrompt(
            context=context,
            frameworks=frameworks,
            framework_code_trace_files=framework_code_trace_files,
            code_port_plan_file=final_code_port_plan_file,
            test_plan_file=test_plan_prompt.output_file,
            previous_code_gen_attempt_file=previous_code_gen_attempt_file,
            output_info_file=code_pr_info_file,
            output_pr_file=code_pr_file,
        )

        code_pr_review_file = "{}_V{}_PR_REVIEW_from_{}_to_{}.txt".format(
            CODE_GEN_FILE_PREFIX, i + 1, frameworks[0], frameworks[1]
        )

        code_pr_fixed_file = "{}_V{}_PR_FIXED_from_{}_to_{}.txt".format(
            CODE_GEN_FILE_PREFIX, i + 1, frameworks[0], frameworks[1]
        )

        code_pr_total_review_summary_file = (
            "{}_V{}_PR_TOTAL_REVIEW_EVOLUTION_from_{}_to_{}.txt".format(
                CODE_GEN_FILE_PREFIX, i + 1, frameworks[0], frameworks[1]
            )
        )

        code_review_prompt = gen_ReviewCodeGenPrompt(
            context=context,
            frameworks=frameworks,
            framework_code_trace_files=framework_code_trace_files,
            code_port_plan_file=final_code_port_plan_file,
            test_plan_file=test_plan_prompt.output_file,
            code_pr_info_file=code_pr_info_file,
            code_pr_file=code_pr_file,
            output_review_file=code_pr_review_file,
            output_fixed_file=code_pr_fixed_file,
            output_total_review_summary_file=code_pr_total_review_summary_file,
        )
        previous_code_gen_attempt_file = code_pr_fixed_file

        code_gen_and_review_prompts.append(clear_vllm_dir_cmd)
        code_gen_and_review_prompts.append(code_gen_prompt.prompt())
        code_gen_and_review_prompts.append(code_review_prompt.prompt())

    prompts = []

    prompts.append(clear_vllm_dir_cmd)
    prompts.append(sglang_code_trace_prompt.prompt())
    prompts.append(vllm_code_trace_prompt.prompt())
    prompts.extend(code_port_plan_and_review_prompts)
    prompts.append(test_plan_prompt.prompt())
    prompts.extend(code_gen_and_review_prompts)

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
