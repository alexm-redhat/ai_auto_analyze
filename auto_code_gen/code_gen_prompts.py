from dataclasses import dataclass
from typing import ClassVar

from auto_code_gen.code_gen_configs import ClaudeConfig, CodeGenConfig


@dataclass
class HighLevelCodePlanPrompt:
    claude_config: ClaudeConfig
    code_gen_config: CodeGenConfig
    output_file: str
    prompt_template: ClassVar[str] = """
<cwd> = {cwd}

<model> = {model}
<precision> = {precision}
<gpu_type> = {gpu_type}
<batch_size> = {batch_size}
<isl> = {isl}
<osl> = {osl}
<framework_names> = {framework_names}
<framework_source_codes> = {framework_source_codes}
<plan_file> = {plan_file}
<plan_step> = {plan_step}
<faster_framework> = {faster_framework}
<slower_framework> = {slower_framework}
<faster_framework_test_dir> = {faster_framework_test_dir}
<slower_framework_test_dir> = {slower_framework_test_dir}
<faster_transformer_block_high_level_ops_file> = {faster_transformer_block_high_level_ops_file}
<slower_transformer_block_high_level_ops_file> = {slower_transformer_block_high_level_ops_file}
<faster_median_transformer_block_file> = {faster_median_transformer_block_file}
<slower_median_transformer_block_file> = {slower_median_transformer_block_file}
<output_file> = {output_file}

You are performance expert for LLM inference engines and your goal is to fix performance issues in <slower_framework> by porting code pieces from <faster_framework> with minimal code changes to both frameworks, while focusing on the actual performance issue that needs to be fixed.

- <faster_transformer_block_high_level_ops_file> is a list of high-level transformer block operations of the faster framework.
- <slower_transformer_block_high_level_ops_file> is a list of high-level transformer block operations of the slower framework.
- <faster_median_transformer_block_file> is a median transformer block of the faster framework, with low-level kernel to high-level source code per-operation correlations. 
- <slower_median_transformer_block_file> is a median transformer block of the slower framework, with low-level kernel to high-level source code per-operation correlations. 

<test_id>=[model]-tp_$[num_gpus]-isl_[input_len]-osl_[output_len]
<test_id_with_batch>=<test_id>-b_[concurrency]
<full_test_id>=[framework]-<test_id_with_batch>

The <faster_framework_test_dir> and <slower_framework_test_dir> are the test/profile/run-logs directories for the faster and slower framework respectively. Each directory has the following format (that encodes test parameters): ../<test_id_with_batch>/[framework], and it includes the following files:
    - bench-<full_test_id>.json file that has the benchmark results of running the test on the framework
    - run-log-<full_test_id>.txt file that has the run log of executing the framework
    - run-log-profile-<full_test_id>.txt file that has the profile run log of executing the framework
    - trace-<full_test_id>.nsys-rep file that has the NSYS profile results
    - trace-<full_test_id>.sqlite file that is the SQLite version of the nsys-rep file	

The traces/logs provided are only for pure decode operations (no prefill).

- <framework_names> is a list of framework names involved
- <framework_source_codes> is a list of framework source codes that match respectively to <framework_names>
- The file <plan_file> has a sequence of improvement steps for <slower_framework> based on the comparison between frameworks running the model <model> in precision <precision> on <gpu_type> GPU, with ISL <isl>, OSL <osl> and batch size <batch_size>.

The goal of this task is to generate a multi-step high level coding plan for <slower_framework>, where the steps go from lower-level code changes to higher-level, that will implement the improvement step <plan_step> from <plan_file> where the faster framework is <faster_framework> and the slower is <slower_framework>. For the implementation, prefer to port code pieces from <faster_framework> to <slower_framework> with minimal changes to the code pieces, while adjusting the code pieces for <slower_framework> (if needed). Do the following and think hard:
- Analyze and understand in-detail the specific improvement <plan_step> by inspecting all relevant files and data.
- Focus on making the <slower_framework> faster for the specific execution that was tested here, so that code changes to <slower_framework> are minimal. 
- Inspect in-detail source codes, run logs, high-level transformer block files, and median transformer block files (low-level => high-level per-op) (and anything else needed) to determine the exact code-paths / code-pieces taken in both <faster_framework> and <slower_framework> for this specific test here.
- Detect and analyze in-detail the code-pieces from <faster_framework> that make it faster for this specific improvement. 
- Detect and analyze in-detail the code-pieces from <slower_framework> that make it slower for this specific improvement.
- Determine which code pieces from <faster_framework> make sense to port to the <slower_framework> with minimal changes to both the code pieces and the <slower_framework> structure.
- For the sequence of ported code pieces to <slower_framework>, document how the code pieces are ported from <faster_framework> to <slower_framework>, what is unchanged and what is changed and why. Be professional, clear and concise, so it is easy to understand. 
- Avoid importing code/modules/kernels directly from <faster_framework>, but instead port the codes to <slower_framework> by duplicating and adjusting the code in the <slower_framework>.
- Propose a multi-step high-level validation plan, where each step is a group of unit tests, and the steps go from lower-level code changes to higher-level. Ensure the following:
    - Full test code coverage, correctness and speed gains. 
    - End-to-end tests that execute the modified/new code-paths by providing highest level input tensors and verifying output tensors (vs baseline or known previous versions). If necessary, instantiate the modified/new classes/objects with the prepare/finalize codes that are required.
    - If possible, reuse existing code pieces from <slower_framework> or <faster_framework>. 

Finally:
- Review the work here for issues, and fix them. Repeat the review 3 times or more till high confidence is reached.

Output:
- Dump results to <cwd>/{output_file}
"""

    def prompt(self):
        return self.prompt_template.format(
            cwd=self.claude_config.cwd,
            model=self.code_gen_config.model,
            precision=self.code_gen_config.precision,
            gpu_type=self.code_gen_config.gpu_type,
            batch_size=self.code_gen_config.batch_size,
            isl=self.code_gen_config.isl,
            osl=self.code_gen_config.osl,
            framework_names=self.code_gen_config.framework_names,
            framework_source_codes=self.code_gen_config.framework_source_codes,
            plan_file=self.code_gen_config.plan_file,
            plan_step=self.code_gen_config.plan_step,
            faster_framework=self.code_gen_config.faster_framework,
            slower_framework=self.code_gen_config.slower_framework,
            faster_framework_test_dir=self.code_gen_config.faster_framework_test_dir,
            slower_framework_test_dir=self.code_gen_config.slower_framework_test_dir,
            faster_transformer_block_high_level_ops_file=self.code_gen_config.faster_transformer_block_high_level_ops_file,
            slower_transformer_block_high_level_ops_file=self.code_gen_config.slower_transformer_block_high_level_ops_file,
            faster_median_transformer_block_file=self.code_gen_config.faster_median_transformer_block_file,
            slower_median_transformer_block_file=self.code_gen_config.slower_median_transformer_block_file,
            output_file=self.output_file,
        )


HIGH_LEVEL_CODE_PLAN_FILE = "high_level_code_plan.txt"


def gen_HighLevelCodePlanPrompt(
    claude_config: ClaudeConfig, code_gen_config: CodeGenConfig
):
    return HighLevelCodePlanPrompt(
        claude_config=claude_config,
        code_gen_config=code_gen_config,
        output_file=HIGH_LEVEL_CODE_PLAN_FILE,
    )


@dataclass
class SmallPRsPrompt:
    claude_config: ClaudeConfig
    code_gen_config: CodeGenConfig
    high_level_code_plan_file: str
    output_file_prefix: str
    prompt_template: ClassVar[str] = """
<cwd> = {cwd}

<model> = {model}
<precision> = {precision}
<gpu_type> = {gpu_type}
<batch_size> = {batch_size}
<isl> = {isl}
<osl> = {osl}
<framework_names> = {framework_names}
<framework_source_codes> = {framework_source_codes}
<plan_file> = {plan_file}
<plan_step> = {plan_step}
<faster_framework> = {faster_framework}
<slower_framework> = {slower_framework}
<faster_framework_test_dir> = {faster_framework_test_dir}
<slower_framework_test_dir> = {slower_framework_test_dir}
<faster_transformer_block_high_level_ops_file> = {faster_transformer_block_high_level_ops_file}
<slower_transformer_block_high_level_ops_file> = {slower_transformer_block_high_level_ops_file}
<faster_median_transformer_block_file> = {faster_median_transformer_block_file}
<slower_median_transformer_block_file> = {slower_median_transformer_block_file}
<high_level_code_plan_file> = {high_level_code_plan_file}
<output_file_prefix> = {output_file_prefix}

- <faster_transformer_block_high_level_ops_file> is a list of high-level transformer block operations of the faster framework.
- <slower_transformer_block_high_level_ops_file> is a list of high-level transformer block operations of the slower framework.
- <faster_median_transformer_block_file> is a median transformer block of the faster framework, with low-level kernel to high-level source code per-operation correlations. 
- <slower_median_transformer_block_file> is a median transformer block of the slower framework, with low-level kernel to high-level source code per-operation correlations. 

<test_id>=[model]-tp_$[num_gpus]-isl_[input_len]-osl_[output_len]
<test_id_with_batch>=<test_id>-b_[concurrency]
<full_test_id>=[framework]-<test_id_with_batch>

The <faster_framework_test_dir> and <slower_framework_test_dir> are the test/profile/run-logs directories for the faster and slower framework respectively. Each directory has the following format (that encodes test parameters): ../<test_id_with_batch>/[framework], and it includes the following files:
    - bench-<full_test_id>.json file that has the benchmark results of running the test on the framework
    - run-log-<full_test_id>.txt file that has the run log of executing the framework
    - run-log-profile-<full_test_id>.txt file that has the profile run log of executing the framework
    - trace-<full_test_id>.nsys-rep file that has the NSYS profile results
    - trace-<full_test_id>.sqlite file that is the SQLite version of the nsys-rep file	

The traces provided are only for pure decode operations (no prefill).

- <framework_names> is a list of framework names involved
- <framework_source_codes> is a list of framework source codes that match respectively to <framework_names>
- The file <plan_file> has a sequence of improvement steps for <slower_framework> based on the comparison between frameworks running the model <model> in precision <precision> on <gpu_type> GPU, with ISL <isl>, OSL <osl> and batch size <batch_size>.
- The file <high_level_code_plan_file> has the multi-step high-level coding plan to apply the improvement <plan_step> to <slower_framework>.

Generate a code patch for <slower_framework> that will apply the improvement <plan_step> from <plan_file> based on the multi-step high-level coding plan from <high_level_code_plan_file>. Do the following and think hard:
- Prefer to port code-pieces from <faster_framework> with minimal modifications to both frameworks, so that the correctness will be preserved and less new code will be generated. Ensure the ported code pieces are adjusted as necessary for <slower_framework> code structure.
- If the code patch is too complex, then break down the problem to a sequence of small code patches. This will make code application and verification simpler.
- For each code patch, write tests to verify correctness, coverage and speed gains. 
- Write tests based on the high-level test plan from <high_level_code_plan_file>. Ensure all tests run (no skip) and all correct.
- Write end-to-end tests for full verification as follows:
    - Must execute all modified/new code-paths by providing input tensors (on the highest level possible)
    - Must verify output tensors by comparing to baseline (old code or known code that works from before the changes/PR)
    - Do not skip these important tests
    - There is no need to execute a full model for these tests, the goal is to instantiate the highest level classes/objects possible (ones that were modified or created), provide input tensors, and verify output tensors for correctness.
    - If possible, reuse existing test code pieces from <slower_framework>.
- Do not skip any tests, and make sure all of them run correctly. If something fails, then fix the related issues.

Finally:
- Apply the new code patches and tests
- Review the code patches for issues, and fix them. Repeat the review 3 times or more till high confidence is reached.
- Review the tests for issues, and fix them. Repeat the review 3 times or more till high confidence is reached.
- Run all of the tests, ensure no SKIP, all works, full correctness and full speed gains.

Output:
- Dump a brief explanation of the sequence of PRs and relevant info to <cwd>/<output_file_prefix>_info.txt so it will be easy to understand the summary of the work here.
- Dump the sequence of small code patches to <cwd>/<output_file_prefix>_[seq_id].patch
- Dump the final full code patch to <cwd>/<output_file_prefix>_full.patch  
- Dump all unit tests to <cwd>/
"""

    def prompt(self):
        return self.prompt_template.format(
            cwd=self.claude_config.cwd,
            model=self.code_gen_config.model,
            precision=self.code_gen_config.precision,
            gpu_type=self.code_gen_config.gpu_type,
            batch_size=self.code_gen_config.batch_size,
            isl=self.code_gen_config.isl,
            osl=self.code_gen_config.osl,
            framework_names=self.code_gen_config.framework_names,
            framework_source_codes=self.code_gen_config.framework_source_codes,
            plan_file=self.code_gen_config.plan_file,
            plan_step=self.code_gen_config.plan_step,
            faster_framework=self.code_gen_config.faster_framework,
            slower_framework=self.code_gen_config.slower_framework,
            faster_framework_test_dir=self.code_gen_config.faster_framework_test_dir,
            slower_framework_test_dir=self.code_gen_config.slower_framework_test_dir,
            faster_transformer_block_high_level_ops_file=self.code_gen_config.faster_transformer_block_high_level_ops_file,
            slower_transformer_block_high_level_ops_file=self.code_gen_config.slower_transformer_block_high_level_ops_file,
            faster_median_transformer_block_file=self.code_gen_config.faster_median_transformer_block_file,
            slower_median_transformer_block_file=self.code_gen_config.slower_median_transformer_block_file,
            high_level_code_plan_file=self.high_level_code_plan_file,
            output_file_prefix=self.output_file_prefix,
        )


PR_FILE_PREFIX = "pr_"


def gen_SmallPRsPrompt(
    claude_config: ClaudeConfig,
    code_gen_config: CodeGenConfig,
    high_level_code_plan_file: str,
):
    return SmallPRsPrompt(
        claude_config=claude_config,
        code_gen_config=code_gen_config,
        high_level_code_plan_file=high_level_code_plan_file,
        output_file_prefix=PR_FILE_PREFIX,
    )
