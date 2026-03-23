from dataclasses import dataclass
from typing import ClassVar

from auto_code_gen.code_gen_configs import ClaudeConfig, CodeGenConfig


def create_context_str(claude_config: ClaudeConfig, code_gen_config: CodeGenConfig):
    return """
<context>

<cwd> 
{cwd} 
</cwd>

<model>
{model}
</model>
<precision>
{precision}
</precision>
<gpu_type>
{gpu_type}
</gpu_type>
<batch_size>
{batch_size}
</batch_size>
<isl>
{isl}
</isl>
<osl>
{osl}
</osl>

<tested_execution>
Execution of model <model> in precision <precision> on <gpu_type> GPU with ISL <isl>, OSL <osl> and batch size <batch_size>
</tested_execution>

<framework_names>
{framework_names}
</framework_names>
<framework_source_codes>
{framework_source_codes}
</framework_source_codes>
<framework_test_dirs>
{framework_test_dirs}
</framework_test_dirs>
<transformer_block_high_level_ops_files>
{transformer_block_high_level_ops_files}
</transformer_block_high_level_ops_files>
<median_transformer_block_files>
{median_transformer_block_files}
</median_transformer_block_files>

<plan_file>
{plan_file}
</plan_file>
<plan_step>
{plan_step}
</plan_step>

</context>

<definitions>
<code_trace> 
code-paths, code-pieces, and their associated call-chains 
</code_trace>
</definitions>

<context_explanations>
- <cwd> is the current working directory

- <framework_names> is the list of frameworks involved.
- <framework_source_codes> is the list of framework source codes for <framework_names> respectively.
- <framework_test_dirs> is the list of test directories for <framework_names> respectively, testing the <tested_execution>. Each directory has a run-log-*.txt that can be inspected to detect the active code pieces during the run of <tested_execution>.
- <transformer_block_high_level_ops_files> is the list of high-level transformer block operation files for <framework_names> respectively.
- <median_transformer_block_files> is the list of median low-level => high-level transformer block operation files for <framework_names> respectively. 

- The file <plan_file> has a sequence of improvement steps for <slower_framework> based on the comparison between frameworks running <tested_execution>.
- The improvement plan step <plan_step> from <plan_file> is what we want to implement for <slower_framework>
</context_explanations>

""".format(
        cwd=claude_config.cwd,
        model=code_gen_config.model,
        precision=code_gen_config.precision,
        gpu_type=code_gen_config.gpu_type,
        batch_size=code_gen_config.batch_size,
        isl=code_gen_config.isl,
        osl=code_gen_config.osl,
        framework_names=code_gen_config.framework_names,
        framework_source_codes=code_gen_config.framework_source_codes,
        framework_test_dirs=code_gen_config.framework_test_dirs,
        transformer_block_high_level_ops_files=code_gen_config.transformer_block_high_level_ops_files,
        median_transformer_block_files=code_gen_config.median_transformer_block_files,
        plan_file=code_gen_config.plan_file,
        plan_step=code_gen_config.plan_step,
    )


@dataclass
class CodeTracePrompt:
    context: str
    framework: str
    output_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<framework>
{framework}
</framework>
</definitions>

<instructions>
The goal of this task is to detect the <code_trace> inside <framework> that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>). Think hard for this task and follow these guidelines:
- Analyze and understand in-detail the improvement plan step <plan_step> by inspecting all relevant files and data.
- Detect the <code_trace> inside <framework> that is relevant for improvement plan step <plan_step>.
- For the detected <code_trace>, perform a detailed analysis of how exactly the <code_trace> executes for the following iteration modes:
    - decode-only runs
    - prefill-only runs
    - a mix of prefill and decode runs
- Document each execution mode with its details, step-by-step from the high-level point of the <code_trace> to the lower-level point. Provide all critical details, which includes:
    - For each function in the trace, provide its goal, info, inputs (+shapes), and outputs (+shapes)
    - For the lowest level functions, inspect their source code in detail, and understand their inputs/outputs, including shapes and function assumptions.
    - Document classes/objects and their relations
    - Be professional, clear and consice, while documenting the trace gradually, in an incremental way from top to bottom.
- Document how cuda graphs affect each execution mode as well and how it is handled in the codebase
</instructions>

<output>
- Dump results to <cwd>/{output_file}
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            framework=self.framework,
            output_file=self.output_file,
        )


CODE_TRACE_FILE = "code_trace.txt"


def gen_CodeTracePrompt(
    context: str,
    framework: str,
):
    return CodeTracePrompt(
        context=context,
        framework=framework,
        output_file="{}_{}".format(framework, CODE_TRACE_FILE),
    )


@dataclass
class CodePortPlanPrompt:
    context: str
    frameworks: list[str]
    framework_code_trace_files: list[str]
    output_file: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<frameworks>
{frameworks}
</frameworks>
<framework_code_trace_files>
{framework_code_trace_files}
</framework_code_trace_files>
</definitions>

<definition_explanations>
- <frameworks> is a list of 2 frameworks, where the first framework is the "source" and the second is the "target". 
- <framework_code_trace_files> is a list of code trace files for <frameworks> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
</definition_explanations>

<instructions>
The goal of this task is to provide a high-level multi-step coding plan that implements the improvement plan step <plan_step> (from <plan_file>) inside "target" framework by porting <code_trace> parts from the "source" framework to "target" framework. Think hard for this task and follow these guidelines:
- Analyze and understand in-detail the code traces in <framework_code_trace_files>.
- Detect <code_trace> parts of the "source" framework to port to "target" framework. Maximize the amount of parts that are ported AS IS with MINIMAL necessary changes to frameworks. Take into account the execution modes (and their effects on inputs/outputs/shapes and call-chains):
    - decode-only
    - prefill-only
    - mixed prefill and decode
    - For each mode above, iterate over various sizes of tokens, including edge cases.
- For each ported code part, ensure minimal changes to the part and minimal changes to the "target" framework. 
- For each ported code part, if code adjustments are necessary due to "target" framework constraints, then try best to minimize these code adjustments as much as possible.
- For each ported code part, provide step-by-step integration coding details into "target" framework. Make sure to take care of decode-only, prefill-only and mix execution modes correctly. 
- For each ported code part, provide porting idea documentation, with what is ported, why, what is unchanged, and what is changed, and why. Be clear, concise and professional.
- For any code part in "target" framework that is used here, ensure it is used correctly with the ported code parts. 
    - Verify inputs/outputs for all execution modes: decode, prefill, and mix
    - Verify all constraints and dependencies
- Ensure cuda graphs are handled properly for all execution modes
- Ensure the end-to-end multi-step plan is coherent, bug-free and works for all execution modes:
    - Sanity and verify shapes of inputs/outputs
    - Verify API usage is fully correct and coherent
    - Verify the lowest level parts are used correctly
    - Inspect and trace all of the necessary source code points that are sensitive.
- Provide risk analysis and potential sensitive breaking points
</instructions>

<output>
- Dump the high-level multi-step coding plan to <cwd>/{output_file}
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            frameworks=self.frameworks,
            framework_code_trace_files=self.framework_code_trace_files,
            output_file=self.output_file,
        )


CODE_PORT_PLAN_PREFIX = "code_port_plan"


def gen_CodePortPlanPrompt(
    context: str,
    frameworks: list[str],
    framework_code_trace_files: list[str],
):
    assert len(frameworks) == 2
    assert len(frameworks) == len(framework_code_trace_files)

    return CodePortPlanPrompt(
        context=context,
        frameworks=frameworks,
        framework_code_trace_files=framework_code_trace_files,
        output_file="{}_from_{}_to_{}.txt".format(
            CODE_PORT_PLAN_PREFIX, frameworks[0], frameworks[1]
        ),
    )

@dataclass
class ReviewCodePortPlanPrompt:
    context: str
    frameworks: list[str]
    framework_code_trace_files: list[str]
    code_port_plan_file: list[str]
    output_file_1: str
    output_file_2: str
    prompt_template: ClassVar[str] = """

{context}

<definitions>
<frameworks>
{frameworks}
</frameworks>
<framework_code_trace_files>
{framework_code_trace_files}
</framework_code_trace_files>
<code_port_plan_file>
{code_port_plan_file}
</code_port_plan_file>
</definitions>

<definition_explanations>
- <frameworks> is a list of 2 frameworks, where the first framework is the "source" and the second is the "target". 
- <framework_code_trace_files> is a list of code trace files for <frameworks> respectively. Each code trace file describes the <code_trace> of the specific framework that is active during the execution of <tested_execution> for improvement plan step <plan_step> (from <plan_file>).
- <code_port_plan_file> is a file that describes the high-level multi-step coding plan that implements the improvement plan step <plan_step> (from <plan_file>) inside "target" framework.
</definition_explanations>

<instructions>
The goal of this task is to perform a critical review of the high-level multi-step coding plan in <code_port_plan_file>. Do the following and think hard:
- Understand in-detail the plan in <code_port_plan_file>, and restate the plan step-by-step.
- Review the plan for:
    - End-to-end execution modes: decode-only, prefill-only and mixed. Fully trace each execution, with inputs/outputs/shapes, assumptions, connections etc, to verify all is correct.
    - Execution with cuda graph enabled
    - If a low-level (or third party) kernel is invoked, then verify all input parameters and assumption of this kernel. Go deep and analyze kernel's source code as well, by fetching it from whether it is located.
    - Ensure the plan fully implements the related improvement plan step, so that default execution will trigger it.
    - Ensure proper memory management of CPU/GPU buffers
    - Ensure proper scheduling constraints and separation between prefill/decode/mixed (if needed)
    - Any other issue that is important for "target" framework integrity
    - In general, these points too:
        - incorrect assumptions
        - missing steps
        - bad ordering or sequencing
        - ambiguity or vagueness
        - missing edge cases
        - architectural risks
        - hidden dependencies
        - places where the plan is not actionable enough
        - places where validation/testing is missing
        - places where the plan may solve the wrong problem
        - places where the plan is too broad or too low-level
        - any part that is not aligned with the original task or issue
- For each issue found, document:
    - The affected part of the plan
    - What is wrong
    - Why it matters
    - How it should be improved
- Produce a corrected plan with all issues fixed.

</instructions>

<output>
- Dump the documentation of the issues fixed to <cwd>/{output_file_1}
- Dump the corrected plan to <cwd>/{output_file_2}
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.context,
            frameworks=self.frameworks,
            framework_code_trace_files=self.framework_code_trace_files,
            code_port_plan_file=self.code_port_plan_file,
            output_file_1=self.output_file_1,
            output_file_2=self.output_file_2,
        )


def gen_ReviewCodePortPlanPrompt(
    context: str,
    frameworks: list[str],
    framework_code_trace_files: list[str],
    code_port_plan_file: str,
):
    assert len(frameworks) == 2
    assert len(frameworks) == len(framework_code_trace_files)

    return ReviewCodePortPlanPrompt(
        context=context,
        frameworks=frameworks,
        framework_code_trace_files=framework_code_trace_files,
        code_port_plan_file=code_port_plan_file,
        output_file_1="{}_from_{}_to_{}_review_info.txt".format(
            CODE_PORT_PLAN_PREFIX, frameworks[0], frameworks[1]
        ),
        output_file_2="{}_v2_from_{}_to_{}.txt".format(
            CODE_PORT_PLAN_PREFIX, frameworks[0], frameworks[1]
        ),
    )


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
- Ensure to read PDFs if exist. Install necessary tools or codes. Do not skip it.

The goal of this task is to generate a multi-step high level coding plan for <slower_framework>, where the steps go from lower-level code changes to higher-level, that will implement the improvement step <plan_step> from <plan_file> where the faster framework is <faster_framework> and the slower is <slower_framework>. For the implementation, prefer to port code pieces from <faster_framework> to <slower_framework> with minimal changes to the code pieces, while adjusting the code pieces for <slower_framework> (if needed). Do the following and think hard:
- Analyze and understand in-detail the specific improvement <plan_step> by inspecting all relevant files and data.
- Focus on making the <slower_framework> faster for the specific execution that was tested here, so that code changes to <slower_framework> are minimal. 
- Inspect in-detail source codes, run logs, high-level transformer block files, and median transformer block files (low-level => high-level per-op) (and anything else needed) to determine the exact code-paths / code-pieces taken in both <faster_framework> and <slower_framework> for this specific test and improvement step here.
- Detect and analyze in-detail the code-pieces from <faster_framework> that make it faster for this specific improvement. 
    - Ensure to detect and understand in-detail the call-chain that is invoked from high-level functions/objects to lower-level functions/objects for this specific test and improvement step here. 
    - Find the related commits and their descriptions to get more info/documentation about the active/found code-pieces and their related call-chain.  
    - Consider and understand in-detail how the detected call-chain works for prefill-only, decode-only, and mix of prefill-decode execution modes.
- Detect and analyze in-detail the code-pieces from <slower_framework> that make it slower for this specific improvement. 
    - Ensure to detect and understand in-detail the call-chain that is invoked from high-level functions/objects to lower-level functions/objects for this specific test and improvement step here. 
    - Find the related commits and their descriptions to get more info/documentation about the active/found code-pieces and their related call-chain. 
    - Consider and understand in-detail how the detected call-chain works for prefill-only, decode-only, and mix of prefill-decode execution modes.
- Determine which code pieces from <faster_framework> make sense to port to the <slower_framework> with minimal changes to both the code pieces and the <slower_framework> structure.
    - Ensure prefill-only, decode-only and mix of prefill-decode execution modes are handled in <slower_framework> in a similar way to <faster_framework>.
- For the sequence of ported code pieces to <slower_framework>, document how the code pieces are ported from <faster_framework> to <slower_framework>, what is unchanged and what is changed and why. Be professional, clear and concise. Ensure to explain the changes vs the call-chain that is invoked.
- Avoid importing code/modules/kernels directly from <faster_framework>, but instead port the codes to <slower_framework> by duplicating and adjusting the code in the <slower_framework>.
- Propose a multi-step high-level validation plan, where each step is a group of unit tests, and the steps go from lower-level code changes to higher-level. Ensure the following:
    - Full test code coverage, correctness and speed gains. 
    - Ensure end-to-end tests for prefill-only, decode-only and mix of prefill-decode execution modes.
    - Ensure end-to-end tests for cuda graph modes.
    - End-to-end tests that execute the modified/new code-paths by providing highest level input tensors and verifying output tensors (vs baseline or known previous versions). If necessary, instantiate the modified/new classes/objects with the prepare/finalize codes that are required.
    - If possible, reuse existing code pieces from <slower_framework> or <faster_framework>. 
    - Ensure there are multiple tests inside each file, and if some fails, then do not run the tests separately, instead ensure they all pass together. For example, CUDA illegal memory access in the middle means that the code is incorrect, so fix it. Do not skip anything. This is critical for correctness.

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

The goal is to generate a code patch for <slower_framework> that will apply the improvement <plan_step> from <plan_file> based on the multi-step high-level coding plan from <high_level_code_plan_file>. Do the following and think hard:
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
    - Ensure to test many use-cases and edge-cases.
- Do not skip any tests, and make sure all of them run correctly. If something fails, then fix the related issues.
- Ensure all of the tests are part of the full patch and that all tests are added inside <slower_framework> in the related test sub-directories.

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


@dataclass
class FixIssuePrompt:
    claude_config: ClaudeConfig
    code_gen_config: CodeGenConfig
    high_level_code_plan_file: str
    prs_dir: str
    issue_to_fix_file: str
    issue_cwd: str
    prompt_template: ClassVar[str] = """
<cwd> = {cwd}
<issue_cwd> = {issue_cwd}

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
<prs_dir> = {prs_dir}
<issue_to_fix_file> = {issue_to_fix_file}

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
- Inside <prs_dir> directory are the code patches that implement the multi-step high-level coding plan from <high_level_code_plan_file> that applies the improvement <plan_step> to <slower_framework>.

The goal of this task is to fix the issue described in <issue_to_fix_file> that happens after the code patches from <prs_dir> are applied to <slower_framework> source code (that is inside the related directory from <framework_source_codes>), and the new code is executed. Do the following and think hard:
- Analyze and understand in-detail the issue <issue_to_fix>.
- Detect and analyze in-detail the root causes that make issue <issue_to_fix> to appear in <slower_framework>. Ensure to correlate the findings here with the multi-step high-level coding plan from <high_level_code_plan_file>.
- Detect and analyze in-detail the root causes that make issue <issue_to_fix> to NOT appear in <faster_framework>. Ensure to correlate the findings here with the multi-step high-level coding plan from <high_level_code_plan_file>.
- Fix <slower_framework> code patches to remove the issue <issue_to_fix> with minimal changes to the code patches.
- Strongly prefer to port fixes/ideas/code-pieces from <faster_framework> to fix issue <issue_to_fix> in <slower_framework>.
- For the sequence of changes made to fix the issue, write a sequence of tests that verify each small step taken for correctness, and eventually verifying the full fix end-to-end as follows: 
    - The sequence of tests must go from smaller low-level tests, to larger higher-level tests to gradually test the correctness. 
    - Make sure everything passes, for prefill, decode and mix of prefill and decode executions. 
    - Make sure to test cuda graph modes enabled with the new code-paths for full correctness. 
    - Make sure to have a large end-to-end tests that compare vs baseline and test all possible modes of execution (prefill/decode, cuda graph and more). 
    - Ensure the issue <issue_to_fix> is fixed in <slower_framework> with the new fix. 
- Ensure all existing tests are still working. Run them and verify.
- Review the new fixed code patches again for issues and fix anything necessary. Repeat the review multiple times till high confidence is reached.

Finally:
- Apply the new fixed code patches, run all of the tests (do NOT SKIP any tests), and verify all works and the issue is fixed.
- Ensure there is more than 1 test inside each test file, and ensure that all of them pass when run together in one file. If there is an error due to CUDA illegal memory, then it means that the fix is incorrect. Do not try to run the tests separately. Make sure they pass all together while fixing the relevant issues.
- Try to verify your changes by running a sequence of small tests that verify the sequence of small changes, that together compromise the full change. Be detailed, precise and ensure everything works.

Output:
- Dump a brief explanation of the fix applied to <issue_cwd>/issue_info.txt so a professional programmer can understand the fix applied here step-by-step.
- Dump the sequence of new fixed small code patches to <issue_cwd>/<output_file_prefix>_[seq_id].patch
- Dump the new fixed final full code patch to <issue_cwd>/<output_file_prefix>_full.patch  
- Dump all new/fixed and old tests to <issue_cwd>/

Do not modify any files inside <prs_dir>.

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
            prs_dir=self.prs_dir,
            issue_to_fix_file=self.issue_to_fix_file,
            issue_cwd=self.issue_cwd,
        )


def gen_FixIssuePrompt(
    claude_config: ClaudeConfig,
    code_gen_config: CodeGenConfig,
    high_level_code_plan_file: str,
    prs_dir: str,
    issue_to_fix_file: str,
    issue_cwd: str,
):
    return FixIssuePrompt(
        claude_config=claude_config,
        code_gen_config=code_gen_config,
        high_level_code_plan_file=high_level_code_plan_file,
        prs_dir=prs_dir,
        issue_to_fix_file=issue_to_fix_file,
        issue_cwd=issue_cwd,
    )
