import os
from dataclasses import dataclass
from typing import ClassVar

from auto_analyze.configs.cross_trace_config import (
    CrossTraceConfig,
    TraceResult,
    PERF_MATCHING_FILE,
    PERF_ANALYSIS_CROSS_FILE,
)


def _build_traces_context(
    traces: list[TraceResult], target_idx: int, output_dir: str
) -> str:
    lines = []
    for i, tr in enumerate(traces):
        marker = " (TARGET)" if i == target_idx else ""
        lines.append(f"Trace [{i}]{marker}:")
        lines.append(f"  trace_id: {tr.trace_id}")
        lines.append(f"  framework_name: {tr.framework_name}")
        lines.append(f"  framework_source_code: {tr.framework_source_code}")
        lines.append(f"  commit_id: {tr.commit_id}")
        lines.append(f"  model: {tr.model}")
        lines.append(f"  gpu_type: {tr.gpu_type}")
        lines.append(
            f"  execution_params: BS={tr.batch_size_range} ISL={tr.prefill_size_range} OSL={tr.output_size_range}"
        )
        lines.append(f"  median_block_file: {tr.get_median_block_file()}")
        lines.append(f"  high_level_ops_file: {tr.get_high_level_ops_file()}")
        perf_file = tr.get_perf_analysis_file()
        if os.path.exists(perf_file):
            lines.append(f"  perf_analysis_file: {perf_file}")
        lines.append(f"  gpu_ops_to_blocks_file: {tr.get_gpu_ops_to_blocks_file()}")
        lines.append("")
    abs_output = os.path.abspath(output_dir)
    lines.append(
        f"IMPORTANT: Write ALL output files to {abs_output} only. Do not create files anywhere else."
    )
    return "\n".join(lines)


@dataclass
class MatchMedianTransformerBlocksPrompt:
    config: CrossTraceConfig
    output_file: str
    prompt_template: ClassVar[str] = """
{traces_context}
[output_file] = {output_file}

This is a {analysis_type} matching.
{analysis_description}

The goal of this step is to MATCH the operations between the median transformer blocks of all traces. 

Each trace has its own low-level GPU operations correlated to high-level operations, and since high-level operations as similar between traces (they all implement the same logical LLM transformer block), exploit this to help match the low-level operations between the transformer block traces.

Do the following:
- Read and analyze in-detail the high-level transformer block file of each framework
- Read and analyze in-detail the median transformer block file of each framework, with their low-level => high-level operation matchings.
- For each block, understand its operation sequence: the high level operation names, the low-level kernel names, and what each operation does.
- Match operations across blocks, one-by-one, with the help of:
    - High-level logical purpose
    - Order in the logical sequence of a transformer block
    - Source code inspections and deep dives
- Take into account separate GPU streams and their synchronization points.
- When operations do not match cleanly, perform a deeper analysis and source code deep-dive to fully understand it and fix it.

Produce a structured output with:
1. A matched operations table showing, for each logical operation, the corresponding high_level_op name, low-level kernel name, duration, and stream for each block — side by side.
2. Operations that exist in one trace but not the other (added/removed/fused differently).
3. Overall wall time for each trace.

Dump results to [output_file].
"""

    def prompt(self):
        analysis_type = self.config.infer_analysis_type()
        if analysis_type == "cross-framework":
            desc = "The traces come from different frameworks running the same model."
        else:
            desc = "The traces come from the SAME framework but different versions/commits."

        return self.prompt_template.format(
            traces_context=_build_traces_context(
                self.config.traces, self.config.target_trace_id, self.config.output_dir
            ),
            output_file=self.output_file,
            analysis_type=analysis_type,
            analysis_description=desc,
        )


@dataclass
class PerfDiffAnalysisPrompt:
    config: CrossTraceConfig
    matching_file: str
    output_file: str
    prompt_template: ClassVar[str] = """
{traces_context}
[matching_file] = {matching_file}
[output_file] = {output_file}
- {analysis_description}
- [matching_file] contains a detailed per-operation matching of median transformer blocks across traces.
- The TARGET trace is [{target_idx}] ({target_label}) — the trace we want to optimize/fix.

Do the following:
- Read, analyze in-detail, and understand the matching of blocks in [matching_file].
- Detect the cases where the target trace is slower than the other trace(s), and for each such case:
    - Understand in-detail the performance difference in the context of the high-level logical operation involved and the low-level operations produced.
    - Perform source code analysis to get all of the details necessary to understand the performance difference.
    - Understand in-detail the root cause, grounded in the source code and concrete precise reasons.
    - Understand in-detail why the other framework is faster and see how to fix this performance difference by porting the code from the other framework. I.e we want to COPY-PASTE as much as possible from the other framework while doing MINIMAL changes to the target framework. THIS IS IMPORTANT!

- Generate a summary improvement plan for the target trace that fully recovers its performance as follows:
    - Executive summary with gaps, key findings, key ideas, and ranked plan summary (that also has impact per each improvement in % of total block time).
    - A detailed ranked sequence of improvement proposals (from most impactful to less), where each proposal has:
        - A summary of root causes, key findings, key ideas, and estimated improvement/impact in % of total block time. 
        - A step-by-step guide that describes how to port code from the other framework with minimal changes to the target framework, in order to fix the performance difference fully (THIS IS IMPORTANT!). For each step provide:
            - An explanation of the step
            - A code snippet example that shows what needs to be changed. Ensure the code snippet is properly formatted and detailed enough.
    - Make sure the summary is clear, concise and professional so that it can be executed on by a professional {target_framework} programmer.

Dump the summary to [output_file].
"""

    def prompt(self):
        target = self.config.get_target_result()
        analysis_type = self.config.infer_analysis_type()

        if analysis_type == "cross-framework":
            desc = "This is a CROSS-FRAMEWORK comparison of different frameworks running the same model."
            label = target.framework_name
        else:
            desc = f"This is a REGRESSION analysis of {target.framework_name} across different commits."
            label = f"{target.framework_name} commit {target.commit_id[:12]}"

        return self.prompt_template.format(
            traces_context=_build_traces_context(
                self.config.traces, self.config.target_trace_id, self.config.output_dir
            ),
            matching_file=self.matching_file,
            output_file=self.output_file,
            analysis_description=desc,
            target_idx=self.config.target_trace_id,
            target_label=label,
            target_framework=target.framework_name,
        )


def gen_cross_trace_prompts(config: CrossTraceConfig, file_prefix=""):
    import os

    output_dir = os.path.abspath(config.output_dir)

    def _path(rel):
        return os.path.join(output_dir, f"{file_prefix}{rel}")

    matching_file = _path(PERF_MATCHING_FILE)
    diff_analysis_file = _path(PERF_ANALYSIS_CROSS_FILE)

    matching_prompt = MatchMedianTransformerBlocksPrompt(
        config=config,
        output_file=matching_file,
    )

    diff_prompt = PerfDiffAnalysisPrompt(
        config=config,
        matching_file=matching_file,
        output_file=diff_analysis_file,
    )

    step_names = [
        "Matching median transformer blocks across traces",
        "Analyzing performance differences and generating improvement plan",
    ]

    prompts = []
    prompt_objects = [matching_prompt, diff_prompt]

    for i, (step_name, prompt_obj) in enumerate(zip(step_names, prompt_objects), 1):
        prompts.append(
            {
                "cmd": (
                    f"print('\\n=== [Step {i}/{len(step_names)}] {step_name}... ===')"
                )
            }
        )
        prompts.append(prompt_obj.prompt())

    output_files = {
        "perf_matching": matching_file,
        "perf_diff_analysis": diff_analysis_file,
    }

    return prompts, output_files
