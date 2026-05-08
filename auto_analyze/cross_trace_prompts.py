from dataclasses import dataclass
from typing import ClassVar

from auto_analyze.cross_trace_config import (
    CrossTraceConfig,
    ANALYSIS_TYPE_CROSS_FRAMEWORK,
    ANALYSIS_TYPE_REGRESSION,
    PERF_COMPARE_FILE,
    PERF_DIFF_ANALYSIS_FILE,
    IMPROVEMENT_PLAN_FILE,
)


@dataclass
class CompareMedianTransformerBlocksPrompt:
    model: str
    gpu_type: str
    analysis_type: str
    transformer_blocks: list[str]
    trace_ids: list[str]
    framework_names: list[str]
    framework_source_codes: list[str]
    output_file: str
    prompt_template: ClassVar[str] = """
[transformer_blocks] = {transformer_blocks}

[trace_ids] = {trace_ids}
[framework_names] = {framework_names}
[framework_source_codes] = {framework_source_codes}

[output_file] = {output_file}

{context_section}

- [trace_ids] is the list of trace identifiers
- [framework_names] is a list of framework names that match respectively to [trace_ids]
- [framework_source_codes] is a list of framework source codes that match respectively to [trace_ids]
- [transformer_blocks] is a list of median transformer block files that match respectively to [trace_ids]

Do the following and think hard:
- Read all median transformer block files from [transformer_blocks].
- Match the sequence of operations of each median transformer block with the sequence of operations of other median transformer blocks, so that it will be possible to compare all of the median transformer blocks. Base the matching on the high-level implementation details of how transformer blocks are implemented. Ensure to take into account the separate GPU streams and their start and end synchronization points.
- Analyze in-depth and compare in-detail the performance of the matched median transformer blocks, in order to find the performance differences.
- Inside [transformer_blocks] there is also performance and timing information of the components that are not the transformer blocks, and also have effect on the inference pass total latency:
    - Analyze and compare these components in-detail to understand their effect
- Summarize in a single table all of the found performance differences, on both the transformer block level and on the non-transformer block level (the general system overhead). For each difference, provide:
    - a short description of the difference.
    - a short source code reference for all traces.
    - a short description how each trace can improve vs the other.
- Ensure that the summary table covers the full transformer block GPU operation sequences with their high-level correlations, while properly correlating separate GPU streams, and also fully covers the non-transformer block pieces with full details that necessary to understand and pinpoint the differences.
- Dump results to [output_file] in the current working directory, including the summary table and other relevant information.

"""

    def prompt(self):
        if self.analysis_type == ANALYSIS_TYPE_CROSS_FRAMEWORK:
            context_section = (
                "This is a CROSS-FRAMEWORK comparison. "
                "The traces come from different frameworks running the same model. "
                "The goal is to understand performance differences between framework implementations."
            )
        else:
            context_section = (
                "This is a REGRESSION analysis. "
                "The traces come from the SAME framework but different versions/commits. "
                "The goal is to understand what changed between versions that caused performance differences."
            )

        return self.prompt_template.format(
            model=self.model,
            gpu_type=self.gpu_type,
            transformer_blocks=self.transformer_blocks,
            trace_ids=self.trace_ids,
            framework_names=self.framework_names,
            framework_source_codes=self.framework_source_codes,
            output_file=self.output_file,
            context_section=context_section,
        )


@dataclass
class PerfDiffAnalysisPrompt:
    model: str
    gpu_type: str
    analysis_type: str
    target_trace_id: str
    target_framework_name: str
    target_source_code: str
    comparison_file: str
    transformer_blocks: list[str]
    trace_ids: list[str]
    framework_names: list[str]
    framework_source_codes: list[str]
    output_file: str
    prompt_template_cross_framework: ClassVar[str] = """
[model] = {model}
[gpu_type] = {gpu_type}
[target_trace_id] = {target_trace_id}
[target_framework] = {target_framework_name}
[target_source_code] = {target_source_code}
[comparison_file] = {comparison_file}
[transformer_blocks] = {transformer_blocks}
[trace_ids] = {trace_ids}
[framework_names] = {framework_names}
[framework_source_codes] = {framework_source_codes}
[output_file] = {output_file}

- [comparison_file] contains a detailed per-operation comparison of median transformer blocks across frameworks.
- [transformer_blocks] are the median transformer block files for each trace.
- [target_trace_id] / [target_framework] is the framework we want to analyze and understand.

Do the following and think hard:
- Read and analyze the [comparison_file] in depth.
- Read the relevant median transformer block files from [transformer_blocks].
- For each source code reference, fetch and analyze the actual source code from [framework_source_codes].

Produce a FULL EXPLANATION of the performance differences FROM THE PERSPECTIVE OF [target_framework]:

1. OVERALL PERFORMANCE GAP
   - Total wall time comparison: [target_framework] vs each other framework
   - Percentage difference and absolute time difference
   - Which framework is faster overall and by how much

2. OPERATION-BY-OPERATION ANALYSIS
   For each operation where [target_framework] differs significantly:
   - What the operation does (high-level purpose)
   - How [target_framework] implements it vs how other frameworks implement it
   - Source code references showing the key implementation differences
   - Why [target_framework] is slower/faster for this specific operation
   - Quantified time difference (absolute and percentage)

3. SYSTEM OVERHEAD DIFFERENCES
   - Non-transformer-block overhead comparison
   - Scheduling, synchronization, and framework-level differences
   - Impact on total inference latency

4. ROOT CAUSE SUMMARY
   - Ranked list of all performance differences (largest impact first)
   - For each: root cause category (algorithmic, implementation, configuration, kernel selection)
   - Clear technical explanation of WHY the difference exists

Dump the full explanation to [output_file].
"""

    prompt_template_regression: ClassVar[str] = """
[model] = {model}
[gpu_type] = {gpu_type}
[target_trace_id] = {target_trace_id}
[target_framework] = {target_framework_name}
[target_source_code] = {target_source_code}
[comparison_file] = {comparison_file}
[transformer_blocks] = {transformer_blocks}
[trace_ids] = {trace_ids}
[framework_names] = {framework_names}
[framework_source_codes] = {framework_source_codes}
[output_file] = {output_file}

- This is a REGRESSION analysis of the same framework across different versions/commits.
- [comparison_file] contains a detailed per-operation comparison of median transformer blocks across versions.
- [transformer_blocks] are the median transformer block files for each version.
- [target_trace_id] is the version we want to analyze (typically the newer/regressed version).

Do the following and think hard:
- Read and analyze the [comparison_file] in depth.
- Read the relevant median transformer block files from [transformer_blocks].
- For each source code reference, fetch and analyze the actual source code from [framework_source_codes].

Produce a FULL REGRESSION ANALYSIS for [target_trace_id]:

1. OVERALL REGRESSION SUMMARY
   - Total wall time comparison between versions
   - Percentage regression/improvement
   - Is this a regression or improvement?

2. OPERATION-BY-OPERATION CHANGES
   For each operation that changed significantly:
   - What the operation does
   - How it changed between versions (implementation, kernel selection, parameters)
   - Source code references showing what changed
   - Quantified time difference

3. NEW/REMOVED OPERATIONS
   - Operations that appear in one version but not the other
   - Their impact on total timing

4. ROOT CAUSE ANALYSIS
   - Ranked list of changes causing the regression/improvement
   - For each: what code change likely caused it
   - Source file and function references

Dump the full regression analysis to [output_file].
"""

    def prompt(self):
        fmt_args = dict(
            model=self.model,
            gpu_type=self.gpu_type,
            target_trace_id=self.target_trace_id,
            target_framework_name=self.target_framework_name,
            target_source_code=self.target_source_code,
            comparison_file=self.comparison_file,
            transformer_blocks=self.transformer_blocks,
            trace_ids=self.trace_ids,
            framework_names=self.framework_names,
            framework_source_codes=self.framework_source_codes,
            output_file=self.output_file,
        )

        if self.analysis_type == ANALYSIS_TYPE_CROSS_FRAMEWORK:
            return self.prompt_template_cross_framework.format(**fmt_args)
        else:
            return self.prompt_template_regression.format(**fmt_args)


@dataclass
class ImprovementPlanPrompt:
    model: str
    gpu_type: str
    analysis_type: str
    target_trace_id: str
    target_framework_name: str
    target_source_code: str
    comparison_file: str
    diff_analysis_file: str
    transformer_blocks: list[str]
    trace_ids: list[str]
    framework_names: list[str]
    framework_source_codes: list[str]
    output_file: str
    prompt_template_cross_framework: ClassVar[str] = """
[transformer_blocks] = {transformer_blocks}
[trace_ids] = {trace_ids}
[framework_names] = {framework_names}
[framework_source_codes] = {framework_source_codes}
[comparison_file] = {comparison_file}
[diff_analysis_file] = {diff_analysis_file}
[target_trace_id] = {target_trace_id}
[target_framework] = {target_framework_name}
[output_file] = {output_file}

- [trace_ids] is the list of trace identifiers
- [framework_names] is a list of framework names that match respectively to [trace_ids]
- [framework_source_codes] is a list of framework source codes that match respectively to [trace_ids]
- [transformer_blocks] is a list of median transformer blocks that match respectively to [trace_ids]

- The file [comparison_file] has an operation by operation comparison of [transformer_blocks]
- The file [diff_analysis_file] has a detailed explanation of the performance differences for [target_framework]
- The [target_framework] is the framework that we want to optimize and improve

Do the following and think hard:
- Read, analyze and understand in-detail the performance comparisons in [comparison_file]
- Read and understand the performance difference analysis in [diff_analysis_file]
- Based on these, for [target_framework], detect all performance issues (where [target_framework] is slower) that need to be fixed to fully recover performance for the transformer block (skip anything outside of the block).
- For each detected performance issue, generate an improvement plan as follows:
    - Fetch related source code files from the associated [framework_source_codes] to analyze the performance issue in detail, in order to understand exactly why [target_framework] is slower than the other framework. Analyze the related call chains and participating classes/objects/functions that are key to the performance difference.
    - Analyze how the performance issue can be fixed in the [target_framework] and plan it in detail for the [target_framework]. Ensure the step-by-step plan is clear, concise and detailed enough to execute on.
    - Provide a high-level coding step-by-step summary of the previous plan to fix the performance issue for the [target_framework]. For each step, provide source code file/line references and related code snippets to illustrate the key step points, so expert programmers can execute on it.
- Order the resulting sequence of plans by priority and their impact
- Ensure the performance is fully recovered in the [target_framework] for the transformer block, DO NOT MISS OPTIMIZATIONS.

Dump the resulting sequence of plans to [output_file].

"""

    prompt_template_regression: ClassVar[str] = """
[transformer_blocks] = {transformer_blocks}
[trace_ids] = {trace_ids}
[framework_names] = {framework_names}
[framework_source_codes] = {framework_source_codes}
[comparison_file] = {comparison_file}
[diff_analysis_file] = {diff_analysis_file}
[target_trace_id] = {target_trace_id}
[output_file] = {output_file}

- This is a REGRESSION analysis of the same framework across different versions/commits.
- [trace_ids] identifies each version being compared
- [framework_source_codes] are the source code trees for each version
- [transformer_blocks] are the median transformer blocks for each version
- [comparison_file] has the operation-by-operation comparison
- [diff_analysis_file] has the detailed regression analysis
- [target_trace_id] is the version we want to fix (the regressed version)

Do the following and think hard:
- Read and understand the comparisons in [comparison_file] and the regression analysis in [diff_analysis_file]
- For [target_trace_id], detect all operations that regressed vs the other version(s)
- For each regression, generate a fix plan:
    - Analyze the source code changes between versions that caused the regression
    - Identify the specific code changes, configurations, or kernel selections that need to be reverted or fixed
    - Provide step-by-step instructions with source code file/line references and code snippets
    - Estimate the impact of each fix (how much time it would recover)
- Order plans by impact (largest regression first)
- Ensure all regressions are covered

Dump the resulting sequence of fix plans to [output_file].

"""

    def prompt(self):
        fmt_args = dict(
            model=self.model,
            gpu_type=self.gpu_type,
            target_trace_id=self.target_trace_id,
            target_framework_name=self.target_framework_name,
            target_source_code=self.target_source_code,
            comparison_file=self.comparison_file,
            diff_analysis_file=self.diff_analysis_file,
            transformer_blocks=self.transformer_blocks,
            trace_ids=self.trace_ids,
            framework_names=self.framework_names,
            framework_source_codes=self.framework_source_codes,
            output_file=self.output_file,
        )

        if self.analysis_type == ANALYSIS_TYPE_CROSS_FRAMEWORK:
            return self.prompt_template_cross_framework.format(**fmt_args)
        else:
            return self.prompt_template_regression.format(**fmt_args)


def gen_cross_trace_prompts(config: CrossTraceConfig, file_prefix=""):
    compare_file = f"{file_prefix}{PERF_COMPARE_FILE}"
    diff_analysis_file = f"{file_prefix}{PERF_DIFF_ANALYSIS_FILE}"
    plan_file = f"{file_prefix}{IMPROVEMENT_PLAN_FILE}"

    trace_ids = [tr.trace_id for tr in config.trace_results]
    framework_names = [tr.framework_name for tr in config.trace_results]
    framework_source_codes = [
        tr.framework_source_code for tr in config.trace_results
    ]
    transformer_blocks = [
        tr.get_median_block_file() for tr in config.trace_results
    ]

    target = config.get_target_result()

    compare_prompt = CompareMedianTransformerBlocksPrompt(
        model=config.model,
        gpu_type=config.gpu_type,
        analysis_type=config.analysis_type,
        transformer_blocks=transformer_blocks,
        trace_ids=trace_ids,
        framework_names=framework_names,
        framework_source_codes=framework_source_codes,
        output_file=compare_file,
    )

    diff_prompt = PerfDiffAnalysisPrompt(
        model=config.model,
        gpu_type=config.gpu_type,
        analysis_type=config.analysis_type,
        target_trace_id=config.target_trace_id,
        target_framework_name=target.framework_name,
        target_source_code=target.framework_source_code,
        comparison_file=compare_file,
        transformer_blocks=transformer_blocks,
        trace_ids=trace_ids,
        framework_names=framework_names,
        framework_source_codes=framework_source_codes,
        output_file=diff_analysis_file,
    )

    plan_prompt = ImprovementPlanPrompt(
        model=config.model,
        gpu_type=config.gpu_type,
        analysis_type=config.analysis_type,
        target_trace_id=config.target_trace_id,
        target_framework_name=target.framework_name,
        target_source_code=target.framework_source_code,
        comparison_file=compare_file,
        diff_analysis_file=diff_analysis_file,
        transformer_blocks=transformer_blocks,
        trace_ids=trace_ids,
        framework_names=framework_names,
        framework_source_codes=framework_source_codes,
        output_file=plan_file,
    )

    step_names = [
        "Comparing median transformer blocks across traces",
        "Analyzing performance differences for target trace",
        "Generating improvement/fix plan for target trace",
    ]

    prompts = []
    prompt_objects = [compare_prompt, diff_prompt, plan_prompt]

    for i, (step_name, prompt_obj) in enumerate(
        zip(step_names, prompt_objects), 1
    ):
        prompts.append(
            {
                "cmd": (
                    f"print('\\n=== [Step {i}/{len(step_names)}]"
                    f" {step_name}... ===')"
                )
            }
        )
        prompts.append(prompt_obj.prompt())

    output_files = {
        "perf_compare": compare_file,
        "perf_diff_analysis": diff_analysis_file,
        "improvement_plan": plan_file,
    }

    return prompts, output_files
