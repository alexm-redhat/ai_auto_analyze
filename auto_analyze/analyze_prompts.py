from dataclasses import dataclass
from typing import ClassVar

from auto_analyze.analyze_configs import AnalyzeConfig


@dataclass
class TransformerBlockHighLevelPrompt:
    model: str
    precision: str
    gpu_type: str
    framework_name: str
    framework_source_code: str
    test_dir: str
    output_file: str
    prompt_template: ClassVar[str] = """
<model> = {model}
<precision> = {precision}
<gpu_type> = {gpu_type}
<framework_name> = {framework_name}
<framework_source_code> = {framework_source_code}
<test_dir> = {test_dir}
<output_file> = {output_file}

- <cwd> is the current working directory

You specialize in analyzing the inference performance of the model <model> in precision <precision> running via the <framework_name> framework on <gpu_type> GPU.

<test_id>=[model]-tp_$[num_gpus]-isl_[input_len]-osl_[output_len]
<test_id_with_batch>=<test_id>-b_[concurrency]
<full_test_id>=[framework]-<test_id_with_batch>

The <test_dir> test/profile directory has the following format (that encodes test parameters): ../<test_id_with_batch>/[framework], and it includes the following files:
    - bench-<full_test_id>.json file that has the benchmark results of running the test on the framework
	- run-log-<full_test_id>.txt file that has the run log of executing the framework
	- run-log-profile-<full_test_id>.txt file that has the profile run log of executing the framework
	- trace-<full_test_id>.nsys-rep file that has the NSYS profile results
	- trace-<full_test_id>.sqlite file that is the SQLite version of the nsys-rep file	

The traces provided are only for pure decode operations (no prefill).

The source code of <framework_name> framework is located here: <framework_source_code>

The model <model> is a sequence of transformer blocks.
The goal of this task is to generate the sequence of high-level operations of a single transformer block.

Do the following plan and think hard:
- Inspect the source code in <framework_source_code> and find the set of files that implement the model <model>. Use run-log* files from the test directory to get hints for what classes and source files are being used to run model <model>.
- Find the sequence of high-level operations of a single transformer block of <model> based on the previous code inspection.
- Each high-level operation may have multiple execution modes. Detect these execution modes for each high-level operation.
- Based on different execution modes of each high-level operation, detect the types of transformer blocks that can run. 
- For each transformer block type, summarize in a table the sequence of high-level operations of this transformer block, so that each operation is shown in a row and each row has: high-level operation name, short source code reference, and execution mode used (if exists).
- Dump to <output_file> (in current working directory) all the resulting tables, which represent all possible transformer blocks.
"""

    def prompt(self):
        return self.prompt_template.format(
            model=self.model,
            precision=self.precision,
            gpu_type=self.gpu_type,
            framework_name=self.framework_name,
            framework_source_code=self.framework_source_code,
            test_dir=self.test_dir,
            output_file=self.output_file,
        )


TRANSFORMER_BLOCK_HIGH_LEVEL_OPS_FILE = "transformer_block_high_level_ops.txt"


@dataclass
class GpuOpsPrompt:
    model: str
    gpu_type: str
    framework_name: str
    test_dir: str
    output_file: str
    output_max_gpu_ops: int
    output_filter_ops: str = ""
    prompt_template: ClassVar[str] = """
<model> = {model}
<gpu_type> = {gpu_type}
<framework_name> = {framework_name}
<test_dir> = {test_dir}
<output_file> = {output_file}
<output_filter_ops> = {output_filter_ops}
<output_max_gpu_ops> = {output_max_gpu_ops}

<test_id>=[model]-tp_$[num_gpus]-isl_[input_len]-osl_[output_len]
<test_id_with_batch>=<test_id>-b_[concurrency]
<full_test_id>=[framework]-<test_id_with_batch>


- The file trace-*.sqlite from <test_dir> is an nsys profile result file in SQLite format of <framework_name> running <model> on <gpu_type> GPU.

Do the following plan and think hard:
- Understand the basic structure of the SQLite trace file.
- Understand how GPU streams are represented.
- Understand how GPU operations are represented. Ignore any CPU operations.

For below, focus only on GPU 0 and ignore other GPUs (since we use tensor-parallel and all GPUs are running in a similar way):
- Find the sequence of all GPU streams in this trace file.
- Find the sequence of all GPU operations, across all GPU streams, including overlapping ones. Keep the maximum number of GPU operations found to <output_max_gpu_ops>. 
- If <output_filter_ops> is not empty string, then filter-out GPU operations that match the pattern <output_filter_ops>.
- Dump the found sequence of GPU operations to file <output_file>, sorted by their start time, where each row in the file will have: a short GPU operation name (max 50 chars), start and end times, duration, source GPU stream number, and full original GPU operation name (limited to 200 chars).
"""

    def prompt(self):
        return self.prompt_template.format(
            model=self.model,
            gpu_type=self.gpu_type,
            framework_name=self.framework_name,
            test_dir=self.test_dir,
            output_file=self.output_file,
            output_max_gpu_ops=self.output_max_gpu_ops,
            output_filter_ops=self.output_filter_ops,
        )


GPU_OPS_FILE = "gpu_ops.txt"
MAX_GPU_OPS = 2000


@dataclass
class GpuOpsToTransformerBlocksPrompt:
    transformer_block_high_level_ops_file: str
    gpu_ops_file: str
    output_file: str
    prompt_template: ClassVar[str] = """
<transformer_block_high_level_ops_file> = {transformer_block_high_level_ops_file}
<gpu_ops_file> = {gpu_ops_file}

<output_file> = {output_file}

- The file <transformer_block_high_level_ops_file> describes different types of transformer blocks, and for each transformer block type it provides the sequence of high-level operations inside this block type.
- The file <gpu_ops_file> provides the sequence of low-level GPU operations that represent an execution of a sequence of transformer blocks, where each transformer block can be of a different type (from <transformer_block_high_level_ops_file>).

Do the following plan and think hard (and ultra hard):
- Detect ranges of GPU operations inside <gpu_ops_file> that are full transformer blocks based on the high-level operations from <transformer_block_high_level_ops_file>. Ensure no operation is missed, and that the start and end operations of each block are consistent.
- For each detected block, do the following:
    - Determine the transformer block type it represents
    - Correlate the sequence of low-level GPU operations of the block with the high-level operations from <transformer_block_high_level_ops_file> based on the transformer block type. Ensure every low-level GPU operation is correlated with a high-level operation, while taking into account different GPU streams and their high-level implementation details.
    - Review the results of the previous correlation step for mismatches, including the correlation of separate GPU streams. Fix found errors. Repeat this process 3 times to ensure results are precise, clear and concise.
- Dump the detected blocks to <output_file>. For each block provide:
    - Summary with start/end/duration/wall times and transformer block type.
    - A table with the sequence of low-level GPU operations of this block, where each row has: correlated high-level operation name, start and end times, duration, source GPU stream, and original low-level GPU operation name.
"""

    def prompt(self):
        return self.prompt_template.format(
            transformer_block_high_level_ops_file=self.transformer_block_high_level_ops_file,
            gpu_ops_file=self.gpu_ops_file,
            output_file=self.output_file,
        )


GPU_OPS_TO_BLOCKS_FILE = "gpu_ops_to_blocks.txt"


@dataclass
class MedianTransformerBlockPrompt:
    gpu_ops_to_blocks_file: str
    output_file: str
    prompt_template: ClassVar[str] = """
<gpu_ops_to_blocks_file> = {gpu_ops_to_blocks_file}
<output_file> = {output_file}

- The file <gpu_ops_to_blocks_file> provides the sequence of transformer blocks, where each block has execution statistics and its corresponding range of GPU operations.

Do the following and think hard: 
- Pick a median wall time transformer block from <gpu_ops_to_blocks_file> that appears most of the time.
- Dump the picked transformer block content lines to <output_file> 

"""

    def prompt(self):
        return self.prompt_template.format(
            gpu_ops_to_blocks_file=self.gpu_ops_to_blocks_file,
            output_file=self.output_file,
        )


MEDIAN_BLOCK_FILE = "median_block.txt"


@dataclass
class CompareMedianTransformerBlocksPrompt:
    model: str
    gpu_type: str
    transformer_blocks: list[str]
    framework_names: list[str]
    framework_source_codes: list[str]
    output_file: str
    prompt_template: ClassVar[str] = """
<transformer_blocks> = {transformer_blocks}

<framework_names> = {framework_names} 
<framework_source_codes> = {framework_source_codes}

<output_file> = {output_file}

- <framework_names> is a list of framework names
- <framework_source_codes> is a list of framework source codes that match respectively to <framework_names>
- <transformer_blocks> is a list of median transformer blocks that match respectively to <framework_names>

Do the following and think hard:
- Match the sequence of operations of each median transformer block with the sequence of operations of other median transformer blocks, so that it will be possible to compare all of the median transformer blocks across frameworks. Base the matching on the high-level implementation details of how a transformer blocks are implemented in each framework. Ensure to take into account the separate GPU streams and their start and end synchronization points.
- Analyze in-depth and compare in-detail the performance of the matched median transformer blocks from before, in order to find the performance differences.
- Inside <transformer_blocks> there is also performance and timing information of the components that are not the transformer blocks, and also have effect on the inference pass total latency:
    - Analyze and compare these components in-detail to understand their effect
- Summarize in a single table all of the found performance differences, on both the transformer block level and on the non-transformer block level (the general system overhead). For each difference, provide:
    - a short description of the difference.
    - a short source code reference for all frameworks.
    - a short description how each framework can improve vs the other.
- Ensure that the summary table covers the full transformer block GPU operation sequences with their high-level correlations, while properly correlating separate GPU streams, and also fully covers the non-transformer block pieces with full details that necessary to understand and pinpoint the differences.
- Dump results to <output_file> in the current working directory, including the summary table and other relevant information.

"""

    def prompt(self):
        return self.prompt_template.format(
            model=self.model,
            gpu_type=self.gpu_type,
            transformer_blocks=self.transformer_blocks,
            framework_names=self.framework_names,
            framework_source_codes=self.framework_source_codes,
            output_file=self.output_file,
        )


PERF_COMPARE_BLOCK_FILE = "perf_compare_blocks.txt"


# @dataclass
# class CompareGeneralOverheadsPrompt:
#     model: str
#     gpu_type: str
#     transformer_blocks: list[str]
#     blocks_perf_cmp_file: str
#     framework_names: list[str]
#     framework_source_codes: list[str]
#     test_dirs: list[str]
#     output_file: str
#     prompt_template: ClassVar[str] = """
# <transformer_blocks> = {transformer_blocks}
# <blocks_perf_cmp_file> = {blocks_perf_cmp_file}
# <framework_names> = {framework_names}
# <framework_source_codes> = {framework_source_codes}
# <test_dirs> = {test_dirs}
# <output_file> = {output_file}

# - <framework_names> is a list of framework names
# - <framework_source_codes> is a list of framework source codes that match respectively to <framework_names>
# - <transformer_blocks> is a list of median transformer blocks that match respectively to <framework_names>
# - <blocks_perf_cmp_file> compares the median transformer blocks of <framework_names> respectively
# - <test_dirs> are the test result directories for <framework_names> respectively. They include run logs with performance results and profile traces.

# Do the following and think hard:
# - Look at the run logs of <framework_names> and find the final performance results (token/secs) for both of them. Compare this performance result to see the total difference.
# - Look at the <blocks_perf_cmp_file> and the performance difference that is reported there (wall time) and compare it vs the performance difference that is in run logs.
# - There may be a gap between the difference in <blocks_perf_cmp_file> and the difference in the run logs.
# - Analyze and understand in-detail the overheads of the relevant forward-pass mode here (for the test here), for both transformer block and non-transformer block components
# - DO NOT CONSIDER any CPU related overheads in this analysis, they are not relevant. ONLY FOCUS on the components that are represented by GPU low-level operations inside the profile trace files of the frameworks:
#     - Re-read and analyze the profile trace files GPU operation sequences to see the operations that are transformer blocks and non transformer blocks to be sure that all of them are covered in the analysis and properly compared.
# - Use all of the observations about the forward pass components, both transformer block and non-transformer blocks to explain the performance differences between the frameworks.
#     - Ensure differences are grounded in profile traces, actual code, and full understandings and learnings
# - Dump the extended performance comparison summary between the frameworks to <output_file> in the current working directory. Ensure this extended performance comparison includes both transformer block comparison and the non-transformer block components comparison and analysis.

# """

#     def prompt(self):
#         return self.prompt_template.format(
#             model=self.model,
#             gpu_type=self.gpu_type,
#             transformer_blocks=self.transformer_blocks,
#             blocks_perf_cmp_file=self.blocks_perf_cmp_file,
#             framework_names=self.framework_names,
#             framework_source_codes=self.framework_source_codes,
#             test_dirs=self.test_dirs,
#             output_file=self.output_file,
#         )


# GENERAL_PERF_COMPARE_BLOCK_FILE = "general_perf_compare_blocks.txt"


@dataclass
class PlanPrompt:
    model: str
    gpu_type: str
    transformer_blocks: list[str]
    framework_names: list[str]
    framework_source_codes: list[str]
    comparison_file: str
    target_framework: str
    output_file: str
    prompt_template: ClassVar[str] = """
<transformer_blocks>
{transformer_blocks}
</transformer_blocks>

<framework_names>
{framework_names}
</framework_names> 
<framework_source_codes>
{framework_source_codes}
</framework_source_codes>

<comparison_file>
{comparison_file}
</comparison_file>
<target_framework>
{target_framework}
</target_framework>

<output_file>
{output_file}
</output_file>

<definitions>
- <framework_names> is a list of framework names
- <framework_source_codes> is a list of framework source codes that match respectively to <framework_names>
- <transformer_blocks> is a list of median transformer blocks that match respectively to <framework_names>

- The file <comparison_file> has an operation by operation comparison of <transformer_blocks>
- The <target_framework> is the framework that we want to optimize and improve 
</definitions>

<instructions>
Do the following and think hard:
- Read, analyze and understand in-detail the performance comparisons in <comparison_file>  
- Based on these comparisons, for the <target_framework>, detect all of the performance issues (where <target_framework> is slower) that need to be fixed to fully recover performance for the transformer block (skip anything outside of the block).
- For each detected performance issue, generate an improvement plan as follows:
    - Fetch related source code files from the associated <framework_source_codes> to analyze the performance issue in detail, in order to understand exactly why <target_framework> is slower than the other framework. Analyze the related call chains and participating classes/objects/functions that are key to the performance difference.
    - Analyze how the performance issue can be fixed in the <target_framework> and plan it in detail for the <target_framework>. Ensure the step-by-step plan is clear, concise and detailed enough to execute on. 
    - Provide a high-level coding step-by-step summary of the previous plan to fix the performance issue for the <target_framework>. For each step, provide source code file/line references and related code snippets to illustrate the key step points, so expert programmers can execute on it.
- Order the resulting sequence of plans by priority and their impact
- Ensure the performance is fully recovered in the <target_framework> for the transformer block, DO NOT MISS OPTIMIZATIONS.
</instructions>

<output>
- Dump the resulting sequence of plans to <output_file>
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            model=self.model,
            gpu_type=self.gpu_type,
            transformer_blocks=self.transformer_blocks,
            framework_names=self.framework_names,
            framework_source_codes=self.framework_source_codes,
            comparison_file=self.comparison_file,
            target_framework=self.target_framework,
            output_file=self.output_file,
        )


PLAN_FILE = "plan.txt"


@dataclass
class SummaryPDFPrompt:
    model: str
    precision: str
    gpu_type: str
    transformer_blocks: list[str]
    cmp_file: str
    plan_file: str
    target_framework: str
    framework_names: list[str]
    framework_source_codes: list[str]
    output_file: str
    prompt_template: ClassVar[str] = """
<model>{model}</model>
<precision>{precision}</precision>
<gpu_type>{gpu_type}</gpu_type>
<transformer_blocks>{transformer_blocks}</transformer_blocks>
<cmp_file>{cmp_file}</cmp_file>
<plan_file>{plan_file}</plan_file>
<target_framework>{target_framework}</target_framework>
<framework_names>{framework_names}</framework_names>
<framework_source_codes>{framework_source_codes}</framework_source_codes>
<output_file>{output_file}</output_file>

<definitions>
- Benchmark: <model> in <precision> on <gpu_type> GPU.
- <framework_names>: list of frameworks; <framework_source_codes>: their source trees.
- <transformer_blocks>: median transformer blocks per framework.
- <cmp_file>: op-by-op comparison of the median blocks.
- <plan_file>: ranked improvement plans for <target_framework>.
</definitions>

<instructions>
Generate a professional PDF presenting a ranked improvement plan for <target_framework>.
Audience: C-level executives AND kernel-level engineers.

Write and execute ONE self-contained Python script using reportlab and pygments.

=== ANALYSIS (before generating) ===

1. Read <plan_file> — extract every issue where <target_framework> is slower.
2. Read <cmp_file> — parse op-by-op comparison data.
3. For each issue, inspect relevant source files in <framework_source_codes>.
   Do NOT fabricate code — only use real code from the source.
4. Rank issues by (estimated_ns_savings x ease_of_implementation).
5. Per issue, identify 2-4 implementation steps with file:line refs and real code snippets.

=== VISUAL DESIGN ===

Color palette (use these exact hex values):
  primary:       #1B2A4A  (dark navy — titles, headers)
  secondary:     #2E86AB  (steel blue — section accents, rule lines)
  accent:        #E8630A  (burnt orange — impact badges)
  success:       #1A7F37  (forest green — positive deltas)
  danger:        #CF222E  (signal red — regressions)
  bg_light:      #F6F8FA  (ghost white — backgrounds)
  bg_alt_row:    #EEF2F7  (pale blue-gray — alternating table rows)
  border:        #D0D7DE  (silver — gridlines)
  text_primary:  #1F2328  (near-black — body text)
  text_secondary:#656D76  (slate gray — captions)

Page: LETTER portrait, margins 0.75in left/right, 0.6in top/bottom.

Typography:
  - Title: Helvetica-Bold 22pt, primary color.
  - Section headers: Helvetica-Bold 15pt, primary color, with a secondary-colored
    rule line underneath.
  - Subsection headers: Helvetica-Bold 12pt, secondary color.
  - Body: Helvetica 10pt, text_primary, 14pt leading.
  - Captions: Helvetica-Oblique 8.5pt, text_secondary, centered.

Tables:
  - Header row: primary background, white bold text, 9pt.
  - Body: Helvetica 9pt, alternating row backgrounds (white / bg_alt_row).
  - Grid: 0.5pt border color. Cell padding: 5pt vertical, 6pt horizontal.
  - Performance deltas: green for improvements, red for regressions.
  - Appendix tables: use smaller font (7.5pt) if needed to fit columns.
  - Bold rows where timing delta exceeds 10%.
  - Always add spacing before and after every table to prevent overlap.

Code blocks — LIGHT THEME:
  - Background: #F6F8FA (ghost white). Border: 1pt #D0D7DE (silver), rounded feel.
  - Default text: Courier 8.5pt, color #1F2328 (near-black).
  - Use pygments tokenization with XPreformatted (NOT Preformatted) for syntax coloring.
  - Token colors (light-theme friendly, high contrast on white):
      Keywords:          #CF222E  (red)
      Strings/chars:     #0A3069  (deep blue)
      Comments:          #656D76  (gray, italic)
      Functions/builtins:#8250DF  (purple)
      Numbers:           #0550AE  (blue)
      Decorators:        #953800  (brown)
      Operators/punct:   #1F2328  (near-black)
      Class names/types: #953800  (brown)
      Default:           #1F2328  (inherits from style)
  - Wrap the XPreformatted in a Table cell for reliable background rendering.
  - Hard-wrap lines at 88 chars; use KeepTogether to prevent page splits.
  - Add spacing above and below each code block.
  - Inline code refs: Courier 9pt, #1F2328 text on #F6F8FA background.
  - CRITICAL: Escape XML entities in all text going into Paragraph/XPreformatted.

Impact badges (small colored table cell before issue title):
  - HIGH IMPACT: danger background, white text.
  - MEDIUM IMPACT: accent background, white text.
  - LOW IMPACT: secondary background, white text.

=== DOCUMENT STRUCTURE ===

1. COVER BLOCK
   - Title: "<target_framework> Performance Optimization Report"
   - Subtitle: "<model> | <precision> | <gpu_type>"
   - Date line: "Generated: [today's date]"
   - Horizontal rule separator.

2. EXECUTIVE SUMMARY (max 1 page)
   - 3-5 bullets: what was compared, total perf gap, top 3 issues, estimated total gain.
   - Summary table: [Issue #, Title, Impact (ns), Difficulty, Priority].
   - Page break.

3. DETAILED IMPROVEMENT PLANS (one sub-section per issue, ranked)
   Per issue:
   - Impact badge + title.
   - "Problem": 2-3 executive-friendly sentences.
   - "Root Cause": technical explanation with inline code refs.
   - "Performance Impact" table: [Metric, <target_framework>, Other, Delta].
   - "Implementation Guide": numbered steps, each with description,
     source file ref, syntax-highlighted code snippet, and brief explanation.
   - Spacing between issues.

4. APPENDIX A: OP-BY-OP COMPARISON
   - Render the COMPLETE contents of <cmp_file> as formatted tables.
   - Do NOT omit any operations.

5. APPENDIX B: SOURCE REFERENCES
   - All referenced source files, grouped by framework.

=== GENERATION RULES ===

- One self-contained Python script, deterministic, no network calls.
- Use Paragraph for wrapping text; XPreformatted for code (never Preformatted).
- Test that the script runs without errors before finishing.
- Verify: no overlapping elements, no clipped tables, code blocks have multi-color
  syntax highlighting on the light background, appendix is complete.
</instructions>

<output>
Write and execute the Python script. Save the PDF to <output_file>.
</output>

"""

    def prompt(self):
        return self.prompt_template.format(
            model=self.model,
            precision=self.precision,
            gpu_type=self.gpu_type,
            transformer_blocks=self.transformer_blocks,
            cmp_file=self.cmp_file,
            plan_file=self.plan_file,
            target_framework=self.target_framework,
            framework_names=self.framework_names,
            framework_source_codes=self.framework_source_codes,
            output_file=self.output_file,
        )


@dataclass
class CombinedTracePrompt:
    model: str
    precision: str
    gpu_type: str
    isl: int
    osl: int
    batch_size: int
    framework_names: list[str]
    framework_source_codes: list[str]
    transformer_blocks: list[str]
    transformer_high_level_blocks: list[str]
    test_dirs: list[str]
    cmp_file: str
    plan_file: str
    output_file: str
    prompt_template: ClassVar[str] = """
<model> = {model}
<precision> = {precision}
<gpu_type> = {gpu_type}
<isl> = {isl}
<osl> = {osl}
<batch_size> = {batch_size}
<framework_names> = {framework_names} 
<framework_source_codes> = {framework_source_codes}
<transformer_blocks> = {transformer_blocks}
<transformer_high_level_blocks> = {transformer_high_level_blocks}
<test_dirs> = {test_dirs}
<cmp_file> = {cmp_file}
<plan_file> = {plan_file}
<output_file> = {output_file}

Make sure to do all work in Claude's current working directory.

- The model benchmarked and compared is <model> in precision <precision> running on <gpu_type> gpu with ISL <isl>, OSL <osl>, and batch size <batch_size>. 
- <framework_names> is a list of framework names
- <framework_source_codes> is a list of framework source codes that match respectively to <framework_names>
- <transformer_blocks> is a list of median transformer blocks that match respectively to <framework_names>
- <transformer_high_level_blocks> is a list of high-level transformer block operations (correlated to source code) that match respectively to <framework_names>, and <transformer_blocks>.
- The file <cmp_file> provides a performance comparison of a transformer block between the frameworks
- The file <plan_file> provides an improvement plan for each performance issue for each framework
- <test_dirs> is a list of test directories that match respectively to <framework_names>, where each test directory has:
    - The file trace-*.sqlite - an nsys profile result file in SQLite format of the respective framework running <model> in precision <precision> on <gpu_type> gpu

The goal of this task is to generate a new combined trace file, in Chrome trace json format, that provides a visual/timeline comparison of the median transformer blocks <transformer_blocks> for frameworks <framework_names> (in this order). Do the following and think hard:
- Color-code similar type operations with same color
- Provide a single top level bar with the test information/params used. This includes that the model tested is <model> in precision <precision> running on <gpu_type> gpu with ISL <isl>, OSL <osl>, and batch size <batch_size>. Put this bar first before anything else.
- Ensure all median transformer blocks start at the same time point 0.
- For each framework's trace-*.sqlite NSYS profile trace, focus on GPU 0, and find the range that represents the median transformer block of this framework, while detecting the cuda streams in this range.
- For each framework, show the cuda streams as they appear in the above found range (one after another). I.e. each framework's lanes / cuda-streams are grouped together. Ensure all operations are showed across all cuda streams of the found range (per framework).
- For each cuda stream:
    - Overlapping operations inside the same cuda stream (which is mainly due to PDL), cannot be propertly shown in perfetto viewer (it errors and drops them). To avoid this, use multiple lanes to represent a cuda stream and place the overlapping operations on different lanes of this stream, so that there are NO OVERLAPPING operations inside the same lane. Annotate the lanes of the same cuda stream accordingly so it will be clear that they belong to the same cuda stream.
    - Ensure no operation is missed or skipped due to overlaps.
    - Ensure all overlaps are fixed by using separate lanes
- For each operation, ensure that all NSYS GPU utilization details and params (if exists) are also transferred to a GPU utilization section.
- For each operation name, prepend a prefix high-level name, based on the median transformer block correlation of operation names to their high-level names. Ensure the high-level name is brief and clear.
- For each operation, add a detailed description field that has the following information:
    - Source code references
    - Explanation of the operations (based on median transformer block file and high level transformer block file)
    - Any other relevant details that are important for understanding the operation in the context of the trace and the comparison.
    - If this is a vLLM operation, then provide an explanation of the relevant improvement plan for this operation, and a reference to the specific improvement.

- Follow all of the previous guidelines strictly, DO NOT SKIP ANYTHING.
- Critically review the work, find issues and fix them.
    - Ensure that there are NO OVERLAPPING operations on the same lane.
    - Ensure all operations have the high level context that is brief, concise and clear.
    - Ensure all operations have source code references in their details section.
    - Ensure that all operations of both frameworks are shown. I.e no operation is missed or skipped (THIS IS IMPORTANT!), and the operation overlaps another operation then it is represented in a different lane.

Dump the new combined trace file to <output_file> in Claude's current working directory.

"""

    def prompt(self):
        return self.prompt_template.format(
            model=self.model,
            precision=self.precision,
            gpu_type=self.gpu_type,
            isl=self.isl,
            osl=self.osl,
            batch_size=self.batch_size,
            framework_names=self.framework_names,
            framework_source_codes=self.framework_source_codes,
            transformer_blocks=self.transformer_blocks,
            transformer_high_level_blocks=self.transformer_high_level_blocks,
            test_dirs=self.test_dirs,
            cmp_file=self.cmp_file,
            plan_file=self.plan_file,
            output_file=self.output_file,
        )


SUMMARY_PDF_FILE = "cmp_and_plan_summary.pdf"

TRACE_COMBINED_FILE = "trace_combined_transformer_blocks.json"

@dataclass
class JiraTasksPrompt:
    model: str
    precision: str
    gpu_type: str
    isl: int
    osl: int
    batch_size: int
    framework_names: list[str]
    framework_source_codes: list[str]
    transformer_blocks: list[str]
    transformer_high_level_blocks: list[str]
    test_dirs: list[str]
    cmp_file: str
    plan_file: str
    output_file: str
    prompt_template: ClassVar[str] = """
<model> = {model}
<precision> = {precision}
<gpu_type> = {gpu_type}
<isl> = {isl}
<osl> = {osl}
<batch_size> = {batch_size}
<framework_names> = {framework_names} 
<framework_source_codes> = {framework_source_codes}
<transformer_blocks> = {transformer_blocks}
<transformer_high_level_blocks> = {transformer_high_level_blocks}
<test_dirs> = {test_dirs}
<cmp_file> = {cmp_file}
<plan_file> = {plan_file}
<output_file> = {output_file}

<jira_epyc> = "Auto generated tasks for vLLM improvement"
<jira_team_value> = INFERENG Runtime
<jira_activity_type> = "Learning & Enablement"

Make sure to do all work in Claude's current working directory.

- The model benchmarked and compared is <model> in precision <precision> running on <gpu_type> gpu with ISL <isl>, OSL <osl>, and batch size <batch_size>. 
- <framework_names> is a list of framework names
- <framework_source_codes> is a list of framework source codes that match respectively to <framework_names>
- <transformer_blocks> is a list of median transformer blocks that match respectively to <framework_names>
- <transformer_high_level_blocks> is a list of high-level transformer block operations (correlated to source code) that match respectively to <framework_names>, and <transformer_blocks>.
- The file <cmp_file> provides a performance comparison of a transformer block between the frameworks
- The file <plan_file> provides an improvement plan for each performance issue for each framework
- <test_dirs> is a list of test directories that match respectively to <framework_names>, where each test directory has:
    - The file trace-*.sqlite - an nsys profile result file in SQLite format of the respective framework running <model> in precision <precision> on <gpu_type> gpu


The goal of this task is to create a set of JIRA tasks for each improvement plan in <plan_file>. Do the following and think hard:
- Use JIRA MCP for this task
- The <plan_file> is associated with results_[test_name]/test_results/[test_name_with_batch] test directory (name it <plan_test_dir>). This directory includes:
    - run logs (non-profile and profile)
    - NSYS traces per-framework
    - Analyze results in ./analyze
        
- Copy the [test_name_with_batch] subdirectory from <plan_test_dir> to /raid/engine/auto_analyze/ and remember links that use the /raid/... location to the following pieces:
    - Links to run logs (not run log profile ones)
    - Links to NSYS trace files (nsys-rep files only)
    - Link to the combined trace (json file only)
    - Link to the summary PDF (pdf file only)

- Analyze and understand in-depth the improvement plans in <plan_file>
- Verify that epyc <jira_epyc> exists. If it does not exist, then STOP and DO NOT continue this task.
- To represent the <plan_file> in JIRA, we will create a (master) task that will have sub-tasks, where each sub-task will represent an individual improvement proposal (from the plan).
    - Create a (master) task as follows:
        - The parent epyc of the master task must be <jira_epyc>.
        - The team value must be <jira_team_value>.
        - The activity type must be <jira_activity_type>.
        - Name the task as: tasks_for_[model]_tp_$[num_gpus]_isl_[input_len]_osl_[output_len]_b_[concurrency]__TIME_[current_datetime]. Detect the test parameters from this specific test, and also use a timestamp for [current_datetime] that is formatted as YYYY-MM-DD-HH-MM-SECS (so that the folders can be sorted easily)
        - Add a task description that describes:
            - High-level results of the improvement plan <plan_file>. Add any important details to make it clear and crisp for high-level executives and low-level programmers, so that the description provides a good picture of the results.
            - Info about the per-framework run metadata (can be fetched from the metadata files in results_[test_name]/test_results/) which includes:
                - Docker image used
                - Execution time 
                - OS type
                - Number of GPUs used and their type
        - In the description provide full links (based on previously remembered links) to:
            - General links:
                - Per-framework execution run logs (not profile run logs)
                - Per-framework NSYS trace files (nsys-rep)
                - Combined trace (json file)
                - Summary PDF file
            - Also state that all intermediate analysis results are in the related "analysis" sub-directory, which includes: 
                - Per-framework high-level transformer blocks
                - Per-framework low-level GPU kernel => high-level source-code correlated transformer blocks
                - Per-framework median transformer block
                - Cross-framework transformer block comparison file
                - Improvement plan file for vLLM
        - For the links mention on what machine they reside: hostname and ip address 
    - For each specific improvement plan, create a sub-task as follows:
        - The parent task is the master task from above.
        - The team value must be <jira_team_value>.
        - Name the task as: plan_[id]_[plan_topic] where the [id] is the serial id of the plan and the [plan_topic] is the topic of the plan.
        - Add a sub-task description that describes this specific plan. Make sure to include:
            - High-level description, including the general metrics of impact, difficulty and more.
            - Step-by-step guide with code snippets and maximum details per-step so that expert programmer can execute on it (make sure to be consistent with what is inside <plan_file>)

- Make sure that all new jira tasks and sub-tasks created are completely new and are not overriding any existing tasks or sub-tasks. I.e DO NOT modify anything existing in Jira currently.
- Provide a summary of what was done

"""

    def prompt(self):
        return self.prompt_template.format(
            model=self.model,
            precision=self.precision,
            gpu_type=self.gpu_type,
            isl=self.isl,
            osl=self.osl,
            batch_size=self.batch_size,
            framework_names=self.framework_names,
            framework_source_codes=self.framework_source_codes,
            transformer_blocks=self.transformer_blocks,
            transformer_high_level_blocks=self.transformer_high_level_blocks,
            test_dirs=self.test_dirs,
            cmp_file=self.cmp_file,
            plan_file=self.plan_file,
            output_file=self.output_file,
        )


JIRA_TASKS_FILE = "jira_tasks_output.txt"


def gen_analyze_prompts(config: AnalyzeConfig):
    transformer_block_high_level_prompt = TransformerBlockHighLevelPrompt(
        model=config.model,
        precision=config.precision,
        gpu_type=config.gpu_type,
        framework_name=config.framework_name,
        framework_source_code=config.framework_source_code,
        test_dir=config.test_dir,
        output_file="{}_{}".format(
            config.framework_name, TRANSFORMER_BLOCK_HIGH_LEVEL_OPS_FILE
        ),
    )

    gpu_ops_prompt = GpuOpsPrompt(
        model=config.model,
        gpu_type=config.gpu_type,
        framework_name=config.framework_name,
        test_dir=config.test_dir,
        output_file="{}_{}".format(config.framework_name, GPU_OPS_FILE),
        output_max_gpu_ops=MAX_GPU_OPS,
        output_filter_ops=config.gpu_ops_filter,
    )

    gpu_ops_to_blocks_prompt = GpuOpsToTransformerBlocksPrompt(
        transformer_block_high_level_ops_file=transformer_block_high_level_prompt.output_file,
        gpu_ops_file=gpu_ops_prompt.output_file,
        output_file="{}_{}".format(config.framework_name, GPU_OPS_TO_BLOCKS_FILE),
    )

    median_block_prompt = MedianTransformerBlockPrompt(
        gpu_ops_to_blocks_file=gpu_ops_to_blocks_prompt.output_file,
        output_file="{}_{}".format(config.framework_name, MEDIAN_BLOCK_FILE),
    )

    return (
        [
            transformer_block_high_level_prompt.prompt(),
            gpu_ops_prompt.prompt(),
            gpu_ops_to_blocks_prompt.prompt(),
            median_block_prompt.prompt(),
        ],
        median_block_prompt.output_file,
        transformer_block_high_level_prompt.output_file,
    )


def gen_perf_compare_prompt(configs: list[AnalyzeConfig], block_files: list[str]):
    assert len(configs) >= 2
    assert len(block_files) >= 2

    first_model = configs[0].model
    all_same_model = all(config.model == first_model for config in configs)

    first_gpu_type = configs[0].gpu_type
    all_same_gpu_type = all(config.gpu_type == first_gpu_type for config in configs)

    assert all_same_model and all_same_gpu_type, (
        "all_same_model = {} and all_same_gpu_type = {}".format(
            all_same_model, all_same_gpu_type
        )
    )

    framework_names = [config.framework_name for config in configs]
    framework_source_codes = ([config.framework_source_code for config in configs],)

    perf_cmp_prompt = CompareMedianTransformerBlocksPrompt(
        model=configs[0].model,
        gpu_type=configs[0].gpu_type,
        transformer_blocks=block_files,
        framework_names=framework_names,
        framework_source_codes=framework_source_codes,
        output_file="{}__{}".format(
            "_".join(framework_names),
            PERF_COMPARE_BLOCK_FILE,
        ),
    )

    return perf_cmp_prompt.prompt(), perf_cmp_prompt.output_file


# def gen_general_perf_compare_prompt(
#     configs: list[AnalyzeConfig],
#     block_files: list[str],
#     blocks_perf_cmp_file: str,
#     test_dirs: list[str],
# ):
#     assert len(configs) >= 2
#     assert len(block_files) >= 2

#     first_model = configs[0].model
#     all_same_model = all(config.model == first_model for config in configs)

#     first_gpu_type = configs[0].gpu_type
#     all_same_gpu_type = all(config.gpu_type == first_gpu_type for config in configs)

#     assert all_same_model and all_same_gpu_type, (
#         "all_same_model = {} and all_same_gpu_type = {}".format(
#             all_same_model, all_same_gpu_type
#         )
#     )

#     framework_names = [config.framework_name for config in configs]
#     framework_source_codes = ([config.framework_source_code for config in configs],)

#     general_perf_cmp_prompt = CompareGeneralOverheadsPrompt(
#         model=configs[0].model,
#         gpu_type=configs[0].gpu_type,
#         transformer_blocks=block_files,
#         blocks_perf_cmp_file=blocks_perf_cmp_file,
#         framework_names=framework_names,
#         framework_source_codes=framework_source_codes,
#         test_dirs=test_dirs,
#         output_file="{}__{}".format(
#             "_".join(framework_names),
#             GENERAL_PERF_COMPARE_BLOCK_FILE,
#         ),
#     )

#     return general_perf_cmp_prompt.prompt(), general_perf_cmp_prompt.output_file


def gen_plan_prompt(
    configs: list[AnalyzeConfig],
    block_files: list[str],
    perf_compare_file: str,
    target_framework: str,
):
    assert len(configs) >= 2
    assert len(block_files) >= 2

    first_model = configs[0].model
    all_same_model = all(config.model == first_model for config in configs)

    first_gpu_type = configs[0].gpu_type
    all_same_gpu_type = all(config.gpu_type == first_gpu_type for config in configs)

    assert all_same_model and all_same_gpu_type, (
        "all_same_model = {} and all_same_gpu_type = {}".format(
            all_same_model, all_same_gpu_type
        )
    )

    framework_names = [config.framework_name for config in configs]
    framework_source_codes = ([config.framework_source_code for config in configs],)

    plan_prompt = PlanPrompt(
        model=configs[0].model,
        gpu_type=configs[0].gpu_type,
        transformer_blocks=block_files,
        framework_names=framework_names,
        framework_source_codes=framework_source_codes,
        comparison_file=perf_compare_file,
        target_framework=target_framework,
        output_file="{}__{}".format(
            "_".join(framework_names),
            PLAN_FILE,
        ),
    )

    return plan_prompt.prompt(), plan_prompt.output_file


def gen_combined_trace_prompt(
    model,
    precision,
    gpu_type,
    isl,
    osl,
    batch_size,
    configs: list[AnalyzeConfig],
    target_framework: str,
):
    framework_names = [config.framework_name for config in configs]
    framework_source_codes = [config.framework_source_code for config in configs]
    test_dirs = [config.test_dir for config in configs]

    block_files = []
    block_high_level_files = []
    for config in configs:
        _, block_file, block_high_level_file = gen_analyze_prompts(config)
        block_files.append(block_file)
        block_high_level_files.append(block_high_level_file)

    _, perf_cmp_file = gen_perf_compare_prompt(configs, block_files)
    _, plan_file = gen_plan_prompt(
        configs, block_files, perf_cmp_file, target_framework
    )

    combined_trace_prompt = CombinedTracePrompt(
        model=model,
        precision=precision,
        gpu_type=gpu_type,
        isl=isl,
        osl=osl,
        batch_size=batch_size,
        framework_names=framework_names,
        framework_source_codes=framework_source_codes,
        transformer_blocks=block_files,
        transformer_high_level_blocks=block_high_level_files,
        test_dirs=test_dirs,
        cmp_file=perf_cmp_file,
        plan_file=plan_file,
        output_file=TRACE_COMBINED_FILE,
    )

    return combined_trace_prompt.prompt()


def gen_jira_tasks_prompt(
    model,
    precision,
    gpu_type,
    isl,
    osl,
    batch_size,
    configs: list[AnalyzeConfig],
    target_framework: str,
):
    framework_names = [config.framework_name for config in configs]
    framework_source_codes = [config.framework_source_code for config in configs]
    test_dirs = [config.test_dir for config in configs]

    block_files = []
    block_high_level_files = []
    for config in configs:
        _, block_file, block_high_level_file = gen_analyze_prompts(config)
        block_files.append(block_file)
        block_high_level_files.append(block_high_level_file)

    _, perf_cmp_file = gen_perf_compare_prompt(configs, block_files)
    _, plan_file = gen_plan_prompt(
        configs, block_files, perf_cmp_file, target_framework
    )

    jira_tasks_prompt = JiraTasksPrompt(
        model=model,
        precision=precision,
        gpu_type=gpu_type,
        isl=isl,
        osl=osl,
        batch_size=batch_size,
        framework_names=framework_names,
        framework_source_codes=framework_source_codes,
        transformer_blocks=block_files,
        transformer_high_level_blocks=block_high_level_files,
        test_dirs=test_dirs,
        cmp_file=perf_cmp_file,
        plan_file=plan_file,
        output_file=JIRA_TASKS_FILE,
    )

    return jira_tasks_prompt.prompt()
