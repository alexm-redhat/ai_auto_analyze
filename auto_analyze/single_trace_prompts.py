from dataclasses import dataclass
from typing import ClassVar

from auto_analyze.single_trace_config import (
    SingleTraceParams,
    SingleTraceConfig,
    TRACE_FILE_TYPE_NSYS,
    TRACE_FILE_TYPE_PYTORCH,
    HIGH_LEVEL_OPS_FILE,
    GPU_OPS_TXT_FILE,
    GPU_OPS_TO_BLOCKS_FILE,
    MEDIAN_BLOCK_FILE,
    TRANSFORMER_BLOCK_TRACE_JSON_FILE,
    TRANSFORMER_BLOCK_TRACE_TXT_FILE,
    PERF_ANALYSIS_FILE,
)


TRACE_DESCRIPTIONS = {
    TRACE_FILE_TYPE_NSYS: (
        "- The file <trace_file> is an NSYS profile result file in SQLite"
        " format of <framework_name> running <model> on <gpu_type> GPU."
    ),
    TRACE_FILE_TYPE_PYTORCH: (
        "- The file <trace_file> is a PyTorch Profiler trace in Chrome trace"
        " JSON format of <framework_name> running <model> on <gpu_type> GPU."
    ),
}


# ---------------------------------------------------------------------------
# Context header fields (emitted by SingleTraceParams.context_header())
# These use <angle_brackets> and are defined in the context block.
#
# Always present:
#   <model>                 HuggingFace model tag (e.g. "nvidia/DeepSeek-R1-NVFP4")
#   <model_hf_url>          Derived: https://huggingface.co/<model>
#   <gpu_type>              GPU hardware type (e.g. "B200")
#   <framework_name>        Framework being analyzed (e.g. "vllm", "sglang")
#   <framework_source_code> Path to the framework source tree
#   <trace_file>            Path to the trace file (.sqlite or .json)
#   <trace_file_type>       Auto-detected: "NSYS" (.sqlite) or "PYTORCH" (.json)
#   <trace_gpu_focus>       "ALL" or a GPU ID (e.g. "0")
#   <batch_size_range>      Batch size(s) used in the run
#   <prefill_size_range>    Prefill / input sequence length(s)
#   <output_size_range>     Output / decode sequence length(s)
#   <max_gpu_ops>           Max GPU operations to extract (default 2000)
#   <run_command>           Command line used to launch the framework
#
# Conditional (included when set):
#   <run_log>               Path to the framework run log
#   <high_level_focus>      Free-text guidance on what to focus the analysis on
#   <output_dir>            Absolute path to the output directory for all results
#
# Step-specific file parameters use [square_brackets] and are resolved
# via Python .format() substitution per prompt.
# ---------------------------------------------------------------------------


SCRIPT_ERROR_HANDLING = (
    "If the script fails with an error, read the error message, "
    "diagnose the root cause, fix the script, and re-execute. "
    "Repeat until the script runs successfully."
)


@dataclass
class TransformerBlockHighLevelPrompt:
    params: SingleTraceParams
    output_file: str
    prompt_template: ClassVar[str] = """
{context}
[output_file] = {output_file}

TASK — IDENTIFY TRANSFORMER BLOCK HIGH-LEVEL OPERATIONS:

The model <model> is a sequence of transformer blocks. Your main task is to investigate the source code at <framework_source_code>, find the files that implement <model>, and determine the sequence of high-level operations in each transformer block type.

This output will serve as a GUIDE for a subsequent step that correlates actual GPU trace operations to these high-level operations using both this summary and independent source code analysis. Your goal is to provide an accurate logical map of the transformer block structure — not to predict exact GPU kernel names or counts.

First, inspect <model_hf_url> (the HuggingFace model page for <model>) to learn about this model's architecture, precision/quantization, and key config values (num_layers, num_experts, hidden_size, etc.).

To guide your source code investigation, use the run command and run logs to understand which code paths are actually taken:
- The command used to run the framework is: <run_command>
{run_log_section}

{focus_section}

Read and analyze in-detail the source code to detect the pieces that implement the transformer block types for the currently executed model.

For each transformer block type:
- Find the sequence of operations by reading the decoder layer's forward() method and all sub-modules it calls.
- Each operation should be a distinct logical compute step (e.g. layernorm, projection GEMM, attention, quantization, allreduce, activation function).
- Note fusions: where the framework or compiler fuses multiple logical steps into one operation (e.g. allreduce + layernorm + quantize fused into one call).
- Note stream parallelization: where operations run on different CUDA streams concurrently.
- Note cross-layer boundary effects: where the last operation of one block is fused with the first operation of the next block.
- For each operation provide: source code reference (file:line), what it computes, and any relevant details (dimensions, data types, parallelism).

OUTPUT FORMAT:
Dump to [output_file] a well-structured human-readable text file with:

Per block type: a section with block type name, layer range, and a properly aligned pipe-separated table where each row is one operation with columns: index, operation name, source code reference, execution details. Ensure all "|" column separators are aligned across every row.

"""

    def prompt(self):
        p = self.params

        run_log_section = ""
        if p.run_log:
            run_log_section = (
                "- The run log of executing the framework is at: <run_log>\n"
                "- Read the run log to identify which modules, classes, backends, and code paths "
                "were actually activated during execution. The log is ground truth — use it to "
                "confirm or override the code path analysis from flags."
            )

        focus_section = ""
        if p.high_level_focus:
            focus_section = (
                "Focus area for this analysis: <high_level_focus>"
            )

        return self.prompt_template.format(
            context=p.context_header(),
            output_file=self.output_file,
            run_log_section=run_log_section,
            focus_section=focus_section,
        )


@dataclass
class GpuOpsPrompt:
    params: SingleTraceParams
    output_txt_file: str
    prompt_template: ClassVar[str] = """
{context}
[output_txt_file] = {output_txt_file}
{trace_description}

Write and execute a single self-contained Python script that does the following:

1. Read the trace file at <trace_file>.
2. Extract all GPU operations with the fields listed below.
3. Write the output file.

The script must handle the trace format (<trace_file_type>):
- For NSYS SQLite traces:
    - Query CUPTI_ACTIVITY_KIND_KERNEL (with grid/block/registers/shared memory columns).
    - Check if CUPTI_ACTIVITY_KIND_MEMCPY exists; if so, query and merge memory copy operations.
    - Query StringIds to resolve kernel name IDs.
- For PyTorch JSON traces: parse the Chrome trace JSON and extract GPU kernel events.

{gpu_focus_instruction}

For each GPU operation, extract:
    a. short_name (max 50 chars), start_ns, end_ns, duration_ns, stream, full_name (max 200 chars)
    b. category: "compute_kernel", "memory_transfer", "communication", or "synchronization"
    c. Kernel launch parameters (NSYS only): grid, block, registers_per_thread, shared_memory_bytes
    d. For communication ops: comm_algorithm, comm_size_hint
    e. For memory transfers: transfer_bytes, transfer_kind

Keep max <max_gpu_ops> operations. If more exist, keep the LAST <max_gpu_ops> (steady-state). Sort by start_ns.

OUTPUT FILE:
[output_txt_file]: A properly aligned table with a header row and one row per GPU operation. Include all extracted fields as columns. Use consistent column widths and pipe-separated format so the table is easy to read.

{script_error_handling}

VERIFICATION CHECKLIST:
- [ ] Script executes successfully
- [ ] Output file generated with properly aligned table
- [ ] Sorted by start_ns, count <= <max_gpu_ops>, every op has non-null category
"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.params.context_header(),
            output_txt_file=self.output_txt_file,
            gpu_focus_instruction=self.params.gpu_focus_instruction(),
            trace_description=TRACE_DESCRIPTIONS[self.params.trace_file_type],
            script_error_handling=SCRIPT_ERROR_HANDLING,
        )


@dataclass
class GpuOpsToTransformerBlocksPrompt:
    params: SingleTraceParams
    high_level_ops_file: str
    gpu_ops_txt_file: str
    output_file: str
    median_block_file: str
    prompt_template: ClassVar[str] = """
{context}
[high_level_ops_file] = {high_level_ops_file}
[gpu_ops_txt_file] = {gpu_ops_txt_file}
[output_file] = {output_file}
[median_block_file] = {median_block_file}
- [high_level_ops_file] describes the transformer block types with per-kernel operations and expected kernels.
- [gpu_ops_txt_file] contains a table of all GPU operations sorted by start time, with kernel names, timestamps, stream IDs, and launch parameters.
- <framework_source_code> is the source code of <framework_name>.

Your task is to analyze the sequence of GPU low-level operations in [gpu_ops_txt_file] and detect the transformer blocks and their types. Use the high-level block types from [high_level_ops_file] as a guide, but do NOT rely solely on them — you must perform your own independent source code deep-dive analysis for each and every low-level GPU operation.

THIS IS THE MOST CRITICAL STEP IN THE ENTIRE ANALYSIS. The goal is to find the 100% precise correlation between every low-level GPU operation and its corresponding high-level transformer block operation. No mistakes are acceptable.

STEP 1 — SOURCE CODE DEEP-DIVE FOR EACH GPU OPERATION:
For each distinct GPU kernel name that appears in [gpu_ops_txt_file]:
- Search the source code at <framework_source_code> to find where this kernel is launched.
- Trace the call chain from the kernel launch point UP to the decoder layer's forward() method.
- Determine which high-level transformer block operation this kernel implements.
- Use the context of surrounding operations (what comes before and after in the GPU ops sequence) to disambiguate kernels that could serve multiple roles.
- Cross-reference with [high_level_ops_file] to validate your finding, but trust your source code analysis if there is a conflict.

IMPORTANT: The [high_level_ops_file] from the previous step provides a SUMMARY of the expected high-level operations, but it may have missed operations, mis-identified kernel names, or made errors. YOUR analysis here must be MORE thorough because:
1. You can see the ACTUAL GPU operations and their actual kernel names (not just expected names).
2. You can see the sequence and context of surrounding operations.
3. You have the high-level summary as a starting reference point.

STEP 2 — BLOCK BOUNDARY DETECTION:
- Using your kernel-to-operation mappings from Step 1, identify where each transformer block starts and ends in the GPU ops sequence.
- A transformer block starts with its FIRST operation and ends with its LAST operation.
- Verify that the operations within each detected block follow the expected order from your analysis.
- BEWARE OF AMBIGUOUS BOUNDARY KERNELS: The same kernel type may appear multiple times within a single transformer block. Look at WHAT FOLLOWS each candidate boundary kernel to determine if it's a block start or a mid-block operation.

STEP 3 — CLASSIFICATION AND WARMUP:
- Classify each block by type. Flag first 2 and last 2 as WARMUP/COOLDOWN, rest as REPRESENTATIVE.

STEP 4 — PER-BLOCK CORRELATION:
- For each block, list every GPU operation with its correlated high-level operation name.
- Each kernel must have a UNIQUE specific high_level_op name within its block.
- The high_level_op name must be descriptive enough to understand what the operation does (e.g. "q_b_proj_gemm", "moe_gate_router_gemm", "fused_allreduce_rms_norm_fp4_quant"), but concise — aim for 3-6 words in snake_case, max ~50 characters.
- VALIDATE: the operation sequence within each block must follow a logical order consistent with the decoder layer's forward() method.

STEP 5 — PER-BLOCK TIMING:
- Calculate wall_time (from first op start to last op end), compute_time (sum of all op durations), idle_time, and idle_pct per block.

STEP 6 — FULL CONSISTENCY VALIDATION:
Perform a thorough validation across ALL blocks and ALL operations:

a. BLOCK-LEVEL CONSISTENCY:
   - Verify that all blocks of the same type have the SAME number of operations.
   - Verify that all blocks of the same type have the SAME sequence of high_level_op names in the same order.
   - If any block differs in op count or sequence from the others of its type, flag it as an anomaly and investigate — either the block boundary is wrong or an operation was missed/misclassified.

b. ZERO UNCORRELATED OPERATIONS:
   - Every single GPU operation in [gpu_ops_txt_file] must be accounted for — either assigned to a transformer block or explicitly classified as an inter-block operation (e.g. embedding, sampling, LM head, scheduling overhead).
   - There must be NO operations left without a high-level correlation. If any operation lacks a correlation, perform a source code deep-dive for that specific kernel to determine what it does.
   - Report the total number of GPU operations, number assigned to blocks, and number classified as inter-block, and verify they sum correctly.

c. SOURCE CODE GROUNDING:
   - Every high_level_op name used in the correlation must be traceable back to a specific location in the source code at <framework_source_code>.
   - No operation should be labeled with a generic name like "unknown" or "misc" — if you cannot identify what a kernel does, search harder in the source code.

STEP 7 — MEDIAN BLOCK SELECTION:
- From REPRESENTATIVE blocks, select median wall-time block per type. Primary = most frequent type.
- Compute timing distribution: min, p25, median, p75, max, CoV. Flag HIGH VARIANCE if CoV > 10%.

STEP 8 — WRITE OUTPUTS:

FILE 1 — [output_file]: Structured text with:
    - Header (model, framework, GPU, block counts, warmup/cooldown, consistency)
    - Properly aligned pipe-separated per-block tables where each kernel has columns for:
        - Unique high_level_op (do not skip)
        - source_code_ref: the file:line references where this kernel is launched (do not skip, do not truncate)
        - source_code_explanation: a detailed explanation of what this kernel does and its context in the call chain (use a long text line if needed — do not truncate)
      Ensure all "|" column separators are aligned across every row.
    - Median selection section.

FILE 2 — [median_block_file]: Just the selected median block content.

{script_error_handling}

VERIFICATION CHECKLIST:
- [ ] Both output files generated
- [ ] Every GPU op accounted for
- [ ] Block boundaries are correct: verified by source code analysis, not just pattern matching
- [ ] Operation order within each block matches the decoder layer's forward() method
- [ ] Every kernel has UNIQUE high_level_op name within its block, determined by source code deep-dive
- [ ] Warmup/cooldown flagged, median selected correctly
- [ ] Source code was consulted for EACH distinct kernel type, not just the ones listed in [high_level_ops_file]
"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.params.context_header(),
            high_level_ops_file=self.high_level_ops_file,
            gpu_ops_txt_file=self.gpu_ops_txt_file,
            output_file=self.output_file,
            median_block_file=self.median_block_file,
            script_error_handling=SCRIPT_ERROR_HANDLING,
        )


@dataclass
class PerfAnalysisPrompt:
    params: SingleTraceParams
    median_block_file: str
    high_level_ops_file: str
    gpu_ops_to_blocks_file: str
    gpu_ops_txt_file: str
    output_file: str
    prompt_template: ClassVar[str] = """
{context}
[median_block_file] = {median_block_file}
[high_level_ops_file] = {high_level_ops_file}
[gpu_ops_to_blocks_file] = {gpu_ops_to_blocks_file}
[gpu_ops_txt_file] = {gpu_ops_txt_file}
[output_file] = {output_file}
- The model is <model> running via <framework_name> on <gpu_type> GPU.
- Execution parameters: batch size <batch_size_range>, prefill/input size <prefill_size_range>, output size <output_size_range>.
- IMPORTANT: The execution parameters above are FIXED constraints. All analysis and improvements must target the framework implementation under EXACTLY these parameters. Every observation, classification, and proposal must be specific to this batch size, prefill size, and output size — not generic advice. For example, at BS=1 decode, most GEMMs are memory-bound because the matrices are too small to saturate compute; at BS=64 the same GEMMs would be compute-bound. Your analysis must reflect the actual regime.
- [median_block_file] is the median transformer block with correlated high-level operations.
- [high_level_ops_file] describes high-level operations with source code references.
- [gpu_ops_to_blocks_file] has all detected blocks with timing statistics and median selection.
- [gpu_ops_txt_file] is the full GPU operations table with kernel launch parameters.
- <framework_source_code> is the source code of <framework_name>.

{perf_analysis_focus_section}

BEFORE WRITING THE OUTPUT, perform the following analysis steps:

PHASE 1 — SOURCE CODE DEEP-DIVE:
For each operation in the median block:
- Read the source code at <framework_source_code> and trace the full call chain from the decoder layer's forward() method down to the kernel launch point.
- Understand not just this operation in isolation, but its surrounding context: what runs before it, what runs after it, what data it consumes and produces, and how it interacts with adjacent operations.
- Look up the kernel launch parameters from [gpu_ops_txt_file] (grid, block, registers, shared memory).
- Classify as compute-bound, memory-bound, or communication-bound for the specific execution parameters (BS=<batch_size_range>, ISL=<prefill_size_range>, OSL=<output_size_range>).
- Read surrounding source code for TODO/FIXME comments, alternative code paths, optimization flags, and fused/optimized variants that exist but may not be activated.

Additionally, determine the execution context from the run command, run log, and source code:
- Is CUDA graph capture enabled? If so, what mode (full, piecewise, or other)? Which batch sizes are captured?
- Is torch.compile enabled? What compilation passes and fusions are active?
- What other framework-level optimizations are active (e.g., async scheduling, custom allreduce backends)?
This context is essential — proposals must not suggest improvements that are already handled by the active execution mode. For example, if CUDA graphs are used in full-capture mode for decode, kernel launch overhead is already eliminated and should not be proposed as an improvement.

PHASE 2 — IMPROVEMENT ANALYSIS:
Using the understanding from Phase 1, analyze improvement opportunities:
- For EACH individual kernel: can it be replaced with a faster variant, better configured, or fused with an adjacent kernel?
- For SEQUENCES of kernels: are there fusion opportunities, unnecessary dtype conversions, reducible inter-kernel gaps, or stream parallelization possibilities?
- For the BLOCK as a whole: are there scheduling improvements, stream overlap opportunities, or algorithmic changes?
- Every kernel must have at least one improvement proposal, even if the improvement is small — small gains compound across all operations.
- Derive all impact estimates from actual trace timing data, not speculation.
- IMPORTANT: Do not propose improvements that are already handled by the active execution context. For example, do not propose reducing kernel launch gaps if CUDA graphs are already capturing the execution path, and do not propose kernel fusions that are already performed by active torch.compile passes.

PHASE 3 — REVIEW AND VALIDATE:
Before writing the output, review your analysis:
- Verify that every source code reference (file:line) is accurate.
- Verify that every improvement proposal is grounded in what you actually saw in the source code, not generic advice.
- Verify that impact estimates are consistent with the trace data (durations, gaps, percentages sum correctly).
- Ensure no operation was skipped — every kernel in the median block must have been analyzed and have at least one proposal.
- Verify that no proposal conflicts with the active execution context (CUDA graphs, torch.compile, etc.).

Now, write the output document.

The document is structured from high-level overview down to deep details. Produce the following sections in order:

1. EXECUTIVE SUMMARY
   - Model, framework, GPU, execution config
   - Median block: index, layer, wall time, variance
   - One paragraph: top 3 findings and total estimated improvement potential

2. MEDIAN BLOCK TIMING OVERVIEW
   - Wall time, compute vs idle, timing distribution (min/p25/median/p75/max/CoV)
   - Per-stream breakdown if multiple streams are used

3. KEY OBSERVATIONS AND ACTION PLAN
   - Ranked list of ALL bottlenecks and improvement opportunities for BS=<batch_size_range> / ISL=<prefill_size_range> / OSL=<output_size_range>, from largest to smallest impact. Include every opportunity, even small ones — improvements compound.
   - For each: one-line description, estimated savings (ns and % of block wall time), difficulty, and proposal ID reference (P1, P2, ...).
   - Projected median block wall time if all low-difficulty proposals were implemented: current → projected (% improvement).
   - Projected median block wall time if ALL proposals were implemented: current → projected (% improvement).

4. MEDIAN BLOCK OPERATIONS TABLE WITH IMPROVEMENTS
   THIS IS THE MOST IMPORTANT SECTION — it provides the at-a-glance overview that readers will focus on.

   Produce a properly aligned pipe-separated table with ALL operations from [median_block_file] in execution order. Ensure all columns are aligned using consistent column widths. Columns:
   | Idx | High-Level Op | Dur (us) | % Blk | Stream | Proposal | Impact % | Improvement Summary | Low-Level Kernel | Details |

   Column descriptions:
   - Idx: operation index in the block
   - High-Level Op: the correlated high_level_op name
   - Dur (us): duration in microseconds
   - % Blk: percentage of total block wall time
   - Stream: CUDA stream ID
   - Proposal: short reference (P1, P2, ...) linking to section 5; comma-separated if multiple (e.g., "P3, P7")
   - Impact %: estimated savings as percentage of total block wall time
   - Improvement Summary: one-line explanation of what to do
   - Low-Level Kernel: the actual GPU kernel name (short_name from trace)
   - Details: additional context that helps understand the improvement — this can be longer text, do not truncate

   Rules:
   - Every row must have a proposal. Even if the kernel is already near-optimal, propose something (e.g., "Could save ~0.1us via fusion with adjacent op" or "Near-optimal; consider stream overlap"). Small gains compound.
   - Keep all column separators "|" aligned across every row.

   After the table, include a TOTALS row:
   ```
   TOTAL ESTIMATED SAVINGS: ~<ns> ns (~<pct>% of block wall time)
   Key ideas: <2-3 sentence summary of the biggest improvement themes>
   ```

5. IMPROVEMENT PROPOSALS
   Each proposal is referenced by its ID (P1, P2, ...) from the table in section 4.
   Include the impact % in the title so it is immediately visible:
   ```
   P<N>: <Title> [<pct>% of block wall time]
   ```

   For each proposal:
   a. AFFECTED OPERATIONS — Which operations from the median block this targets.
   b. EVIDENCE — Specific timing data from the trace. Why this matters at BS=<batch_size_range>/ISL=<prefill_size_range>/OSL=<output_size_range>.
   c. ROOT CAUSE — Why it is slow, grounded in source code. Reference specific files and lines.
   d. PROPOSED FIX — What to change.
      - Type: "kernel_fusion", "kernel_replacement", "scheduling", "config_tuning", or "algorithmic".
      - Step-by-step implementation guide. Each step must be concrete and actionable:
        1. State what to change and where (file:line).
        2. Show a BEFORE snippet: the actual current code from the source (5-15 lines, enough to see context).
        3. Show an AFTER snippet: the proposed modified code with the change applied.
        4. Briefly explain why the change helps.
        Example format:
        ```
        Step 1: Fuse RMSNorm + residual add
          File: vllm/model_executor/layers/layernorm.py:42-48
          BEFORE:
            norm_out = rms_norm(hidden_states, weight)
            output = norm_out + residual
          AFTER:
            output = fused_add_rms_norm(hidden_states, residual, weight)
          Why: Eliminates one kernel launch and one intermediate buffer.
        ```
   e. IMPACT ESTIMATE — Derived from actual trace data:
      "<operation>: <current_ns> ns -> <projected_ns> ns (savings: <saved_ns> ns, <pct>% of block wall time)"
   f. Difficulty: low / medium / high
   g. Risks/trade-offs

   Rank proposals by (impact * ease), largest first.

   End with:
   ```
   TOTAL ESTIMATED SAVINGS
   =======================
   All proposals: ~<ns> ns (~<pct>%)
   Low-difficulty only: ~<ns> ns (~<pct>%)
   ```

6. SOURCE CODE ANALYSIS AND UTILIZATION
   For each operation in the median block (ordered by duration, largest first):
   - Trace the call chain from forward() to the kernel launch.
   - Source code references (file:line).
   - Kernel launch parameters from [gpu_ops_txt_file] (grid, block, registers, shared memory).
   - Classify as compute-bound, memory-bound, or communication-bound for the specific execution parameters.
   - Utilization estimate vs <gpu_type> specs.
   - Gap to next op with cause attribution.
   - Performance observations grounded in source code or trace data.

   Format per operation:
   ```
   <operation_name> (<duration_ns> ns, <pct>% of block)
     Call chain: <class.method> -> ... -> <kernel>
     Source: <file1:line>, <file2:line>
     Kernel params: grid=<>, block=<>, regs=<>, smem=<>
     Classification: <type> — <rationale for these execution params>
     Utilization: <estimate vs GPU specs>
     Gap to next op: <gap_ns> ns on stream <S> (<cause>)
     Performance observations:
       - <observation>
   ```

7. SUMMARY
   - Concise recap: total block wall time, total estimated savings, top 3 proposals by impact.
   - Regime analysis: would bottleneck priorities shift at a different batch size?
   - Any unusual patterns or anomalies in the trace.

OUTPUT FORMAT: Structured text, sections 1-7.
Write the output to [output_file].
"""

    def prompt(self):
        focus = self.params.perf_analysis_focus
        focus_section = ""
        if focus:
            focus_section = (
                f"PERFORMANCE ANALYSIS FOCUS:\n{focus}\n"
                "Pay special attention to the above focus areas, "
                "especially in sections 3-5."
            )

        return self.prompt_template.format(
            context=self.params.context_header(),
            median_block_file=self.median_block_file,
            high_level_ops_file=self.high_level_ops_file,
            gpu_ops_to_blocks_file=self.gpu_ops_to_blocks_file,
            gpu_ops_txt_file=self.gpu_ops_txt_file,
            output_file=self.output_file,
            perf_analysis_focus_section=focus_section,
        )


@dataclass
class TransformerBlockTracePrompt:
    params: SingleTraceParams
    median_block_file: str
    high_level_ops_file: str
    gpu_ops_txt_file: str
    perf_analysis_file: str
    output_json_file: str
    output_txt_file: str
    prompt_template: ClassVar[str] = """
{context}
[median_block_file] = {median_block_file}
[high_level_ops_file] = {high_level_ops_file}
[gpu_ops_txt_file] = {gpu_ops_txt_file}
[perf_analysis_file] = {perf_analysis_file}
[output_json_file] = {output_json_file}
[output_txt_file] = {output_txt_file}
- [median_block_file] is the median transformer block with correlated operations.
- [high_level_ops_file] describes the block operations with source code references.
- [gpu_ops_txt_file] is the full GPU operations table — use it for exact kernel timestamps and launch parameters.
- [perf_analysis_file] has source code analysis, utilization, and improvement proposals.
- <framework_source_code> is the source code of <framework_name>.

Generate a Chrome trace JSON and human-readable summary of the median transformer block. In Perfetto, a user should:
- See execution context in the info bar
- See ALL operations on their CUDA streams — NO operation may be dropped or hidden
- Click any operation for source code and improvement details

Write and execute a single self-contained Python script:

1. Read all input files. Match median block ops to [gpu_ops_txt_file] by start_ns/end_ns.

{gpu_focus_instruction}

2. Build Chrome trace JSON:

INFO BAR (pid=0, tid=0): full block duration, name = "<model> | <framework_name> | <gpu_type> | BS=<batch_size_range> ISL=<prefill_size_range> OSL=<output_size_range>", args with execution_config, median_block info, total_proposals count.

CUDA STREAM LANES:
- Each CUDA stream gets one pid (starting from pid=2; pid=0 is reserved for the info bar).
- HANDLING OVERLAPS (PDL): Operations on the same CUDA stream can overlap in time (common with Programmatic Dependent Launch). Use multiple tids within the same pid to separate overlapping operations into lanes:
    - tid=0 is the main lane. tid=1 is the first overflow lane, tid=2 the second, etc.
    - Use a greedy lane assignment: for each operation sorted by start time, find the first tid for this stream where the operation does not overlap with any existing event on that tid. If no tid works, allocate a new one.
    - Use the MINIMUM number of tids needed to avoid all within-tid overlaps.
    - Overlap check: event_A.ts + event_A.dur > event_B.ts means they overlap.
- LANE NAMING:
    - Use process_name metadata for the pid: "Stream 23".
    - Use thread_name metadata for each tid: tid=0 -> "Stream 23", tid=1 -> "Stream 23 (PDL lane 2)", etc.
    - Use process_sort_index metadata to order streams logically in Perfetto.
- Every single operation from the median block MUST appear in the trace. Verify at the end that the number of kernel events in the JSON equals the number of operations in [median_block_file].

OPERATION NAMING: Prepend a brief high-level prefix per kernel (e.g., "Attn: ...", "MoE: ...", "Comm: ..."). Keep prefix short.

OPERATION ARGS: For each kernel event, add "args" with:
- "high_level_op": correlated operation name
- "source_code": file:line references with call chain explanation
- "proposed_improvements": relevant proposals from [perf_analysis_file], or "No improvements proposed for this operation."

TIMING: Block starts at time 0.

3. Write outputs:
[output_json_file]: Chrome trace JSON, displayTimeUnit: "ns", all ts/dur in nanoseconds.
[output_txt_file]: header, properly aligned pipe-separated op table with all operations (ensure all "|" column separators are aligned across every row), per-stream summary.

4. VALIDATE the output (in the script, before finishing):
- Load the generated JSON back and count kernel events (ph="X", excluding info bar).
- Compare against the number of operations in [median_block_file]. They must be equal.
- For every (pid, tid) pair, collect all events (ph="X"), sort by ts, and verify ZERO overlaps: event_A.ts + event_A.dur <= event_B.ts for all consecutive pairs.
- If any validation fails, print the error, fix the lane assignment, and re-generate.

{script_error_handling}

VERIFICATION CHECKLIST:
- [ ] Script executes, both files generated, JSON valid
- [ ] Info bar present
- [ ] ALL operations from [median_block_file] are present in the trace (count matches)
- [ ] One pid per CUDA stream (starting from pid=2), multiple tids for PDL overflow lanes
- [ ] ZERO overlaps within any (pid, tid) pair — validated programmatically in the script
- [ ] Every kernel has high_level_op, source_code, proposed_improvements in args
- [ ] Block starts at t=0, all ts/dur in nanoseconds
"""

    def prompt(self):
        return self.prompt_template.format(
            context=self.params.context_header(),
            gpu_focus_instruction=self.params.gpu_focus_instruction(),
            median_block_file=self.median_block_file,
            high_level_ops_file=self.high_level_ops_file,
            gpu_ops_txt_file=self.gpu_ops_txt_file,
            perf_analysis_file=self.perf_analysis_file,
            output_json_file=self.output_json_file,
            output_txt_file=self.output_txt_file,
            script_error_handling=SCRIPT_ERROR_HANDLING,
        )


def gen_single_trace_prompts(config: SingleTraceConfig, file_prefix=""):
    import os
    output_dir = os.path.abspath(config.output_dir)

    def _path(rel):
        return os.path.join(output_dir, f"{file_prefix}{rel}")

    high_level_ops_file = _path(HIGH_LEVEL_OPS_FILE)
    gpu_ops_txt_file = _path(GPU_OPS_TXT_FILE)
    gpu_ops_to_blocks_file = _path(GPU_OPS_TO_BLOCKS_FILE)
    median_block_file = _path(MEDIAN_BLOCK_FILE)
    perf_analysis_file = _path(PERF_ANALYSIS_FILE)
    trace_json_file = _path(TRANSFORMER_BLOCK_TRACE_JSON_FILE)
    trace_txt_file = _path(TRANSFORMER_BLOCK_TRACE_TXT_FILE)

    params: SingleTraceParams = config

    prompt_objects = [
        TransformerBlockHighLevelPrompt(
            params=params,
            output_file=high_level_ops_file,
        ),
        GpuOpsPrompt(
            params=params,
            output_txt_file=gpu_ops_txt_file,
        ),
        GpuOpsToTransformerBlocksPrompt(
            params=params,
            high_level_ops_file=high_level_ops_file,
            gpu_ops_txt_file=gpu_ops_txt_file,
            output_file=gpu_ops_to_blocks_file,
            median_block_file=median_block_file,
        ),
        PerfAnalysisPrompt(
            params=params,
            median_block_file=median_block_file,
            high_level_ops_file=high_level_ops_file,
            gpu_ops_to_blocks_file=gpu_ops_to_blocks_file,
            gpu_ops_txt_file=gpu_ops_txt_file,
            output_file=perf_analysis_file,
        ),
        TransformerBlockTracePrompt(
            params=params,
            median_block_file=median_block_file,
            high_level_ops_file=high_level_ops_file,
            gpu_ops_txt_file=gpu_ops_txt_file,
            perf_analysis_file=perf_analysis_file,
            output_json_file=trace_json_file,
            output_txt_file=trace_txt_file,
        ),
    ]

    step_names = [
        "Extracting high-level transformer block operations from source code",
        "Extracting GPU operations from trace",
        "Correlating GPU operations to transformer blocks and selecting median",
        "Generating performance analysis",
        "Generating transformer block trace",
    ]

    total_steps = len(step_names)
    prompts = []
    for i, (step_name, prompt_obj) in enumerate(
        zip(step_names, prompt_objects), 1
    ):
        prompts.append(
            {
                "cmd": (
                    f"print('\\n=== [Step {i}/{total_steps}]"
                    f" {step_name}... ===')"
                )
            }
        )
        prompts.append(prompt_obj.prompt())

    output_files = {
        "high_level_ops": high_level_ops_file,
        "gpu_ops_txt": gpu_ops_txt_file,
        "gpu_ops_to_blocks": gpu_ops_to_blocks_file,
        "median_block": median_block_file,
        "transformer_block_trace_json": trace_json_file,
        "transformer_block_trace_txt": trace_txt_file,
        "perf_analysis": perf_analysis_file,
    }

    return prompts, output_files
