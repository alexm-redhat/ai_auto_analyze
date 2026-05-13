# Cross-Commit Performance Comparison

## Motivation

When a framework like vLLM ships a new version, performance can change in both
directions — some kernels get faster through new fusions or specialized kernels,
while others may regress due to library updates or code path changes. Identifying
exactly which operations improved and which regressed requires comparing GPU traces
from two commits kernel-by-kernel, which is extremely tedious to do manually.

The cross-commit analysis pipeline automates this. Given two single-trace analysis
results from different commits of the same framework, it:

1. **Matches operations** across the two median transformer blocks one-by-one
2. **Identifies all performance differences** (both positive and negative) with
   root cause analysis grounded in source code
3. **Optionally generates an improvement plan** with step-by-step coding guides
   to recover any regressions

This guide walks through the complete process using a concrete example: comparing
vLLM `v0.16.0` against the latest `main` branch running Kimi-K2.5-NVFP4 on B200
GPUs in a low-latency pure-decode configuration.

## Prerequisites

- Two completed single-trace analyses (one per commit) with matching execution
  parameters (same model, GPU type, batch size, prefill size, output size)
- See [Single-Trace Analysis and High-Level Trace Annotation](single_trace_analysis_and_annotation_guide.md)
  for how to run single-trace analysis for each commit

## End-to-End Walkthrough

### Step 1: Run Single-Trace Analysis for Each Commit

Each commit needs its own trace capture and single-trace analysis. The execution
parameters must be identical across both runs so the comparison is meaningful.

```bash
# --- Shared parameters (must be identical for both commits) ---
MODEL="nvidia/Kimi-K2.5-NVFP4"
GPU_TYPE="B200"
BATCH_RANGE="1"
PREFILL_RANGE="4"
OUTPUT_RANGE="1024"
SOURCE_CODE="/path/to/vllm"
```

#### Commit A — latest main

```bash
COMMIT_A="8f89381fc6b2d54591a7a560e20ee5211ce1ac33"
TRACE_FILE_A="/path/to/traces/commit_a/rank0.pt.trace.json.gz"
RUN_LOG_A="/path/to/traces/commit_a/run_log.txt"
ANALYZE_DIR_A="/path/to/results/vllm_latest/analyze"
CONFIG_A="/path/to/results/vllm_latest/single_trace_config"

python -m auto_analyze.create_single_trace_config \
    --model $MODEL \
    --gpu-type $GPU_TYPE \
    --batch-size-range $BATCH_RANGE \
    --prefill-size-range $PREFILL_RANGE \
    --output-size-range $OUTPUT_RANGE \
    --trace-file $TRACE_FILE_A \
    --run-log-file $RUN_LOG_A \
    --clean-source-code-path $SOURCE_CODE \
    --commit-id $COMMIT_A \
    --analyze-output-dir $ANALYZE_DIR_A \
    --output-config-file $CONFIG_A

python -m auto_analyze.run_single_trace --config ${CONFIG_A}.json
```

#### Commit B — v0.16.0 release

```bash
COMMIT_B="89a77b10846fd96273cce78d86d2556ea582d26e"
TRACE_FILE_B="/path/to/traces/commit_b/rank0.pt.trace.json.gz"
RUN_LOG_B="/path/to/traces/commit_b/run_log.txt"
ANALYZE_DIR_B="/path/to/results/vllm_v0.16.0/analyze"
CONFIG_B="/path/to/results/vllm_v0.16.0/single_trace_config"

python -m auto_analyze.create_single_trace_config \
    --model $MODEL \
    --gpu-type $GPU_TYPE \
    --batch-size-range $BATCH_RANGE \
    --prefill-size-range $PREFILL_RANGE \
    --output-size-range $OUTPUT_RANGE \
    --trace-file $TRACE_FILE_B \
    --run-log-file $RUN_LOG_B \
    --clean-source-code-path $SOURCE_CODE \
    --commit-id $COMMIT_B \
    --analyze-output-dir $ANALYZE_DIR_B \
    --output-config-file $CONFIG_B

python -m auto_analyze.run_single_trace --config ${CONFIG_B}.json
```

**Notes:**
- Both commits use the **same source code directory** (`$SOURCE_CODE`). The
  analysis automatically creates separate branches for each commit ID.
- The source code directory must be a clean git repo with no uncommitted changes.
- Both runs must use the same model, GPU type, and execution parameters — the
  cross-trace validation will reject mismatched configurations.

### Step 2: Create the Cross-Trace Config

Once both single-trace analyses are complete, create the cross-trace config.
The `--target-trace-id` specifies which trace you want to analyze from — typically
the newer commit (index 0 below):

```bash
CROSS_OUTPUT_DIR="/path/to/results/cross_analyze"
CROSS_CONFIG="/path/to/results/cross_config"

python -m auto_analyze.create_cross_trace_config \
    --trace-result-dir $ANALYZE_DIR_A \
    --trace-result-dir $ANALYZE_DIR_B \
    --target-trace-id 0 \
    --analyze-output-dir $CROSS_OUTPUT_DIR \
    --output-config-file $CROSS_CONFIG
```

The script validates that both result directories contain the required files
(`run_params.txt`, `median_block.txt`, `transformer_block_high_level_ops.txt`).

**Optional — include an improvement plan** with step-by-step coding guides for
recovering any regressions in the target trace:

```bash
python -m auto_analyze.create_cross_trace_config \
    --trace-result-dir $ANALYZE_DIR_A \
    --trace-result-dir $ANALYZE_DIR_B \
    --target-trace-id 0 \
    --analyze-output-dir $CROSS_OUTPUT_DIR \
    --output-config-file $CROSS_CONFIG \
    --make-improvement-plan
```

### Step 3: Run the Cross-Trace Analysis

```bash
python -m auto_analyze.run_cross_trace --config ${CROSS_CONFIG}.json
```

The analysis pipeline executes these steps automatically:

1. **Block matching** — Matches operations between the two median transformer
   blocks one-by-one based on high-level logical purpose, sequence order, and
   source code analysis. Handles differences in fusion, kernel naming, and
   stream layout.

2. **Performance comparison** — Analyzes every matched operation pair to identify
   all performance differences (both positive and negative) from the target
   trace's perspective. For each difference, provides root cause analysis with
   source code references and code snippets.

3. **Improvement plan** *(optional, when `make_improvement_plan` is enabled)* —
   Generates a ranked sequence of improvement proposals for recovering regressions,
   each with step-by-step coding guides showing exactly what to change and where.

### Step 4: Review the Results

The output directory contains:

| File | Description |
|------|-------------|
| `cross_matching_blocks.txt` | Operation-by-operation matching table showing each logical operation with kernel names, durations, and streams side by side |
| `cross_compare_blocks.txt` | Detailed comparison of all performance differences (positive and negative) with root causes, code snippets, and source references |
| `cross_improvement_plan.txt` | *(only when enabled)* Ranked improvement proposals with step-by-step coding guides |
| `run_params_cross.txt` | Cross-trace run parameters for downstream tools |

The comparison report (`cross_compare_blocks.txt`) includes:
- **Executive summary** — overall wall time delta, key findings, and gross
  improvement vs. regression breakdown
- **Detailed differences** — sorted from worst regression to best improvement,
  each with root cause analysis, code snippets, and source references

## Example Results

From the included example comparing vLLM `main` (commit `8f89381f`) against
`v0.16.0` (commit `89a77b10`) on Kimi-K2.5-NVFP4 / B200 / BS=1 decode:

```
Block wall time:  TARGET = 96,992 ns  |  Baseline = 123,232 ns  |  Delta = -26,240 ns (-21.3%)

The TARGET (v0.20.2) is 21.3% faster per MoE decoder layer than the baseline (v0.16.0).
```

**Top improvements identified:**

1. **QKV-A Projection GEMM** — saves 10.8 us (41% of total improvement)
   - Replaced generic cuBLASLt CUTLASS BF16 GEMM with `dsv3_fused_a_gemm`, a
     hand-tuned warp-specialized CUDA kernel adapted from TRT-LLM's DeepSeek V3
     minimum-latency kernels. Uses compile-time shape specialization for
     [M<=16, K=7168, N=2112] and PDL for kernel pipelining.
   - PR #34758 by Robert Shaw, Feb 2026

2. **MoE Gate Router GEMM** — saves 8.0 us (30% of total improvement)
   - Replaced generic cuBLASLt GEMM with `dsv3_router_gemm`, a one-block-per-expert
     kernel with compile-time template specialization for the extremely skinny
     [M<=16, N=384, K=7168] shape. Uses warp-level shuffle reduction instead of
     shared memory.
   - PR #34302 by Robert Shaw, Feb 2026

3. **MoE Routing Dispatch Consolidation** — saves 7.6 us (29% of total improvement)
   - Consolidated 4 separate PyTorch-dispatched kernels (sigmoid, bias correction,
     top-k selection, index clustering) into a single monolithic TRT-LLM kernel
     via FlashInfer. Data stays in shared memory between routing stages, eliminating
     3 kernel launch overheads and global memory round-trips.
   - Enabled by FlashInfer upgrade (0.6.3 to 0.6.8) and vLLM backend selection
     routing to `Fp8MoeBackend.FLASHINFER_TRTLLM` on SM100

**Partially offset by:**
- New buffer-init and dtype-cast ops added for TRT-LLM MoE (+6.0 us)
- cuBLASLt nvjet GEMM regressions from CUDA 13.0 vs 12.9 library change (+1.8 us)

The full example with detailed source code analysis is at
`auto_analyze/examples/cross_commit_cmp_example_kimi/`.

## Directory Structure

A complete cross-commit comparison has this layout:

```
results/
  vllm_latest/
    run_log.txt                     # Run log from trace capture
    trace/                          # Raw trace files
    single_trace_config.json        # Single-trace config
    analyze/                        # Single-trace analysis output
      median_block.txt
      transformer_block_high_level_ops.txt
      single_trace_transformer_block.json
      ...
  vllm_v0.16.0/
    run_log.txt
    trace/
    single_trace_config.json
    analyze/
      ...
  cross_config.json                 # Cross-trace config
  cross_analyze/                    # Cross-trace analysis output
    cross_matching_blocks.txt
    cross_compare_blocks.txt
    cross_improvement_plan.txt      # (if enabled)
    run_params_cross.txt
```
