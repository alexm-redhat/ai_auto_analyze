# AI Auto Performance Analysis

An AI-powered framework for automated performance analysis, comparison, and optimization of LLM inference frameworks (vLLM, SGLang, TensorRT-LLM). Uses Claude as an AI agent to profile inference workloads, analyze NSYS GPU traces, generate improvement plans, and produce code patches that port optimizations between frameworks.

## Overview

Manual analysis of GPU profile traces is labor-intensive and error-prone, particularly when correlating low-level GPU operations with high-level Python/C++/CUDA code across multiple inference frameworks. This tool automates the entire workflow:

1. **Profile** -- Run inference benchmarks inside Docker containers with NSYS GPU profiling
2. **Analyze** -- Use Claude to correlate GPU operations with source code, compare transformer blocks across frameworks, and generate ranked improvement plans
3. **Generate Code** -- Use Claude to port optimizations from the faster framework to the slower one, with iterative review cycles

## Project Structure

```
ai_auto_perf_analysis/
|-- common/                        # Shared utilities
|   |-- utils.py                   #   Tee, setup_logging, safe_clean_dir
|-- auto_profile/                  # Profiling orchestration
|   |-- run_profile.sh             #   Main profiling script (runs inside Docker)
|   |-- run_summary.py             #   Summarize profiling results into analyze configs
|   |-- config.sh                  #   Framework constants, NSYS flags, test params
|   |-- parse_run_config.py        #   Config parser and validator
|   |-- parse_prompts.py           #   Claude prompts for result summarization
|   |-- utils.sh                   #   Shell utility functions
|   |-- vllm/                      #   vLLM Docker runner, bench scripts, mode configs
|   |-- sgl/                       #   SGLang Docker runner, bench scripts, mode configs
|   |-- trt/                       #   TensorRT-LLM Docker runner, bench scripts
|   +-- test_configs/              #   JSON configs (infra, run, docker images, GPUs)
|-- auto_analyze/                  # 4-step analysis pipeline
|   |-- run_analyze.py             #   Step 1: Analyze traces and generate improvement plan
|   |-- run_summary_pdf.py         #   Step 2: Generate PDF report
|   |-- run_combined_trace.py      #   Step 3: Generate combined Chrome trace
|   |-- run_create_jiras.py        #   Step 4: Create JIRA tasks
|   |-- analyze_configs.py         #   AnalyzeConfig, ClaudeConfig, config loading
|   |-- analyze_prompts.py         #   All prompt templates for the 4 steps
|   +-- analyze_config_example.json #  Example analysis config
|-- auto_code_gen/                 # AI-based code generation pipeline
|   |-- run_code_gen.py            #   Generate code patches from improvement plans
|   |-- run_fix_issue.py           #   Fix issues encountered during code gen
|   |-- run_investigate_issue.py   #   Deep-dive investigation of runtime issues
|   |-- run_work_items.py          #   Execute work items (rebase, PR review, etc.)
|   |-- run_summary.py             #   Generate PPTX summary of code gen process
|   |-- code_gen_configs.py        #   CodeGenConfig and hardcoded test parameters
|   +-- code_gen_prompts.py        #   All prompt templates for code generation
|-- claude_utils.py                # Claude Agent SDK wrapper (async prompt execution)
|-- env.sh                         # Shared environment variables
|-- run_all.sh                     # Full pipeline orchestrator
|-- config_deepseek_r1_nvfp4.sh # Example pipeline config
|-- run_profile_core.sh            # Run profiling step
|-- run_profile_summary.sh         # Run profile summarization step
|-- run_analyze_core.sh            # Run analysis step (standalone)
|-- run_analyze_summary_pdf.sh     # Run PDF generation step (standalone)
|-- run_analyze_combined_trace.sh  # Run combined trace step (standalone)
|-- run_analyze_create_jiras.sh    # Run JIRA creation step (standalone)
|-- run_code_gen.sh                # Run code generation
|-- logs/                          # Runtime logs (auto-created)
+-- docs/                          # Documentation
```

## Prerequisites

- Python 3.8+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)
- [Claude Agent Python SDK](https://pypi.org/project/claude-agent-sdk/) (`pip install claude-agent-sdk`)
- Docker (for running inference framework containers)
- NVIDIA Nsight Systems (`nsys`) for GPU profiling
- `colorama` Python package (`pip install colorama`)
- `reportlab` and `pygments` Python packages (for PDF generation)

## Pipeline

### Phase 1: Profiling (`auto_profile/`)

Runs inference benchmarks inside Docker containers with NSYS GPU tracing enabled.

#### Configuration

Two JSON config files are required:

**Infrastructure config** (`auto_profile/test_configs/infra_config.json`):

```json
{
    "docker_images_file": "auto_profile/test_configs/docker_images.json",
    "gpu_configs_file": "auto_profile/test_configs/gpu_configs.json",
    "exec_modes_file": "auto_profile/test_configs/exec_modes.json"
}
```

**Run config** (e.g., `auto_profile/test_configs/run_deepseek_r1_nvfp4.json`):

```json
{
    "profiles": [
        {
            "model": "nvidia/DeepSeek-R1-NVFP4",
            "gpu_ids": "gpu_ids_4",
            "exec_mode": "pure_decode",
            "modes": ["vllm_kimi_k25_nvfp4_b200", "trt_moe_trtllm_b200"]
        }
    ],
    "concurrencies": [1],
    "frameworks": ["vllm", "sgl"],
    "enable_traces": ["vllm", "sgl"]
}
```

#### Running

```bash
./run_profile_core.sh
```

This calls `auto_profile/run_profile.sh` which launches Docker containers for each framework, runs benchmarks, and collects NSYS traces.

#### Summarization

After profiling, summarize results into per-test-case analyze configs:

```bash
./run_profile_summary.sh
```

This uses Claude to parse profiling results and generate `analyze_*.json` config files in the output directory.

### Phase 2: Analysis (`auto_analyze/`)

A 4-step pipeline that uses Claude to analyze NSYS GPU traces and generate improvement plans.

#### Configuration

Analysis requires a JSON config file. See `auto_analyze/analyze_config_example.json`:

```json
{
    "model": "nvidia/Kimi-K2.5-NVFP4",
    "precision": "NVFP4",
    "gpu_type": "B200",
    "isl": 4,
    "osl": 1024,
    "batch_size": 1,
    "claude_output_dir": "/path/to/output",
    "target_framework": "vllm",
    "frameworks": [
        {
            "name": "vllm",
            "source_code": "/path/to/vllm",
            "test_dir": "/path/to/test_dir/vllm",
            "gpu_ops_filter": ""
        },
        {
            "name": "sglang",
            "source_code": "/path/to/sglang",
            "test_dir": "/path/to/test_dir/sgl",
            "gpu_ops_filter": ""
        }
    ]
}
```

| Field | Description |
|-------|-------------|
| `model` | Model identifier (e.g., `nvidia/DeepSeek-R1-NVFP4`) |
| `precision` | Quantization precision (`FP8`, `NVFP4`, etc.) |
| `gpu_type` | GPU hardware (`B200`, `H200`, etc.) |
| `isl` | Input sequence length |
| `osl` | Output sequence length |
| `batch_size` | Batch size / concurrency |
| `claude_output_dir` | Directory where Claude writes analysis results |
| `target_framework` | The framework to optimize (e.g., `vllm`) |
| `frameworks[].name` | Framework identifier |
| `frameworks[].source_code` | Path to framework source code repository |
| `frameworks[].test_dir` | Path to profiling results (contains `trace-*.sqlite`, `bench-*.json`, `run-log-*.txt`) |
| `frameworks[].gpu_ops_filter` | Glob pattern to exclude GPU operations from analysis |

#### Step 1: Analyze (`run_analyze_core.sh <config.json>`)

For each framework, Claude executes four sequential prompts:

1. **Transformer Block High-Level Ops** -- Inspects framework source code to extract the sequence of high-level operations in a transformer block
2. **GPU Ops Extraction** -- Reads the NSYS SQLite trace to extract GPU operation sequences (GPU 0 only, up to 2000 ops)
3. **GPU Ops to Transformer Blocks** -- Maps low-level GPU operations to transformer blocks using the high-level operation sequence
4. **Median Block Selection** -- Selects the median wall-time transformer block as the representative sample

Then two cross-framework prompts run:

5. **Compare Median Blocks** -- Operation-by-operation performance comparison across frameworks
6. **Generate Plan** -- Ranked improvement plan for the target framework

**Output files** (written to `claude_output_dir`):

| File | Description |
|------|-------------|
| `{fw}_transformer_block_high_level_ops.txt` | High-level operation sequence per framework |
| `{fw}_gpu_ops.txt` | Raw GPU operation sequence from trace |
| `{fw}_gpu_ops_to_blocks.txt` | GPU ops mapped to transformer blocks |
| `{fw}_median_block.txt` | Representative median transformer block |
| `{fw1}_{fw2}__perf_compare_blocks.txt` | Operation-by-operation comparison |
| `{fw1}_{fw2}__plan.txt` | Ranked improvement plan for target framework |

#### Step 2: Summary PDF (`run_analyze_summary_pdf.sh <config.json>`)

Generates a professional PDF report (`cmp_and_plan_summary.pdf`) containing:

- Executive summary with impact badges (HIGH/MEDIUM/LOW)
- Detailed improvement plans per issue with syntax-highlighted code snippets
- Performance impact tables with color-coded deltas
- Appendix with complete operation-by-operation comparison
- Source code references grouped by framework

#### Step 3: Combined Trace (`run_analyze_combined_trace.sh <config.json>`)

Generates a Chrome trace JSON file (`trace_combined_transformer_blocks.json`) that can be viewed in [Perfetto](https://ui.perfetto.dev/):

- Side-by-side timeline of median transformer blocks from each framework
- CUDA streams shown as separate lanes with overlap handling
- Each operation annotated with high-level context, source code references, and improvement plan details

#### Step 4: Create JIRA Tasks (`run_analyze_create_jiras.sh <config.json>`)

Creates JIRA tasks from the improvement plan:

- One master task under the configured epic
- Sub-tasks for each individual improvement, with step-by-step implementation guides and code snippets

#### Running All Steps

To run the full analysis pipeline across all test cases:

```bash
./run_all.sh config_deepseek_r1_nvfp4.sh
```

The pipeline config (`config_deepseek_r1_nvfp4.sh`) defines:

```bash
test_name="deepseek_r1_nvfp4"
output_dir="./auto_analyze/results/results_analyze_${test_name}"
```

`run_all.sh` iterates over all `analyze_*.json` files in the output directory and runs all 4 steps for each.

### Phase 3: Code Generation (`auto_code_gen/`)

Uses Claude to automatically port optimizations from the faster framework to the slower one.

The pipeline consists of iterative generate-review-fix cycles:

1. **Code Trace** -- Analyze call chains for decode, prefill, and mixed execution modes in both frameworks
2. **Code Port Plan** -- Generate a multi-step porting plan (4 review iterations)
3. **Test Plan** -- Generate unit, integration, and performance tests
4. **Code Generation** -- Implement the code patch following the porting plan
5. **Code Review** -- Review and fix generated code (3 review iterations)
6. **Issue Investigation** -- Deep-dive into runtime issues with root cause analysis and fixes

```bash
./run_code_gen.sh
```

## Claude Agent Integration

All AI-driven steps use the Claude Agent SDK via `claude_utils.py`. Key configuration:

| Parameter | Value |
|-----------|-------|
| Model | `claude-opus-4-6[1m]` (1M context) |
| Allowed tools | `Read`, `Write`, `Bash` |
| Permission mode | `acceptEdits` |
| Max thinking tokens | 1,048,576 |
| Thinking mode | Adaptive |
| Effort | Max |
| Max output tokens | 120,000 (set via `env.sh`) |

Each step sends one or more prompts to Claude sequentially. Claude reads source code, analyzes traces, and writes results to the configured output directory.

## Logging

All pipeline steps log to `logs/run_{step_name}.log`. Each run overwrites the previous log. Log output is also printed to stdout via the `Tee` mechanism.

| Step | Log file |
|------|----------|
| Analyze | `logs/run_analyze.log` |
| Summary PDF | `logs/run_summary_pdf.log` |
| Combined Trace | `logs/run_combined_trace.log` |
| Create JIRAs | `logs/run_create_jiras.log` |
| Profile Summary | `logs/run_parse.log` |

## Troubleshooting

- Verify all paths in the analyze config JSON are correct and accessible
- Ensure NSYS trace files are in SQLite format (`trace-*.sqlite`)
- Ensure framework source code versions match those used to generate traces
- Check `logs/` for detailed execution logs if a step fails
- Review Claude's permissions if file access issues occur
