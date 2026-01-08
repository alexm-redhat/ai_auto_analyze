# Automatic PyTorch Profile Trace Analysis

An AI-powered tool for automated analysis and comparison of PyTorch profile traces from Large Language Model (LLM) inference frameworks such as **vLLM** and **SGLang**.

## Overview

Manual analysis of PyTorch profile traces is labor-intensive and error-prone, particularly when correlating low-level GPU operations with high-level Python/C++ code. This tool automates the correlation process with high precision, enabling efficient performance analysis and cross-framework comparisons.

## Features

### Transformer Block Extraction & Analysis

- **Automatic Block Identification**: Extracts the most representative transformer block (the most frequently executed) from the trace
- **Operation Sequencing**: Provides the complete sequence of low-level GPU operations within the identified block
- **Source Code Correlation**: Maps low-level GPU kernels to their corresponding high-level Python/C++ implementations by inspecting the framework's source code

### Cross-Framework Comparison

- **Side-by-Side Analysis**: Compares two traces from different frameworks running the same model
- **Operation-Level Mapping**: Provides operation-by-operation comparison at the transformer block level
- **Source Code Attribution**: Correlates each operation to specific source code lines in both frameworks

## Getting Started

### Prerequisites

Ensure the following are installed:

- Python 3.8 or higher
- Access to Claude Agent and the Claude Agent Python SDK
- PyTorch profiling traces from your target framework(s)

### Installation

#### 1. Install Claude Agent

See the [Official Red Hat Claude Install](https://source.redhat.com/departments/it/itx/document_management_and_collaboration/claude_code) guide. For a quick summary, refer to [Quick Claude Install Steps](https://docs.google.com/document/d/1pSZSeM1w_7uvxpPpHAABV1pu0KH5w-YxSUICVrCZ_48/edit?usp=sharing) (VSCode steps can be skipped).

#### 2. Install Claude Agent Python SDK

```bash
pip install claude-agent-sdk
```

## Configuration

### Preparing Trace Files

#### Generate Profile Traces

Use your framework's profiling tools to generate PyTorch trace files.

**Example**: See the [vLLM trace generation guide](https://github.com/alexm-redhat/auto_profile/blob/main/examples/vLLM_generate_trace.md) for instructions on profiling `deepseek-ai/DeepSeek-R1-0528` with 8-GPU tensor parallelism.

#### Set Up Working Directory

For each trace you want to analyze:

1. Copy the trace file to Claude's current working directory (`cwd`)
2. Clone the framework's source code repository into `cwd`

### Configuring `analyze_configs.py`

The analysis uses two main configuration classes:

#### `ClaudeConfig` Class

Defines Claude's general execution environment:

| Parameter       | Description                                          | Default           |
|-----------------|------------------------------------------------------|-------------------|
| `model`         | The Anthropic LLM model to use                       | `claude-opus-4-5` |
| `allowed_tools` | Set of tools Claude is permitted to use              | (see config file) |
| `perm_mode`     | Permissions mode for Claude operations               | (see config file) |
| `cwd`           | Current working directory containing traces and code | (user-specified)  |

#### `AnalyzeConfig` Class

Create one `AnalyzeConfig` instance per trace file. Add all instances to the `analyze_configs` list.

| Parameter              | Description                                        | Example                                                |
|------------------------|----------------------------------------------------|--------------------------------------------------------|
| `model`                | Model identifier used in profiling                 | `deepseek-ai/DeepSeek-R1-0528`                         |
| `gpu_type`             | GPU hardware used during profiling                 | `B200`                                                 |
| `framework_name`       | Framework being analyzed                           | `vLLM`                                                 |
| `framework_code`       | Path to framework source code (relative to `cwd`)  | `{cwd}/vllm`                                           |
| `framework_model_code` | Path to model implementation file                  | `{cwd}/vllm/vllm/model_executor/models/deepseek_v2.py` |
| `trace_file`           | Trace filename (relative to `cwd`)                 | `vllm_trace.json.gz`                                   |
| `gpu_ops_filter`       | Glob pattern to filter/skip specific GPU operations| `*execute_new*` (for vLLM)                             |

**Note**: The `gpu_ops_filter` parameter accepts glob patterns to exclude specific operations from analysis. For example, vLLM's `*execute_new*` operations are typically skipped.

## Usage

### Running the Analysis

Execute the analysis script from the project directory:

```bash
python run.py
```

### Output

The full execution log (Claude prompts and responses) is written to `run_log.txt`.

Analysis results are saved to Claude's `cwd` directory. For vLLM, the following files are generated:

| File | Description |
|------|-------------|
| `vllm_transformer_block_high_level_ops.txt` | Source code breakdown of operations in a transformer block |
| `vllm_gpu_ops.txt` | GPU operation sequences from the trace |
| `vllm_gpu_ops_to_blocks.txt` | GPU operations correlated to transformer blocks |
| `vllm_median_block.txt` | The representative transformer block |

For cross-framework comparisons (e.g., vLLM vs. SGLang):

| File | Description |
|------|-------------|
| `vllm_vs_sglang_perf_compare_blocks.txt` | Operation-by-operation comparison of representative transformer blocks |

## Troubleshooting

- Verify all paths in `analyze_configs.py` are correct and accessible
- Confirm trace files are in the correct format (PyTorch profiler JSON)
- Ensure framework source code versions match those used to generate traces
- Review Claude's permissions if file access issues occur