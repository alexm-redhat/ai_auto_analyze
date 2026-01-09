from dataclasses import dataclass
from typing import ClassVar

from configs import AnalyzeConfig


def system_prompt():
    return "You are a performance programming expert in GPU, CUDA, LLMs, pytorch, profiling and tracing."


@dataclass
class TransformerBlockHighLevelPrompt:
    model: str
    precision: str
    gpu_type: str
    framework_name: str
    framework_code: str
    framework_model_code: str
    output_file: str
    prompt_template: ClassVar[str] = """
<model> = {model}
<precision> = {precision}
<gpu_type> = {gpu_type}
<framework_name> = {framework_name}
<framework_code> = {framework_code}
<framework_model_code> = {framework_model_code}
<output_file> = {output_file}

You specialize in analyzing the inference performance of the model <model> in precision <precision> running via the <framework_name> framework on <gpu_type> GPU.

The code of <framework_name> framework is located here: <framework_code>
The code that implements the model <model> inside <framework_name> is located here: <framework_model_code>

The model <model> is a sequence of transformer blocks.
The goal of this task is to generate the sequence of high-level operations of a single transformer block.

Do the following plan and think hard:
- Inspect the code in <framework_model_code>, and any other relevant code files there.
- Find the sequence of high-level operations of a single transformer block of <model> based on the previous code inspection.
- Each high-level operation may have multiple execution modes. Detect these execution modes for each high-level operation.
- Based on different execution modes of each high-level operation, detect the types of transformer blocks that can run. 
- For each transformer block type, summarize in a table the sequence of high-level operations of this transformer block, so that each operation is shown in a row and each row has: high-level operation name, short source code reference, and execution mode used (if exists).
- Dump the resulting tables (that represent all possible transformer blocks) to <output_file> inside the current working directory.
"""

    def prompt(self):
        return self.prompt_template.format(
            model=self.model,
            precision=self.precision,
            gpu_type=self.gpu_type,
            framework_name=self.framework_name,
            framework_code=self.framework_code,
            framework_model_code=self.framework_model_code,
            output_file=self.output_file,
        )


TRANSFORMER_BLOCK_HIGH_LEVEL_OPS_FILE = "transformer_block_high_level_ops.txt"


@dataclass
class GpuOpsPrompt:
    model: str
    gpu_type: str
    framework_name: str
    trace_file: str
    output_file: str
    output_max_gpu_ops: int
    output_filter_ops: str = ""
    prompt_template: ClassVar[str] = """
<model> = {model}
<gpu_type> = {gpu_type}
<framework_name> = {framework_name}

<trace_file> = {trace_file}

<output_file> = {output_file}
<output_filter_ops> = {output_filter_ops}
<output_max_gpu_ops> = {output_max_gpu_ops}

- The file <trace_file> is a pytorch trace file of <framework_name> running <model> on <gpu_type> GPU.

Do the following plan and think hard:
- Understand the basic structure of the pytorch trace file <trace_file>.
- Understand how GPU streams are represented.
- Understand how GPU operations are represented. Ignore any CPU operations.
- Find the sequence of all GPU streams in this trace file.
- Find the sequence of all GPU operations, across all GPU streams, including overlapping ones. Keep the maximum number of GPU operations found to <output_max_gpu_ops>. 
- If <output_filter_ops> is not empty string, then filter-out GPU operations that match the pattern <output_filter_ops>.
- Dump the found sequence of GPU operations to file <output_file>, sorted by their start time, where each row in the file will have: a short GPU operation name (max 50 chars), start and end times, duration, source GPU stream number, and full original GPU operation name.
"""

    def prompt(self):
        return self.prompt_template.format(
            model=self.model,
            gpu_type=self.gpu_type,
            framework_name=self.framework_name,
            trace_file=self.trace_file,
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
    transformer_block_1: str
    framework_name_1: str
    framework_code_1: str
    framework_model_code_1: str
    transformer_block_2: str
    framework_name_2: str
    framework_code_2: str
    framework_model_code_2: str
    output_file: str
    prompt_template: ClassVar[str] = """
<transformer_block_1> = {transformer_block_1}
<transformer_block_2> = {transformer_block_2}

<framework_name_1> = {framework_name_1} 
<framework_code_1> = {framework_code_1}
<framework_model_code_1> = {framework_model_code_1}

<framework_name_2> = {framework_name_2} 
<framework_code_2> = {framework_code_2}
<framework_model_code_2> = {framework_model_code_2}

<output_file> = {output_file}

- The source code of <framework_name_1> is here: <framework_code_1>
- The source code of <framework_name_2> is here: <framework_code_2>
- The source code file of <framework_name_1> that implements model <model> is here: <framework_model_code_1> 
- The source code file of <framework_name_2> that implements model <model> is here: <framework_model_code_2>

- The file <transformer_block_1> has a median transformer block of <model> running via <framework_name_1>
- The file <transformer_block_2> has a median transformer block of <model> running via <framework_name_2>

Do the following and think hard (and ultra hard): 
- Match the sequence of operations of the median transformer block from <transformer_block_1> with the median transformer block from <transformer_block_2> based on the high-level implementation details of how a transformer block is implemented in both frameworks. Ensure to take into account the separate GPU streams and their start and end synchronization points.
- Analyze in-depth and compare in-detail the performance of the two matched blocks from before to find the performance differences.
- Summarize in a single table found performance differences. For each difference, provide:
    - a short description of the difference.
    - a short source code reference for both frameworks.
    - a short description how each framework can improve vs the other.
- Ensure that the summary table covers the full transformer block GPU operation sequences with their high-level correlations, while properly correlating separate GPU streams.
- Dump results to <output_file> in the current working directory, including the summary table and other relevant information.

"""

    def prompt(self):
        return self.prompt_template.format(
            model=self.model,
            gpu_type=self.gpu_type,
            transformer_block_1=self.transformer_block_1,
            framework_name_1=self.framework_name_1,
            framework_code_1=self.framework_code_1,
            framework_model_code_1=self.framework_model_code_1,
            transformer_block_2=self.transformer_block_2,
            framework_name_2=self.framework_name_2,
            framework_code_2=self.framework_code_2,
            framework_model_code_2=self.framework_model_code_2,
            output_file=self.output_file,
        )


PERF_COMPARE_BLOCK_FILE = "perf_compare_blocks.txt"

@dataclass
class PlanPrompt:
    model: str
    gpu_type: str
    transformer_block_1: str
    framework_name_1: str
    framework_code_1: str
    framework_model_code_1: str
    transformer_block_2: str
    framework_name_2: str
    framework_code_2: str
    framework_model_code_2: str
    comparison_file: str
    output_file: str
    prompt_template: ClassVar[str] = """
<transformer_block_1> = {transformer_block_1}
<transformer_block_2> = {transformer_block_2}

<framework_name_1> = {framework_name_1} 
<framework_code_1> = {framework_code_1}
<framework_model_code_1> = {framework_model_code_1}

<framework_name_2> = {framework_name_2} 
<framework_code_2> = {framework_code_2}
<framework_model_code_2> = {framework_model_code_2}

<comparison_file> = {comparison_file}

<output_file> = {output_file}

- The source code of <framework_name_1> is here: <framework_code_1>
- The source code of <framework_name_2> is here: <framework_code_2>
- The source code file of <framework_name_1> that implements model <model> is here: <framework_model_code_1> 
- The source code file of <framework_name_2> that implements model <model> is here: <framework_model_code_2>

- The file <transformer_block_1> has a median transformer block of <model> running via <framework_name_1>
- The file <transformer_block_2> has a median transformer block of <model> running via <framework_name_2>
- The file <comparison_file> has an operation by operation comparison of <transformer_block_1> vs <transformer_block_2>

Do the following and think hard (and ultra hard):
- For each performance issue in <comparison_file> where <framework_name_1> performs worse, generate an improvement plan as follows:
    - Fetch related source code files from <framework_code_1> and <framework_code_1> to analyze the performance issue in detail.
    - Think how the performance issue can be fixed in <framework_code_1> and plan it in detail. Review your plan 3 times to improve it.
    - Provide a high-level step-by-step summary of the previous plan to fix the performance issue for <framework_name_1>. Ensure the summary is clear and detailed.
- Order the resulting sequence of plans by priority and their impact
- Dump the resulting sequence of plans to <output_file>

"""

    def prompt(self):
        return self.prompt_template.format(
            model=self.model,
            gpu_type=self.gpu_type,
            transformer_block_1=self.transformer_block_1,
            framework_name_1=self.framework_name_1,
            framework_code_1=self.framework_code_1,
            framework_model_code_1=self.framework_model_code_1,
            transformer_block_2=self.transformer_block_2,
            framework_name_2=self.framework_name_2,
            framework_code_2=self.framework_code_2,
            framework_model_code_2=self.framework_model_code_2,
            comparison_file=self.comparison_file,
            output_file=self.output_file,
        )

PLAN_FILE = "plan.txt"

def gen_analyze_prompts(config: AnalyzeConfig):
    transformer_block_high_level_prompt = TransformerBlockHighLevelPrompt(
        model=config.model,
        precision=config.precision,
        gpu_type=config.gpu_type,
        framework_name=config.framework_name,
        framework_code=config.framework_code,
        framework_model_code=config.framework_model_code,
        output_file="{}_{}".format(
            config.framework_name, TRANSFORMER_BLOCK_HIGH_LEVEL_OPS_FILE
        ),
    )

    gpu_ops_prompt = GpuOpsPrompt(
        model=config.model,
        gpu_type=config.gpu_type,
        framework_name=config.framework_name,
        trace_file=config.trace_file,
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
    )


def gen_perf_compare_prompt(configs: list[AnalyzeConfig], block_files: list[str]):
    assert len(configs) == 2
    assert len(block_files) == 2

    assert configs[0].model == configs[1].model
    assert configs[0].gpu_type == configs[1].gpu_type

    perf_cmp_prompt = CompareMedianTransformerBlocksPrompt(
        model=configs[0].model,
        gpu_type=configs[0].gpu_type,
        transformer_block_1=block_files[0],
        framework_name_1=configs[0].framework_name,
        framework_code_1=configs[0].framework_code,
        framework_model_code_1=configs[0].framework_model_code,
        transformer_block_2=block_files[1],
        framework_name_2=configs[1].framework_name,
        framework_code_2=configs[1].framework_code,
        framework_model_code_2=configs[1].framework_model_code,
        output_file="{}_vs_{}_{}".format(
            configs[0].framework_name,
            configs[1].framework_name,
            PERF_COMPARE_BLOCK_FILE,
        ),
    )

    return perf_cmp_prompt.prompt(), perf_cmp_prompt.output_file


def gen_plan_prompt(configs: list[AnalyzeConfig], block_files: list[str], perf_compare_file: str):
    assert len(configs) == 2
    assert len(block_files) == 2

    assert configs[0].model == configs[1].model
    assert configs[0].gpu_type == configs[1].gpu_type

    plan_prompt = PlanPrompt(
        model=configs[0].model,
        gpu_type=configs[0].gpu_type,
        transformer_block_1=block_files[0],
        framework_name_1=configs[0].framework_name,
        framework_code_1=configs[0].framework_code,
        framework_model_code_1=configs[0].framework_model_code,
        transformer_block_2=block_files[1],
        framework_name_2=configs[1].framework_name,
        framework_code_2=configs[1].framework_code,
        framework_model_code_2=configs[1].framework_model_code,
        comparison_file=perf_compare_file,
        output_file="{}_{}".format(
            configs[0].framework_name,
            PLAN_FILE,
        ),
    )

    return plan_prompt.prompt()
