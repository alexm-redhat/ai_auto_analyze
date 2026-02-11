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

You specialize in analyzing the inference performance of the model <model> in precision <precision> running via the <framework_name> framework on <gpu_type> GPU.

<test_id>=[model]-tp_$[num_gpus]-isl_[input_len]-osl_[output_len]
<test_id_with_batch>=<test_id>-b_[concurrency]
<full_test_id>=[framework]-<test_id_with_batch>

The <test_dir> test/profile directory has the following format (that encodes test parameters): test-<full_test_id>, and it includes the following files:
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
- Dump the resulting tables (that represent all possible transformer blocks) to <output_file> inside the current working directory.
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
MAX_GPU_OPS = 1000


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
- Summarize in a single table all of the found performance differences. For each difference, provide:
    - a short description of the difference.
    - a short source code reference for all frameworks.
    - a short description how each framework can improve vs the other.
- Ensure that the summary table covers the full transformer block GPU operation sequences with their high-level correlations, while properly correlating separate GPU streams.
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


@dataclass
class PlanPrompt:
    model: str
    gpu_type: str
    transformer_blocks: list[str]
    framework_names: list[str]
    framework_source_codes: list[str]
    comparison_file: str
    output_file: str
    prompt_template: ClassVar[str] = """
<transformer_blocks> = {transformer_blocks}

<framework_names> = {framework_names} 
<framework_source_codes> = {framework_source_codes}

<comparison_file> = {comparison_file}

<output_file> = {output_file}

- <framework_names> is a list of framework names
- <framework_source_codes> is a list of framework source codes that match respectively to <framework_names>
- <transformer_blocks> is a list of median transformer blocks that match respectively to <framework_names>

- The file <comparison_file> has an operation by operation comparison of <transformer_blocks>

Do the following and think hard:
- For each performance issue in <comparison_file>, generate an improvement plan as follows:
    - Fetch related source code files from <framework_source_codes> to analyze the performance issue in detail, in order to understand exactly why some frameworks are slower and why some frameworks are faster.
    - Analyze how the performance issue can be fixed in the slower frameworks and plan it in detail for the slower frameworks. Ensure the step-by-step plan is clear, concise and detailed enough to execute on. 
    - Provide a high-level step-by-step summary of the previous plan to fix the performance issue for the slower frameworks. Ensure the summary is clear and detailed.
- Order the resulting sequence of plans by priority and their impact
- Dump the resulting sequence of plans to <output_file>

"""
    def prompt(self):
        return self.prompt_template.format(
            model=self.model,
            gpu_type=self.gpu_type,
            transformer_blocks=self.transformer_blocks,
            framework_names=self.framework_names,
            framework_source_codes=self.framework_source_codes,
            comparison_file=self.comparison_file,
            output_file=self.output_file,
        )


PLAN_FILE = "plan.txt"

@dataclass
class SummaryPDFPrompt:
    cmp_file: str
    plan_file: str
    output_file: str
    prompt_template: ClassVar[str] = """
<cmp_file> = {cmp_file}
<plan_file> = {plan_file} 
<output_file> = {output_file}

- The file <cmp_file> provides a performance comparison of a transformer block between the frameworks
- The file <plan_file> provides an improvement plan for each performance issue for each framework

The goal of this task is to generate a summary PDF file that has the comparison and planning information. Think hard for this task.

Generate a summary PDF as follows:
- It has information from <cmp_file> and <plan_file
- The PDF is technical and nicely formatted
- PDF preseves good alignment of tables
- PDF uses PDF-style formatting instead of TXT formatting
- Focus only on issues where vllm is slower than the other frameworks

Dump the resulting PDF to <output_file>.

"""
    def prompt(self):
        return self.prompt_template.format(
            cmp_file=self.cmp_file,
            plan_file=self.plan_file,
            output_file=self.output_file,
        )


PLAN_FILE = "plan.txt"


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


def gen_plan_prompt(
    configs: list[AnalyzeConfig], block_files: list[str], perf_compare_file: str
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
        output_file="{}__{}".format(
            "_".join(framework_names),
            PLAN_FILE,
        ),
    )

    return plan_prompt.prompt()
