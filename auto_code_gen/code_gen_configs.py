from dataclasses import dataclass
from claude_utils import ClaudeConfig


@dataclass
class CodeGenConfig:
    model: str
    precision: str
    gpu_type: str
    batch_size: int
    isl: int
    osl: int
    framework_names: list[str]
    framework_source_codes: list[str]
    framework_test_dirs: list[str]
    transformer_block_high_level_ops_files: list[str]
    median_transformer_block_files: list[str]
    plan_file: str
    plan_step: int
    faster_framework: str
    slower_framework: str


MODEL = "deepseek-ai/DeepSeek-V3.2"
PRECISION = "FP8"
GPU_TYPE = "H200"
BATCH_SIZE = 1
ISL = 4
OSL = 1024


VLLM = "vllm"
SGLANG = "sglang"

framework_names = [VLLM, SGLANG]

VLLM_CODE = "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_code_gen/prs/vllm"
SGLANG_CODE = "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_code_gen/prs/sglang"

framework_source_codes = [VLLM_CODE, SGLANG_CODE]

PLAN_FILE = "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_analyze/results/vllm_sglang_trt_cmp_and_plan_summary.pdf"
PLAN_STEP = 1
FASTER_FRAMEWORK = SGLANG
SLOWER_FRAMEWORK = VLLM

FRAMEWORK_TEST_DIRS = [
    "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_profile/results/parse/test_results/deepseek-ai__DeepSeek-V3.2-tp_8-isl_4-osl_1024-b_1-mode_none/vllm",
    "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_profile/results/parse/test_results/deepseek-ai__DeepSeek-V3.2-tp_8-isl_4-osl_1024-b_1-mode_none/sgl",
]

TRANSFORMER_BLOCK_HIGH_LEVEL_OPS_FILES = [
    "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_analyze/results/vllm_transformer_block_high_level_ops.txt",
    "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_analyze/results/sglang_transformer_block_high_level_ops.txt",
]

MEDIAN_TRANSFORMER_BLOCK_FILES = [
    "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_analyze/results/vllm_median_block.txt",
    "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_analyze/results/sglang_median_block.txt",
]

# TODO: Now we need export CLAUDE_CODE_ALLOWED_PATHS=/home/alexm-redhat/code, remove this requirement
claude_config = ClaudeConfig(
    model="claude-opus-4-6[1m]",
    allowed_tools=["Read", "Write", "Bash"],
    perm_mode="acceptEdits",  # "bypassPermissions",
    cwd="/home/alexm-redhat/code/ai_auto_perf_analysis/auto_code_gen/prs",
)

code_gen_config = CodeGenConfig(
    model=MODEL,
    precision=PRECISION,
    gpu_type=GPU_TYPE,
    batch_size=BATCH_SIZE,
    isl=ISL,
    osl=OSL,
    framework_names=framework_names,
    framework_source_codes=framework_source_codes,
    framework_test_dirs=FRAMEWORK_TEST_DIRS,
    transformer_block_high_level_ops_files=TRANSFORMER_BLOCK_HIGH_LEVEL_OPS_FILES,
    median_transformer_block_files=MEDIAN_TRANSFORMER_BLOCK_FILES,
    plan_file=PLAN_FILE,
    plan_step=PLAN_STEP,
    faster_framework=FASTER_FRAMEWORK,
    slower_framework=SLOWER_FRAMEWORK,
)
