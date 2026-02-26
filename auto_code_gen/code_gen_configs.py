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
    plan_file: str
    plan_step: int
    faster_framework: str
    slower_framework: str
    faster_framework_test_dir: str
    slower_framework_test_dir: str
    faster_transformer_block_high_level_ops_file: str
    slower_transformer_block_high_level_ops_file: str
    faster_median_transformer_block_file: str
    slower_median_transformer_block_file: str



MODEL = "deepseek-ai/DeepSeek-V3.2"
PRECISION = "FP8"
GPU_TYPE = "H200"
BATCH_SIZE = 1
ISL = 4
OSL = 1024


VLLM = "vllm"
SGLANG = "sglang"
TRT = "trt"

framework_names = [VLLM, SGLANG, TRT]

VLLM_CODE = "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_code_gen/prs/vllm"
SGLANG_CODE = "/home/alexm-redhat/code/sglang"
TRT_CODE = "/home/alexm-redhat/code/TensorRT-LLM"

framework_source_codes = [VLLM_CODE, SGLANG_CODE, TRT_CODE]

PLAN_FILE = "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_analyze/results/vllm_sglang_trt_cmp_and_plan_summary.pdf"
PLAN_STEP = 1
FASTER_FRAMEWORK = SGLANG
SLOWER_FRAMEWORK = VLLM

FASTER_FRAMEWORK_TEST_DIR = "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_profile/results/parse/test_results/deepseek-ai__DeepSeek-V3.2-tp_8-isl_4-osl_1024-b_1-mode_none/sgl"
SLOWER_FRAMEWORK_TEST_DIR = "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_profile/results/parse/test_results/deepseek-ai__DeepSeek-V3.2-tp_8-isl_4-osl_1024-b_1-mode_none/vllm"

FASTER_TRANSFORMER_BLOCK_HIGH_LEVEL_OPS_FILE = "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_analyze/results/sglang_transformer_block_high_level_ops.txt"
SLOWER_TRANSFORMER_BLOCK_HIGH_LEVEL_OPS_FILE = "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_analyze/results/vllm_transformer_block_high_level_ops.txt"
FASTER_MEDIAN_TRANSFORMER_BLOCK_FILE = "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_analyze/results/sglang_median_block.txt.txt"
SLOWER_MEDIAN_TRANSFORMER_BLOCK_FILE = "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_analyze/results/vllm_median_block.txt.txt"

# TODO: Now we need export CLAUDE_CODE_ALLOWED_PATHS=/home/alexm-redhat/code, remove this requirement
claude_config = ClaudeConfig(
    # model="claude-opus-4-6",
    model="claude-opus-4-6[1m]",
    # model="claude-opus-4-5",
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
    plan_file=PLAN_FILE,
    plan_step=PLAN_STEP,
    faster_framework=FASTER_FRAMEWORK,
    slower_framework=SLOWER_FRAMEWORK,
    faster_framework_test_dir=FASTER_FRAMEWORK_TEST_DIR,
    slower_framework_test_dir=SLOWER_FRAMEWORK_TEST_DIR,
    faster_transformer_block_high_level_ops_file=FASTER_TRANSFORMER_BLOCK_HIGH_LEVEL_OPS_FILE,
    slower_transformer_block_high_level_ops_file=SLOWER_TRANSFORMER_BLOCK_HIGH_LEVEL_OPS_FILE,
    faster_median_transformer_block_file=FASTER_MEDIAN_TRANSFORMER_BLOCK_FILE,
    slower_median_transformer_block_file=SLOWER_MEDIAN_TRANSFORMER_BLOCK_FILE,
)
