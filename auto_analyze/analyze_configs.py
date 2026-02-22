from dataclasses import dataclass
from claude_utils import ClaudeConfig

@dataclass
class AnalyzeConfig:
    model: str
    precision: str
    gpu_type: str
    framework_name: str
    framework_source_code: str
    test_dir: str
    gpu_ops_filter: str = ""

MODEL = "deepseek-ai/DeepSeek-V3.2"
PRECISION = "FP8"
GPU_TYPE = "H200"

VLLM = "vllm"
SGLANG = "sglang"
TRT = "trt"

VLLM_TEST_DIR = "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_profile/results/parse/test_results/deepseek-ai__DeepSeek-V3.2-tp_8-isl_4-osl_1024-b_1-mode_none/vllm"
SGLANG_TEST_DIR = "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_profile/results/parse/test_results/deepseek-ai__DeepSeek-V3.2-tp_8-isl_4-osl_1024-b_1-mode_none/sgl"
TRT_TEST_DIR = "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_profile/results/parse/test_results/deepseek-ai__DeepSeek-V3.2-tp_8-isl_4-osl_1024-b_1-mode_none/trt"

claude_config = ClaudeConfig(
    model="claude-opus-4-6",
    # model="claude-opus-4-5",
    allowed_tools=["Read", "Write", "Bash"],
    perm_mode="acceptEdits",  # "bypassPermissions",
    cwd="/home/alexm-redhat/code/ai_auto_perf_analysis/auto_analyze/results",
)

analyze_configs = [
    AnalyzeConfig(
        model=MODEL,
        precision=PRECISION,
        gpu_type=GPU_TYPE,
        framework_name=VLLM,
        framework_source_code="/home/alexm-redhat/code/vllm",
        test_dir=VLLM_TEST_DIR,
    ),
    AnalyzeConfig(
        model=MODEL,
        precision=PRECISION,
        gpu_type=GPU_TYPE,
        framework_name=SGLANG,
        framework_source_code="/home/alexm-redhat/code/sglang",
        test_dir=SGLANG_TEST_DIR,
    ),
    AnalyzeConfig(
        model=MODEL,
        precision=PRECISION,
        gpu_type=GPU_TYPE,
        framework_name=TRT,
        framework_source_code="/home/alexm-redhat/code/TensorRT-LLM",
        test_dir=TRT_TEST_DIR,
    ),
]
