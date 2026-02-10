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

MODEL = "nvidia/DeepSeek-R1-NVFP4"
PRECISION = "NVFP4"
GPU_TYPE = "B200"

VLLM = "vllm"
SGLANG = "sglang"
TRT = "trt"

VLLM_TEST_DIR = "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_profile/results/vllm/nvidia__DeepSeek-R1-NVFP4/test-vllm-nvidia__DeepSeek-R1-NVFP4-tp_4-isl_4-osl_1024-b_1-mode_moe_fp4_trtllm_fa_mla_b200"
SGLANG_TEST_DIR = "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_profile/results/sgl/nvidia__DeepSeek-R1-NVFP4/test-sgl-nvidia__DeepSeek-R1-NVFP4-tp_4-isl_4-osl_1024-b_1-mode_none"
TRT_TEST_DIR = "/home/alexm-redhat/code/ai_auto_perf_analysis/auto_profile/results/trt/nvidia__DeepSeek-R1-NVFP4/test-trt-nvidia__DeepSeek-R1-NVFP4-tp_4-isl_4-osl_1024-b_1-mode_moe_trtllm_b200"

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
