from dataclasses import dataclass
from claude_utils import ClaudeConfig

@dataclass
class AnalyzeConfig:
    model: str
    precision: str
    gpu_type: str
    framework_name: str
    framework_code: str
    framework_model_code: str
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

# VLLM_TRACE_FILE = (
#     "{}_1767817178193635761-rank-0.1767818040975629317.pt.trace.json.gz".format(VLLM)
# )
# SGLANG_TRACE_FILE = "{}_1767821176.8858435-TP-0.trace.json.gz".format(SGLANG)

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
        framework_code="./source_code/vllm",
        framework_model_code="./source_code/vllm/vllm/model_executor/models/deepseek_v2.py",
        test_dir=VLLM_TEST_DIR,
        gpu_ops_filter="*execute_new*",
    ),
    # AnalyzeConfig(
    #     model=MODEL,
    #     precision=PRECISION,
    #     gpu_type=GPU_TYPE,
    #     framework_name=SGLANG,
    #     framework_code="./source_code/sglang",
    #     framework_model_code="./source_code/sglang/python/sglang/srt/models/deepseek_v2.py",
    #     test_dir=SGLANG_TEST_DIR,
    # ),
    # AnalyzeConfig(
    #     model=MODEL,
    #     precision=PRECISION,
    #     gpu_type=GPU_TYPE,
    #     framework_name=TRT,
    #     framework_code="./source_code/TensorRT-LLM",
    #     framework_model_code="./source_code/TensorRT-LLM/python/sglang/srt/models/deepseek_v2.py",
    #     test_dir=SGLANG_TEST_DIR,
    # ),
]
