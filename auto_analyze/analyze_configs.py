from gen_configs import ClaudeConfig

@dataclass
class AnalyzeConfig:
    model: str
    precision: str
    gpu_type: str
    framework_name: str
    framework_code: str
    framework_model_code: str
    trace_file: str
    gpu_ops_filter: str = ""

MODEL = "nvidia/DeepSeek-R1-NVFP4"
PRECISION = "NVFP4"
GPU_TYPE = "B200"

VLLM = "vllm"
SGLANG = "sglang"

VLLM_TRACE_FILE = (
    "{}_1767817178193635761-rank-0.1767818040975629317.pt.trace.json.gz".format(VLLM)
)
SGLANG_TRACE_FILE = "{}_1767821176.8858435-TP-0.trace.json.gz".format(SGLANG)

claude_config = ClaudeConfig(
    model="claude-opus-4-6",
    # model="claude-opus-4-5",
    allowed_tools=["Read", "Write", "Bash"],
    perm_mode="acceptEdits",  # "bypassPermissions",
    cwd="./perf_cmp_dsr1_nvfp4",
)

analyze_configs = [
    AnalyzeConfig(
        model=MODEL,
        precision=PRECISION,
        gpu_type=GPU_TYPE,
        framework_name=VLLM,
        framework_code="./{}".format(VLLM),
        framework_model_code="./{}/vllm/model_executor/models/deepseek_v2.py".format(
            VLLM
        ),
        trace_file=VLLM_TRACE_FILE,
        gpu_ops_filter="*execute_new*",
    ),
    AnalyzeConfig(
        model=MODEL,
        precision=PRECISION,
        gpu_type=GPU_TYPE,
        framework_name=SGLANG,
        framework_code="./{}".format(SGLANG),
        framework_model_code="./{}/python/sglang/srt/models/deepseek_v2.py".format(
            SGLANG
        ),
        trace_file=SGLANG_TRACE_FILE,
    ),
]
