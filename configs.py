from dataclasses import dataclass


@dataclass
class ClaudeConfig:
    model: str
    allowed_tools: list[str]
    perm_mode: str
    cwd: str


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


MODEL = "deepseek-ai/DeepSeek-R1-0528"
GPU_TYPE = "B200"

VLLM_TRACE_FILE = "vllm_1764877581490541553-rank-0.1764877934150253928.pt.trace.json.gz"
SGLANG_TRACE_FILE = "sglang_1764879711.240358-TP-0.trace.json.gz"
